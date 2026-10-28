"""Worker-owned capture, detection and media timing for the desktop application.

All OpenCV/model operations occur in ``run`` or its worker-only helpers. Public
control methods enqueue commands; Qt signals are never emitted under that lock.
A stop completes the frame already in inference, then flushes every accepted
sample exactly once. File preview holds a chosen start frame for calibration.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from copy import deepcopy
from math import isfinite
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .calibration import Calibration
from .detection import ColorMarkerDetector, YoloDetector
from .tracking import Detection, TrackSelector


class MediaClock:
    """Prefer decoded presentation times; explicitly flag frame/FPS fallback.

    Time is relative to the first decoded frame. File processing speed never
    enters this calculation. Without usable PTS or FPS, analysis stops rather
    than inventing seconds from inference duration.
    """

    def __init__(self, fps: float):
        self.fps = float(fps) if isfinite(fps) and fps > 0 else None
        self._origin: float | None = None
        self._last: float | None = None
        self.fallback_used = False
        self.last_method = "media_pts"

    def timestamp(self, frame_index: int, position_msec: float) -> float:
        valid = isfinite(position_msec) and position_msec >= 0
        position = position_msec / 1000 if valid else None
        if self._last is None:
            if position is None and self.fps is None:
                raise ValueError("Video has no usable media timestamps or frame rate for timing")
            self._origin = position
            timestamp = 0.0
            use_fallback = position is None
        elif self._origin is None and position is not None:
            # Initial missing PTS must not make the backend's nonzero source
            # offset appear as elapsed time. Anchor the first usable PTS to
            # the nominal timeline already used for the preceding frames.
            timestamp = max(frame_index / self.fps, self._last + 1 / self.fps)
            self._origin = position - timestamp
            use_fallback = True
        else:
            timestamp = position - self._origin if position is not None else -1.0
            use_fallback = timestamp <= self._last
            if use_fallback:
                if self.fps is None:
                    raise ValueError(
                        "Video timestamps repeat or decrease, and its frame rate is unavailable"
                    )
                timestamp = max(frame_index / self.fps, self._last + 1 / self.fps)
        self.last_method = "frame_fps" if use_fallback else "media_pts"
        self.fallback_used |= use_fallback
        self._last = timestamp
        return timestamp


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    camera_ready = Signal(int, int)
    message = Signal(str)
    sample_ready = Signal(object)
    recording_finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int)
    target_ready = Signal(bool)
    seek_finished = Signal(int, int)

    def __init__(
        self,
        camera_index: int = 0,
        model_name: str = "yolo11n.pt",
        device: str = "cpu",
        confidence: float = 0.35,
        *,
        video_path: str | Path | None = None,
        tracker: str = "color",
        parent=None,
    ):
        super().__init__(parent)
        if not isinstance(camera_index, int) or isinstance(camera_index, bool) or camera_index < 0:
            raise ValueError("Camera index must be a nonnegative integer")
        if tracker not in {"color", "yolo"}:
            raise ValueError("Tracker must be 'color' or 'yolo'")
        if not isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Detection confidence must be finite and between zero and one")
        self.camera_index = camera_index
        self.video_path = str(Path(video_path).expanduser()) if video_path is not None else None
        self.model_name, self.device, self.confidence = model_name, device, confidence
        self.tracker = tracker
        self._condition = threading.Condition()
        self._commands: deque[tuple] = deque()
        self._closing = threading.Event()
        self._metadata: dict = {}
        self._completed_metadata: dict | None = None
        self._cap = None
        self._detector = None
        self._frame = None
        self._frame_size: tuple[int, int] | None = None
        self._frame_index = -1
        self._preview_start_index = 0
        self._timestamp = 0.0
        self._clock: MediaClock | None = None
        self._total_frames = 0
        self._recording_active = False
        self._calibration: Calibration | None = None
        self._class_id = 32
        self._selector = TrackSelector()
        self._selection_point: tuple[float, float] | None = None
        self._samples: list[dict] = []
        self._start_time: float | None = None
        self._needs_processing = False
        self._paused_for_selection = False
        self._last_status = ""
        self._last_preview = 0.0
        self._read_failures = 0
        self._last_processed_index: int | None = None
        self._last_frame_missing = False

    def _enqueue(self, *command) -> None:
        with self._condition:
            if not self._closing.is_set():
                self._commands.append(command)
                self._condition.notify_all()

    def begin_tracking(self, calibration: Calibration, coco_class: int = 32) -> None:
        if not isinstance(calibration, Calibration):
            raise ValueError("A valid calibration is required before tracking")
        if not isinstance(coco_class, int) or isinstance(coco_class, bool) or coco_class < 0:
            raise ValueError("Detector class must be a nonnegative integer")
        self._enqueue("start", calibration, coco_class)

    def stop_tracking(self) -> None:
        self._enqueue("stop")

    def select_target(self, x: float, y: float) -> None:
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Selection point must be finite")
        self._enqueue("select", float(x), float(y))

    def seek_frame(self, frame_index: int) -> None:
        """Choose a zero-based video start frame while recording is stopped.

        Sequential decoding preserves original-file timestamps, including for
        variable-rate footage. Completion reports the actual frame and total;
        requests beyond the decodable end stop on the last available frame.
        """
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise ValueError("Preview frame must be a nonnegative integer")
        if self.video_path is None:
            raise ValueError("Preview frames can only be selected in a video file")
        self._enqueue("seek", frame_index)

    def stop_camera(self) -> None:
        self._closing.set()
        with self._condition:
            self._condition.notify_all()

    def snapshot_metadata(self) -> dict:
        """Return capture diagnostics suitable for the exported run context."""
        with self._condition:
            return deepcopy(
                self._completed_metadata if self._completed_metadata is not None else self._metadata
            )

    def _warning(self, text: str) -> None:
        with self._condition:
            warnings = self._metadata.setdefault("warnings", [])
            if text in warnings:
                return
            warnings.append(text)
        self.message.emit(text)

    def _status(self, text: str) -> None:
        if text != self._last_status:
            self._last_status = text
            self.message.emit(text)

    def _make_detector(self):
        if self.tracker == "color":
            return ColorMarkerDetector()
        self.message.emit("Loading optional YOLO detector; first use may download model weights")
        return YoloDetector(self.model_name, self.device, self.confidence)

    def _ensure_detector(self):
        if self._detector is None:
            self._detector = self._make_detector()
        return self._detector

    def _open_capture(self) -> None:
        source = self.video_path if self.video_path is not None else self.camera_index
        self._cap = self._cv2.VideoCapture(source)
        if not self._cap.isOpened():
            kind = f"video {self.video_path}" if self.video_path else f"camera {self.camera_index}"
            raise RuntimeError(
                f"Could not open {kind}. Check the file, device, permissions, and other camera apps."
            )
        if self.video_path is None:
            self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, 960)
            self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, 540)
        fps = float(self._cap.get(self._cv2.CAP_PROP_FPS))
        count = float(self._cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
        self._total_frames = int(count) if isfinite(count) and count > 0 else 0
        self._clock = MediaClock(fps) if self.video_path is not None else None
        self._frame_index = -1
        self._read_failures = 0
        with self._condition:
            self._metadata = {
                "source_type": "video" if self.video_path is not None else "camera",
                "video_path": self.video_path,
                "camera_index": self.camera_index,
                "tracker": self.tracker,
                "media_fps": fps if isfinite(fps) and fps > 0 else None,
                "frame_size": None,
                "total_frames": self._total_frames,
                "decoded_frames": 0,
                "processed_frames": 0,
                "accepted_samples": 0,
                "missed_frames": 0,
                "failed_reads": 0,
                "timing_sources": [],
                "warnings": [],
                "finish_reason": None,
            }

    def _read_frame(self, *, preview: bool = False) -> bool:
        ok, frame = self._cap.read()
        captured_at = time.monotonic()  # Read completion, before detection/inference.
        if not ok or frame is None or frame.size == 0:
            self._read_failures += 1
            if self.video_path is not None:
                if self._frame_index < 0:
                    raise RuntimeError("Video contains no decodable frames")
                if self._total_frames and self._frame_index + 1 < self._total_frames - 1:
                    self._warning(
                        "Video ended before its reported frame count; the file may be truncated or damaged"
                    )
                if not preview:
                    self._finish_recording("end_of_video")
                    self.message.emit(
                        "End of video; all accepted observations are ready for analysis"
                    )
                return False
            if self._recording_active:
                self._selector.choose([], captured_at)
                with self._condition:
                    self._metadata["failed_reads"] += 1
            if self._read_failures >= 15:
                raise RuntimeError(
                    "Camera stopped delivering frames. Reopen the camera and recalibrate."
                )
            with self._condition:
                self._condition.wait(timeout=0.075)
            return True
        self._read_failures = 0
        self._frame_index += 1
        self._frame = frame
        height, width = frame.shape[:2]
        size = width, height
        if self._clock is not None:
            self._timestamp = self._clock.timestamp(
                self._frame_index, float(self._cap.get(self._cv2.CAP_PROP_POS_MSEC))
            )
            if self._clock.last_method == "frame_fps":
                self._warning(
                    "Video media timestamps were unavailable or non-increasing; affected frames use nominal FPS timing. Variable-rate footage may be inaccurate."
                )
        else:
            self._timestamp = captured_at
        with self._condition:
            self._metadata["decoded_frames"] = self._metadata.get("decoded_frames", 0) + 1
        if self._frame_size != size:
            changed = self._frame_size is not None
            self._frame_size = size
            self.camera_ready.emit(width, height)
            if changed:
                self._selection_point = None
                self.target_ready.emit(False)
                if self.tracker == "color":
                    self._detector = None
                if self._recording_active:
                    text = "Source resolution changed; recording stopped before using the old calibration. Recalibrate the new frame."
                    self._warning(text)
                    self.error.emit(text)
                    self._finish_recording("resolution_changed")
                self._calibration = None
            if not self._recording_active:
                with self._condition:
                    self._metadata["frame_size"] = list(size)
        self._needs_processing = self._recording_active
        return True

    def _move_to_frame(self, frame_index: int) -> bool:
        """Decode to a preview frame without relying on backend random seeking."""
        if frame_index < self._frame_index:
            self._cap.release()
            self._open_capture()
        while self._frame_index < frame_index and not self._closing.is_set():
            if not self._read_frame(preview=True):
                break
        return self._frame_index >= 0 and not self._closing.is_set()

    def _seek_preview(self, frame_index: int) -> None:
        try:
            if self._recording_active:
                self._status("Stop tracking before choosing a video start frame")
                return
            self._selection_point = None
            self._selector = TrackSelector()
            self._paused_for_selection = False
            self._needs_processing = False
            if self.tracker == "color":
                self._detector = None
            self.target_ready.emit(False)
            self._status(f"Finding video frame {frame_index}…")
            if self._move_to_frame(frame_index):
                self._preview_start_index = self._frame_index
                self._render(force=True)
                prefix = "End of video. " if self._frame_index < frame_index else ""
                self._status(
                    f"{prefix}Preview at frame {self._frame_index}. Select the target before tracking."
                )
        finally:
            if not self._closing.is_set():
                self.seek_finished.emit(self._frame_index, self._total_frames)

    def _finish_recording(self, reason: str) -> None:
        if not self._recording_active:
            return
        self._recording_active = False
        self._paused_for_selection = False
        with self._condition:
            self._metadata["finish_reason"] = reason
            self._metadata["accepted_samples"] = len(self._samples)
            self._completed_metadata = deepcopy(self._metadata)
        # Copies isolate storage from GUI slots that might mutate their payload.
        recorded = [dict(sample) for sample in self._samples]
        self.recording_finished.emit(recorded)

    def _start_recording(self, calibration: Calibration, class_id: int) -> None:
        self._finish_recording("restarted")
        restarting_video = (
            self.video_path is not None and self._frame_index != self._preview_start_index
        )
        if restarting_video:
            if not self._move_to_frame(self._preview_start_index):
                return
        self._samples = []
        self._selector = TrackSelector()
        self._last_processed_index = None
        self._last_frame_missing = False
        self._start_time = None
        self._recording_active = True
        self._paused_for_selection = False
        self._calibration, self._class_id = calibration, class_id
        with self._condition:
            self._completed_metadata = None
            self._metadata.update(
                {
                    "processed_frames": 0,
                    "accepted_samples": 0,
                    "missed_frames": 0,
                    "failed_reads": 0,
                    "timing_sources": [],
                    "finish_reason": None,
                    "frame_size": list(self._frame_size) if self._frame_size else None,
                    "start_frame": self._frame_index,
                }
            )
        try:
            if self._frame_size != (calibration.frame_width, calibration.frame_height):
                raise ValueError(
                    "Calibration frame size differs from the source; recalibrate before tracking"
                )
            detector = self._ensure_detector()
            if self.tracker == "color" and not detector.selected:
                raise ValueError("Select the interior of a colored marker before starting")
            if self._selection_point is not None and not restarting_video:
                self._selector.request_selection(*self._selection_point)
            self._selection_point = None
            self._needs_processing = self.video_path is not None
            self._last_status = ""
            self._status("Tracking started")
        except Exception as exc:
            self.error.emit(f"Tracking could not start: {exc}")
            self._finish_recording("start_error")

    def _select(self, x: float, y: float) -> None:
        if self._frame is None:
            return
        try:
            if self.tracker == "color":
                detector = self._ensure_detector()
                detector.select_color(self._frame, x, y)
                self.target_ready.emit(True)
                self._status("Marker color selected. Start tracking or click to change the target.")
            if self._recording_active:
                self._selector.request_selection(x, y)
                if self._paused_for_selection:
                    self._paused_for_selection = False
                    self._needs_processing = True
            else:
                self._selection_point = (x, y)
                if self.tracker == "color":
                    self._render(self._detector.detect(self._frame), force=True)
        except Exception as exc:
            self.target_ready.emit(False)
            # A failed click must not silently retain a previous target identity.
            self._selector.lost = True
            self._selection_point = None
            self._status(f"Marker selection failed: {exc}")

    def _process_commands(self) -> None:
        while True:
            if self._closing.is_set():
                return
            with self._condition:
                if not self._commands:
                    return
                command = self._commands.popleft()
            if command[0] == "start":
                self._start_recording(command[1], command[2])
            elif command[0] == "stop":
                self._finish_recording("stopped")
                self._render(force=True)
            elif command[0] == "select":
                self._select(command[1], command[2])
            elif command[0] == "seek":
                self._seek_preview(command[1])

    def _process_frame(self) -> None:
        self._needs_processing = False
        try:
            if self._frame_size != (self._calibration.frame_width, self._calibration.frame_height):
                raise ValueError("Source frame size differs from calibration")
            if self._start_time is None:
                self._start_time = self._timestamp
            detections = self._ensure_detector().detect(self._frame, self._class_id)
            chosen, status = self._selector.choose(detections, self._timestamp)
            timing_source = self._clock.last_method if self._clock else "monotonic"
            new_frame = self._frame_index != self._last_processed_index
            with self._condition:
                if new_frame:
                    self._metadata["processed_frames"] += 1
                    self._metadata["missed_frames"] += int(chosen is None)
                elif self._last_frame_missing and chosen is not None:
                    self._metadata["missed_frames"] -= 1
                if timing_source not in self._metadata["timing_sources"]:
                    self._metadata["timing_sources"].append(timing_source)
            self._last_processed_index = self._frame_index
            self._last_frame_missing = chosen is None
            if chosen is not None:
                x_m, y_m = self._calibration.pixel_to_meters(*chosen.center)
                sample = {
                    "Time": self._timestamp - self._start_time,
                    "X": x_m,
                    "Y": y_m,
                    "Confidence": chosen.confidence,
                    "TrackID": self._selector.track_id,
                    "Frame": self._frame_index,
                    "SourceTime": self._timestamp,
                    "TimingSource": timing_source,
                }
                self._samples.append(sample)
                with self._condition:
                    self._metadata["accepted_samples"] = len(self._samples)
                self.sample_ready.emit(dict(sample))
            else:
                # Hold visible ambiguity for an actionable click; empty frames
                # continue through occlusion until the object is visible again.
                if self.video_path is not None and detections and self._selector.lost:
                    self._paused_for_selection = True
                    status += " (video paused)"
            self._status(status)
            self._render(detections, chosen, force=self._paused_for_selection)
        except Exception as exc:
            self._warning(f"Tracking stopped after a detector or processing error: {exc}")
            self.error.emit(f"Tracking failed: {exc}")
            self._finish_recording("processing_error")
            self._render(force=True)

    def _render(
        self,
        detections: list[Detection] | None = None,
        chosen: Detection | None = None,
        *,
        force=False,
    ) -> None:
        if self._frame is None:
            return
        now = time.monotonic()
        # Decode and analyze every frame; only the displayed preview is throttled.
        if not force and now - self._last_preview < 1 / 30:
            return
        self._last_preview = now
        display = self._frame.copy()
        for item in detections or []:
            x1, y1, x2, y2 = (round(value) for value in item.box)
            color = (75, 190, 75) if item is chosen else (220, 180, 70)
            self._cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            if item is chosen:
                self._cv2.circle(
                    display, tuple(round(value) for value in item.center), 3, color, -1
                )
        rgb = self._cv2.cvtColor(display, self._cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.frame_ready.emit(image)
        self.progress.emit(self._frame_index + 1, self._total_frames)

    def run(self) -> None:
        if self._closing.is_set():
            return
        try:
            import cv2

            self._cv2 = cv2
            self._open_capture()
            if not self._read_frame():
                return
            self._render(force=True)
            self.message.emit(
                "Video paused at its first frame; calibrate and select a marker"
                if self.video_path is not None
                else "Camera opened; calibrate and select a marker"
            )
            while not self._closing.is_set():
                self._process_commands()
                if self._closing.is_set():
                    break
                if self._recording_active and self._needs_processing:
                    self._process_frame()
                    continue
                if self.video_path is not None and (
                    not self._recording_active or self._paused_for_selection
                ):
                    with self._condition:
                        if not self._commands and not self._closing.is_set():
                            self._condition.wait(timeout=0.25)
                    continue
                if not self._read_frame():
                    self._render(force=True)
                    break
                if not self._recording_active:
                    self._render()
        except ImportError as exc:
            self.error.emit(
                f"Desktop capture dependency missing. Install physics-motion-analyzer[desktop]: {exc}"
            )
        except Exception as exc:
            self._warning(f"Capture stopped: {exc}")
            self.error.emit(f"Capture failed: {exc}")
        finally:
            self._finish_recording("source_closed" if self._closing.is_set() else "capture_error")
            if self._cap is not None:
                self._cap.release()
            self.message.emit("Video closed" if self.video_path is not None else "Camera closed")
