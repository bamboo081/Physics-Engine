"""Worker tests with real synthetic video and deterministic external boundaries."""

import importlib.util
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("PySide6")
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from physics_motion.calibration import Calibration
from physics_motion.tracking import Detection


def capture_module():
    assert importlib.util.find_spec("physics_motion.capture") is not None, (
        "Video capture worker has not been extracted"
    )
    from physics_motion import capture

    return capture


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def until(app, predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    app.processEvents()
    assert predicate(), "Worker condition was not reached before timeout"


def close_worker(app, worker):
    worker.stop_camera()
    assert worker.wait(3000), "Worker did not stop cooperatively"
    app.processEvents()


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "marker.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (180, 120))
    assert writer.isOpened(), "OpenCV MJPG encoder is required for this integration test"
    for index in range(8):
        frame = np.zeros((120, 180, 3), np.uint8)
        cv2.circle(frame, (25 + 4 * index, 35 + 2 * index), 9, (0, 0, 240), -1)
        writer.write(frame)
    writer.release()
    return path


def calibration(size=(180, 120)):
    return Calibration.from_points((20, 100), (120, 100), 1.0, size)


def collect(worker):
    events = {
        "frames": [],
        "sizes": [],
        "samples": [],
        "records": [],
        "errors": [],
        "targets": [],
        "messages": [],
    }
    worker.frame_ready.connect(events["frames"].append)
    worker.camera_ready.connect(lambda w, h: events["sizes"].append((w, h)))
    worker.sample_ready.connect(events["samples"].append)
    worker.recording_finished.connect(events["records"].append)
    worker.error.connect(events["errors"].append)
    worker.target_ready.connect(events["targets"].append)
    worker.message.connect(events["messages"].append)
    return events


def test_file_freezes_preview_then_uses_media_time_and_flushes_at_eof(app, video, monkeypatch):
    capture = capture_module()
    from physics_motion.detection import ColorMarkerDetector

    original = ColorMarkerDetector.detect

    def delayed_detect(self, frame, class_id=32):
        time.sleep(0.015)  # Inference time intentionally differs from 40 ms media time.
        return original(self, frame, class_id)

    monkeypatch.setattr(ColorMarkerDetector, "detect", delayed_detect)
    worker = capture.CameraWorker(video_path=video, tracker="color")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: len(events["frames"]) == 1)
        QTest.qWait(100)
        app.processEvents()
        assert len(events["frames"]) == 1
        assert events["samples"] == []
        worker.select_target(25, 35)
        until(app, lambda: events["targets"] and events["targets"][-1])
        worker.begin_tracking(calibration(), 32)
        until(app, lambda: not worker.isRunning())
        assert events["errors"] == []
        assert len(events["records"]) == 1
        records = events["records"][0]
        assert records == events["samples"]
        assert len(records) == 8
        assert [r["Time"] for r in records] == pytest.approx(
            [0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28]
        )
        assert [r["Frame"] for r in records] == list(range(8))
        assert [r["TrackID"] for r in records] == [1] * 8
        assert [r["X"] for r in records] == pytest.approx(
            [0.25, 0.29, 0.33, 0.37, 0.41, 0.45, 0.49, 0.53], abs=0.015
        )
        assert [r["Y"] for r in records] == pytest.approx(
            [0.85, 0.83, 0.81, 0.79, 0.77, 0.75, 0.73, 0.71], abs=0.015
        )
    finally:
        close_worker(app, worker)


def test_media_clock_preserves_variable_frame_intervals():
    clock = capture_module().MediaClock(30)
    assert [
        clock.timestamp(i, milliseconds) for i, milliseconds in enumerate([0, 40, 110, 155])
    ] == pytest.approx([0, 0.04, 0.11, 0.155])
    assert clock.last_method == "media_pts"


def test_media_clock_falls_back_explicitly_when_backend_timestamps_repeat():
    clock = capture_module().MediaClock(25)
    assert [clock.timestamp(i, 0) for i in range(4)] == pytest.approx([0, 0.04, 0.08, 0.12])
    assert clock.fallback_used
    assert clock.last_method == "frame_fps"
    clock = capture_module().MediaClock(float("nan"))
    clock.timestamp(0, 0)
    with pytest.raises(ValueError, match="timestamp|frame rate|timing"):
        clock.timestamp(1, 0)


@pytest.mark.parametrize(
    "positions, expected",
    [
        ([float("nan"), 1040, 1080, 1120], [0, 0.04, 0.08, 0.12]),
        ([-1, float("nan"), 2080, 2130], [0, 0.04, 0.08, 0.13]),
    ],
)
def test_missing_initial_media_time_anchors_later_pts_to_nominal_elapsed_time(positions, expected):
    clock = capture_module().MediaClock(25)
    observed = [clock.timestamp(index, value) for index, value in enumerate(positions)]
    assert observed == pytest.approx(expected)
    assert clock.fallback_used
    assert clock.last_method == "media_pts"


class ControlledDetector:
    def __init__(self, block_at=None, fail_at=None):
        self.calls = 0
        self.block_at = block_at
        self.fail_at = fail_at
        self.entered = threading.Event()
        self.release = threading.Event()

    def detect(self, frame, class_id=32):
        self.calls += 1
        if self.calls == self.block_at:
            self.entered.set()
            assert self.release.wait(3), "Test did not release detector"
        if self.calls == self.fail_at:
            raise RuntimeError("Deliberate external detector failure")
        return [Detection((20, 30, 30, 40), 0.95)]


def test_stop_during_inference_keeps_inflight_frame_and_finishes_once(app, video, monkeypatch):
    capture = capture_module()
    detector = ControlledDetector(block_at=2)
    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: detector)
    worker = capture.CameraWorker(video_path=video, tracker="yolo")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration(), 32)
        until(app, detector.entered.is_set)
        worker.stop_tracking()
        detector.release.set()
        until(app, lambda: bool(events["records"]))
        assert events["errors"] == []
        assert len(events["records"]) == 1
        assert len(events["records"][0]) == 2
        assert events["records"][0] == events["samples"]
        assert worker.isRunning()  # File stays paused for another run.
        worker.stop_camera()
        until(app, lambda: not worker.isRunning())
        assert len(events["records"]) == 1
    finally:
        detector.release.set()
        close_worker(app, worker)


def test_stop_and_immediate_restart_do_not_replace_previous_recording(app, video, monkeypatch):
    capture = capture_module()
    detector = ControlledDetector(block_at=2)
    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: detector)
    worker = capture.CameraWorker(video_path=video, tracker="yolo")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration(), 32)
        until(app, detector.entered.is_set)
        worker.stop_tracking()
        worker.begin_tracking(calibration(), 32)
        detector.release.set()
        until(app, lambda: not worker.isRunning())
        assert events["errors"] == []
        assert [len(records) for records in events["records"]] == [2, 8]
        assert events["records"][0][0]["Time"] == 0
        assert events["records"][1][0]["Time"] == 0
        assert events["samples"] == events["records"][0] + events["records"][1]
    finally:
        detector.release.set()
        close_worker(app, worker)


def test_detector_failure_surfaces_error_and_preserves_accepted_samples(app, video, monkeypatch):
    capture = capture_module()
    detector = ControlledDetector(fail_at=3)
    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: detector)
    worker = capture.CameraWorker(video_path=video, tracker="yolo")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration(), 32)
        until(app, lambda: bool(events["errors"]) and bool(events["records"]))
        assert "Deliberate external detector failure" in events["errors"][0]
        assert len(events["records"]) == 1
        assert len(events["records"][0]) == 2
        assert events["records"][0] == events["samples"]
    finally:
        close_worker(app, worker)
    assert len(events["records"]) == 1


class ChangingSizeCapture:
    def __init__(self):
        self.frames = [
            np.zeros((120, 180, 3), np.uint8),
            np.zeros((120, 180, 3), np.uint8),
            np.zeros((240, 360, 3), np.uint8),
        ]
        self.index = -1

    def isOpened(self):
        return True

    def read(self):
        self.index += 1
        if self.index == len(self.frames):
            return False, None
        return True, self.frames[self.index].copy()

    def get(self, field):
        return {
            cv2.CAP_PROP_FPS: 25,
            cv2.CAP_PROP_FRAME_COUNT: 3,
            cv2.CAP_PROP_POS_MSEC: self.index * 40,
        }.get(field, 0)

    def release(self):
        pass


def test_resolution_change_invalidates_calibration_before_any_bad_sample(app, monkeypatch):
    capture = capture_module()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: ChangingSizeCapture())
    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: ControlledDetector())
    worker = capture.CameraWorker(video_path="controlled.avi", tracker="yolo")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration(), 32)
        until(app, lambda: bool(events["records"]))
        assert len(events["records"][0]) == 2
        assert events["sizes"] == [(180, 120), (360, 240)]
        assert any(
            "resolution" in error.lower() or "size" in error.lower() for error in events["errors"]
        )
        assert events["samples"] == events["records"][0]
    finally:
        close_worker(app, worker)


def test_mismatched_calibration_is_rejected_without_observations(app, video):
    capture = capture_module()
    worker = capture.CameraWorker(video_path=video, tracker="color")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.select_target(25, 35)
        until(app, lambda: events["targets"] and events["targets"][-1])
        worker.begin_tracking(calibration((640, 480)), 32)
        until(app, lambda: bool(events["errors"]))
        assert events["samples"] == []
        assert len(events["records"]) == 1
    finally:
        close_worker(app, worker)


def test_visible_ambiguity_pauses_file_until_click_without_losing_frame_zero(
    app, video, monkeypatch
):
    capture = capture_module()

    class TwoTargets:
        def detect(self, frame, class_id=32):
            return [Detection((20, 30, 30, 40), 0.95), Detection((90, 30, 100, 40), 0.99)]

    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: TwoTargets())
    worker = capture.CameraWorker(video_path=video, tracker="yolo")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration(), 32)
        until(app, lambda: any("paused" in message for message in events["messages"][1:]))
        assert events["samples"] == []
        assert events["records"] == []
        worker.select_target(25, 35)
        until(app, lambda: not worker.isRunning())
        records = events["records"][0]
        assert len(records) == 8
        assert records[0]["Frame"] == 0
        assert records[0]["Time"] == 0
        assert all(record["X"] == 0.25 for record in records)
        metadata = worker.snapshot_metadata()
        assert metadata["processed_frames"] == 8
        assert metadata["missed_frames"] == 0
        assert metadata["finish_reason"] == "end_of_video"
        metadata["warnings"].append("external mutation")
        assert "external mutation" not in worker.snapshot_metadata()["warnings"]
    finally:
        close_worker(app, worker)


def test_camera_uses_monotonic_acquisition_time_and_releases_capture(app, monkeypatch):
    capture = capture_module()

    class Camera:
        released = False

        def isOpened(self):
            return True

        def set(self, field, value):
            return True

        def get(self, field):
            return 25 if field == cv2.CAP_PROP_FPS else 0

        def read(self):
            time.sleep(0.01)
            return True, np.zeros((120, 180, 3), np.uint8)

        def release(self):
            self.released = True

    source = Camera()
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: source)
    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: ControlledDetector())
    worker = capture.CameraWorker(tracker="yolo")
    events = collect(worker)
    before = time.monotonic()
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration())
        until(app, lambda: len(events["samples"]) >= 3)
        worker.stop_tracking()
        until(app, lambda: bool(events["records"]))
        metadata = worker.snapshot_metadata()
        previews = len(events["frames"])
        until(app, lambda: len(events["frames"]) >= previews + 3)
        assert worker.snapshot_metadata() == metadata
        worker.stop_camera()
        until(app, lambda: not worker.isRunning())
        records = events["records"][0]
        assert records == events["samples"]
        assert records[0]["Time"] == 0
        assert records[0]["SourceTime"] >= before
        assert all(item["TimingSource"] == "monotonic" for item in records)
        assert np.diff([item["SourceTime"] for item in records]).min() > 0
        assert source.released
    finally:
        close_worker(app, worker)


def test_invalid_video_source_finishes_with_clear_error(app, tmp_path):
    worker = capture_module().CameraWorker(video_path=tmp_path / "missing.mp4")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: not worker.isRunning())
        assert events["errors"] and "Could not open" in events["errors"][0]
        assert events["samples"] == []
    finally:
        close_worker(app, worker)


def test_stop_queued_during_model_loading_does_not_begin_inference(app, video, monkeypatch):
    capture = capture_module()
    loading = threading.Event()
    release = threading.Event()

    def load_detector(self):
        loading.set()
        assert release.wait(3)
        return ControlledDetector()

    monkeypatch.setattr(capture.CameraWorker, "_make_detector", load_detector)
    worker = capture.CameraWorker(video_path=video, tracker="yolo")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration())
        until(app, loading.is_set)
        worker.stop_tracking()
        release.set()
        until(app, lambda: bool(events["records"]))
        assert events["records"] == [[]]
        assert events["samples"] == []
    finally:
        release.set()
        close_worker(app, worker)


def test_camera_read_failure_breaks_identity_even_before_time_gap_limit(app, monkeypatch):
    capture = capture_module()

    class Camera:
        fail_next = False

        def isOpened(self):
            return True

        def set(self, field, value):
            return True

        def get(self, field):
            return 25 if field == cv2.CAP_PROP_FPS else 0

        def read(self):
            time.sleep(0.01)
            if self.fail_next:
                self.fail_next = False
                return False, None
            return True, np.zeros((120, 180, 3), np.uint8)

        def release(self):
            pass

    source = Camera()

    class ReadFailureDetector(ControlledDetector):
        def detect(self, frame, class_id=32):
            result = super().detect(frame, class_id)
            if self.calls == 1:
                source.fail_next = True
            return result

    monkeypatch.setattr(cv2, "VideoCapture", lambda index: source)
    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: ReadFailureDetector())
    worker = capture.CameraWorker(tracker="yolo")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration())
        until(app, lambda: len(events["samples"]) >= 3)
        worker.stop_camera()
        until(app, lambda: not worker.isRunning())
        records = events["records"][0]
        assert [record["TrackID"] for record in records[:3]] == [1, 2, 2]
        assert worker.snapshot_metadata()["failed_reads"] == 1
    finally:
        close_worker(app, worker)


def test_preview_seek_finds_late_marker_and_starts_at_selected_media_frame(app, tmp_path):
    path = tmp_path / "late-marker.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (180, 120))
    assert writer.isOpened()
    for index in range(8):
        frame = np.zeros((120, 180, 3), np.uint8)
        if index >= 3:
            cv2.circle(frame, (25 + 4 * index, 35 + 2 * index), 9, (0, 0, 240), -1)
        writer.write(frame)
    writer.release()
    worker = capture_module().CameraWorker(video_path=path, tracker="color")
    events = collect(worker)
    seeks = []
    worker.seek_finished.connect(lambda frame, total: seeks.append((frame, total)))
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.seek_frame(3)
        until(app, lambda: bool(seeks))
        assert seeks == [(3, 8)]
        assert events["targets"][-1] is False
        assert events["samples"] == []
        worker.select_target(37, 41)
        until(app, lambda: events["targets"][-1] is True)
        worker.begin_tracking(calibration())
        until(app, lambda: not worker.isRunning())
        assert events["errors"] == []
        records = events["records"][0]
        assert [record["Frame"] for record in records] == [3, 4, 5, 6, 7]
        assert [record["SourceTime"] for record in records] == pytest.approx(
            [0.12, 0.16, 0.20, 0.24, 0.28]
        )
        assert [record["Time"] for record in records] == pytest.approx([0, 0.04, 0.08, 0.12, 0.16])
        assert worker.snapshot_metadata()["start_frame"] == 3
    finally:
        close_worker(app, worker)


def test_preview_seek_backward_resets_target_and_preserves_original_media_clock(app, video):
    worker = capture_module().CameraWorker(video_path=video, tracker="color")
    events = collect(worker)
    seeks = []
    worker.seek_finished.connect(lambda frame, total: seeks.append((frame, total)))
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.select_target(25, 35)
        until(app, lambda: events["targets"] and events["targets"][-1])
        worker.seek_frame(6)
        until(app, lambda: len(seeks) == 1)
        assert seeks[-1] == (6, 8)
        assert events["targets"][-1] is False
        worker.seek_frame(2)
        until(app, lambda: len(seeks) == 2)
        assert seeks[-1] == (2, 8)
        worker.select_target(33, 39)
        until(app, lambda: events["targets"][-1] is True)
        worker.begin_tracking(calibration())
        until(app, lambda: not worker.isRunning())
        assert events["errors"] == []
        records = events["records"][0]
        assert [record["Frame"] for record in records] == [2, 3, 4, 5, 6, 7]
        assert [record["SourceTime"] for record in records] == pytest.approx(
            [0.08, 0.12, 0.16, 0.20, 0.24, 0.28]
        )
        assert records[0]["Time"] == 0
    finally:
        close_worker(app, worker)


def test_preview_seek_past_eof_keeps_last_frame_and_can_return_to_an_earlier_frame(app, video):
    worker = capture_module().CameraWorker(video_path=video)
    events = collect(worker)
    seeks, progress = [], []
    worker.seek_finished.connect(lambda frame, total: seeks.append((frame, total)))
    worker.progress.connect(lambda frame, total: progress.append((frame, total)))
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.seek_frame(1000)
        until(app, lambda: bool(seeks))
        assert seeks == [(7, 8)]
        assert progress[-1] == (8, 8)
        assert worker.isRunning()
        assert events["records"] == []
        worker.seek_frame(1)
        until(app, lambda: len(seeks) == 2)
        assert seeks[-1] == (1, 8)
        assert worker.isRunning()
        assert events["errors"] == []
    finally:
        close_worker(app, worker)


def test_preview_seek_during_recording_does_not_skip_measurements(app, video, monkeypatch):
    capture = capture_module()
    detector = ControlledDetector(block_at=2)
    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: detector)
    worker = capture.CameraWorker(video_path=video, tracker="yolo")
    events = collect(worker)
    seeks = []
    worker.seek_finished.connect(lambda frame, total: seeks.append((frame, total)))
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.begin_tracking(calibration())
        until(app, detector.entered.is_set)
        worker.seek_frame(6)
        worker.stop_tracking()
        detector.release.set()
        until(app, lambda: bool(events["records"]))
        assert seeks == [(1, 8)]
        assert [record["Frame"] for record in events["records"][0]] == [0, 1]
        assert events["errors"] == []
    finally:
        detector.release.set()
        close_worker(app, worker)


@pytest.mark.parametrize("frame", [-1, True, 1.5, "1"])
def test_invalid_preview_frame_is_rejected_before_queuing(frame):
    worker = capture_module().CameraWorker(video_path="unused.avi")
    with pytest.raises(ValueError, match="frame|integer"):
        worker.seek_frame(frame)


def test_stopped_video_retry_retains_chosen_start_frame(app, video, monkeypatch):
    capture = capture_module()
    detector = ControlledDetector(block_at=2)
    monkeypatch.setattr(capture.CameraWorker, "_make_detector", lambda self: detector)
    worker = capture.CameraWorker(video_path=video, tracker="yolo")
    events = collect(worker)
    seeks = []
    worker.seek_finished.connect(lambda frame, total: seeks.append((frame, total)))
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.seek_frame(3)
        until(app, lambda: bool(seeks))
        worker.begin_tracking(calibration())
        until(app, detector.entered.is_set)
        worker.stop_tracking()
        worker.begin_tracking(calibration())
        detector.release.set()
        until(app, lambda: not worker.isRunning())
        assert events["errors"] == []
        assert [[record["Frame"] for record in records] for records in events["records"]] == [
            [3, 4],
            [3, 4, 5, 6, 7],
        ]
        assert [records[0]["SourceTime"] for records in events["records"]] == pytest.approx(
            [0.12, 0.12]
        )
        assert [records[0]["Time"] for records in events["records"]] == [0, 0]
    finally:
        detector.release.set()
        close_worker(app, worker)


def test_closing_during_long_preview_seek_stops_at_the_current_decode(app, monkeypatch):
    capture = capture_module()

    class LongVideo:
        def __init__(self):
            self.index = -1
            self.entered = threading.Event()
            self.resume = threading.Event()
            self.released = False

        def isOpened(self):
            return True

        def get(self, field):
            return {
                cv2.CAP_PROP_FPS: 25,
                cv2.CAP_PROP_FRAME_COUNT: 10000,
                cv2.CAP_PROP_POS_MSEC: self.index * 40,
            }.get(field, 0)

        def read(self):
            self.index += 1
            if self.index == 1:
                self.entered.set()
                assert self.resume.wait(3)
            return True, np.zeros((120, 180, 3), np.uint8)

        def release(self):
            self.released = True

    source = LongVideo()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: source)
    worker = capture.CameraWorker(video_path="controlled.avi")
    events = collect(worker)
    worker.start()
    try:
        until(app, lambda: bool(events["frames"]))
        worker.seek_frame(9999)
        until(app, source.entered.is_set)
        worker.stop_camera()
        source.resume.set()
        until(app, lambda: not worker.isRunning())
        assert events["samples"] == []
        assert events["errors"] == []
        assert source.released
        assert source.index == 1
    finally:
        source.resume.set()
        close_worker(app, worker)
