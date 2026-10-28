"""Optional detectors. Heavy libraries are loaded only when a detector is created.

The color detector measures the area centroid of each connected mask component.
Its confidence value is a deterministic mask match, not a calibrated probability.
YOLO is deliberately optional and is never imported for color tracking.
"""

from __future__ import annotations

from math import isfinite

import numpy as np

from .tracking import Detection


class ColorMarkerDetector:
    """Track a saturated marker color sampled by clicking its interior.

    Hue is circular (OpenCV's 0..179 scale). Saturation and brightness floors
    exclude neutral backgrounds; small components are discarded as image noise.
    Multiple similarly colored components are returned for explicit association.
    """

    def __init__(self, hue_tolerance: float = 12.0, min_area: int = 16):
        import cv2

        if not isfinite(hue_tolerance) or not 0 < hue_tolerance < 90:
            raise ValueError("Hue tolerance must be between zero and 90")
        if not isinstance(min_area, int) or isinstance(min_area, bool) or min_area < 1:
            raise ValueError("Marker minimum area must be a positive integer")
        self._cv2 = cv2
        self.hue_tolerance = hue_tolerance
        self.min_area = min_area
        self._hue: float | None = None
        self._saturation_floor = 60.0
        self._value_floor = 45.0
        self._kernel = np.ones((3, 3), np.uint8)

    @property
    def selected(self) -> bool:
        return self._hue is not None

    def _hsv(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8:
            raise ValueError("Color marker detector requires a nonempty 8-bit BGR frame")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2HSV)

    def select_color(self, frame: np.ndarray, x: float, y: float) -> None:
        self._hue = None
        height, width = frame.shape[:2]
        if not isfinite(x) or not isfinite(y) or not (0 <= x < width and 0 <= y < height):
            raise ValueError("Click a marker inside the video frame")
        hsv = self._hsv(frame)
        px, py = int(x), int(y)
        patch = hsv[max(0, py - 2) : min(height, py + 3), max(0, px - 2) : min(width, px + 3)]
        saturation = float(np.median(patch[:, :, 1]))
        brightness = float(np.median(patch[:, :, 2]))
        if saturation < 60 or brightness < 45:
            raise ValueError("Choose the interior of a bright, saturated color marker")
        # Circular mean prevents a red patch spanning hue 179/0 becoming cyan.
        colored = patch[(patch[:, :, 1] >= 60) & (patch[:, :, 2] >= 45)]
        angles = colored[:, 0].astype(float) * (2 * np.pi / 180)
        self._hue = (
            float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) * 180 / (2 * np.pi))
            % 180
        )
        self._saturation_floor = max(60.0, saturation * 0.35)
        self._value_floor = max(45.0, brightness * 0.2)

    def detect(self, frame: np.ndarray, class_id: int = 32) -> list[Detection]:
        if self._hue is None:
            return []
        hsv = self._hsv(frame)
        delta = np.abs(hsv[:, :, 0].astype(float) - self._hue)
        delta = np.minimum(delta, 180 - delta)
        mask = (
            (delta <= self.hue_tolerance)
            & (hsv[:, :, 1] >= self._saturation_floor)
            & (hsv[:, :, 2] >= self._value_floor)
        ).astype(np.uint8) * 255
        mask = self._cv2.morphologyEx(mask, self._cv2.MORPH_OPEN, self._kernel)
        count, _, stats, centers = self._cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections = []
        for index in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[index])
            if area >= self.min_area:
                detections.append(
                    Detection(
                        (x, y, x + width, y + height),
                        1.0,
                        centroid=tuple(float(value) for value in centers[index]),
                    )
                )
        return detections


class YoloDetector:
    """Small adapter around Ultralytics; importing this module does not load it."""

    def __init__(self, model_name: str, device: str, confidence: float):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO is optional. Install physics-motion-analyzer[desktop,detector], or select Color marker"
            ) from exc
        self.model = YOLO(model_name)
        self.device = device
        self.confidence = confidence

    def detect(self, frame: np.ndarray, class_id: int = 32) -> list[Detection]:
        predictions = self.model.predict(
            frame,
            verbose=False,
            conf=self.confidence,
            classes=[class_id],
            device=self.device,
            max_det=30,
        )
        detections = []
        for result in predictions:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                detections.append(Detection(xyxy, float(box.conf[0])))
        return detections
