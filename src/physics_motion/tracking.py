"""Conservative, independently testable single-object association.

This is not a multi-object identity tracker. Nearby competing detections and
long losses require a fresh click. Track IDs also separate shorter missing
intervals so downstream fitting cannot silently join unobserved motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    centroid: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        try:
            box = tuple(float(value) for value in self.box)
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("Detection coordinates and confidence must be numeric") from exc
        if len(box) != 4 or not all(isfinite(value) for value in box):
            raise ValueError("Detection box must contain four finite coordinates")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Detection box must have positive width and height")
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Detection confidence must be finite and between zero and one")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "confidence", confidence)
        if self.centroid is not None:
            try:
                centroid = tuple(float(value) for value in self.centroid)
            except (TypeError, ValueError) as exc:
                raise ValueError("Detection centroid must contain two finite coordinates") from exc
            if len(centroid) != 2 or not all(isfinite(value) for value in centroid):
                raise ValueError("Detection centroid must contain two finite coordinates")
            if not self.contains(*centroid):
                raise ValueError("Detection centroid must lie inside its box")
            object.__setattr__(self, "centroid", centroid)

    @property
    def center(self) -> tuple[float, float]:
        if self.centroid is not None:
            return self.centroid
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2


class TrackSelector:
    def __init__(
        self,
        max_speed_px_s: float = 750.0,
        max_gap_s: float = 0.75,
        ambiguity_px: float = 8.0,
    ):
        if not all(isfinite(value) and value > 0 for value in (max_speed_px_s, max_gap_s)):
            raise ValueError("Association speed and gap must be finite and positive")
        if not isfinite(ambiguity_px) or ambiguity_px < 0:
            raise ValueError("Association ambiguity margin must be finite and nonnegative")
        self.max_speed_px_s = max_speed_px_s
        self.max_gap_s = max_gap_s
        self.ambiguity_px = ambiguity_px
        self.last_center: tuple[float, float] | None = None
        self.last_time: float | None = None
        self.pending_click: tuple[float, float] | None = None
        self.lost = False
        self.track_id = 0
        self._new_track = True
        self._velocity: tuple[float, float] = (0.0, 0.0)

    def request_selection(self, x: float, y: float) -> None:
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Selection point must be finite")
        self.pending_click = (x, y)

    def _accept(self, choice: Detection, timestamp: float) -> None:
        if self._new_track:
            self.track_id += 1
            self._velocity = (0.0, 0.0)
        elif self.last_center is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self._velocity = (
                (choice.center[0] - self.last_center[0]) / dt,
                (choice.center[1] - self.last_center[1]) / dt,
            )
        self.last_center, self.last_time = choice.center, timestamp
        self.lost = self._new_track = False

    def choose(self, detections: list[Detection], timestamp: float) -> tuple[Detection | None, str]:
        if not isfinite(timestamp):
            raise ValueError("Invalid timestamp")
        if self.last_time is not None and timestamp <= self.last_time:
            self.lost = self._new_track = True
            return None, "Non-increasing frame time; click the object to reselect"
        if self.pending_click is not None:
            x, y = self.pending_click
            self.pending_click = None
            clicked = [d for d in detections if d.contains(x, y)]
            self._new_track = True
            if len(clicked) != 1:
                self.lost = True
                if not clicked:
                    return None, "Click missed the detected objects; click the object again"
                return None, "Selection overlaps multiple objects; click when they separate"
            self._accept(clicked[0], timestamp)
            return clicked[0], "Object selected manually"
        if not detections:
            self._new_track = True
            if self.last_time is not None and timestamp - self.last_time > self.max_gap_s:
                self.lost = True
            return None, "Target not detected"
        if self.lost:
            return None, "Tracking lost or ambiguous; click the object to reselect"
        if self.last_center is None:
            if len(detections) > 1:
                self.lost = True
                return None, "Multiple matching objects; click the intended object"
            choice = detections[0]
        else:
            dt = timestamp - self.last_time
            if dt > self.max_gap_s:
                self.lost = self._new_track = True
                return None, "Tracking lost; click the object to reselect"
            prediction = (
                self.last_center[0] + self._velocity[0] * dt,
                self.last_center[1] + self._velocity[1] * dt,
            )
            possible = [
                (hypot(d.center[0] - prediction[0], d.center[1] - prediction[1]), d)
                for d in detections
                if hypot(d.center[0] - self.last_center[0], d.center[1] - self.last_center[1])
                <= self.max_speed_px_s * dt + 12
            ]
            possible.sort(key=lambda pair: pair[0])
            if not possible:
                self.lost = self._new_track = True
                return None, "Target jump rejected; click the intended object to reselect"
            if len(possible) > 1 and possible[1][0] - possible[0][0] <= self.ambiguity_px:
                self.lost = self._new_track = True
                return None, "Nearby objects are ambiguous; click the intended object to continue"
            choice = possible[0][1]
        self._accept(choice, timestamp)
        return choice, "Tracking"
