"""A uniform planar pixel-to-meter scale for an unrotated, fixed camera.

The reference and tracked object must lie in the same plane, approximately
parallel to the sensor. A two-point reference cannot correct perspective,
out-of-plane motion, lens distortion, camera tilt, or localization error.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isclose, isfinite
from numbers import Integral, Real


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _point(value: object) -> tuple[float, float]:
    try:
        x, y = value
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration points must each contain exactly two coordinates") from exc
    return _finite_real(x, "Coordinate"), _finite_real(y, "Coordinate")


def _frame_size(value: object) -> tuple[int, int]:
    try:
        width, height = value
    except (TypeError, ValueError) as exc:
        raise ValueError("Frame dimensions must be two positive integers") from exc
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in (width, height)):
        raise ValueError("Frame dimensions must be two positive integers")
    return int(width), int(height)


@dataclass(frozen=True)
class Calibration:
    frame_width: int
    frame_height: int
    point_a: tuple[float, float]
    point_b: tuple[float, float]
    reference_length_m: float
    pixels_per_meter: float

    def __post_init__(self) -> None:
        """Validate direct construction as well as the recommended factory."""
        width, height = _frame_size((self.frame_width, self.frame_height))
        a, b = _point(self.point_a), _point(self.point_b)
        length = _finite_real(self.reference_length_m, "Reference length")
        scale = _finite_real(self.pixels_per_meter, "Pixels per meter")
        if length <= 0 or scale <= 0:
            raise ValueError("Reference length and pixels per meter must be greater than zero")
        if not all(0 <= x < width and 0 <= y < height for x, y in (a, b)):
            raise ValueError("Calibration points must be inside the original video frame")
        distance = hypot(b[0] - a[0], b[1] - a[1])
        expected = distance / length
        if distance <= 0 or not isfinite(expected) or expected <= 0:
            raise ValueError("Calibration scale must be finite and greater than zero")
        if not isclose(scale, expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Pixels per meter must agree with the reference points and length")
        try:
            extent_is_finite = isfinite(width / scale) and isfinite(height / scale)
        except OverflowError:
            extent_is_finite = False
        if not extent_is_finite:
            raise ValueError("Calibration magnitude exceeds finite coordinate limits")
        # Normalize NumPy scalar inputs and mutable point lists for safe serialization.
        for name, value in (
            ("frame_width", width),
            ("frame_height", height),
            ("point_a", a),
            ("point_b", b),
            ("reference_length_m", length),
            ("pixels_per_meter", scale),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_points(
        cls,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
        reference_length_m: float,
        frame_size: tuple[int, int],
        min_pixel_distance: float = 10.0,
    ) -> Calibration:
        """Measure scale from two points in the original, unresized frame."""
        width, height = _frame_size(frame_size)
        a, b = _point(point_a), _point(point_b)
        length = _finite_real(reference_length_m, "Reference length")
        minimum = _finite_real(min_pixel_distance, "Minimum reference span")
        if length <= 0:
            raise ValueError("Reference length must be finite and greater than zero")
        if minimum <= 0:
            raise ValueError("Minimum reference span must be greater than zero")
        distance = hypot(b[0] - a[0], b[1] - a[1])
        if distance < minimum:
            raise ValueError(f"Reference must span at least {minimum:g} pixels")
        return cls(width, height, a, b, length, distance / length)

    def pixel_to_meters(self, x: float, y: float) -> tuple[float, float]:
        """Convert original-frame coordinates; include the geometric frame edges.

        The bottom geometric edge is y=frame_height, so it maps exactly to zero.
        Pixel centers occupy 0..width-1 and 0..height-1 in this convention.
        """
        x = _finite_real(x, "Coordinate")
        y = _finite_real(y, "Coordinate")
        if not (0 <= x <= self.frame_width and 0 <= y <= self.frame_height):
            raise ValueError("Coordinates must be inside the calibrated frame")
        return x / self.pixels_per_meter, (self.frame_height - y) / self.pixels_per_meter

    def to_dict(self) -> dict:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "point_a": list(self.point_a),
            "point_b": list(self.point_b),
            "reference_length_m": self.reference_length_m,
            "pixels_per_meter": self.pixels_per_meter,
            "coordinate_system": "origin at bottom-left geometric frame edge; x right; y up",
        }
