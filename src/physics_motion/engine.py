"""Offline planar kinematics on actual observation timestamps.

General motion uses moving local cubic least-squares fits. Operator-selected
ballistic modes use an unconstrained quadratic in each coordinate. Fits never
cross a large time gap or a TrackID change. Only explicit bounce mode proposes
impact boundaries; a direction reversal alone does not establish a collision.

Coordinates are meters, x right and y up. Energies assume a point mass and use
9.81 m/s²: PE uses measured y relative to the input datum, KE uses fitted velocity.
Fit standard errors assume independent, equal-variance position residuals and
exact timestamps, conditional on the polynomial. They do not quantify calibration,
tracking, or timing errors. Detector confidence is not a position-error variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from math import factorial
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

G_STANDARD = 9.81
_BOUNCE_MODES = {"free-fall / bounce", "free fall / bounce"}
_BALLISTIC_MODES = _BOUNCE_MODES | {"horizontal toss", "free fall", "projectile motion"}
_MODEL_COLUMNS = (
    "X_model",
    "Y_model",
    "Vx_model",
    "Vy_model",
    "Ax_model",
    "Ay_model",
    "Speed_model",
    "PE",
    "KE",
    "E_total",
    "X_fit_se",
    "Y_fit_se",
    "Vx_fit_se",
    "Vy_fit_se",
    "Ax_fit_se",
    "Ay_fit_se",
)


def _is_finite_real(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, ValueError):
        return False


@dataclass(frozen=True)
class AnalysisSettings:
    mass_kg: float = 0.20
    experiment_type: str = "General motion"
    min_data_points: int = 12
    min_segment_points: int = 8
    max_gap_seconds: float = 1.0
    bounce_velocity_threshold: float = 0.20
    fit_quality_threshold: float = 0.95
    smoothing_window: int = 9

    def validate(self) -> None:
        if not _is_finite_real(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("Mass must be a finite positive number in kg")
        for count in (self.min_data_points, self.min_segment_points):
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 4:
                raise ValueError("Minimum sample counts must be integers of at least four")
        if not _is_finite_real(self.max_gap_seconds) or self.max_gap_seconds <= 0:
            raise ValueError("Maximum sampling gap must be a finite positive number")
        if (
            not _is_finite_real(self.fit_quality_threshold)
            or not 0 <= self.fit_quality_threshold <= 1
        ):
            raise ValueError("Fit quality threshold must be finite and between zero and one")
        if (
            not _is_finite_real(self.bounce_velocity_threshold)
            or self.bounce_velocity_threshold <= 0
        ):
            raise ValueError("Bounce threshold must be a finite positive number")
        if not isinstance(self.experiment_type, str) or not self.experiment_type.strip():
            raise ValueError("Experiment type must be a nonempty string")
        if (
            isinstance(self.smoothing_window, (bool, np.bool_))
            or not isinstance(self.smoothing_window, Integral)
            or self.smoothing_window < 5
            or self.smoothing_window % 2 != 1
        ):
            raise ValueError("Smoothing window must be an odd integer of at least five samples")


@dataclass
class Segment:
    index: int
    start_index: int
    end_index: int  # exclusive
    t_start: float
    t_end: float
    motion_type: str
    # Descending powers of tau = Time - time_origin_s, NOT absolute Time.
    # Both lists are empty for a local model, which has no single polynomial.
    x_coefficients: list[float]
    y_coefficients: list[float]
    r2_x: float
    r2_y: float
    gravity_est: float | None = None
    time_origin_s: float = 0.0
    model_type: str = "quadratic_ballistic"
    boundary_reason: str = "recording_start"
    continuity_id: int = 1
    smoothing_window: int | None = None
    rms_residual_x_m: float = 0.0
    rms_residual_y_m: float = 0.0
    gravity_fit_standard_error_m_s2: float | None = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def _evaluate(self, time: Any, derivative: int) -> tuple[Any, Any]:
        if not self.x_coefficients or not self.y_coefficients:
            raise ValueError("Local models must be read from the per-sample model columns")
        values = np.asarray(time, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Evaluation timestamps must be finite")
        tau = values - self.time_origin_s
        x = np.polyval(np.polyder(self.x_coefficients, derivative), tau)
        y = np.polyval(np.polyder(self.y_coefficients, derivative), tau)
        if values.ndim == 0:
            return float(x), float(y)
        return x, y

    def evaluate_position(self, time: Any) -> tuple[Any, Any]:
        """Evaluate a ballistic polynomial; extrapolation is the caller's choice."""
        return self._evaluate(time, 0)

    def evaluate_velocity(self, time: Any) -> tuple[Any, Any]:
        return self._evaluate(time, 1)

    def evaluate_acceleration(self, time: Any) -> tuple[Any, Any]:
        return self._evaluate(time, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "sample_count": self.end_index - self.start_index,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "motion_type": self.motion_type,
            "model_type": self.model_type,
            "time_origin_s": self.time_origin_s,
            "coefficient_convention": "descending powers of tau = Time - time_origin_s",
            "x_coefficients": self.x_coefficients,
            "y_coefficients": self.y_coefficients,
            "r2_x": self.r2_x,
            "r2_y": self.r2_y,
            "rms_residual_x_m": self.rms_residual_x_m,
            "rms_residual_y_m": self.rms_residual_y_m,
            "boundary_reason": self.boundary_reason,
            "continuity_id": self.continuity_id,
            "smoothing_window": self.smoothing_window,
            "gravity_est_m_s2": self.gravity_est,
            "gravity_fit_standard_error_m_s2": self.gravity_fit_standard_error_m_s2,
        }


@dataclass
class AnalysisResult:
    data: pd.DataFrame
    segments: list[Segment]
    stats: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    raw_samples: pd.DataFrame | None = None
    settings: AnalysisSettings | None = None


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    centered = observed - observed[0]
    total = float(np.sum((centered - np.mean(centered)) ** 2))
    if total == 0:
        tolerance = 32 * np.finfo(float).eps * max(1.0, abs(float(observed[0])))
        return 1.0 if np.max(np.abs(observed - predicted)) <= tolerance else 0.0
    return float(1.0 - residual / total)


def _basis(u: np.ndarray, degree: int, derivative: int = 0) -> np.ndarray:
    columns = []
    for power in range(degree + 1):
        if power < derivative:
            columns.append(np.zeros_like(u))
        else:
            factor = factorial(power) / factorial(power - derivative)
            columns.append(factor * u ** (power - derivative))
    return np.column_stack(columns)


@dataclass
class _PolynomialFit:
    """SVD fit in a translated, scaled time domain, with separate x/y residuals."""

    origin: float
    scale: float
    coefficients: np.ndarray  # ascending powers of normalized time; shape (degree+1, 2)
    anchor: np.ndarray
    inverse_gram: np.ndarray
    residual_variance: np.ndarray
    rms_residual: np.ndarray

    def evaluate(self, time: np.ndarray, derivative: int = 0) -> tuple[np.ndarray, np.ndarray]:
        design = (
            _basis(
                (np.atleast_1d(time) - self.origin) / self.scale,
                len(self.coefficients) - 1,
                derivative,
            )
            / self.scale**derivative
        )
        value = design @ self.coefficients
        if derivative == 0:
            value = value + self.anchor
        factor = np.einsum("ij,jk,ik->i", design, self.inverse_gram, design)
        variance = np.maximum(factor, 0)[:, None] * self.residual_variance
        return value, np.sqrt(variance)

    def physical_coefficients(self) -> np.ndarray:
        coefficients = self.coefficients / self.scale ** np.arange(len(self.coefficients))[:, None]
        coefficients[0] += self.anchor
        return coefficients[::-1]


def _fit_polynomial(
    t: np.ndarray, positions: np.ndarray, degree: int, origin: float | None = None
) -> _PolynomialFit:
    origin = float(t[0]) if origin is None else origin
    offsets = t - origin
    scale = float(np.max(np.abs(offsets)))
    if scale <= 0:
        raise ValueError("Timestamp spacing cannot support a polynomial fit")
    design = _basis(offsets / scale, degree)
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    if singular[-1] <= singular[0] * max(design.shape) * np.finfo(float).eps:
        raise ValueError("Timestamp spacing is too ill-conditioned for reliable derivatives")
    anchor = positions[int(np.argmin(np.abs(offsets)))].copy()
    centered = positions - anchor
    coefficients = vh.T @ ((u.T @ centered) / singular[:, None])
    residual = centered - design @ coefficients
    sum_squares = np.sum(residual**2, axis=0)
    degrees_of_freedom = len(t) - degree - 1
    residual_variance = (
        sum_squares / degrees_of_freedom if degrees_of_freedom > 0 else np.full(2, np.nan)
    )
    inverse_gram = (vh.T / singular**2) @ vh
    return _PolynomialFit(
        origin,
        scale,
        coefficients,
        anchor,
        inverse_gram,
        residual_variance,
        np.sqrt(sum_squares / len(t)),
    )


def _local_kinematics(
    t: np.ndarray, positions: np.ndarray, window: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    count = len(t)
    window = min(window, count)
    estimates = [np.empty((count, 2)) for _ in range(3)]
    errors = [np.empty((count, 2)) for _ in range(3)]
    for i in range(count):
        left = min(max(i - window // 2, 0), count - window)
        right = left + window
        fit = _fit_polynomial(t[left:right], positions[left:right], 3, float(t[i]))
        for derivative in range(3):
            value, error = fit.evaluate(t[i : i + 1], derivative)
            estimates[derivative][i] = value[0]
            errors[derivative][i] = error[0]
    return estimates, errors


def _bounce_boundaries(
    t: np.ndarray, positions: np.ndarray, settings: AnalysisSettings
) -> list[int]:
    """Propose down-to-up changes only within an explicit bounce experiment.

    Local minima nominate two possible cuts, since the closest observed point
    may be on either side of contact. Separate quadratic fits choose the cut.
    Qualification and impact-time intersection are performed independently later.
    """
    y = positions[:, 1]
    minima = np.flatnonzero((y[1:-1] <= y[:-2]) & (y[1:-1] < y[2:])) + 1
    count, minimum = len(t), settings.min_segment_points
    cuts: list[int] = []
    last = 0
    for i in minima:
        candidates: list[tuple[float, int]] = []
        for cut in (int(i), int(i) + 1):
            if cut - last < minimum or count - cut < minimum:
                continue
            left = max(last, cut - minimum)
            right = min(count, cut + minimum)
            before = _fit_polynomial(t[left:cut], positions[left:cut], 2)
            after = _fit_polynomial(t[cut:right], positions[cut:right], 2)
            midpoint = t[cut - 1] + (t[cut] - t[cut - 1]) / 2
            velocity_before = before.evaluate(np.array([midpoint]), 1)[0][0, 1]
            velocity_after = after.evaluate(np.array([midpoint]), 1)[0][0, 1]
            acceleration_before = before.evaluate(np.array([midpoint]), 2)[0][0, 1]
            acceleration_after = after.evaluate(np.array([midpoint]), 2)[0][0, 1]
            pred_before = before.evaluate(t[left:cut])[0][:, 1]
            pred_after = after.evaluate(t[cut:right])[0][:, 1]
            if (
                velocity_before < -settings.bounce_velocity_threshold
                and velocity_after > settings.bounce_velocity_threshold
                and acceleration_before < 0
                and acceleration_after < 0
                and _r_squared(y[left:cut], pred_before) >= 0.8
                and _r_squared(y[cut:right], pred_after) >= 0.8
            ):
                candidates.append(
                    (float(before.rms_residual[1] ** 2 + after.rms_residual[1] ** 2), cut)
                )
        if candidates:
            _, chosen = min(candidates)
            cuts.append(chosen)
            last = chosen
    return cuts


def _validate_samples(samples: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    if not isinstance(samples, pd.DataFrame):
        raise ValueError("Samples must be a pandas DataFrame with Time, X, Y columns")
    if samples.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names")
    if not {"Time", "X", "Y"}.issubset(samples.columns):
        raise ValueError("Input CSV must contain columns Time, X, Y (seconds, meters, meters)")
    if len(samples) < settings.min_data_points:
        raise ValueError(
            f"Need at least {settings.min_data_points} measured positions; got {len(samples)}"
        )
    df = samples.copy(deep=True).reset_index(drop=True)
    for name in ("Time", "X", "Y", "Confidence"):
        if name not in df:
            continue
        series = df[name]
        if (
            pd.api.types.is_complex_dtype(series.dtype)
            or pd.api.types.is_datetime64_any_dtype(series.dtype)
            or pd.api.types.is_timedelta64_dtype(series.dtype)
            or any(isinstance(value, (bool, np.bool_, complex)) for value in series)
        ):
            raise ValueError(f"{name} must contain real numeric measurements")
        try:
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(
                f"{name} contains blank, infinite, or nonnumeric measurements"
            ) from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f"{name} contains blank, infinite, or nonnumeric measurements")
        df[name] = numeric
    t = df["Time"].to_numpy()
    if t[0] < 0:
        raise ValueError("Timestamps must be nonnegative seconds")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be strictly increasing with no duplicate values")
    if "Confidence" in df and not df["Confidence"].between(0, 1).all():
        raise ValueError("Confidence values must be between 0 and 1")
    if "TrackID" in df:
        for value in df["TrackID"]:
            if isinstance(value, str):
                valid = bool(value.strip())
            else:
                valid = _is_finite_real(value)
            if not valid:
                raise ValueError(
                    "TrackID must contain nonempty strings or finite numeric identifiers"
                )
    return df


def _qualified_gravity(
    t: np.ndarray,
    positions: np.ndarray,
    fit: _PolynomialFit,
    r2_y: float,
    settings: AnalysisSettings,
) -> tuple[float | None, float | None]:
    acceleration, standard_error = fit.evaluate(t[:1], 2)
    ax, ay = acceleration[0]
    ax_se, ay_se = standard_error[0]
    g = float(-ay)
    if (
        r2_y < settings.fit_quality_threshold
        or not 1 <= g <= 30
        or not np.isfinite(standard_error).all()
        or ay_se > 0.2 * g
        or abs(ax) + 2 * ax_se > max(0.25, 0.1 * g)
    ):
        return None, None
    # A high position R² can conceal changing acceleration. A quartic catches
    # symmetric curvature that a cubic cannot. This is a compatibility screen,
    # not a formal post-selection significance test.
    if len(t) >= 5:
        try:
            diagnostic = _fit_polynomial(t, positions, 4 if len(t) >= 6 else 3)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            # An unsupported auxiliary model must not discard valid primary
            # quadratic kinematics; only the gravity interpretation is withheld.
            return None, None
        lower = (t[0] - diagnostic.origin) / diagnostic.scale
        upper = (t[-1] - diagnostic.origin) / diagnostic.scale
        locations = [lower, upper]
        if len(diagnostic.coefficients) == 5:
            for axis in range(2):
                c3, c4 = diagnostic.coefficients[3:, axis]
                if c4 != 0:
                    vertex = -c3 / (4 * c4)
                    if lower < vertex < upper:
                        locations.append(vertex)
        design = (
            _basis(np.asarray(locations), len(diagnostic.coefficients) - 1, derivative=2)
            / diagnostic.scale**2
        )
        accelerations = design @ diagnostic.coefficients
        for axis in range(2):
            high = int(np.argmax(accelerations[:, axis]))
            low = int(np.argmin(accelerations[:, axis]))
            contrast = design[high] - design[low]
            variance = max(float(contrast @ diagnostic.inverse_gram @ contrast), 0.0)
            contrast_se = np.sqrt(variance * diagnostic.residual_variance[axis])
            spread = accelerations[high, axis] - accelerations[low, axis]
            if not np.isfinite(contrast_se) or (spread > 0.1 * g and spread > 3 * contrast_se):
                return None, None
    return g, float(ay_se)


def _impact_event(
    previous: Segment, following: Segment, t: np.ndarray, settings: AnalysisSettings
) -> tuple[dict[str, Any] | None, str | None]:
    if (
        following.boundary_reason != "bounce_candidate"
        or previous.continuity_id != following.continuity_id
        or previous.end_index != following.start_index
        or previous.gravity_est is None
        or following.gravity_est is None
    ):
        return None, None
    lower, upper = previous.t_end, following.t_start
    interval = upper - lower
    cadence = float(
        np.median(
            np.r_[
                np.diff(t[previous.start_index : previous.end_index]),
                np.diff(t[following.start_index : following.end_index]),
            ]
        )
    )
    if interval > 3 * cadence or interval > 0.2 * min(previous.duration, following.duration):
        return None, None
    g_mean = (previous.gravity_est + following.gravity_est) / 2
    if abs(previous.gravity_est - following.gravity_est) > 0.2 * g_mean:
        return None, None
    midpoint = lower + interval / 2
    x_before, y_before = previous.evaluate_position(midpoint)
    x_after, y_after = following.evaluate_position(midpoint)
    vx_before, vy_before = previous.evaluate_velocity(midpoint)
    vx_after, vy_after = following.evaluate_velocity(midpoint)
    ax_before, ay_before = previous.evaluate_acceleration(midpoint)
    ax_after, ay_after = following.evaluate_acceleration(midpoint)
    # Solve y_before(t) = y_after(t) in normalized time within the one observed
    # interval around contact; never extrapolate over an unobserved recording gap.
    difference = np.array(
        [
            0.5 * (ay_before - ay_after) * interval**2,
            (vy_before - vy_after) * interval,
            y_before - y_after,
        ]
    )
    if np.max(np.abs(difference)) == 0:
        return None, None
    tolerance = 64 * np.finfo(float).eps * np.max(np.abs(difference))
    first = int(np.flatnonzero(np.abs(difference) > tolerance)[0])
    roots = np.roots(difference[first:])
    impact_candidates = [
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-8 and -0.500001 <= root.real <= 0.500001
    ]
    if len(impact_candidates) != 1:
        return None, None
    offset = float(np.clip(impact_candidates[0], -0.5, 0.5)) * interval
    # Evaluate in offsets from midpoint to retain precision for large time origins.
    v_pre = vy_before + ay_before * offset
    v_post = vy_after + ay_after * offset
    x_pre = x_before + vx_before * offset + 0.5 * ax_before * offset**2
    x_post = x_after + vx_after * offset + 0.5 * ax_after * offset**2
    continuity_tolerance = max(
        3 * np.hypot(previous.rms_residual_x_m, following.rms_residual_x_m),
        128 * np.finfo(float).eps * max(1.0, abs(x_pre), abs(x_post)),
    )
    if (
        abs(x_pre - x_post) > continuity_tolerance
        or v_pre >= -settings.bounce_velocity_threshold
        or v_post <= settings.bounce_velocity_threshold
    ):
        return None, None
    estimate = float(v_post / abs(v_pre))
    if estimate > 1 + 1e-10:
        return (
            None,
            f"Estimated restitution {estimate:.3f} exceeds 1 for a passive bounce; excluded.",
        )
    return {
        "previous_segment": previous.index,
        "next_segment": following.index,
        "impact_time_s": float(midpoint + offset),
        "observation_interval_s": interval,
        "vertical_velocity_before_m_s": float(v_pre),
        "vertical_velocity_after_m_s": float(v_post),
        "restitution": min(estimate, 1.0),
    }, None


def _analyze_validated(df: pd.DataFrame, settings: AnalysisSettings) -> AnalysisResult:
    raw_samples = df.copy(deep=True)
    t = df["Time"].to_numpy()
    positions = df[["X", "Y"]].to_numpy()
    mode = settings.experiment_type.strip().casefold()
    ballistic = mode in _BALLISTIC_MODES
    warnings = [
        "Positions assume planar motion and a fixed, level camera; perspective, depth, and lens distortion are uncorrected.",
        "Potential energy uses measured Y and the chosen y=0 datum, not a measured ground level; energies use g=9.81 m/s².",
        "Fit standard errors assume independent, equal-variance position residuals, exact timestamps, and the chosen model; they are not measurement uncertainty and exclude calibration, tracking, and timing errors.",
    ]
    if ballistic:
        warnings.append(
            "Ballistic estimates assume the selected interval is unforced free flight. A high R² cannot establish that physical assumption."
        )
    else:
        warnings.append(
            "Local cubic derivatives use neighboring measured samples; window size trades temporal detail for noise suppression, and endpoints are less constrained."
        )
    if mode in _BOUNCE_MODES:
        warnings.append(
            "Bounce detection is heuristic. Restitution assumes a stationary horizontal surface; contact impulses, spin, and contact duration are not measured."
        )
    gaps = np.diff(t)
    gap_indices = set((np.flatnonzero(gaps > settings.max_gap_seconds) + 1).tolist())
    track_indices: set[int] = set()
    if "TrackID" in df:
        identifiers = df["TrackID"].to_numpy()
        track_indices = set((np.flatnonzero(identifiers[1:] != identifiers[:-1]) + 1).tolist())
    if gap_indices:
        warnings.append(
            f"Recording has {len(gap_indices)} large time gap(s); fits do not span those gaps."
        )
    if track_indices:
        warnings.append(
            f"Recording has {len(track_indices)} track identity boundary/boundaries; fits do not cross them."
        )
    if gaps.max() > 0.2 or gaps.max() > 3 * np.median(gaps):
        warnings.append(
            "Slow or irregular sampling limits time resolution, especially near impacts and rapid acceleration changes."
        )
    df["SegmentID"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["ContinuityID"] = pd.Series(0, index=df.index, dtype="int64")
    for name in _MODEL_COLUMNS:
        df[name] = np.nan
    segments: list[Segment] = []
    boundaries = [0, *sorted(gap_indices | track_indices), len(t)]
    candidate_count = 0
    for continuity_id, (start, end) in enumerate(pairwise(boundaries), start=1):
        df.loc[start : end - 1, "ContinuityID"] = continuity_id
        if end - start < settings.min_segment_points:
            continue
        reason = (
            "recording_start"
            if start == 0
            else "time_gap"
            if start in gap_indices
            else "track_change"
        )
        local_cuts = (
            _bounce_boundaries(t[start:end], positions[start:end], settings)
            if mode in _BOUNCE_MODES
            else []
        )
        candidate_count += len(local_cuts)
        cuts = [start, *(start + value for value in local_cuts), end]
        for part, (left, right) in enumerate(pairwise(cuts)):
            local_t = t[left:right]
            local_positions = positions[left:right]
            fit = None
            coefficients = np.empty((0, 2))
            if ballistic:
                fit = _fit_polynomial(local_t, local_positions, 2)
                pairs = [fit.evaluate(local_t, order) for order in range(3)]
                estimates, errors = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                coefficients = fit.physical_coefficients()
            else:
                estimates, errors = _local_kinematics(
                    local_t, local_positions, settings.smoothing_window
                )
            predicted, velocity, acceleration = estimates
            r2_x = _r_squared(local_positions[:, 0], predicted[:, 0])
            r2_y = _r_squared(local_positions[:, 1], predicted[:, 1])
            rms = np.sqrt(np.mean((local_positions - predicted) ** 2, axis=0))
            g_est, g_se = (
                _qualified_gravity(local_t, local_positions, fit, r2_y, settings)
                if fit is not None
                else (None, None)
            )
            speed = np.hypot(velocity[:, 0], velocity[:, 1])
            pe = settings.mass_kg * G_STANDARD * local_positions[:, 1]
            ke = 0.5 * settings.mass_kg * speed**2
            model = np.column_stack(
                (
                    predicted,
                    velocity,
                    acceleration,
                    speed,
                    pe,
                    ke,
                    pe + ke,
                    errors[0],
                    errors[1],
                    errors[2],
                )
            )
            if not np.isfinite(model[:, :10]).all():
                raise ValueError(
                    "Input magnitude exceeds finite numerical limits for modeled physics"
                )
            index = len(segments) + 1
            df.loc[left : right - 1, "SegmentID"] = index
            df.loc[left : right - 1, list(_MODEL_COLUMNS)] = model
            segment = Segment(
                index,
                left,
                right,
                float(local_t[0]),
                float(local_t[-1]),
                "constant fitted planar acceleration"
                if ballistic
                else "general motion (local cubic)",
                coefficients[:, 0].tolist(),
                coefficients[:, 1].tolist(),
                r2_x,
                r2_y,
                g_est,
                time_origin_s=float(local_t[0]),
                model_type="quadratic_ballistic" if ballistic else "local_cubic",
                boundary_reason=reason if part == 0 else "bounce_candidate",
                continuity_id=continuity_id,
                smoothing_window=None
                if ballistic
                else min(int(settings.smoothing_window), right - left),
                rms_residual_x_m=float(rms[0]),
                rms_residual_y_m=float(rms[1]),
                gravity_fit_standard_error_m_s2=g_se,
            )
            segments.append(segment)
            if min(r2_x, r2_y) < 0.8:
                warnings.append(
                    f"Segment {index}: weak position fit (R²x={r2_x:.3f}, R²y={r2_y:.3f}); derivative estimates may be unreliable."
                )
            if not np.isfinite(model[:, 10:]).all():
                warnings.append(
                    f"Segment {index}: too few residual degrees of freedom to estimate fit standard errors."
                )
    if not segments:
        raise ValueError(
            "No contiguous interval contains enough samples; improve sampling or adjust the gap threshold"
        )
    unmodeled = int(df["SegmentID"].isna().sum())
    if unmodeled:
        warnings.append(
            f"{unmodeled} points lie in short intervals and have no modeled velocities/energies."
        )
    qualified = [segment for segment in segments if segment.gravity_est is not None]
    g_values = [segment.gravity_est for segment in qualified]
    g_errors = [segment.gravity_fit_standard_error_m_s2 for segment in qualified]
    events = []
    if mode in _BOUNCE_MODES:
        for previous, following in pairwise(segments):
            event, warning = _impact_event(previous, following, t, settings)
            if event is not None:
                events.append(event)
            if warning is not None:
                warnings.append(warning)
    stats: dict[str, Any] = {
        "samples": len(df),
        "modeled_samples": len(df) - unmodeled,
        "segments": len(segments),
        "duration_s": float(t[-1] - t[0]),
        "max_height_m": float(np.max(positions[:, 1])),
        "max_modeled_speed_m_s": float(df["Speed_model"].max()),
        "mass_kg": float(settings.mass_kg),
        "model_type": "quadratic_ballistic" if ballistic else "local_cubic",
        "smoothing_window": None if ballistic else int(settings.smoothing_window),
        "gravity_estimate_m_s2": float(np.mean(g_values)) if g_values else None,
        "gravity_estimate_std_m_s2": float(np.std(g_values, ddof=1)) if len(g_values) > 1 else None,
        "gravity_fit_standard_error_m_s2": float(np.linalg.norm(g_errors) / len(g_errors))
        if g_errors
        else None,
        "gravity_segment_count": len(qualified),
        "restitution_estimate": float(np.mean([event["restitution"] for event in events]))
        if events
        else None,
        "restitution_count": len(events),
        "restitution_events": events,
        "bounce_candidate_count": candidate_count,
        "max_sample_gap_s": float(gaps.max()),
        "continuity_intervals": len(boundaries) - 1,
    }
    if ballistic and not g_values:
        warnings.append(
            "Insufficient compatible, well-resolved free-flight model evidence for a gravity estimate."
        )
    if mode in _BOUNCE_MODES and not events:
        warnings.append("No qualified continuous bounce pair: restitution is not reported.")
    return AnalysisResult(df, segments, stats, warnings, raw_samples, settings)


def analyze_samples(
    samples: pd.DataFrame, settings: AnalysisSettings | None = None
) -> AnalysisResult:
    """Analyze Time/X/Y in s/m/m, optional Confidence (0..1) and TrackID.

    Samples must have finite coordinates and strictly increasing nonnegative
    times. Missing positions must remain absent; mark reacquisition with a new
    TrackID so local windows do not bridge it. Unknown experiment names use the
    general model. Raw observations are preserved separately from derived data.
    """
    if settings is None:
        settings = AnalysisSettings()
    if not isinstance(settings, AnalysisSettings):
        raise ValueError("Settings must be an AnalysisSettings instance")
    settings.validate()
    df = _validate_samples(samples, settings)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return _analyze_validated(df, settings)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "Input magnitude or timestamp spacing exceeds stable numerical limits"
        ) from exc
