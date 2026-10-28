# Measurement method and interpretation

This document describes the implemented version 2 model. It separates observations, numerical estimates, and physical assumptions so a good fit is not mistaken for a verified experiment.

## 1. Coordinates and calibration

Enter a measured reference length `L > 0` in meters, then select two endpoints in the source image. For endpoints `(x1,y1)` and `(x2,y2)`, the pixel scale is:

```text
s = hypot(x2 - x1, y2 - y1) / L          [pixels per meter]
X = cx / s                             [meters]
Y = (frame_height - cy) / s             [meters]
```

The origin is the **bottom-left edge of the source frame**, x points right, and y points up. Pixel coordinates refer to the original decoded frame; resizing or letterboxing the preview does not change calibration. The distance between endpoints must be at least 10 pixels. Invalid dimensions, nonfinite values, coincident endpoints, nonpositive lengths, and points outside the frame are rejected.

The reference may be diagonal: its Euclidean length sets the scale. It does **not** rotate the axes. Keep the image vertical aligned with physical vertical for gravity, restitution, and potential-energy interpretation. A rotated camera can mix horizontal and vertical acceleration.

A two-point scale assumes planar motion, comparable reference/object depth, a stationary camera, and negligible perspective and lens distortion. It does not correct perspective, depth changes, rolling shutter, optical zoom, or camera motion. A source-resolution change stops the run before observations can use a stale scale.

## 2. Measured target position

**Color marker:** a click samples a small interior patch in HSV space. Hue matching is circular, so red near the 0/179 boundary is handled consistently. Saturation/brightness filters and a small morphological opening reject neutral background and isolated noise. Each connected region yields its area centroid. Multiple plausible regions are retained for explicit association. The reported color-tracker `Confidence = 1` indicates a mask match; it is **not** a probability or accuracy claim.

**Optional YOLO:** measurements use detected bounding-box centers, not independently measured centers of mass. Box changes can look like motion. Detector confidence is a model score, not metrological confidence, and is never used as an inverse measurement variance.

The single-target associator limits implausible jumps and asks for selection when candidates are ambiguous. It cannot guarantee identity through every occlusion or crossing. Manual reselection and reacquisition after a lost detection start a new `TrackID`; fitting never joins different track IDs. No position is invented for a frame that yields no accepted detection.

## 3. Time

For cameras, timestamps use a monotonic clock immediately after acquisition returns and before inference. They reflect software acquisition timing, not a hardware exposure clock. Buffered frames, driver latency, auto-exposure, and rolling shutter remain possible systematic errors.

For recorded files, timestamps come from decoded media time (`CAP_PROP_POS_MSEC`), normalized to the first frame. Inference speed and display refresh rate do not enter the time scale. If timestamps are missing or fail to increase, the affected frames use the file's nominal FPS and emit an explicit capture warning. If neither usable timestamps nor FPS is available, tracking stops. **Nominal-FPS fallback is unsuitable for assuming precise variable-frame-rate timing.** The timing method is preserved in each sample and capture metadata. Slow-motion footage must be interpreted using the intended real-world time scale; the app cannot infer whether a phone has retimed a clip.

Run-relative `Time` starts at the first detector-processed frame, which is the selected starting frame for a video. If initial detections are missing, the first accepted observation can have a positive Time. `SourceTime`, `Frame`, and `TimingSource` retain the relationship to the original source. The preview holds a chosen source image for calibration and target selection. **Start frame / Show** navigates before recording, using sequential decoding when necessary to preserve the original clock. Setup time is not added to recorded media time.

OpenCV exposes timestamp, FPS, and frame-position properties, but their reliability depends on the backend, codec, and device. See the [OpenCV video-I/O reference](https://docs.opencv.org/4.11.0/d4/d15/group__videoio__flags__base.html).

## 4. Input and interval boundaries

`Time`, `X`, and `Y` must be finite real numbers with strictly increasing, nonnegative times. Duplicate timestamps are rejected instead of silently averaged, sorted, or discarded. Optional `Confidence` values must lie in [0,1]. Optional `TrackID` values must be nonempty strings or finite numeric identifiers. The analysis leaves the caller's data unchanged and keeps a separate copy of validated observations.

A time gap larger than `max_gap_seconds` (default 1.0 s) or a `TrackID` change begins a new continuity interval. Intervals shorter than `min_segment_points` (default 8) keep their observations but have no modeled derivatives or energy. The complete input needs at least `min_data_points` (default 12). The gap threshold is an operator setting; a gap below it may still reduce time resolution, which is why a new `TrackID` on reacquisition matters.

Bounce candidates are considered **only** in the explicit `Free-fall / bounce` mode. A projectile apex, ordinary lift/lower motion, or a reversal in general motion does not automatically become a collision.

## 5. Position models and derivatives

### General motion

`General motion`, `Lift and lower`, custom labels, and unrecognized experiment labels use a local degree-three least-squares polynomial in each axis. The default window contains 9 measured samples, centered where possible and shifted inward at the endpoints. It is shortened when an interval contains fewer samples; observations are never padded or borrowed across a boundary.

For one local coordinate model, with `tau = t - t_i`:

```text
p(t)   = c0 + c1*tau + c2*tau² + c3*tau³
v(t_i) = c1
a(t_i) = 2*c2
```

The fitted value and derivatives are evaluated at the actual sample time. Fits use the actual irregular timestamps. This is a local polynomial smoother, not a fixed-frame-rate finite-difference formula. Increasing the window suppresses derivative noise but can blur fast changes. Windows are measured in samples, so their duration varies with sampling cadence. Edge derivatives are less constrained. Local windows overlap, so neighboring estimates are correlated.

### Ballistic motion

`Horizontal toss`, `Free-fall / bounce`, `Free fall`, and `Projectile motion` use a separate quadratic fit in **both** axes for each interval:

```text
x(t) = bx2*tau² + bx1*tau + bx0
vx   = 2*bx2*tau + bx1
ax   = 2*bx2

y(t) = by2*tau² + by1*tau + by0
vy   = 2*by2*tau + by1
ay   = 2*by2
```

The numerical model does not force `ax = 0` or `ay = -9.81`. Horizontal acceleration is estimated and can disqualify a gravity interpretation. A global quadratic still assumes constant acceleration within the selected interval; it should not be used for arbitrary driven motion just because its overall R² appears high.

### Numerical conditioning and coefficients

Time is translated and scaled before solving each least-squares problem with singular value decomposition. Position offsets are also removed before fitting. Rank-deficient timestamp configurations and finite-number overflow fail clearly. This avoids the poor conditioning of fitting powers of a large absolute timestamp directly. The numerical basis follows [NumPy's SVD definitions](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html); the objective is the usual [least-squares sum of squared residuals](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd431.htm).

In `summary.json`, ballistic `x_coefficients` and `y_coefficients` are **descending powers of `tau = Time - time_origin_s`**. They are not coefficients in absolute `Time`. Local models have empty coefficient lists because no single polynomial describes the whole interval; read `X_model` and `Y_model` instead. Python consumers can use `Segment.evaluate_position`, `evaluate_velocity`, and `evaluate_acceleration` for ballistic segments.

**Migration from version 1:** coefficient time convention changed, x is now quadratic in ballistic mode, and general motion now uses local models. Machine-readable exports have `schema_version: 2`.

## 6. Fit quality and fit standard errors

For each coordinate, `R² = 1 - SSE/SST`; a constant observed coordinate uses an explicit numerical-tolerance convention. Negative R² is retained when appropriate. RMS residuals are in meters. A high R² does not prove the physical assumptions, validate the scale, or imply that derivative estimates are accurate.

For a polynomial design matrix `A`, conditional coefficient covariance is based on:

```text
sigma² = SSE / (n - p)
Cov(c) = sigma² * (Aᵀ A)^-1
Var(derivative) = dᵀ Cov(c) d
```

Here `p` is the number of fitted coefficients and `d` is the appropriate derivative basis vector, including time-scaling factors. The implementation computes the inverse Gram matrix from the SVD. When there are no residual degrees of freedom, fit standard errors are unavailable, not zero.

The `_fit_se` columns are **one-standard-error estimates conditional on the model and exact timestamps**, assuming independent equal-variance position residuals within that fit. They exclude scale/reference-length error, perspective, lens distortion, target/detector bias, correlated motion blur, and timing error. They are not a total uncertainty budget or a confidence interval on experimental truth. Reported precision should reflect your experiment's actual uncertainty.

## 7. Gravity qualification

A segment contributes a gravity estimate `g_est = -ay` only when all implemented checks pass:

- The operator selected a ballistic experiment mode.
- Vertical R² meets `fit_quality_threshold` (default 0.95).
- `1 <= g_est <= 30 m/s²`, a broad compatibility screen rather than an Earth-gravity guarantee.
- The vertical-acceleration fit standard error is finite and no more than 20% of `g_est`.
- `abs(ax) + 2*SE(ax) <= max(0.25, 0.1*g_est)`.
- A supported higher-order fit does not resolve an acceleration change larger than both 10% of `g_est` and three contrast standard errors in either coordinate. The diagnostic is quartic when enough samples exist, otherwise cubic; extrema are evaluated at the endpoints and any internal acceleration vertex. If the diagnostic is rank-deficient, the primary kinematics remain available but gravity is withheld.

The higher-order check is a heuristic compatibility screen, not a formal statistical significance test or proof of constant acceleration. Passing these gates does not establish that the object was unforced or that the camera was calibrated correctly. The overall reported gravity is the arithmetic mean of qualifying intervals. `gravity_estimate_std_m_s2` describes sample scatter among two or more interval estimates; it is unavailable for a single interval. `gravity_fit_standard_error_m_s2` combines independent interval fit errors for the mean and does not include shared calibration bias.

## 8. Bounce and restitution

A down-to-up local minimum nominates neighboring cuts. Candidate cuts require adequately populated downward-accelerating quadratic fits on each side, downward pre-impact velocity, upward post-impact velocity, and local vertical R² of at least 0.8. This is a heuristic proposal, not a contact sensor.

Restitution is reported only for adjacent qualified free-flight segments inside the **same continuity interval**, with a `bounce_candidate` boundary and no missing interval. Further checks require compatible fitted gravity, a short contact sampling interval relative to cadence and segment duration, horizontal position continuity, and a unique intersection of the fitted vertical paths inside the two observations surrounding contact. This intersection sets the common impact time used for both velocities:

```text
e = vy_after / abs(vy_before)
```

The model assumes a stationary horizontal surface and passive vertical bounce. Values above 1 are excluded with a warning. Values are not clamped to an expected answer except for floating-point roundoff at 1. Contact forces, spin, deforming objects, finite contact duration, moving/sloped surfaces, and missed contact frames can invalidate this model. No restitution uncertainty is claimed. Mean restitution and individual qualified events are recorded in JSON.

## 9. Speed and energy

```text
speed = hypot(vx_model, vy_model)
KE = 0.5 * mass_kg * speed²
PE = mass_kg * 9.81 * Y_measured
E_total = KE + PE
```

Energy is calculated only where a motion model exists. PE deliberately uses measured y; KE uses modeled velocity. Their sum can therefore retain position noise even when the fitted motion is smooth. The gravitational field in energy is the fixed 9.81 m/s² reference, not the inferred gravity value. The app does not prove energy conservation. The datum may be an image edge or a user-chosen CSV origin and need not be the physical floor.

## 10. Output fields

| Fields | Meaning |
|---|---|
| `Time`, `X`, `Y` | Observation time (s) and calibrated positions (m). |
| `Confidence` | Detector score/mask match, if available; not physical confidence. |
| `TrackID` | Source target-continuity identifier, if available. |
| `Frame`, `SourceTime`, `TimingSource` | Original decoded-frame index, source clock value, and timing method when captured in the app. |
| `ContinuityID` | Interval bounded by time gaps or target changes. |
| `SegmentID` | Fitted interval; blank for short unmodeled stretches. |
| `X_model`, `Y_model` | Fitted positions (m). |
| `Vx_model`, `Vy_model`, `Speed_model` | Fitted velocities and speed (m/s). |
| `Ax_model`, `Ay_model` | Fitted acceleration components (m/s²). |
| `X_fit_se`, `Y_fit_se` | Conditional position fit standard errors (m). |
| `Vx_fit_se`, `Vy_fit_se` | Conditional velocity fit standard errors (m/s). |
| `Ax_fit_se`, `Ay_fit_se` | Conditional acceleration fit standard errors (m/s²). |
| `PE`, `KE`, `E_total` | Datum-relative potential, modeled kinetic, and their sum (J). |

For quantitative experiments, compare the complete capture pipeline with independently measured reference motion and assess the uncertainty sources above. Passing software tests establishes the tested computations, not the accuracy of an arbitrary camera setup.
