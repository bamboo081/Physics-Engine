# Release notes

## 2.0.0 — 2026-10-28

This release keeps the local Python desktop architecture and upgrades the measurement pipeline, numerical model, interface, and distribution.

### Measurement and mathematics

- General motion uses local cubic position models against actual sample times. It no longer forces constant horizontal speed.
- Ballistic modes fit a quadratic in both coordinates, with stable translated/scaled SVD solves.
- Added conditional position, velocity, and acceleration fit standard errors and explicit assumptions.
- Time gaps, missing-target reacquisition, and target changes create separate fitting intervals.
- Bounce detection is limited to explicit bounce mode. Restitution uses a common fitted impact time and cannot span a recording gap or a different target.
- Gravity qualification checks fit quality, acceleration resolution, horizontal compatibility, and higher-order acceleration drift; unsupported auxiliary fits withhold gravity without discarding valid kinematics.
- Calibration and sample/settings validation reject malformed, nonfinite, ambiguous, or numerically unsupported inputs.

### Capture and interface

- Added offline color-marker tracking using a connected-component centroid. YOLO/PyTorch is an optional extra.
- Added recorded-video input with original media timing, a paused setup preview, and a selectable starting frame.
- Added GUI CSV import and a camera-free synthetic demo.
- Reorganized the native desktop into source/setup controls and Capture, Plots, and Results tabs. Kept visible text focused on the measurement task.
- Added explicit progress/error recovery, file/folder actions, invalidation of stale results, and asynchronous shutdown.
- Capture operations are worker-owned and command-queued. Stop, EOF, detector errors, source failures, and resolution changes preserve accepted observations.
- Rejected target jumps require explicit reselection; they do not silently become eligible later as the association gate expands.

### Outputs and distribution

- Plot models separately by interval; equal geometric scaling for trajectory plots.
- Every export includes raw observations, analysis CSV, configuration/version/units, JSON/text summary, six plots, a formatted PDF, and a checksum manifest.
- Desktop report rendering uses a spawned process with parent-owned staging and cleanup. Abnormal renderer exits preserve samples for a retry and cannot publish a partial report.
- Added strict CSV parsing, schema version 2, and protection against exporting a result with different analysis settings.
- Added source launchers for Windows/macOS/Linux, a project-local setup script, core/desktop/detector dependency separation, and package build configuration.
- Included a complete example report with its raw data, plots, configuration, and checksums.
- Added reproducible coordinate fixtures and a generated colored-marker video with known physical parameters.
- Expanded regression coverage, native Qt interactions, real synthetic-video checks, and CI configuration. See `docs/validation.md` for the checks actually performed for this archive.

### Compatibility changes

- `schema_version` is 2. Ballistic coefficient arrays are descending powers of **`Time - time_origin_s`**, not absolute Time. Both axes are quadratic. Local models have no single coefficient array; use their per-sample model columns.
- `gravity_estimate_std_m_s2` is scatter across at least two qualifying segments, and is unavailable for one segment. A separate field reports conditional fit standard error.
- Unknown/custom experiment labels use the general-motion model. Only explicit bounce mode proposes impacts/restitution.
- The `desktop` extra no longer installs Ultralytics. Install `.[desktop,detector]` if needed.
- The unverified legacy results are preserved under `examples/archive/` and are not expected-answer fixtures.

### Scope of verification

The source and synthetic workflows are verified as recorded in `docs/validation.md`. Physical camera behavior, OS permission dialogs, real YOLO inference, GPU configurations, and other people's footage still need validation on the target system. Numerical tests are not calibration certification or a full measurement uncertainty budget.
