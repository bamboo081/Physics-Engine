# Release verification

## Scope

This release was exercised on Linux with Python 3.12.14. Numerical checks use analytic trajectories and independent covariance calculations. Capture checks use real OpenCV decoding of generated MJPEG videos. Desktop tests use native Qt with its offscreen platform; they do not simulate browser widgets.

Two installation environments were used: the available scientific runtime with separately installed test/desktop dependencies, and a **new virtual environment installed only from the project's declared `desktop,dev` extras**. The latter contains no inherited system-site packages. `pip check` reported no broken requirements. Its exact dependency versions are in `constraints-tested-py312.txt`.

A third fresh virtual environment installed the built wheel with core dependencies only. It contained no Qt, OpenCV, YOLO, or PyTorch; importing the installed package, running `doctor`, and exporting the complete demo all succeeded from outside the source checkout.

Windows/macOS launch scripts and a cross-platform CI matrix are provided, but those platforms were not available for execution in this session. The workflow has not been pushed or run on GitHub as part of this work.

## Automated coverage

The final release gate completed with these results:

| Check | Result |
|---|---|
| Full suite, fresh desktop environment | **195 passed in 47.52 s**, exit 0; no skips or warnings. A separate final run also passed all 195. |
| Ruff lint | Passed. |
| Ruff formatting | All 37 Python files formatted. |
| Python compilation and Unix launcher syntax | Passed. |
| Source distribution and wheel build | Passed with the declared build requirements installed. |
| Fresh installed-wheel core smoke | Correct installed-package location; version and dependency check; complete 66-sample report; no desktop/ML packages installed. |
| Dependency consistency | `pip check` passed in both the clean desktop and core environments. |
| Included sample result | All 12 report files match the saved checksum manifest. |
| Documentation | Internal file links resolve; final report pages and desktop views visually inspected. |

The nine renderer-isolation regressions include an actual desktop worker failure and successful retry. The tests cover:

- Constant velocity, acceleration in both axes, general curved motion, irregular timing, and large timestamp offsets.
- Known projectile gravity, energy definitions, stationary coordinates, true bounces at and between frames, repeated impacts, and rejection of false bounce/gravity interpretations.
- Fit standard errors compared with independently calculated regression covariance; absence of a false uncertainty claim when residual degrees of freedom are missing.
- Gaps, target changes, insufficient intervals, invalid types, nonfinite values, duplicate timestamps/headers, numerical conditioning, and overflow.
- Video timestamps independent of inference delay; missing initial PTS, variable cadence, explicit FPS fallback, and unusable clocks.
- Real color-marker centroids, hue wrapping, ambiguity, lost detection, rejected jumps, and explicit reselection.
- Stop during loading/inference, restart, EOF, camera read failure, detector error, resolution changes, preview navigation, and closure during seeking.
- GUI CSV/demo/video workflows; target selection; enabled/disabled controls; invalidating stale results; retry after failure; source/capture warnings in the report; and asynchronous shutdown.
- Renderer Python exceptions and abrupt process exits before/after a success reply; parent interruption; failed publication; retention of earlier reports and observations; successful retry.
- Complete unique exports, strict JSON, preserved observations, checksum integrity, settings consistency, wrapped/escaped PDF text, and cleanup of failed runs.

## Known-answer integration checks

| Fixture | Expected | Observed during release checks |
|---|---|---|
| Seeded coordinate projectile, 66 samples | g = 9.81 m/s² | 9.8095981455 m/s²; conditional fit SE 0.0024435244 m/s². |
| Included colored-marker video | 97 frames, 0.800 s, g = 9.81 m/s² | 97 accepted observations; 0.800 s; g = 9.8077244914 m/s². |
| Marker first appears later in a clip | Start at source frame 15 of 30 FPS | 25 observations; first SourceTime = 0.5 s; first run-relative Time = 0. |
| Included bounce CSV | g = 9.81 m/s²; e = 0.72; impact at 0.5 s | g = 9.81; e = 0.72; impact time = 0.5 s to floating-point precision. |
| Included circular-motion CSV | Constant speed 1.2 m/s; gravity unavailable | Maximum modeled speed 1.2000033868 m/s; gravity unavailable. |
| Independent irregular-time, three-bounce test | e = 0.8 for each bounce | All three values and contact times recovered to floating-point precision. |
| Symmetric driven quartic motion | Gravity must be unavailable despite R² > 0.999 | Gravity withheld; primary quadratic kinematics retained for inspection. |

The video check includes pixel rounding and MJPEG compression; it is not expected to equal the analytic acceleration bit-for-bit. These results verify the exercised software pipeline and do not certify a physical camera's calibration.

## Independent review

A separate scientific review checked the formulas, calibration, covariance, continuity, energy, and export contracts. It reproduced an even-order acceleration change that escaped the original cubic-only gravity diagnostic, and an ill-conditioned auxiliary fit that could abort otherwise supported quadratic analysis. Both cases received failing regressions and corrections. Follow-up review accepted the covariance-aware quartic/cubic diagnostic and the auxiliary-fit fallback. It also accepted the immutable used-settings record and export mismatch rejection.

A separate application review reproduced an association defect where a rejected jump could later become an automatic target switch, and a media-clock defect when the first PTS was missing. Both received regression tests and corrections. Preview navigation was added and tested to support clips whose marker is absent from the first frame.

During development, intermittent native aborts occurred inside Matplotlib Agg font/layout processing in combined Qt test runs. Controlled lifetime/affinity probes did not establish a root cause. GUI report rendering is isolated in a separate process so an abnormal renderer exit becomes a recoverable export error while the desktop retains its observations. Renderer process-failure cleanup and retry are explicitly tested. This containment does not claim that the underlying third-party native issue has been diagnosed.

## Visual review

The native interface was rendered and inspected at 1280×800 and 1000×700, covering capture, plots, and results. The sidebar fits at the larger size and scrolls at the smaller size. Plots retain their aspect ratio, source click coordinates map through letterboxing, and result tables provide tooltips for truncated content.

The exported PDF was rendered with Poppler and inspected page by page for readable units, complete text, tables, figures, and page numbering. JSON/CSV data and manifest checksums were also inspected programmatically.

## Reproduce the checks

```bash
python -m pip install -e ".[desktop,dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m compileall -q src
python -m build
physics-motion demo --output output
physics-motion analyze --input examples/bounce.csv --experiment "Free-fall / bounce" --mass 0.2 --output output
physics-motion analyze --input examples/circular_motion.csv --experiment "General motion" --mass 0.2 --output output
```

For an offscreen Linux test run, set `QT_QPA_PLATFORM=offscreen`. Normal desktop use should run in a graphical session. For the specific Python 3.12 dependency set tested here, optionally use `python -m pip install -c constraints-tested-py312.txt -e ".[desktop,dev]"`; that file is not a promise that identical wheels exist for every operating system or Python release.

## Hardware checks still required

A real camera, its OS permission prompts/driver, downloaded YOLO weights, CUDA/GPU inference, and another person's footage were not available. Before a lab session, verify on the intended machine:

1. Camera permission, correct index, preview, disconnect/reconnect, and recovery after unplugging.
2. Reference-length calibration in the actual motion plane, camera level, and stable image dimensions.
3. Reliable marker/object selection under the intended lighting, motion blur, and occlusion.
4. Known reference motion and its time scale, particularly for slow-motion or variable-rate phone footage.
5. A complete saved run and successful opening of its report with the system PDF viewer.

Use those observations to assess measurement uncertainty. Software test coverage alone cannot establish experimental accuracy.
