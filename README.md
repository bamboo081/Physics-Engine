# Physics Motion Analyzer

A desktop tool for measuring **planar motion** from a camera, a recorded video, or a CSV of measured positions. Calibrate a known distance, track one object, and inspect its position, velocity, acceleration, and energy. Analysis runs locally; a saved experiment includes the original observations and a PDF report.

The default **color-marker tracker** works without a machine-learning model. An optional YOLO detector is available for supported object classes. A camera-free demo and a synthetic video are included so you can check the complete workflow before collecting measurements.

## Start the app

Install **Python 3.12** (Python 3.10–3.13 is supported), extract the entire ZIP into a writable folder, then use the launcher for your system:

| System | Start |
|---|---|
| Windows | Double-click `start-windows.bat`. When installing Python, enable **Add Python to PATH**. |
| macOS | Open `start-macos.command`, or run `bash start-macos.command` in Terminal from the project folder. |
| Linux | Run `bash start-linux.sh` from the project folder. |
| Any system | Run `python launch.py` (use `python3` if that is your Python command). |

The first launch creates a `.venv` folder and downloads the Python dependencies into it. It does not install packages into your system Python. Later launches reuse that environment. Basic color tracking, CSV analysis, and reporting work offline after setup. The optional YOLO detector may download model weights on first use.

**This ZIP contains the complete source project, examples, tests, and documentation. It is not a precompiled Windows/macOS application, and it does not bundle Python or platform-specific dependencies.**

### Manual installation

For an existing development environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[desktop]"
physics-motion-gui
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pip ...` and `.venv\Scripts\python.exe -m physics_motion.gui` directly. Activation is optional.

## Try it without a camera

### Built-in coordinate demo

Open the app and choose **Load demo**. It loads synthetic projectile positions with known horizontal speed of 1.4 m/s and downward acceleration of 9.81 m/s². Choose **Analyze samples**, then use **Plots** and **Results** to inspect the output. The simulated position noise is 0.4 mm; this is a software check, not an experimental validation of a camera.

A command-line equivalent needs only the smaller core installation:

```bash
python -m pip install -e .
physics-motion demo --output output
```

Or let the launcher set up the core package:

```bash
python launch.py --demo
```

A complete example is already saved in [`examples/demo_result`](examples/demo_result). You can open its [PDF report](examples/demo_result/Analysis_Report.pdf), CSV files, and plots before installing the app.

### Included video

`examples/marker_projectile.avi` exercises actual video decoding, color detection, calibration, timestamp handling, analysis, and export:

1. Select **Video file**, choose this file, and leave the tracker on **Color marker**.
2. Open the video. Its preview stays still while you set up the measurement. For your own clips, use **Start frame** and **Show** to find an image where the marker is visible.
3. Set **Known length** to **1.0000 m**. Choose **Calibrate**, then click the centers of the two white endpoints on the reference line near the bottom right. They are at native frame positions **(690, 570)** and **(840, 570)**. The scale should be about **150 px/m**.
4. Choose **Select marker** and click the green ball near **(105, 190)**.
5. Select **Horizontal toss**, then **Start tracking**. The file is analyzed when it ends; processing speed does not change its recorded time scale.
6. Inspect the plots and the gravity estimate. The video encodes an acceleration of **−9.81 m/s²**, horizontal speed **2.5 m/s**, and **120 frames/s**. Pixel rounding and video compression introduce a small measurement difference.

Exact positions and setup parameters are in `examples/marker_projectile_truth.csv` and `examples/marker_projectile.json`. See [examples](examples/README.md) for the other fixtures and how to regenerate the video.

## Record your own experiment

1. **Prepare the scene.** Keep the camera stationary, facing the plane of motion. Use a bright, saturated marker that differs from the background. Put a measured reference in that same plane and at the same distance from the camera.
2. **Open the source.** Choose a camera index or a local video file. Use **Color marker** for a colored ball or marker. Camera access is requested only when you open a camera.
3. **Set the scale.** Enter the actual reference length in meters, choose **Calibrate**, and click its endpoints. A reference shorter than ten image pixels is rejected. Recalibrate after moving the camera, changing resolution, changing optical zoom, or changing the motion plane.
4. **Choose the target.** Choose **Select marker**, then click inside the colored object. Avoid its edge, glare, shadows, or background pixels. Several matching objects require explicit selection.
5. **Set the analysis.** Enter your object's mass in kilograms. The initial value is an example. Use **General motion** for arbitrary motion, **Horizontal toss** for an isolated projectile, and **Free-fall / bounce** for a vertical fall or bounce. A free-fall label is an operator assumption, not automatic proof of the physical setup.
6. **Record.** Start tracking. Stop and analyze when finished; video files finish automatically. Lost detections remain missing. Reselecting a target starts a new interval, so unrelated tracks are not fitted together.
7. **Review.** Inspect the measured positions, fit quality, gaps, and measurement notes before using derived quantities. Open the PDF or results folder from the **Results** tab.

Saved desktop experiments default to `Documents/Physics Motion Analyzer`; you can choose another folder. The command-line default is `output` in the current working directory. Every successful analysis gets a new run folder.

### Analysis options

- **Maximum gap:** a larger time gap starts a new fit. Choose a value appropriate to the sampling rate and motion. The setting is in seconds.
- **Smoothing window:** an odd number of samples used by the general-motion local polynomial model. A larger window reduces position noise in derivatives but can hide rapid changes. It does not change the ballistic model.
- **Mass:** affects energy, not position, velocity, or acceleration. The program cannot measure mass from an image.

### Optional YOLO detector

The core and color-marker workflow do not require PyTorch or YOLO. To add them:

```bash
python -m pip install -e ".[desktop,detector]"
```

Or use `python launch.py --with-yolo`. Select **YOLO object detector**, choose the class, and supply a supported model name or a local weights path. CPU is the portable default. CUDA requires a compatible PyTorch installation; use the [official installation selector](https://pytorch.org/get-started/locally/).

YOLO uses a bounding-box center, which can move as the detected box changes. A color-marker centroid is often easier to interpret for controlled lab experiments. Neither method establishes persistent physical identity through every crossing or occlusion. See [third-party notices](docs/third-party-notices.md) before redistributing dependencies or model weights.

## Import measured positions

Use **Import CSV** in the app, or:

```bash
physics-motion analyze --input examples/projectile.csv --experiment "Horizontal toss" --mass 0.2 --output output
```

Required columns are **`Time,X,Y`**, in **seconds, meters, meters**. Positive y is upward. Times must be finite, nonnegative, and strictly increasing. Pixel coordinates must be calibrated before import; the importer does not guess units.

Optional columns include `Confidence` (0–1), `TrackID` (changes split tracks), `Frame`, `SourceTime`, and `TimingSource`. Extra columns are retained in the raw observations. UTF-8 files with a BOM and whitespace around headers are accepted. Duplicate headers, malformed row lengths, and nonnumeric measurement columns are rejected. Import is limited to 128 MiB and 250,000 rows to keep interactive use manageable.

```bash
physics-motion analyze --input measurements.csv --experiment "General motion" --mass 0.15 --smoothing-window 11 --max-gap 0.2
physics-motion doctor
physics-motion --version
```

`--json` prints the result statistics and output path as JSON. `python -m physics_motion` is an equivalent entry point.

## Files in each result

| File | Contents |
|---|---|
| `raw_samples.csv` | The observations supplied to the numerical engine, with source metadata where available. |
| `analysis_data.csv` | Observations plus modeled position, velocity, acceleration, energy, fit error columns, and interval IDs. |
| `config.json` | Version, units, settings, source/calibration context, and capture diagnostics. |
| `summary.json` | Statistics, interval models, quality information, and warnings. |
| `summary.txt` | Human-readable summary. |
| Six PNG plots | Trajectory, position, velocity, acceleration, energy, and fitting intervals. |
| `Analysis_Report.pdf` | A portable report containing the setup, method, results, notes, and all figures. |
| `manifest.json` | SHA-256 checksums and sizes for the other files. Written last on successful completion. |

Desktop report rendering runs in a separate process; an abnormal renderer exit leaves the app available for a retry. A failed export removes its own partial run directory. The desktop retains the captured observations so you can correct the destination and retry. The program does not overwrite an earlier successful run.

## What the numbers mean

General motion uses local polynomial fits; ballistic experiments use quadratic fits in both axes. Velocity and acceleration are derivatives of the chosen position model. Time gaps, changes of target, and accepted impact boundaries separate models. Gravity and restitution are only reported when their specific model checks pass. Every plot draws fitted intervals separately.

A fit standard error describes scatter around the mathematical model. It is **not** a full measurement uncertainty. Two-point calibration assumes a fixed camera and a planar scene. Perspective, lens distortion, rolling shutter, motion blur, exposure timing, detector errors, and an incorrect scale can all bias a result even when a curve fits well. Potential energy is relative to your selected y = 0 datum, not necessarily the floor.

Read [measurement methodology](docs/methodology.md) for equations, coefficient conventions, quality gates, and the limitations of gravity/restitution estimates.

## Development and verification

```bash
python -m pip install -e ".[desktop,dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The core tests run without Qt, OpenCV, a camera, or YOLO. Installing the desktop extra enables the native Qt and synthetic-video tests. GitHub Actions is configured for the core on Windows, macOS, and Linux / Python 3.10–3.13, and for desktop tests on all three operating systems with Python 3.12. A workflow configuration is not a claim that those remote jobs have already run.

The specific checks performed for this release, and the remaining hardware checks, are recorded in [validation](docs/validation.md). See [architecture](docs/architecture.md), [troubleshooting](docs/troubleshooting.md), and [release notes](RELEASE_NOTES.md).

## License

Project source is MIT licensed; see [LICENSE](LICENSE). Dependencies and optional model weights retain their own licenses. This source distribution contains no YOLO weights, copied virtual environment, credentials, or user recordings.
