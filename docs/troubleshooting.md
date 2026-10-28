# Troubleshooting

## Installation

**Python is not found, or a launcher opens the Microsoft Store.** Install Python 3.12 from [python.org](https://www.python.org/downloads/). On Windows select **Add Python to PATH**. Close and reopen the terminal, then run `py -3.12 launch.py` or use the Windows launcher.

**Python 3.14 or newer is selected.** This release targets Python 3.10–3.13. Use `python3.12 launch.py` or `py -3.12 launch.py`. The launchers prefer 3.12 when it is available.

**macOS refuses to open the `.command` file.** Open Terminal, change into the extracted project directory, and run `bash start-macos.command`. You can also run `python3.12 launch.py` directly. The source archive is not a signed native application.

**Permission denied while creating `.venv` or outputs.** Extract into a folder you own, such as your Documents folder. Do not run inside the ZIP or a read-only application directory. Choose a writable output folder in the app.

**First setup cannot download dependencies.** Check the network and proxy settings for your Python installation, then rerun the launcher. Completed package downloads may be cached. Only first setup and optional model acquisition need internet; the default color/CSV workflow does not use a remote inference service.

**PowerShell blocks activation.** Activation is optional. Use `.venv\Scripts\python.exe -m physics_motion.gui` directly, or use `start-windows.bat`.

**Linux reports a missing Qt platform plugin or system library.** A graphical desktop session and Qt's system prerequisites are needed for the native app. Run `python -m physics_motion doctor` inside the environment to inspect packages. `QT_QPA_PLATFORM=offscreen` is for automated tests, not normal use. If OpenCV and Qt packages from unrelated environments are mixed, create a clean environment and install the declared desktop extra. Consult [Qt's Linux prerequisites](https://doc.qt.io/qt-6/linux-requirements.html) for your distribution. The CLI remains available without Qt.

**Dependencies are inconsistent after manual package changes.** Close the app. Create a new environment using the manual installation steps in README. Preserve your result folders. `python -m pip check` reports package-level conflicts; `physics-motion doctor` reports versions.

## Camera or video

**Camera will not open.** Confirm OS camera permission, close other apps using the device, and try the correct camera index. Use a local recorded video to verify the rest of the app independently of the camera. Some drivers block during a failed open/read; closing is cooperative and may need to wait for the driver call to return.

**Video will not open.** Use a locally stored decodable video, not a webpage or streaming URL. Try the included `examples/marker_projectile.avi` first. If the codec is unsupported, transcode a copy while preserving its original frame timing. Keep the original footage and document any retiming.

**The video stays on the first image.** This is intentional during setup. If the marker appears later, set **Start frame** and choose **Show** to inspect that frame. Frame numbers start at zero. Then calibrate the known length, select the marker, and start tracking. Seeking clears the selected marker, so select it again on the displayed image.

**Video stops on an ambiguous frame.** Select the intended target in the preview, or stop and analyze the observations already collected. Multiple similar markers can make association ambiguous.

**Video ends early or uses FPS fallback.** Read the capture warning in the results/report and `config.json`. A damaged file may stop before its advertised frame count. Missing/repeated media timestamps use nominal FPS only when available. That fallback does not establish accurate variable-frame-rate or slow-motion timing.

**The camera changes resolution.** The run stops before using the old calibration on the new image. Recalibrate and select the marker again.

## Calibration and tracking

**Start tracking is disabled.** Open a source, complete two-point calibration, and select a valid color marker. If an import/analysis is still running, wait for it to finish.

**Calibration is rejected.** Use a finite positive reference length. Both endpoints must be in the source frame and at least 10 pixels apart. Spread them farther apart for better scale precision. Recalibrate using the actual measured reference at the object's depth.

**Color selection is rejected.** Click inside a bright saturated marker. White, gray, black, glare, and deep shadow do not provide a useful hue. Use a marker well separated in color from the background.

**The marker disappears or tracking is lost.** Improve illumination, shorten exposure/motion blur where possible, use a larger marker, reduce occlusion, and keep it in the frame. Select it again to begin a new continuity interval. Missing frames are not filled automatically.

**YOLO is not installed.** Color tracking works without it. For YOLO, use `python launch.py --with-yolo` or install `.[desktop,detector]` into the active environment. A GPU is optional. Use model weights from a trusted source. Model names/classes must match the loaded weights.

## Analysis and results

**Too few observations.** At least 12 observations are needed overall, with at least 8 in a modeled interval by default. Short fragments are not enough for stable derivatives. A large total count split among many short tracks may still produce no usable model.

**CSV import fails.** Required headers are `Time,X,Y`, with comma-separated UTF-8 data, seconds and meters, and one consistent number of columns per row. Times must be finite and strictly increasing. Duplicate headers/timestamps and invalid confidence scores are errors. Do not provide pixels as meters.

**Gravity or restitution says “Not established.”** The chosen model did not qualify. Use General motion for driven motion; use ballistic modes only for suitable experiments. Check scale, camera level, fit quality, sampling, and missing detections. Do not change thresholds solely to force an expected value.

**Acceleration is noisy in General motion.** Derivatives amplify position noise. Try a slightly larger smoothing window, compare the position model with observations, and inspect endpoints. A very large window can conceal real rapid motion. Quantitative uncertainty still needs an experiment-specific assessment.

**Energy is not constant.** Potential energy uses measured y and a fixed 9.81 m/s² field; kinetic energy uses modeled speed. The datum, mass, forces, air resistance, sampling, and tracking error all matter. A free-flight label does not prove an isolated conservative system.

**Export fails.** Observations stay in memory. Choose a writable folder with enough space, then choose **Analyze samples** again. A failed run's partial directory is removed; earlier successful runs are retained.

**PDF or folder does not open automatically.** Open the saved path shown in Results with your file manager. The operating system needs a PDF viewer/file-manager association; the report is also available as `summary.txt`, JSON, CSV, and PNG files.

**Closing seems delayed.** The app allows an active report export or blocked device operation to finish. It does not terminate threads while they own a camera or write a report. Numerical data in an unsaved, unfinished run are only in memory.
