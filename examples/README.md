# Example measurements

All files outside `archive/` are synthetic. They test software behavior; they are not camera experiments.

| File | Analysis mode | Ground truth |
|---|---|---|
| `projectile.csv` | Horizontal toss | vx = 1.4 m/s; ay = −9.81 m/s²; 0.4 mm seeded position noise. |
| `circular_motion.csv` | General motion | Radius 0.6 m; angular speed 2 rad/s; speed 1.2 m/s; acceleration magnitude 2.4 m/s². Gravity/restitution are not established by this motion. |
| `bounce.csv` | Free-fall / bounce | Drop from 1.22625 m; impact at 0.5 s; restitution 0.72; ay = −9.81 m/s² on each free-flight interval. |
| `marker_projectile.avi` | Horizontal toss | 960×640 MJPEG, 120 frames/s; vx = 2.5 m/s; vy(0) = 1 m/s; ay = −9.81 m/s². |
| `marker_projectile_truth.csv` | Horizontal toss | Exact analytical positions encoded by the video before pixel rounding/compression. |
| `marker_projectile.json` | Setup reference | Video calibration endpoints, scale, first marker position, and motion constants. |

For the video, enter **1.0 m**, calibrate the white endpoint centers **(690,570)** and **(840,570)**, then select the green marker at **(105,190)**. The scale is **150 px/m**. Click locations are native image coordinates; the app maps them correctly when the preview is resized or letterboxed.

Regenerate the video and truth files after installing the desktop dependencies:

```bash
python scripts/generate_example_video.py
```

The example video has no audio and contains no external or private media. Its time scale is **recorded media time**; processing may run faster or slower than playback.

## Completed example

`demo_result/` contains a complete saved run from the seeded coordinate demo: the [PDF report](demo_result/Analysis_Report.pdf), all six plots, original and analyzed samples, settings, summaries, and checksums. Its estimated gravity is 9.8095981455 m/s² for the synthetic input of 9.81 m/s². Regenerate it with `physics-motion demo --output output`; each run receives a new timestamped directory.
