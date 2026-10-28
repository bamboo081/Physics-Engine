#!/usr/bin/env python3
"""Generate a synthetic, constant-frame-rate color-marker validation fixture.

Run from the project root after installing the desktop extra. No camera, model,
network, or external media is used. The generated CSV contains exact analytical
positions; the AVI contains rounded pixel positions and MJPEG compression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    width, height, fps, pixels_per_meter = 960, 640, 120, 150
    t = np.arange(97, dtype=float) / fps
    x = 0.7 + 2.5 * t
    y = 3.0 + t - 0.5 * 9.81 * t**2
    video = destination / "marker_projectile.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MJPEG video encoder")
    try:
        for elapsed, x_m, y_m in zip(t, x, y, strict=True):
            frame = np.full((height, width, 3), (34, 39, 45), dtype=np.uint8)
            cv2.putText(
                frame,
                "Synthetic projectile - 120 frames/s",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 233, 237),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"t = {elapsed:.3f} s",
                (30, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 189, 201),
                1,
                cv2.LINE_AA,
            )
            cv2.line(frame, (690, 570), (840, 570), (245, 245, 245), 2)
            for point in ((690, 570), (840, 570)):
                cv2.circle(frame, point, 5, (245, 245, 245), -1)
            cv2.putText(
                frame,
                "1.000 m",
                (724, 603),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            center = (round(x_m * pixels_per_meter), round(height - y_m * pixels_per_meter))
            cv2.circle(frame, center, 12, (50, 215, 70), -1, cv2.LINE_8)
            writer.write(frame)
    finally:
        writer.release()
    pd.DataFrame({"Time": t, "X": x, "Y": y}).to_csv(
        destination / "marker_projectile_truth.csv", index=False
    )
    metadata = {
        "synthetic": True,
        "frame_size": [width, height],
        "fps": fps,
        "frames": len(t),
        "reference_length_m": 1.0,
        "reference_points_px": [[690, 570], [840, 570]],
        "first_marker_center_px": [105, 190],
        "pixels_per_meter": pixels_per_meter,
        "x0_m": 0.7,
        "y0_m": 3.0,
        "vx_m_s": 2.5,
        "vy0_m_s": 1.0,
        "ax_m_s2": 0.0,
        "ay_m_s2": -9.81,
        "coordinate_origin": "bottom-left edge of 960x640 frame",
        "note": "Truth CSV is exact; video includes pixel rounding and MJPEG compression.",
    }
    (destination / "marker_projectile.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples"))
    generate(parser.parse_args().output)
