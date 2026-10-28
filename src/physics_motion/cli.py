"""Camera-free command-line entry point; Qt and ML are optional dependencies."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data_io import read_samples_csv
from .engine import AnalysisSettings, analyze_samples


def demo_samples() -> pd.DataFrame:
    """Seeded projectile data in SI units, with 0.4 mm simulated position noise.

    Ground truth: vx=1.4 m/s, vy(0)=1 m/s, ay=-9.81 m/s²; t=0..0.65 s.
    These are generated coordinates, not a real camera measurement.
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0, 0.65, 66)
    return pd.DataFrame(
        {
            "Time": t,
            "X": 0.25 + 1.4 * t + rng.normal(0, 0.0004, len(t)),
            "Y": 3.0 + 1.0 * t - 0.5 * 9.81 * t**2 + rng.normal(0, 0.0004, len(t)),
        }
    )


def _odd_window(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Smoothing window must be an odd integer") from exc
    if number < 5 or number % 2 == 0:
        raise argparse.ArgumentTypeError("Smoothing window must be an odd integer of at least 5")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-motion", description="Analyze 2D position measurements in seconds and meters."
    )
    parser.add_argument(
        "--version", action="version", version=f"Physics Motion Analyzer {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show installed dependencies without opening a camera")
    for name in ("demo", "analyze"):
        cmd = commands.add_parser(
            name, help="Analyze a synthetic projectile" if name == "demo" else "Analyze a local CSV"
        )
        cmd.add_argument(
            "--output",
            type=Path,
            default=Path("output"),
            help="Parent directory for a new, unique run folder (default: output)",
        )
        cmd.add_argument(
            "--mass",
            type=float,
            default=0.2,
            help="Object mass in kg; replace the example default of 0.2 with your measurement",
        )
        cmd.add_argument(
            "--experiment",
            default="Horizontal toss" if name == "demo" else "General motion",
            help='"General motion", "Horizontal toss", or "Free-fall / bounce"',
        )
        cmd.add_argument(
            "--max-gap",
            type=float,
            default=1.0,
            help="Maximum time between samples in one fitted interval, in s",
        )
        cmd.add_argument(
            "--smoothing-window",
            type=_odd_window,
            default=9,
            help="Odd number of samples in a general-motion local fit (default: 9)",
        )
        cmd.add_argument(
            "--json", action="store_true", help="Print machine-readable results to stdout"
        )
        if name == "analyze":
            cmd.add_argument(
                "--input",
                required=True,
                type=Path,
                help="UTF-8 CSV with Time,X,Y in seconds, meters, meters",
            )
    return parser


def dependency_report() -> str:
    lines = [
        f"Physics Motion Analyzer {__version__}",
        f"Python {platform.python_version()} ({platform.system()} {platform.machine()})",
    ]
    for title, packages in (
        ("NumPy", ("numpy",)),
        ("pandas", ("pandas",)),
        ("Matplotlib", ("matplotlib",)),
        ("ReportLab", ("reportlab",)),
        ("Qt desktop", ("PySide6", "PySide6-Essentials")),
        ("OpenCV", ("opencv-python", "opencv-python-headless")),
        ("YOLO (optional)", ("ultralytics",)),
    ):
        version = "Not installed"
        for package in packages:
            try:
                version = importlib.metadata.version(package)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        lines.append(f"{title}: {version}")
    lines.append(
        "Camera access and detector inference are checked only when you open a source in the app."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(dependency_report())
        return 0
    from .output import export_run, summary_text

    try:
        settings = AnalysisSettings(
            mass_kg=args.mass,
            experiment_type=args.experiment,
            max_gap_seconds=args.max_gap,
            smoothing_window=args.smoothing_window,
        )
        settings.validate()
        frame = demo_samples() if args.command == "demo" else read_samples_csv(args.input)
        result = analyze_samples(frame, settings)
        context = {
            "origin": "Synthetic projectile demo" if args.command == "demo" else str(args.input),
            "measurement_origin": "CSV-supplied y = 0 datum",
            "synthetic": args.command == "demo",
        }
        folder = export_run(result, settings, args.output, context=context, raw_data=frame)
    except KeyboardInterrupt:
        parser.exit(130, "Analysis canceled.\n")
    except (OSError, ValueError, TypeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    if args.json:
        print(
            json.dumps(
                {
                    "output_directory": str(folder.resolve()),
                    "stats": result.stats,
                    "warnings": result.warnings,
                },
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(summary_text(result, settings.experiment_type))
        print(f"\nSaved results to: {folder.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
