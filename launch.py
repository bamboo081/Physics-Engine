#!/usr/bin/env python3
"""Create a project-local environment and launch the app using only the stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import venv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up and start Physics Motion Analyzer")
    parser.add_argument("--demo", action="store_true", help="Run the camera-free synthetic demo")
    parser.add_argument("--doctor", action="store_true", help="Check installed dependencies")
    parser.add_argument(
        "--setup-only", action="store_true", help="Install dependencies without opening the app"
    )
    parser.add_argument(
        "--with-yolo", action="store_true", help="Also install the optional YOLO/PyTorch detector"
    )
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        print(
            "Use Python 3.10–3.13 (3.12 recommended). This release has not validated Python 3.14+.",
            file=sys.stderr,
        )
        return 2
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = environment / ".physics-motion-install.json"
    extras = (
        "desktop,detector" if args.with_yolo else ("" if args.demo or args.doctor else "desktop")
    )
    requested = set(filter(None, extras.split(",")))
    signature = hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest()
    try:
        if not python.exists():
            print("Creating .venv in the project folder…", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        try:
            installed = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            installed = {}
        if installed.get("project") != signature or not requested.issubset(
            installed.get("extras", [])
        ):
            # Include previously installed options when upgrading this project.
            requested |= set(installed.get("extras", [])) & {"desktop", "detector"}
            extra_suffix = f"[{','.join(sorted(requested))}]" if requested else ""
            print(
                "Installing dependencies into .venv. First setup requires internet access.",
                flush=True,
            )
            subprocess.run(
                [str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=root, check=True
            )
            subprocess.run(
                [str(python), "-m", "pip", "install", "-e", f".{extra_suffix}"],
                cwd=root,
                check=True,
            )
            marker.write_text(
                json.dumps({"project": signature, "extras": sorted(requested)}) + "\n",
                encoding="utf-8",
            )
        if args.setup_only:
            print("Setup complete. Start the app with: python launch.py")
            return 0
        if args.demo:
            command = [
                str(python),
                "-m",
                "physics_motion",
                "demo",
                "--output",
                str(root / "output"),
            ]
        elif args.doctor:
            command = [str(python), "-m", "physics_motion", "doctor"]
        else:
            command = [str(python), "-m", "physics_motion.gui"]
        return subprocess.run(command, cwd=root, check=False).returncode
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup or launch failed: {exc}", file=sys.stderr)
        print("See docs/troubleshooting.md. Your measurements are not removed.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup canceled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
