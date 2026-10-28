import json
import subprocess
import sys

import numpy as np
import pandas as pd

from physics_motion.cli import demo_samples
from physics_motion.engine import AnalysisSettings, analyze_samples
from physics_motion.output import GRAPH_FILES, export_run


def test_report_outputs_complete_nonoverwriting(tmp_path):
    input_frame = demo_samples()
    settings = AnalysisSettings(experiment_type="Horizontal toss")
    result = analyze_samples(input_frame, settings)
    first = export_run(result, settings, tmp_path, {"object_name": "synthetic"}, input_frame)
    second = export_run(result, settings, tmp_path, {"object_name": "synthetic"}, input_frame)
    assert first != second
    for folder in (first, second):
        assert (folder / "Analysis_Report.pdf").read_bytes().startswith(b"%PDF-")
        assert (folder / "analysis_data.csv").exists()
        assert (folder / "raw_samples.csv").exists()
        assert (folder / "config.json").exists()
        assert (folder / "summary.json").exists()
        assert all((folder / filename).stat().st_size > 3000 for filename in GRAPH_FILES.values())
        restored = pd.read_csv(folder / "analysis_data.csv")
        assert len(restored) == len(input_frame)
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["stats"]["gravity_estimate_m_s2"] is not None
        assert np.allclose(restored.PE + restored.KE, restored.E_total)


def test_cli_demo_runs_without_camera(tmp_path):
    cmd = [sys.executable, "-m", "physics_motion.cli", "demo", "--output", str(tmp_path)]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert "gravity estimate" in completed.stdout.lower()
    assert len(list(tmp_path.glob("run-*/Analysis_Report.pdf"))) == 1


def test_cli_analyze_csv(tmp_path):
    source = tmp_path / "measurements.csv"
    demo_samples().to_csv(source, index=False)
    cmd = [
        sys.executable,
        "-m",
        "physics_motion.cli",
        "analyze",
        "--input",
        str(source),
        "--output",
        str(tmp_path / "results"),
        "--experiment",
        "Horizontal toss",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert len(list((tmp_path / "results").glob("run-*/summary.json"))) == 1
