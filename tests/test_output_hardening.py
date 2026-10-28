"""Regression checks for truthful plots and complete, portable exports."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from physics_motion import output
from physics_motion.cli import demo_samples
from physics_motion.engine import AnalysisSettings, analyze_samples


def test_model_plots_do_not_bridge_recording_gaps(tmp_path, monkeypatch):
    t = np.r_[np.linspace(0, 0.3, 16), np.linspace(2, 2.3, 16)]
    measured = pd.DataFrame({"Time": t, "X": t, "Y": 10 - t * t})
    result = analyze_samples(measured, AnalysisSettings(max_gap_seconds=0.2))
    figures = {}

    def inspect_figure(fig, path):
        figures[path.name] = fig

    monkeypatch.setattr(output, "_save_fig", inspect_figure)
    output._make_plots(result, tmp_path)
    for filename in ("velocity_time.png", "acceleration_time.png", "energy_time.png"):
        for line in figures[filename].axes[0].lines:
            finite_times = np.asarray(line.get_xdata(), dtype=float)
            # Each modeled line belongs to one contiguous segment.
            assert not np.any(np.diff(finite_times) > 0.2), filename
    assert float(figures["trajectory.png"].axes[0].get_aspect()) == 1.0


def test_export_always_includes_raw_observations_and_manifest(tmp_path):
    raw = demo_samples()
    settings = AnalysisSettings(experiment_type="Horizontal toss")
    run = output.export_run(analyze_samples(raw, settings), settings, tmp_path)
    restored = pd.read_csv(run / "raw_samples.csv")
    pd.testing.assert_frame_equal(restored, raw, check_exact=False)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    for name, entry in manifest["files"].items():
        content = (run / name).read_bytes()
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
        assert entry["bytes"] == len(content)
    config = json.loads((run / "config.json").read_text())
    assert config["software"]["version"]
    assert config["units"]["Time"] == "s"


def test_export_cleans_partial_runs_on_invalid_metadata(tmp_path):
    frame = demo_samples()
    settings = AnalysisSettings()
    with pytest.raises((ValueError, TypeError)):
        output.export_run(
            analyze_samples(frame, settings), settings, tmp_path, context={"invalid": float("nan")}
        )
    assert list(tmp_path.iterdir()) == []


def test_report_wraps_and_escapes_user_metadata(tmp_path):
    frame = demo_samples()
    settings = AnalysisSettings(experiment_type="Custom <experiment> & measurement")
    run = output.export_run(
        analyze_samples(frame, settings),
        settings,
        tmp_path,
        context={"object_name": "<ball & marker> " * 35, "origin": Path("measurements.csv")},
    )
    assert (run / "Analysis_Report.pdf").stat().st_size > 1000
    assert json.loads((run / "config.json").read_text())["context"]["origin"] == "measurements.csv"


def test_failed_report_removes_partial_run(tmp_path, monkeypatch):
    frame = demo_samples()
    settings = AnalysisSettings()

    def fail(*args, **kwargs):
        raise OSError("Disk write failed")

    monkeypatch.setattr(output, "_pdf_report", fail)
    with pytest.raises(OSError, match="Disk write failed"):
        output.export_run(analyze_samples(frame, settings), settings, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_summary_labels_fit_standard_error_separately_from_gravity():
    settings = AnalysisSettings(experiment_type="Horizontal toss")
    result = analyze_samples(demo_samples(), settings)
    text = output.summary_text(result, settings.experiment_type)
    assert "Gravity fit standard error:" in text
    assert "measurement uncertainty" in text
