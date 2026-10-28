"""The desktop report renderer cannot take down its parent or publish partial runs."""

import hashlib
import json
import multiprocessing
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from physics_motion import export_worker as renderer
from physics_motion.cli import demo_samples
from physics_motion.engine import AnalysisSettings, analyze_samples
from physics_motion.output import GRAPH_FILES


@pytest.fixture
def measurements():
    raw = demo_samples()
    raw["TrackID"] = "blue-marker"
    raw["Source"] = "measured-positions.csv"
    raw["SourceTime"] = raw["Time"] + 120.0
    settings = AnalysisSettings(experiment_type="Horizontal toss", mass_kg=0.35)
    return raw, settings, analyze_samples(raw, settings)


def _file_hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir()}


def _crash_after_partial_export(connection, result, settings, stage, context, raw_data):
    """A real spawned child bypasses all Python cleanup, as a native crash would."""
    folder = Path(stage) / "run-incomplete"
    folder.mkdir()
    result.data.to_csv(folder / "analysis_data.csv", index=False)
    (folder / "trajectory.png").write_bytes(b"unfinished image")
    if context.get("send_success_before_crash"):
        (folder / "manifest.json").write_text("{}", encoding="utf-8")
        connection.send(("ok", folder.name))
        connection.close()
    os._exit(139)


def test_isolated_export_preserves_report_data_metadata_and_manifest(measurements, tmp_path):
    raw, settings, result = measurements
    expected_raw = raw.copy(deep=True)
    expected_data = result.data.copy(deep=True)
    context = {"object_name": "blue marker", "origin": Path("measured-positions.csv")}

    folder = renderer.export_run_isolated(result, settings, tmp_path, context, raw)

    assert folder.parent == tmp_path
    assert folder.name.startswith("run-")
    assert list(tmp_path.iterdir()) == [folder]
    assert (folder / "Analysis_Report.pdf").read_bytes().startswith(b"%PDF-")
    assert all((folder / name).stat().st_size > 3000 for name in GRAPH_FILES.values())
    pd.testing.assert_frame_equal(pd.read_csv(folder / "raw_samples.csv"), expected_raw)
    pd.testing.assert_frame_equal(
        pd.read_csv(folder / "analysis_data.csv"), expected_data, check_dtype=False
    )
    pd.testing.assert_frame_equal(raw, expected_raw)
    pd.testing.assert_frame_equal(result.data, expected_data)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["settings"] == asdict(settings)
    assert config["context"] == {
        "object_name": "blue marker",
        "origin": "measured-positions.csv",
    }
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["stats"] == result.stats
    assert summary["warnings"] == result.warnings
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert set(manifest["files"]) == {
        "analysis_data.csv",
        "raw_samples.csv",
        "config.json",
        "summary.json",
        "summary.txt",
        "Analysis_Report.pdf",
        *GRAPH_FILES.values(),
    }
    for name, entry in manifest["files"].items():
        contents = (folder / name).read_bytes()
        assert entry == {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}


def test_isolated_python_error_is_reported_without_partial_output(measurements, tmp_path):
    raw, settings, result = measurements
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("existing output", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Report export failed: ValueError"):
        renderer.export_run_isolated(
            result, settings, tmp_path, context={"invalid": float("nan")}, raw_data=raw
        )

    assert list(tmp_path.iterdir()) == [unrelated]
    assert unrelated.read_text(encoding="utf-8") == "existing output"


@pytest.mark.parametrize("send_success_before_crash", [False, True])
def test_native_exit_preserves_previous_run_and_allows_retry(
    measurements, tmp_path, monkeypatch, send_success_before_crash
):
    raw, settings, result = measurements
    first = renderer.export_run_isolated(result, settings, tmp_path, raw_data=raw)
    first_hashes = _file_hashes(first)
    existing_children = {child.pid for child in multiprocessing.active_children()}

    with monkeypatch.context() as patch:
        patch.setattr(renderer, "_render_export", _crash_after_partial_export)
        with pytest.raises(RuntimeError, match="exit code 139"):
            renderer.export_run_isolated(
                result,
                settings,
                tmp_path,
                {"send_success_before_crash": send_success_before_crash},
                raw,
            )

    assert list(tmp_path.iterdir()) == [first]
    assert _file_hashes(first) == first_hashes
    assert {child.pid for child in multiprocessing.active_children()} == existing_children
    second = renderer.export_run_isolated(result, settings, tmp_path, raw_data=raw)
    assert second != first
    assert set(tmp_path.iterdir()) == {first, second}
    assert (second / "Analysis_Report.pdf").read_bytes().startswith(b"%PDF-")
    assert (second / "manifest.json").is_file()
    assert _file_hashes(first) == first_hashes


@pytest.mark.parametrize(
    "settings, message",
    [
        (AnalysisSettings(mass_kg=-1), "Mass"),
        (AnalysisSettings(experiment_type="Horizontal toss", mass_kg=0.2), "settings"),
    ],
)
def test_isolated_export_rejects_invalid_or_mismatched_settings(
    measurements, tmp_path, settings, message
):
    raw, _, result = measurements
    with pytest.raises(ValueError, match=message):
        renderer.export_run_isolated(result, settings, tmp_path, raw_data=raw)
    assert not list(tmp_path.iterdir())


def test_interrupted_parent_reaps_its_child_and_removes_staging(
    measurements, tmp_path, monkeypatch
):
    raw, settings, result = measurements
    existing_children = {child.pid for child in multiprocessing.active_children()}

    def interrupt_wait(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(renderer, "wait", interrupt_wait)
    with pytest.raises(KeyboardInterrupt):
        renderer.export_run_isolated(result, settings, tmp_path, raw_data=raw)
    assert not list(tmp_path.iterdir())
    assert {child.pid for child in multiprocessing.active_children()} == existing_children


def test_publish_failure_removes_only_its_own_new_output(measurements, tmp_path, monkeypatch):
    raw, settings, result = measurements
    preserved = tmp_path / "preserved.txt"
    preserved.write_bytes(b"previous experiment")
    original_replace = Path.replace
    promoted = []

    def fail_during_publish(source, destination):
        if promoted:
            raise OSError("Destination became unavailable")
        promoted.append(source.name)
        assert source.name != "manifest.json", "A partial report must not have a manifest"
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_during_publish)
    with pytest.raises(OSError, match="Destination became unavailable"):
        renderer.export_run_isolated(result, settings, tmp_path, raw_data=raw)
    assert len(promoted) == 1
    assert list(tmp_path.iterdir()) == [preserved]
    assert preserved.read_bytes() == b"previous experiment"


def test_desktop_native_exit_retains_observations_and_allows_retry(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from physics_motion.gui import MainWindow

    app = widgets.QApplication.instance() or widgets.QApplication([])
    window = MainWindow()
    window.output_root.setText(str(tmp_path))

    def finish_analysis():
        deadline = time.monotonic() + 20
        while window.analysis_thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        assert window.analysis_thread is None, "The desktop analysis worker did not finish"

    try:
        window._load_demo()
        original_records = [dict(record) for record in window._records]
        with monkeypatch.context() as patch:
            patch.setattr(renderer, "_render_export", _crash_after_partial_export)
            window._analyze_records()
            finish_analysis()

        assert "exit code 139" in window.notice.text()
        assert window._records == original_records
        assert window._latest_folder is None
        assert window._latest_result is None
        assert window._state == "DATA_READY"
        assert window.analyze_btn.isEnabled()
        assert not window.open_report_btn.isEnabled()
        assert not list(tmp_path.iterdir())

        window._analyze_records()
        finish_analysis()
        assert window._records == original_records
        assert window._latest_result.stats["samples"] == 66
        assert (window._latest_folder / "Analysis_Report.pdf").is_file()
        assert (window._latest_folder / "manifest.json").is_file()
        assert window.open_report_btn.isEnabled()
    finally:
        window.close()
        if window.analysis_thread is not None:
            window.analysis_thread.wait(30000)
        app.processEvents()
