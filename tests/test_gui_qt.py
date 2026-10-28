"""Run when PySide6 is installed. These tests do not open a webcam or load YOLO."""

import os
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from physics_motion.cli import demo_samples
from physics_motion.engine import AnalysisSettings
from physics_motion.gui import AnalysisWorker, MainWindow, VideoCanvas


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


def wait_until(app, condition, timeout=12):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert condition(), "The desktop operation did not finish in time"


@pytest.fixture
def window(app, tmp_path):
    window = MainWindow()
    window.output_root.setText(str(tmp_path / "runs"))
    yield window
    window.close()
    wait_until(
        app,
        lambda: all(
            worker is None or not worker.isRunning()
            for worker in (
                window.camera,
                window.analysis_thread,
                getattr(window, "import_thread", None),
            )
        ),
    )
    app.processEvents()


def test_letterboxed_video_click_mapping(app):
    canvas = VideoCanvas()
    canvas.resize(800, 600)
    image = QImage(960, 540, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    canvas.set_frame(image)
    # The 16:9 image is centered in 4:3 canvas; vertical bars of 75 pixels.
    assert canvas.map_widget_to_frame(QPoint(400, 300)) == (480, 270)
    assert canvas.map_widget_to_frame(QPoint(400, 10)) is None
    assert canvas.map_widget_to_frame(QPoint(0, 300)) == (0, 270)


def test_main_window_calibration_and_recalibration(app):
    window = MainWindow()
    try:
        assert not window.start_btn.isEnabled()
        assert not window.stop_btn.isEnabled()
        window._frame_size = (640, 480)
        window._state = "CALIBRATION"
        window._video_clicked(100, 100)
        assert window._calibration is None
        window._video_clicked(310, 100)
        assert window._calibration.pixels_per_meter == pytest.approx(1000)
        window.reference.setValue(0.42)
        assert window._calibration.pixels_per_meter == pytest.approx(500)
        window._begin_calibration()
        window._video_clicked(110, 110)
        assert window._calibration is None
        assert window._points == [(110, 110)]
    finally:
        window.close()


def test_analysis_worker_preserves_source_columns_without_confidence(app, tmp_path):
    raw = demo_samples()
    raw["TrackID"] = "blue-marker"
    raw["Source"] = "measured-positions.csv"
    raw["SourceTime"] = raw["Time"] + 120.0
    worker = AnalysisWorker(
        raw.to_dict("records"),
        AnalysisSettings(experiment_type="Horizontal toss"),
        tmp_path,
        {"origin": "CSV"},
    )
    completed, failed = [], []
    worker.completed.connect(lambda result, folder: completed.append((result, folder)))
    worker.failed.connect(failed.append)
    worker.run()
    assert not failed
    assert len(completed) == 1
    result, folder = completed[0]
    saved_raw = pd.read_csv(folder / "raw_samples.csv")
    assert "Confidence" not in saved_raw.columns
    assert saved_raw["TrackID"].tolist() == ["blue-marker"] * 66
    assert saved_raw["Source"].iloc[0] == "measured-positions.csv"
    assert saved_raw["SourceTime"].iloc[0] == pytest.approx(120)
    assert result.data["TrackID"].iloc[-1] == "blue-marker"


def test_capture_timing_warning_reaches_export_and_results(app, tmp_path):
    warning = "Nominal FPS used for source timing."
    worker = AnalysisWorker(
        demo_samples(),
        AnalysisSettings(experiment_type="Horizontal toss"),
        tmp_path,
        {"capture": {"warnings": [warning, warning]}},
    )
    completed, failed = [], []
    worker.completed.connect(lambda result, folder: completed.append((result, folder)))
    worker.failed.connect(failed.append)
    worker.run()
    assert not failed
    result, folder = completed[0]
    assert result.warnings.count("Capture: " + warning) == 1
    assert (folder / "summary.txt").read_text().count("Capture: " + warning) == 1


def test_csv_import_is_available_without_camera(app, window, tmp_path):
    path = tmp_path / "motion.csv"
    raw = demo_samples()
    raw["TrackID"] = "target-A"
    raw.to_csv(path, index=False)
    window._load_csv(path)
    wait_until(app, lambda: window.import_thread is None)
    assert len(window._records) == 66
    assert window._records[0]["TrackID"] == "target-A"
    assert window.camera is None
    assert window.analyze_btn.isEnabled()
    assert "motion.csv" in window.source_label.text()
    assert not window.start_btn.isEnabled()


def test_bad_import_clears_old_results_and_allows_another_source(app, window, tmp_path):
    window._latest_folder = tmp_path
    window._set_controls()
    assert window.open_report_btn.isEnabled()
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"height": [1, 2, 3]}).to_csv(path, index=False)
    window._load_csv(path)
    wait_until(app, lambda: window.import_thread is None)
    assert window._latest_folder is None
    assert window._records == []
    assert not window.open_report_btn.isEnabled()
    assert window.import_btn.isEnabled()
    assert window.demo_btn.isEnabled()
    assert window.notice.text()
    assert window._state == "IDLE"


def test_loading_demo_invalidates_previous_plot_and_report(window, tmp_path):
    image = QImage(300, 200, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    path = tmp_path / "plot.png"
    image.save(str(path))
    window._latest_folder = tmp_path
    window.plot_label.set_plot(path)
    window._set_controls()
    window._load_demo()
    assert len(window._records) == 66
    assert window._latest_folder is None
    assert not window.plot_label.has_plot()
    assert not window.open_report_btn.isEnabled()
    assert not window.open_folder_btn.isEnabled()
    assert window.analyze_btn.isEnabled()
    assert window.experiment.currentText() == "Horizontal toss"


def test_demo_analysis_locks_inputs_and_produces_openable_results(app, window):
    window._load_demo()
    window._analyze_records()
    assert window._state == "ANALYZING"
    assert not window.import_btn.isEnabled()
    assert not window.demo_btn.isEnabled()
    assert not window.mass.isEnabled()
    assert not window.analyze_btn.isEnabled()
    wait_until(app, lambda: window.analysis_thread is None)
    assert window._latest_folder is not None
    assert (window._latest_folder / "Analysis_Report.pdf").is_file()
    assert (window._latest_folder / "analysis_data.csv").is_file()
    assert window.open_report_btn.isEnabled()
    assert window.open_folder_btn.isEnabled()
    assert window.analyze_btn.isEnabled()
    assert window._latest_result.stats["samples"] == 66
    assert window.segment_table.rowCount() > 0
    assert window.plot_label.has_plot()


def test_failed_analysis_preserves_samples_for_retry_and_clears_exports(app, window):
    window._load_demo()
    window._records[5]["Time"] = window._records[4]["Time"]
    window._analyze_records()
    wait_until(app, lambda: window.analysis_thread is None)
    assert window._latest_folder is None
    assert len(window._records) == 66
    assert window.analyze_btn.isEnabled()
    assert window.import_btn.isEnabled()
    assert not window.open_report_btn.isEnabled()
    assert window.notice.text()
    assert window._state == "DATA_READY"


def test_natural_recording_end_starts_analysis_without_stop_click(app, window):
    window._settings = AnalysisSettings(experiment_type="Horizontal toss")
    window._state = "TRACKING"
    window._recording_finished(demo_samples().to_dict("records"))
    assert window._state == "ANALYZING"
    wait_until(app, lambda: window.analysis_thread is None)
    assert window._latest_folder is not None
    assert window._latest_result.stats["samples"] == 66


def test_missing_video_recovers_without_starting_capture(window, tmp_path):
    window.source_choice.setCurrentIndex(window.source_choice.findData("video"))
    window.video_path.setText(str(tmp_path / "missing-video.mp4"))
    window._open_camera()
    assert window.camera is None
    assert window.open_btn.isEnabled()
    assert window.notice.text()


def test_changed_analysis_settings_invalidate_previous_results(window, tmp_path):
    window._load_demo()
    window._latest_folder = tmp_path
    window._set_controls()
    assert window.open_report_btn.isEnabled()
    window.mass.setValue(0.3)
    assert window._latest_folder is None
    assert not window.open_report_btn.isEnabled()
    assert window.analyze_btn.isEnabled()
    assert len(window._records) == 66


def test_recalibration_starts_a_fresh_measurement(window, tmp_path):
    window._load_demo()
    window._latest_folder = tmp_path
    window._frame_size = (640, 480)
    window._begin_calibration()
    assert window._records == []
    assert window._latest_folder is None
    assert not window.analyze_btn.isEnabled()
    assert window._state == "CALIBRATION"


def test_close_waits_asynchronously_for_analysis_export(app, window):
    window.show()
    window._load_demo()
    window._analyze_records()
    window.close()
    assert window._closing
    assert window.isVisible()
    wait_until(app, lambda: window.analysis_thread is None and not window.isVisible())
    assert window._latest_folder is not None


def test_recorded_marker_video_runs_through_desktop_controls(app, window):
    pytest.importorskip("cv2")
    video = Path(__file__).parents[1] / "examples" / "marker_projectile.avi"
    window.show()
    window.source_choice.setCurrentIndex(window.source_choice.findData("video"))
    window.video_path.setText(str(video))
    window.open_btn.click()
    wait_until(app, lambda: window._frame_size == (960, 640))
    assert not window.import_btn.isEnabled()
    window.reference.setValue(1.0)
    window.experiment.setCurrentText("Horizontal toss")
    window.calibrate_btn.click()
    window._video_clicked(690, 570)
    window._video_clicked(840, 570)
    assert window._calibration.pixels_per_meter == pytest.approx(150)
    assert not window.start_btn.isEnabled()

    # A neutral background click cannot satisfy marker readiness.
    window.select_btn.click()
    window._video_clicked(10, 10)
    wait_until(app, lambda: "distinct" in window.target_label.text())
    assert not window.start_btn.isEnabled()
    window.select_btn.click()
    window._video_clicked(105, 190)
    wait_until(app, lambda: window.start_btn.isEnabled())
    assert window._calibration.pixels_per_meter == pytest.approx(150)
    window.start_frame.setValue(5)
    assert not window.start_btn.isEnabled()
    window.start_frame.setValue(0)
    assert window.start_btn.isEnabled()
    window.seek_btn.click()
    assert not window.start_btn.isEnabled()
    wait_until(app, lambda: window._state == "READY")
    assert not window._marker_selected
    window.select_btn.click()
    window._video_clicked(105, 190)
    wait_until(app, lambda: window.start_btn.isEnabled())
    window.start_btn.click()
    assert window._state == "TRACKING"
    assert not window.reference.isEnabled()
    wait_until(
        app,
        lambda: (
            window._latest_folder is not None
            and window.analysis_thread is None
            and window.camera is None
        ),
        timeout=20,
    )
    assert len(window._records) == 97
    assert window._latest_result.stats["gravity_estimate_m_s2"] == pytest.approx(9.81, abs=0.1)
    assert window._context["capture"]["finish_reason"] == "end_of_video"
    assert window._context["capture"]["accepted_samples"] == 97
    assert window.open_report_btn.isEnabled()
    assert "Recording" not in window.sample_label.text()


def test_video_can_start_where_a_later_marker_becomes_visible(app, window, tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "late-marker.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 200))
    assert writer.isOpened()
    for index in range(40):
        frame = np.full((200, 320, 3), 30, dtype=np.uint8)
        if index >= 15:
            cv2.circle(frame, (60 + 3 * (index - 15), 60 + index - 15), 8, (0, 230, 0), -1)
        writer.write(frame)
    writer.release()

    window.source_choice.setCurrentIndex(window.source_choice.findData("video"))
    window.video_path.setText(str(video))
    window.open_btn.click()
    wait_until(app, lambda: window._frame_size == (320, 200))
    window.reference.setValue(1)
    window.calibrate_btn.click()
    window._video_clicked(40, 170)
    window._video_clicked(140, 170)
    assert not window.start_btn.isEnabled()
    assert window.start_frame.maximum() == 39
    window.start_frame.setValue(15)
    window.seek_btn.click()
    assert window._state == "SEEKING"
    assert not window.calibrate_btn.isEnabled()
    assert not window.select_btn.isEnabled()
    assert window.disconnect_btn.isEnabled()
    wait_until(app, lambda: window._state == "READY")
    assert window._calibration.pixels_per_meter == pytest.approx(100)
    assert not window._marker_selected
    window.select_btn.click()
    window._video_clicked(60, 60)
    wait_until(app, lambda: window.start_btn.isEnabled())
    window.start_btn.click()
    wait_until(
        app,
        lambda: (
            window._latest_folder is not None
            and window.analysis_thread is None
            and window.camera is None
        ),
        timeout=20,
    )
    assert len(window._records) == 25
    assert window._records[0]["Frame"] == 15
    assert window._records[0]["SourceTime"] == pytest.approx(0.5)
    assert window._records[0]["Time"] == pytest.approx(0)
