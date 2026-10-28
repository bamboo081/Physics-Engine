"""Desktop workspace for capture, measured positions, and reproducible analysis.

OpenCV and optional YOLO operations belong to ``capture.CameraWorker``. CSV reads,
numerical analysis, plots, and report export also run outside the GUI thread.
"""

from __future__ import annotations

import math
import multiprocessing
import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QImage,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .calibration import Calibration
from .capture import CameraWorker
from .cli import demo_samples
from .data_io import read_samples_csv
from .engine import AnalysisResult, AnalysisSettings, analyze_samples
from .export_worker import export_run_isolated
from .output import GRAPH_FILES, GRAPH_LABELS, summary_text

COCO_PRESETS = [
    ("Sports ball", 32),
    ("Bottle", 39),
    ("Person", 0),
    ("Cup", 41),
    ("Custom COCO class", None),
]
EXPERIMENT_TYPES = ["General motion", "Free-fall / bounce", "Horizontal toss"]
BUSY_STATES = {
    "OPENING",
    "LOADING",
    "SEEKING",
    "TRACKING",
    "STOPPING",
    "ANALYZING",
    "DISCONNECTING",
}


def apply_palette(widget: QWidget) -> None:
    """Use explicit light colors, independent of the operating-system theme."""
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#f5f6f7"),
        (QPalette.ColorRole.WindowText, "#20252b"),
        (QPalette.ColorRole.Base, "#ffffff"),
        (QPalette.ColorRole.AlternateBase, "#f6f7f8"),
        (QPalette.ColorRole.Text, "#20252b"),
        (QPalette.ColorRole.Button, "#ffffff"),
        (QPalette.ColorRole.ButtonText, "#20252b"),
        (QPalette.ColorRole.Highlight, "#2367ad"),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.ToolTipBase, "#ffffff"),
        (QPalette.ColorRole.ToolTipText, "#20252b"),
        (QPalette.ColorRole.PlaceholderText, "#727c87"),
    ):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#8b939b"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#8b939b"))
    widget.setPalette(palette)
    widget.setStyleSheet("""
        QWidget { color: #20252b; font-size: 12px; }
        QMainWindow, QScrollArea, QWidget#sidebar { background: #f5f6f7; }
        QScrollArea { border: none; }
        QGroupBox { background: #ffffff; border: 1px solid #d8dde3; border-radius: 5px;
                    margin-top: 10px; padding: 8px 6px 6px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                          color: #39434e; }
        QLabel { background: transparent; }
        QLabel#muted { color: #64707d; }
        QLabel#dataset { font-size: 16px; font-weight: 600; }
        QLabel#error { color: #a42b2b; padding: 7px 0; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #ffffff;
            border: 1px solid #c8cfd7; border-radius: 4px; min-height: 26px; padding: 0 6px; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #2367ad; }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            background: #f2f4f6; color: #8b939b; border-color: #e1e5e9; }
        QComboBox QAbstractItemView { background: #ffffff; color: #20252b;
            selection-background-color: #2367ad; selection-color: #ffffff; }
        QPushButton { background: #ffffff; color: #25313d; border: 1px solid #c6cdd5;
            border-radius: 4px; min-height: 29px; padding: 0 10px; }
        QPushButton:hover { background: #f0f4f8; border-color: #9aa9b8; }
        QPushButton:pressed, QPushButton:checked { background: #e6eef7; border-color: #2367ad; }
        QPushButton:focus { border: 1px solid #2367ad; }
        QPushButton[primary="true"] { background: #2367ad; color: #ffffff; border-color: #2367ad; }
        QPushButton[primary="true"]:hover { background: #1b568f; }
        QPushButton:disabled { color: #929aa3; background: #f2f4f6; border-color: #e1e5e9; }
        QPushButton[primary="true"]:disabled { color: #8b99a6; background: #e5ebf1;
            border-color: #dce3ea; }
        QToolButton { border: none; color: #4e5c6b; background: transparent; padding: 3px 0; }
        QToolButton:hover { color: #2367ad; }
        QTabWidget::pane { background: #ffffff; border: 1px solid #d8dde3; border-radius: 3px; }
        QTabBar::tab { background: #eef1f4; color: #596572; padding: 10px 20px;
            border: 1px solid #d8dde3; border-bottom: none; margin-right: 3px; }
        QTabBar::tab:selected { background: #ffffff; color: #2367ad; }
        QTabBar::tab:focus { color: #2367ad; }
        QTextEdit, QTableWidget { background: #ffffff; color: #26313d;
            border: 1px solid #dce1e6; border-radius: 3px; selection-background-color: #e7eff8; }
        QHeaderView::section { color: #53606e; background: #f2f4f6; border: none;
            border-bottom: 1px solid #dce1e6; padding: 7px 9px; }
        QTableWidget::item { padding: 5px 8px; border: none; }
        QProgressBar { background: #e7ecf1; border: none; border-radius: 2px; max-height: 5px; }
        QProgressBar::chunk { background: #2367ad; border-radius: 2px; }
        QStatusBar { background: #f5f6f7; color: #5f6b77; border-top: 1px solid #dce1e6; }
        QToolTip { background: #ffffff; color: #20252b; border: 1px solid #c8cfd7;
            padding: 5px; }
    """)


class VideoCanvas(QWidget):
    clicked = Signal(int, int)

    def __init__(self):
        super().__init__()
        self._pixmap = QPixmap()
        self._frame_size = (0, 0)
        self._points: list[tuple[int, int]] = []
        self._empty_text = "No camera or video open"
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip(
            "Calibration and target selection use original video pixels, including when resized."
        )

    def set_frame(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._frame_size = image.width(), image.height()
        self.update()

    def clear(self, text: str = "No camera or video open") -> None:
        self._pixmap = QPixmap()
        self._frame_size = (0, 0)
        self._points = []
        self._empty_text = text
        self.update()

    def set_calibration_points(self, points: list[tuple[int, int]]) -> None:
        self._points = list(points)
        self.update()

    def _frame_rect(self) -> QRect:
        if self._pixmap.isNull():
            return QRect()
        size = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(
            (self.width() - size.width()) // 2,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )

    def map_widget_to_frame(self, position: QPoint) -> tuple[int, int] | None:
        rect = self._frame_rect()
        if rect.isNull() or not rect.contains(position):
            return None
        width, height = self._frame_size
        return (
            min(width - 1, int((position.x() - rect.left()) * width / rect.width())),
            min(height - 1, int((position.y() - rect.top()) * height / rect.height())),
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            result = self.map_widget_to_frame(event.position().toPoint())
            if result is not None:
                self.clicked.emit(*result)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f1f3f5"))
        rect = self._frame_rect()
        if rect.isNull():
            painter.setPen(QColor("#697583"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return
        painter.drawPixmap(rect, self._pixmap)
        width, height = self._frame_size
        coords = [
            QPoint(
                rect.left() + round(x * rect.width() / width),
                rect.top() + round(y * rect.height() / height),
            )
            for x, y in self._points
        ]
        painter.setPen(QPen(QColor("#2367ad"), 2))
        if len(coords) == 2:
            painter.drawLine(*coords)
        for index, point in enumerate(coords, 1):
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(point, 6, 6)
            painter.drawText(point + QPoint(10, -8), str(index))


class LiveHeightGraph(QWidget):
    """Small live plot; lines never bridge a lost track or a large time gap."""

    def __init__(self):
        super().__init__()
        self.samples: list[tuple] = []
        self.max_gap_seconds = 1.0
        self.setMinimumHeight(145)
        self.setMaximumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_samples(self, samples: list[tuple]) -> None:
        self.samples = samples[-500:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(68, 15, -16, -30)
        painter.setPen(QPen(QColor("#dfe4e9"), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        if len(self.samples) < 2:
            painter.setPen(QColor("#79838e"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Height (m) vs time (s)")
            return
        valid = [row for row in self.samples if math.isfinite(row[0]) and math.isfinite(row[1])]
        if len(valid) < 2:
            return
        tmin, tmax = valid[0][0], valid[-1][0]
        ymin, ymax = min(row[1] for row in valid), max(row[1] for row in valid)
        pad = max((ymax - ymin) * 0.12, 0.005)
        ymin, ymax = ymin - pad, ymax + pad
        path = QPainterPath()
        previous = None
        for row in valid:
            x = rect.left() + (row[0] - tmin) / max(tmax - tmin, 1e-9) * rect.width()
            y = rect.bottom() - (row[1] - ymin) / (ymax - ymin) * rect.height()
            continuous = previous is not None and 0 < row[0] - previous[0] <= self.max_gap_seconds
            if previous is not None and len(row) > 2 and len(previous) > 2:
                continuous = continuous and row[2] == previous[2]
            path.lineTo(x, y) if continuous else path.moveTo(x, y)
            previous = row
        painter.setPen(QPen(QColor("#2367ad"), 2))
        painter.drawPath(path)
        painter.setPen(QColor("#5f6c7a"))
        painter.drawText(4, rect.top() + 10, f"{ymax:.2f} m")
        painter.drawText(4, rect.bottom(), f"{ymin:.2f} m")
        painter.drawText(rect.left(), self.height() - 8, f"{tmin:.2f} s")
        painter.drawText(rect.right() - 65, self.height() - 8, f"{tmax:.2f} s")


class PlotLabel(QLabel):
    """Rescale from the original image so resize never degrades an exported plot."""

    def __init__(self):
        super().__init__()
        self._original = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(100, 160)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: #ffffff; color: #697583")
        self.setText("Analyze a recording or CSV to view plots")

    def set_plot(self, path: Path) -> None:
        self._original = QPixmap(str(path))
        if self._original.isNull():
            self.setText("This plot is unavailable. Open the results folder to inspect the export.")
        else:
            self._rescale()

    def clear_plot(self, text: str = "Analyze a recording or CSV to view plots") -> None:
        self._original = QPixmap()
        self.clear()
        self.setText(text)

    def has_plot(self) -> bool:
        return not self._original.isNull()

    def _rescale(self) -> None:
        if not self._original.isNull():
            self.setPixmap(
                self._original.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def sizeHint(self):
        return QSize(750, 480)


class ImportWorker(QThread):
    completed = Signal(object, object)
    failed = Signal(str)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            frame = read_samples_csv(self.path)
            self.completed.emit(frame, self.path)
        except Exception as exc:
            self.failed.emit(f"Could not import CSV: {exc}")


class AnalysisWorker(QThread):
    completed = Signal(object, object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        records: list[dict] | pd.DataFrame,
        settings: AnalysisSettings,
        root: Path,
        context: dict,
    ):
        super().__init__()
        self.records = (
            records.copy(deep=True)
            if isinstance(records, pd.DataFrame)
            else [dict(record) for record in records]
        )
        self.settings = settings
        self.root = root
        self.context = dict(context)

    def run(self):
        try:
            # Preserve TrackID, media times, source metadata, and genuinely optional Confidence.
            raw = pd.DataFrame(self.records)
            self.progress.emit("Analyzing measured positions…")
            analysis = analyze_samples(raw, self.settings)
            for message in self.context.get("capture", {}).get("warnings", []):
                warning = f"Capture: {message}"
                if warning not in analysis.warnings:
                    analysis.warnings.append(warning)
            self.progress.emit("Saving plots, data, and PDF report…")
            folder = export_run_isolated(
                analysis, self.settings, self.root, self.context, raw_data=raw
            )
            self.completed.emit(analysis, folder)
        except Exception as exc:
            self.failed.emit(f"Analysis failed: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physics Motion Analyzer")
        self.setMinimumSize(1000, 700)
        self.resize(1280, 800)
        apply_palette(self)
        self.camera: CameraWorker | None = None
        self.analysis_thread: AnalysisWorker | None = None
        self.import_thread: ImportWorker | None = None
        self._state = "IDLE"
        self._closing = False
        self._frame_size = (0, 0)
        self._applied_start_frame = 0
        self._points: list[tuple[int, int]] = []
        self._calibration: Calibration | None = None
        self._marker_selected = False
        self._records: list[dict] = []
        self._settings: AnalysisSettings | None = None
        self._context: dict = {}
        self._latest_folder: Path | None = None
        self._latest_result: AnalysisResult | None = None
        self._build_ui()
        self._set_controls()
        self._graph_timer = QTimer(self)
        self._graph_timer.timeout.connect(self._refresh_graph)
        self._graph_timer.start(250)

    @staticmethod
    def _button(text: str, callback, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("primary", primary)
        button.clicked.connect(callback)
        return button

    @staticmethod
    def _row(*widgets: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for widget in widgets:
            layout.addWidget(widget)
        return row

    @staticmethod
    def _form(group: QWidget) -> QFormLayout:
        form = QFormLayout(group)
        form.setContentsMargins(10, 12, 10, 8)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(16)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFixedWidth(320)
        side_widget = QWidget()
        side_widget.setObjectName("sidebar")
        side = QVBoxLayout(side_widget)
        side.setContentsMargins(0, 0, 5, 0)
        side.setSpacing(9)
        side_scroll.setWidget(side_widget)
        layout.addWidget(side_scroll)

        source_group = QGroupBox("Source")
        source_form = self._form(source_group)
        self.source_choice = QComboBox()
        self.source_choice.addItem("Camera", "camera")
        self.source_choice.addItem("Video file", "video")
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 32)
        self.camera_index.setToolTip(
            "0 is usually the built-in camera. Try 1 or 2 for a USB camera."
        )
        self.camera_row = self._row(self.camera_index)
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Choose a video…")
        self.video_browse_btn = self._button("…", self._browse_video)
        self.video_browse_btn.setFixedWidth(32)
        self.video_browse_btn.setToolTip("Choose a recorded video file")
        self.video_row = self._row(self.video_path, self.video_browse_btn)
        self.start_frame = QSpinBox()
        self.start_frame.setRange(0, 2_147_483_646)
        self.start_frame.setToolTip(
            "Zero-based video index: 0 is the first frame. Show the frame, select a marker, then start tracking from there."
        )
        self.seek_btn = self._button("Show", self._seek_preview)
        self.seek_btn.setToolTip(
            "Move to this frame before recording; calibration is retained at the same frame size."
        )
        self.seek_btn.setFixedWidth(58)
        self.seek_row = self._row(self.start_frame, self.seek_btn)
        self.seek_row_label = QLabel("Start frame")
        self.seek_row_label.setToolTip("Frame numbers start at 0")
        self.tracker = QComboBox()
        self.tracker.addItem("Color marker", "color")
        self.tracker.addItem("YOLO object detector", "yolo")
        self.tracker.setToolTip(
            "Color: click a distinct colored marker. YOLO: optional detector package and model weights."
        )
        source_form.addRow("Input", self.source_choice)
        self.camera_row_label = QLabel("Camera index")
        self.video_row_label = QLabel("File")
        source_form.addRow(self.camera_row_label, self.camera_row)
        source_form.addRow(self.video_row_label, self.video_row)
        source_form.addRow(self.seek_row_label, self.seek_row)
        source_form.addRow("Tracker", self.tracker)

        self.yolo_panel = QWidget()
        yolo_form = QFormLayout(self.yolo_panel)
        yolo_form.setContentsMargins(0, 0, 0, 0)
        yolo_form.setSpacing(8)
        self.object_kind = QComboBox()
        for label, class_id in COCO_PRESETS:
            self.object_kind.addItem(label, class_id)
        self.class_id = QSpinBox()
        self.class_id.setRange(0, 79)
        self.class_id.setValue(32)
        self.class_id_label = QLabel("Class ID")
        self.model_name = QLineEdit("yolo11n.pt")
        self.model_name.setToolTip(
            "Local YOLO weights path or weights name; first use may download weights."
        )
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        self.device.addItem("CUDA GPU 0", "0")
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.99)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        self.confidence.setValue(0.35)
        yolo_form.addRow("Object", self.object_kind)
        yolo_form.addRow(self.class_id_label, self.class_id)
        yolo_form.addRow("Weights", self.model_name)
        yolo_form.addRow("Device", self.device)
        yolo_form.addRow("Confidence", self.confidence)
        source_form.addRow(self.yolo_panel)
        self.open_btn = self._button("Open camera", self._open_camera)
        self.disconnect_btn = self._button("Close source", self._disconnect_camera)
        source_form.addRow(self._row(self.open_btn, self.disconnect_btn))
        self.import_btn = self._button("Import CSV…", self._choose_csv)
        self.import_btn.setToolTip(
            "Measured positions: Time (s), X (m), Y (m). Extra source columns are preserved."
        )
        self.demo_btn = self._button("Load demo", self._load_demo)
        self.demo_btn.setToolTip(
            "Load a seeded synthetic projectile; no camera or detector is needed."
        )
        source_form.addRow(self._row(self.import_btn, self.demo_btn))
        side.addWidget(source_group)

        scale_group = QGroupBox("Scale and target")
        scale_form = self._form(scale_group)
        self.reference = QDoubleSpinBox()
        self.reference.setRange(0.0001, 10000)
        self.reference.setDecimals(4)
        self.reference.setValue(0.21)
        self.reference.setSuffix(" m")
        self.reference.setToolTip(
            "Measured separation of two visible points in the target's motion plane. Keep the camera fixed and perpendicular to that plane."
        )
        scale_form.addRow("Known length", self.reference)
        self.calibrate_btn = self._button("Calibrate", self._begin_calibration)
        self.calibrate_btn.setToolTip(
            "Click the two endpoints of the measured reference in the image."
        )
        self.select_btn = self._button("Select marker", self._select_target)
        self.select_btn.setCheckable(True)
        scale_form.addRow(self._row(self.calibrate_btn, self.select_btn))
        self.calibration_label = QLabel("Scale not set")
        self.calibration_label.setObjectName("muted")
        self.calibration_label.setWordWrap(True)
        self.target_label = QLabel("No marker selected")
        self.target_label.setObjectName("muted")
        self.target_label.setWordWrap(True)
        scale_form.addRow(self.calibration_label)
        scale_form.addRow(self.target_label)
        side.addWidget(scale_group)

        analysis_group = QGroupBox("Analysis")
        analysis_form = self._form(analysis_group)
        self.experiment = QComboBox()
        self.experiment.addItems(EXPERIMENT_TYPES)
        self.experiment.setToolTip(
            "General motion uses local fits. Free-fall and horizontal toss fit ballistic intervals; select only when those physical assumptions apply."
        )
        self.mass = QDoubleSpinBox()
        self.mass.setDecimals(6)
        self.mass.setRange(0.000001, 1000000)
        self.mass.setSingleStep(0.05)
        self.mass.setValue(0.2)
        self.mass.setSuffix(" kg")
        self.mass.setToolTip(
            "Enter the object's measured mass for energy calculations. 0.2 kg is an example."
        )
        analysis_form.addRow("Motion model", self.experiment)
        analysis_form.addRow("Object mass", self.mass)
        self.advanced_btn = QToolButton()
        self.advanced_btn.setText("Analysis options")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_btn.setArrowType(Qt.ArrowType.RightArrow)
        analysis_form.addRow(self.advanced_btn)
        self.advanced_panel = QWidget()
        advanced = QFormLayout(self.advanced_panel)
        advanced.setContentsMargins(0, 0, 0, 0)
        self.max_gap = QDoubleSpinBox()
        self.max_gap.setRange(0.001, 3600)
        self.max_gap.setDecimals(3)
        self.max_gap.setValue(1.0)
        self.max_gap.setSuffix(" s")
        self.max_gap.setToolTip(
            "Do not fit across a gap larger than this. Track changes always break a fit."
        )
        self.smoothing = QSpinBox()
        self.smoothing.setRange(5, 101)
        self.smoothing.setSingleStep(2)
        self.smoothing.setValue(9)
        self.smoothing.setToolTip(
            "Odd sample count for local fits in General motion. Larger windows smooth more detail."
        )
        advanced.addRow("Maximum gap", self.max_gap)
        advanced.addRow("Fit window", self.smoothing)
        analysis_form.addRow(self.advanced_panel)
        self.advanced_panel.hide()
        self.start_btn = self._button("Start tracking", self._start_tracking, primary=True)
        self.stop_btn = self._button("Stop & analyze", self._stop_tracking)
        analysis_form.addRow(self._row(self.start_btn, self.stop_btn))
        self.analyze_btn = self._button("Analyze samples", self._analyze_records, primary=True)
        analysis_form.addRow(self.analyze_btn)
        side.addWidget(analysis_group)

        output_group = QGroupBox("Save results")
        output_form = self._form(output_group)
        self.output_root = QLineEdit(str(Path.home() / "Documents" / "Physics Motion Analyzer"))
        self.output_root.setToolTip(
            "Each analysis creates a separate folder containing CSVs, plots, metadata, and a PDF report."
        )
        self.output_browse_btn = self._button("…", self._browse_output)
        self.output_browse_btn.setFixedWidth(32)
        self.output_browse_btn.setToolTip("Choose the parent directory for analysis results")
        output_form.addRow(self._row(self.output_root, self.output_browse_btn))
        side.addWidget(output_group)
        side.addStretch()

        right = QVBoxLayout()
        right.setSpacing(8)
        layout.addLayout(right, 1)
        self.source_label = QLabel("No data loaded")
        self.source_label.setObjectName("dataset")
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_label.setWordWrap(True)
        self.sample_label = QLabel("Open a camera or video, import positions, or load the demo.")
        self.sample_label.setObjectName("muted")
        self.sample_label.setWordWrap(True)
        right.addWidget(self.source_label)
        right.addWidget(self.sample_label)
        self.tabs = QTabWidget()
        right.addWidget(self.tabs, 1)

        self.capture_page = QWidget()
        capture_layout = QVBoxLayout(self.capture_page)
        capture_layout.setContentsMargins(12, 12, 12, 10)
        capture_layout.setSpacing(8)
        self.canvas = VideoCanvas()
        self.canvas.clicked.connect(self._video_clicked)
        capture_layout.addWidget(self.canvas, 1)
        self.capture_hint = QLabel("Calibrate a known length, then select the target in the image.")
        self.capture_hint.setObjectName("muted")
        self.capture_hint.setWordWrap(True)
        capture_layout.addWidget(self.capture_hint)
        self.live_graph = LiveHeightGraph()
        capture_layout.addWidget(self.live_graph)
        self.tabs.addTab(self.capture_page, "Capture")

        self.plots_page = QWidget()
        plots_layout = QVBoxLayout(self.plots_page)
        plots_layout.setContentsMargins(14, 14, 14, 14)
        plot_toolbar = QHBoxLayout()
        plot_toolbar.addWidget(QLabel("Plot"))
        self.graph_choice = QComboBox()
        for key in GRAPH_FILES:
            self.graph_choice.addItem(GRAPH_LABELS.get(key, key.title()), key)
        self.graph_choice.setMinimumWidth(170)
        plot_toolbar.addWidget(self.graph_choice)
        plot_toolbar.addStretch()
        self.plot_note = QLabel("Measured positions and modeled quantities")
        self.plot_note.setObjectName("muted")
        plot_toolbar.addWidget(self.plot_note)
        plots_layout.addLayout(plot_toolbar)
        self.plot_label = PlotLabel()
        plots_layout.addWidget(self.plot_label, 1)
        self.tabs.addTab(self.plots_page, "Plots")

        self.results_page = QWidget()
        results_layout = QVBoxLayout(self.results_page)
        results_layout.setContentsMargins(16, 16, 16, 14)
        results_layout.setSpacing(12)
        actions = QHBoxLayout()
        self.open_report_btn = self._button("Open PDF report", self._open_report)
        self.open_folder_btn = self._button("Open results folder", self._open_folder)
        self.copy_summary_btn = self._button("Copy summary", self._copy_summary)
        actions.addWidget(self.open_report_btn)
        actions.addWidget(self.open_folder_btn)
        actions.addWidget(self.copy_summary_btn)
        actions.addStretch()
        results_layout.addLayout(actions)
        self.result_path = QLabel("No analysis saved")
        self.result_path.setObjectName("muted")
        self.result_path.setWordWrap(True)
        self.result_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        results_layout.addWidget(self.result_path)
        metrics = QGridLayout()
        metrics.setHorizontalSpacing(28)
        metrics.setVerticalSpacing(9)
        self.metric_labels: dict[str, QLabel] = {}
        for index, (key, title) in enumerate(
            (
                ("samples", "Measured positions"),
                ("duration_s", "Duration (s)"),
                ("modeled_samples", "Modeled positions"),
                ("max_modeled_speed_m_s", "Peak speed (m/s)"),
                ("gravity_estimate_m_s2", "Qualified gravity (m/s²)"),
                ("restitution_estimate", "Qualified restitution"),
            )
        ):
            row, column = divmod(index, 2)
            caption = QLabel(title)
            caption.setObjectName("muted")
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            metrics.addWidget(caption, row, column * 2)
            metrics.addWidget(value, row, column * 2 + 1)
            self.metric_labels[key] = value
        metrics.setColumnStretch(1, 1)
        metrics.setColumnStretch(3, 1)
        results_layout.addLayout(metrics)
        results_layout.addWidget(QLabel("Modeled intervals"))
        self.segment_table = QTableWidget(0, 6)
        self.segment_table.setHorizontalHeaderLabels(
            ["Interval", "Model", "Time (s)", "Positions", "R² x", "R² y"]
        )
        self.segment_table.verticalHeader().hide()
        self.segment_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.segment_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.segment_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.segment_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.segment_table.setAlternatingRowColors(True)
        self.segment_table.setShowGrid(False)
        self.segment_table.setMinimumHeight(110)
        self.segment_table.setMaximumHeight(220)
        results_layout.addWidget(self.segment_table, 1)
        self.interval_note = QLabel("")
        self.interval_note.setObjectName("muted")
        self.interval_note.hide()
        results_layout.addWidget(self.interval_note)
        results_layout.addWidget(QLabel("Measurement notes"))
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(130)
        self.summary.setPlainText("Analysis results and measurement limitations appear here.")
        results_layout.addWidget(self.summary, 1)
        self.tabs.addTab(self.results_page, "Results")
        self.notice = QLabel()
        self.notice.setObjectName("error")
        self.notice.setWordWrap(True)
        self.notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.notice.hide()
        right.addWidget(self.notice)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        right.addWidget(self.progress)
        self.statusBar().setSizeGripEnabled(True)
        self.statusBar().showMessage("Ready")

        self.setup_widgets = [
            self.experiment,
            self.mass,
            self.reference,
            self.max_gap,
            self.smoothing,
            self.output_root,
            self.output_browse_btn,
        ]
        self.source_widgets = [
            self.source_choice,
            self.camera_index,
            self.video_path,
            self.video_browse_btn,
            self.tracker,
            self.model_name,
            self.device,
            self.confidence,
            self.object_kind,
            self.class_id,
        ]
        self.source_choice.currentIndexChanged.connect(self._update_source_ui)
        self.start_frame.valueChanged.connect(self._start_frame_changed)
        self.tracker.currentIndexChanged.connect(self._update_tracker_ui)
        self.object_kind.currentIndexChanged.connect(self._update_class_ui)
        self.reference.valueChanged.connect(self._on_reference_changed)
        self.experiment.currentIndexChanged.connect(self._analysis_options_changed)
        self.mass.valueChanged.connect(self._analysis_options_changed)
        self.max_gap.valueChanged.connect(self._analysis_options_changed)
        self.smoothing.valueChanged.connect(self._analysis_options_changed)
        self.advanced_btn.toggled.connect(self._toggle_advanced)
        self.graph_choice.currentIndexChanged.connect(self._update_graph_choice)
        self._update_source_ui()
        self._update_tracker_ui()

    def _busy(self) -> bool:
        return (
            self._closing
            or self._state in BUSY_STATES
            or any(
                worker is not None and worker.isRunning()
                for worker in (self.analysis_thread, self.import_thread)
            )
        )

    def _camera_live(self) -> bool:
        return self.camera is not None and self.camera.isRunning()

    def _resting_state(self) -> str:
        if self._calibration is not None and self._camera_live():
            return "READY"
        return "DATA_READY" if self._records else "IDLE"

    def _set_controls(self, *_):
        busy, live = self._busy(), self._camera_live()
        for widget in self.setup_widgets:
            widget.setEnabled(not busy)
        self.reference.setEnabled(not busy and live)
        self.smoothing.setEnabled(not busy and self.experiment.currentText() == "General motion")
        self.advanced_btn.setEnabled(not busy)
        for widget in self.source_widgets:
            widget.setEnabled(not busy and not live)
        self._update_class_ui()
        self.open_btn.setEnabled(not busy and not live)
        self.disconnect_btn.setEnabled(
            live and (not busy or self._state == "SEEKING") and not self._closing
        )
        can_seek = (
            live
            and not busy
            and self.source_choice.currentData() == "video"
            and self._frame_size[0] > 0
        )
        self.start_frame.setEnabled(can_seek)
        self.seek_btn.setEnabled(can_seek)
        self.import_btn.setEnabled(not busy and not live)
        self.demo_btn.setEnabled(not busy and not live)
        self.calibrate_btn.setEnabled(live and not busy and self._frame_size[0] > 0)
        self.select_btn.setEnabled(live and not busy and self._frame_size[0] > 0)
        target_ready = self.tracker.currentData() != "color" or self._marker_selected
        frame_ready = (
            self.source_choice.currentData() != "video"
            or self.start_frame.value() == self._applied_start_frame
        )
        self.start_btn.setEnabled(
            live and self._calibration is not None and target_ready and frame_ready and not busy
        )
        self.stop_btn.setEnabled(self._state == "TRACKING" and not self._closing)
        self.analyze_btn.setEnabled(
            not busy and len(self._records) >= AnalysisSettings().min_data_points
        )
        can_open = self._latest_folder is not None and not busy
        self.open_report_btn.setEnabled(can_open)
        self.open_folder_btn.setEnabled(can_open)
        self.copy_summary_btn.setEnabled(can_open and self._latest_result is not None)
        self.graph_choice.setEnabled(self._latest_folder is not None)

    def _update_source_ui(self, *_):
        is_video = self.source_choice.currentData() == "video"
        self.camera_row.setVisible(not is_video)
        self.camera_row_label.setVisible(not is_video)
        self.video_row.setVisible(is_video)
        self.video_row_label.setVisible(is_video)
        self.seek_row.setVisible(is_video)
        self.seek_row_label.setVisible(is_video)
        self.open_btn.setText("Open video" if is_video else "Open camera")
        self._set_controls()

    def _update_tracker_ui(self, *_):
        is_yolo = self.tracker.currentData() == "yolo"
        self.yolo_panel.setVisible(is_yolo)
        self.select_btn.setText("Select target" if is_yolo else "Select marker")
        self.target_label.setText(
            "Click to select when several targets are visible" if is_yolo else "No marker selected"
        )
        self._marker_selected = False
        self._set_controls()

    def _update_class_ui(self, *_):
        custom = self.object_kind.currentData() is None
        self.class_id.setVisible(custom)
        self.class_id_label.setVisible(custom)
        self.class_id.setEnabled(custom and not self._busy() and not self._camera_live())

    def _start_frame_changed(self, value: int):
        if self._camera_live() and not self._busy():
            if value != self._applied_start_frame:
                self.capture_hint.setText(
                    f"Choose Show to apply start frame {value:,} before tracking."
                )
            else:
                self.capture_hint.setText(f"Recording start frame: {value:,}.")
        self._set_controls()

    def _toggle_advanced(self, visible: bool):
        self.advanced_panel.setVisible(visible)
        self.advanced_btn.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )

    def _analysis_options_changed(self, *_):
        if not self._busy() and self._latest_folder is not None:
            self._clear_results()
            self._set_status(
                "Analysis options changed. Analyze samples again to update the results."
            )
        self._set_controls()

    def _set_status(self, text: str):
        if not self._closing:
            self.statusBar().showMessage(text)

    def _show_error(self, text: str):
        self.notice.setText(text)
        self.notice.show()
        self._set_status(text)

    def _clear_notice(self):
        self.notice.clear()
        self.notice.hide()

    def _progress_busy(self):
        self.progress.setRange(0, 0)
        self.progress.show()

    def _clear_results(self):
        self._latest_folder = None
        self._latest_result = None
        self.plot_label.clear_plot()
        self.result_path.setText("No analysis saved")
        self.segment_table.setRowCount(0)
        self.interval_note.hide()
        for value in self.metric_labels.values():
            value.setText("—")
        self.summary.setPlainText("Analysis results and measurement limitations appear here.")
        self._clear_notice()
        self._set_controls()

    def _clear_dataset(self):
        self._records = []
        self._settings = None
        self._context = {}
        self.live_graph.set_samples([])
        self._clear_results()
        self.source_label.setText("No data loaded")
        self.sample_label.setText("Open a camera or video, import positions, or load the demo.")

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            self.video_path.text(),
            "Video files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All files (*)",
        )
        if path:
            self.video_path.setText(path)
            self.video_path.setToolTip(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Save results in", self.output_root.text())
        if path:
            self.output_root.setText(path)

    def _choose_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import measured positions", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._load_csv(Path(path))

    def _load_csv(self, path: Path):
        if self._busy() or self._camera_live():
            return
        self._clear_dataset()
        self.canvas.clear("CSV import in progress")
        self._state = "LOADING"
        self._progress_busy()
        self._set_status("Reading CSV positions…")
        worker = ImportWorker(Path(path))
        self.import_thread = worker
        worker.completed.connect(self._csv_loaded)
        worker.failed.connect(self._import_failed)
        worker.finished.connect(self._import_finished)
        self._set_controls()
        worker.start()

    def _csv_loaded(self, frame: pd.DataFrame, path: Path):
        self._records = frame.to_dict("records")
        self._context = {
            "origin": str(path.resolve()),
            "source_type": "csv",
            "source_file": str(path.resolve()),
            "measurement_origin": "operator-supplied CSV datum",
        }
        self._state = "DATA_READY"
        self.source_label.setText(path.name)
        self.source_label.setToolTip(str(path.resolve()))
        self.canvas.clear("CSV positions loaded")
        self.capture_hint.setText(
            "CSV units: Time in seconds; X and Y in meters; positive Y points upward."
        )
        self._update_sample_count()
        self._refresh_graph()
        self._set_status("CSV loaded. Choose a motion model and object mass, then Analyze samples.")
        self.tabs.setCurrentWidget(self.capture_page)
        if len(self._records) < AnalysisSettings().min_data_points:
            self._show_error(
                f"Only {len(self._records)} positions loaded; analysis needs at least "
                f"{AnalysisSettings().min_data_points}."
            )

    def _import_failed(self, message: str):
        self._state = "IDLE"
        self.canvas.clear()
        self._show_error(message)

    def _import_finished(self):
        worker = self.import_thread
        self.import_thread = None
        if worker is not None:
            worker.deleteLater()
        self.progress.hide()
        self._set_controls()
        self._maybe_finish_close()

    def _load_demo(self):
        if self._busy() or self._camera_live():
            return
        self._clear_dataset()
        self.experiment.setCurrentText("Horizontal toss")
        self._records = demo_samples().to_dict("records")
        self._context = {
            "origin": "seeded synthetic projectile demo",
            "source_type": "synthetic",
            "object_name": "Synthetic projectile",
            "measurement_origin": "synthetic datum",
        }
        self._state = "DATA_READY"
        self.source_label.setText("Synthetic projectile demo")
        self.source_label.setToolTip(
            "Seeded synthetic positions; these are not camera measurements."
        )
        self.canvas.clear("Synthetic positions loaded")
        self.capture_hint.setText(
            "Synthetic data · 1.4 m/s horizontal speed · 9.81 m/s² downward acceleration"
        )
        self._update_sample_count()
        self._refresh_graph()
        self.tabs.setCurrentWidget(self.capture_page)
        self._set_status("Demo loaded. Choose Analyze samples to generate plots and a report.")
        self._set_controls()

    def _open_camera(self):
        if self._busy() or self._camera_live():
            return
        is_video = self.source_choice.currentData() == "video"
        video_path = Path(self.video_path.text().strip()).expanduser() if is_video else None
        if is_video and (not self.video_path.text().strip() or not video_path.is_file()):
            self._show_error("Choose an existing video file before opening it.")
            return
        model = self.model_name.text().strip()
        if self.tracker.currentData() == "yolo" and not model:
            self._show_error("Choose a YOLO weights name or local model path.")
            return
        self._clear_dataset()
        self._calibration = None
        self._points = []
        self._marker_selected = False
        self._frame_size = (0, 0)
        self._applied_start_frame = 0
        self.start_frame.setRange(0, 2_147_483_646)
        self.start_frame.setValue(0)
        self.calibration_label.setText("Scale not set")
        self.target_label.setText(
            "No marker selected"
            if self.tracker.currentData() == "color"
            else "Select a target if several are visible"
        )
        self.canvas.clear("Opening video…" if is_video else "Opening camera…")
        self.source_label.setText(
            video_path.name if video_path else f"Camera {self.camera_index.value()}"
        )
        self.source_label.setToolTip(str(video_path) if video_path else "Live camera capture")
        self.sample_label.setText("Opening source…")
        self.capture_hint.setText(
            "Calibrate a known length, then click Select marker and choose the target."
            if self.tracker.currentData() == "color"
            else "Calibrate a known length. Click a target if several are visible."
        )
        worker = CameraWorker(
            self.camera_index.value(),
            model,
            self.device.currentData(),
            self.confidence.value(),
            video_path=video_path,
            tracker=self.tracker.currentData(),
        )
        self.camera = worker
        worker.frame_ready.connect(self.canvas.set_frame)
        worker.sample_ready.connect(self._sample_received)
        worker.camera_ready.connect(self._camera_ready)
        worker.message.connect(self._set_status)
        worker.recording_finished.connect(self._recording_finished)
        worker.error.connect(self._camera_error)
        worker.progress.connect(self._capture_progress)
        worker.target_ready.connect(self._target_ready)
        worker.seek_finished.connect(self._seek_finished)
        worker.finished.connect(self._camera_stopped)
        self._state = "OPENING"
        self._progress_busy()
        self.tabs.setCurrentWidget(self.capture_page)
        self._set_controls()
        self._set_status("Opening source…")
        worker.start()

    def _disconnect_camera(self):
        if not self._camera_live() or (self._busy() and self._state != "SEEKING"):
            return
        self._state = "DISCONNECTING"
        self.camera.stop_camera()
        self._progress_busy()
        self._set_status("Closing source…")
        self._set_controls()

    def _seek_preview(self):
        if self._busy() or not self._camera_live() or self.source_choice.currentData() != "video":
            return
        self._state = "SEEKING"
        self._marker_selected = False
        self._records = []
        self._settings = None
        self.live_graph.set_samples([])
        self._clear_results()
        self.select_btn.setChecked(False)
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        self.target_label.setText("Select the target in the new preview")
        self._update_sample_count()
        self._progress_busy()
        self._set_status(f"Moving to frame {self.start_frame.value():,}…")
        self.camera.seek_frame(self.start_frame.value())
        self._set_controls()

    def _seek_finished(self, frame: int, total: int):
        if self._state != "SEEKING":
            return
        self._applied_start_frame = frame
        self.start_frame.setMaximum(max(0, total - 1) if total > 0 else 2_147_483_646)
        self.start_frame.setValue(frame)
        self._state = self._resting_state()
        self.progress.hide()
        self.sample_label.setText(f"Preview frame {frame:,} (zero-based) · no positions recorded")
        self.capture_hint.setText("Select the target in this preview, then start tracking.")
        self._set_status(f"Preview ready at frame {frame:,}")
        self._set_controls()

    def _camera_ready(self, width: int, height: int):
        changed = (width, height) != self._frame_size
        if changed:
            had_frame = self._frame_size != (0, 0)
            self._frame_size = width, height
            self._calibration = None
            self._points = []
            self._marker_selected = False
            self.canvas.set_calibration_points([])
            self.calibration_label.setText("Scale not set")
            self.target_label.setText("No marker selected")
            if had_frame:
                self._show_error(
                    "Video dimensions changed. Recalibrate and select the target before another run."
                )
                if self._state == "TRACKING":
                    self._stop_tracking()
        if self._state == "OPENING":
            self._state = "IDLE"
            self.progress.hide()
            self.sample_label.setText(f"{width} × {height} px · no positions recorded")
        self._set_controls()

    def _camera_error(self, message: str):
        self._show_error(message)
        if self._state == "TRACKING":
            self._state = "STOPPING"
        elif self._state == "OPENING":
            self._state = "IDLE"
        self._set_controls()

    def _camera_stopped(self):
        worker = self.camera
        self.camera = None
        if worker is not None:
            worker.deleteLater()
        self._calibration = None
        self._points = []
        self._frame_size = (0, 0)
        self._marker_selected = False
        self.select_btn.setChecked(False)
        self.canvas.set_calibration_points([])
        self.calibration_label.setText("Scale not set · source closed")
        self.target_label.setText("No marker selected")
        if self._state not in {"ANALYZING", "LOADING"}:
            self._state = self._resting_state()
            self.progress.hide()
        self._set_controls()
        self._maybe_finish_close()

    def _begin_calibration(self):
        if self._busy() or self._frame_size[0] <= 0:
            return
        self._records = []
        self._settings = None
        self.live_graph.set_samples([])
        self._clear_results()
        self._state = "CALIBRATION"
        self._points = []
        self._calibration = None
        self.select_btn.setChecked(False)
        self.canvas.set_calibration_points([])
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        self.calibration_label.setText("Click the first reference endpoint")
        self.capture_hint.setText(
            "Click two endpoints of the known length, in the object's motion plane."
        )
        self.tabs.setCurrentWidget(self.capture_page)
        self._clear_notice()
        self._update_sample_count()
        self._set_controls()

    def _select_target(self):
        if self._busy() or not self._camera_live():
            self.select_btn.setChecked(False)
            return
        if self._state == "CALIBRATION":
            self._state = self._resting_state()
        self.canvas.setCursor(
            Qt.CursorShape.CrossCursor
            if self.select_btn.isChecked()
            else Qt.CursorShape.ArrowCursor
        )
        if self.select_btn.isChecked():
            self.capture_hint.setText(
                "Click inside the colored marker."
                if self.tracker.currentData() == "color"
                else "Click the object you want to track."
            )
        self.tabs.setCurrentWidget(self.capture_page)

    def _video_clicked(self, x: int, y: int):
        if self._state == "CALIBRATION":
            if len(self._points) >= 2:
                self._points = []
            self._points.append((x, y))
            self.canvas.set_calibration_points(self._points)
            self._calculate_calibration()
        elif self.camera is not None and (
            self._state == "TRACKING" or (self.select_btn.isChecked() and not self._busy())
        ):
            self.camera.select_target(x, y)
            self._marker_selected = self.tracker.currentData() != "color"
            self.select_btn.setChecked(False)
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self.target_label.setText(f"Sampling target at ({x}, {y}) px…")
            self.capture_hint.setText(
                "Click a target during tracking to select it again; this starts a new interval."
            )
            self._set_status("Target selection requested")
            self._set_controls()

    def _target_ready(self, ready: bool):
        if self.tracker.currentData() == "color":
            self._marker_selected = ready
            self.target_label.setText(
                "Marker color selected" if ready else "Select a distinct colored marker"
            )
            self._set_controls()

    def _calculate_calibration(self):
        self._calibration = None
        self._state = "CALIBRATION"
        if len(self._points) < 2:
            self.calibration_label.setText("Click the second reference endpoint")
        else:
            try:
                calibration = Calibration.from_points(
                    self._points[0], self._points[1], self.reference.value(), self._frame_size
                )
            except ValueError as exc:
                self.calibration_label.setText(str(exc))
                self.capture_hint.setText("Click a new first endpoint to repeat calibration.")
            else:
                self._calibration = calibration
                self._state = "READY"
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
                self.calibration_label.setText(
                    f"{calibration.pixels_per_meter:,.2f} px/m · "
                    f"{calibration.reference_length_m:g} m reference"
                )
                self.capture_hint.setText(
                    "Scale set. Select the marker, then start tracking."
                    if self.tracker.currentData() == "color" and not self._marker_selected
                    else "Scale set. Ready to start tracking."
                )
        self._set_controls()

    def _on_reference_changed(self, *_):
        if self._points and not self._busy():
            if self._records:
                self._records = []
                self.live_graph.set_samples([])
                self._update_sample_count()
            self._clear_results()
            self._calculate_calibration()

    def _make_settings(self) -> AnalysisSettings:
        window = self.smoothing.value()
        if window % 2 == 0:
            window += 1
            self.smoothing.setValue(window)
        settings = AnalysisSettings(
            mass_kg=self.mass.value(),
            experiment_type=self.experiment.currentText(),
            max_gap_seconds=self.max_gap.value(),
            smoothing_window=window,
        )
        settings.validate()
        return settings

    def _start_tracking(self):
        if self._busy() or self._calibration is None or not self._camera_live():
            return
        if (
            self.source_choice.currentData() == "video"
            and self.start_frame.value() != self._applied_start_frame
        ):
            self._show_error("Choose Show to apply the selected video start frame before tracking.")
            return
        if self.tracker.currentData() == "color" and not self._marker_selected:
            self._show_error("Select the colored marker in the preview before tracking.")
            return
        try:
            self._settings = self._make_settings()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        target_class = self.object_kind.currentData()
        target_class = self.class_id.value() if target_class is None else target_class
        is_video = self.source_choice.currentData() == "video"
        self._context = {
            "source_type": "video" if is_video else "camera",
            "origin": self.video_path.text() if is_video else f"camera {self.camera_index.value()}",
            "tracker": self.tracker.currentData(),
            "object_name": self.object_kind.currentText()
            if self.tracker.currentData() == "yolo"
            else "Color marker",
            "calibration": self._calibration.to_dict(),
            "measurement_origin": "frame bottom-left",
            "video_mirrored": False,
        }
        if is_video:
            self._context["source_file"] = str(Path(self.video_path.text()).expanduser().resolve())
        else:
            self._context["camera_index"] = self.camera_index.value()
        if self.tracker.currentData() == "yolo":
            self._context.update(
                target_coco_class_id=target_class,
                model=self.model_name.text().strip(),
                device=self.device.currentData(),
                confidence_threshold=self.confidence.value(),
            )
        self._clear_results()
        self._records = []
        self.live_graph.set_samples([])
        self._state = "TRACKING"
        self.canvas.set_calibration_points([])
        self.select_btn.setChecked(False)
        self.capture_hint.setText("Tracking · click the target to select it again if needed.")
        self.tabs.setCurrentWidget(self.capture_page)
        self._update_sample_count()
        self.camera.begin_tracking(self._calibration, target_class)
        self._set_controls()

    def _sample_received(self, record: dict):
        if self._state in {"TRACKING", "STOPPING"} and not self._closing:
            self._records.append(dict(record))

    def _capture_progress(self, frame: int, total: int):
        if self.source_choice.currentData() == "video" and total > 0:
            self.start_frame.setMaximum(max(0, total - 1))
        if self._state != "TRACKING" or self._closing:
            return
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(frame)
            self.progress.show()

    def _stop_tracking(self):
        if self._state != "TRACKING" or self.camera is None:
            return
        self._state = "STOPPING"
        self.camera.stop_tracking()
        self._progress_busy()
        self._set_status("Finishing recording…")
        self._set_controls()

    def _recording_finished(self, records: list[dict]):
        if self._state not in {"TRACKING", "STOPPING"}:
            return
        self._records = [dict(record) for record in records]
        if self.camera is not None:
            self._context["capture"] = self.camera.snapshot_metadata()
        self._state = self._resting_state()
        self._update_sample_count()
        self._refresh_graph()
        self.capture_hint.setText("Recording finished")
        if self._closing:
            return
        if len(records) < (self._settings or AnalysisSettings()).min_data_points:
            self.progress.hide()
            self._show_error(
                f"Only {len(records)} positions recorded; analysis needs at least "
                f"{(self._settings or AnalysisSettings()).min_data_points}. "
                "Select the target, improve lighting, or record longer."
            )
            self._set_controls()
            return
        self._analyze_records(use_recording_settings=True)

    def _analyze_records(self, checked=False, *, use_recording_settings: bool = False):
        if self._busy() or not self._records:
            return
        try:
            settings = (
                self._settings
                if use_recording_settings and self._settings
                else self._make_settings()
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._settings = settings
        self._clear_results()
        self._state = "ANALYZING"
        self._progress_busy()
        self.plot_label.clear_plot("Analyzing positions and saving plots…")
        self.summary.setPlainText("Analysis in progress…")
        self._set_status("Analyzing measured positions…")
        root = (
            Path(self.output_root.text().strip()).expanduser()
            if self.output_root.text().strip()
            else Path.cwd() / "output"
        )
        worker = AnalysisWorker(self._records, settings, root, self._context)
        self.analysis_thread = worker
        worker.completed.connect(self._analysis_completed)
        worker.failed.connect(self._analysis_failed)
        worker.progress.connect(self._set_status)
        worker.finished.connect(self._analysis_finished)
        self._set_controls()
        worker.start()

    def _analysis_completed(self, result: AnalysisResult, folder: Path):
        self._latest_result = result
        self._latest_folder = Path(folder)
        self._state = self._resting_state()
        self.result_path.setText(str(self._latest_folder.resolve()))
        for key, label in self.metric_labels.items():
            value = result.stats.get(key)
            if value is None or not math.isfinite(float(value)):
                label.setText("Not established")
            elif key in {"samples", "modeled_samples"}:
                label.setText(f"{int(value):,}")
            else:
                label.setText(f"{value:.3f}")
        segments = result.segments[:100]
        self.segment_table.setRowCount(len(segments))
        for row, segment in enumerate(segments):
            values = [
                str(segment.index),
                segment.motion_type,
                f"{segment.t_start:.3f} – {segment.t_end:.3f}",
                str(segment.end_index - segment.start_index),
                f"{segment.r2_x:.3f}",
                f"{segment.r2_y:.3f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.segment_table.setItem(row, column, item)
        self.interval_note.setText(
            f"Showing 100 of {len(result.segments):,} intervals. All intervals are in summary.json."
        )
        self.interval_note.setVisible(len(result.segments) > 100)
        self.summary.setPlainText(
            "\n\n".join(result.warnings) or "No additional measurement notes."
        )
        self._update_graph_choice()
        self.tabs.setCurrentWidget(self.plots_page)
        self._set_status(
            f"Saved {result.stats['samples']:,} positions and analysis to {folder.name}"
        )
        self._set_controls()

    def _analysis_failed(self, message: str):
        self._state = self._resting_state()
        self.plot_label.clear_plot("Analysis could not be completed")
        self.summary.setPlainText(
            message
            + "\n\nYour input samples are retained. Correct the setup or import another file, then analyze again."
        )
        self._show_error(message)
        self.tabs.setCurrentWidget(self.results_page)
        self._set_controls()

    def _analysis_finished(self):
        worker = self.analysis_thread
        self.analysis_thread = None
        if worker is not None:
            worker.deleteLater()
        self.progress.hide()
        self._set_controls()
        self._maybe_finish_close()

    def _update_sample_count(self):
        count = len(self._records)
        duration = None
        if count >= 2:
            try:
                candidate = float(self._records[-1]["Time"]) - float(self._records[0]["Time"])
                duration = candidate if math.isfinite(candidate) and candidate >= 0 else None
            except (KeyError, TypeError, ValueError):
                pass
        suffix = f" · {duration:.3f} s" if duration is not None else ""
        prefix = "Recording · " if self._state == "TRACKING" else ""
        self.sample_label.setText(f"{prefix}{count:,} measured positions{suffix}")

    def _refresh_graph(self):
        if self._state in {"TRACKING", "STOPPING"}:
            self._update_sample_count()
        if not self._records:
            return
        samples = []
        for record in self._records[-500:]:
            try:
                samples.append(
                    (float(record["Time"]), float(record["Y"]), record.get("TrackID", 0))
                )
            except (KeyError, ValueError, TypeError):
                continue
        self.live_graph.max_gap_seconds = self.max_gap.value()
        self.live_graph.set_samples(samples)

    def _update_graph_choice(self, *_):
        key = self.graph_choice.currentData()
        if self._latest_folder is not None and key in GRAPH_FILES:
            self.plot_label.set_plot(self._latest_folder / GRAPH_FILES[key])

    def _open_path(self, path: Path, label: str):
        if not path.exists() or not QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(path.resolve()))
        ):
            self._show_error(
                f"Could not open {label}. Open this path in your file manager: {path.resolve()}"
            )

    def _open_report(self):
        if self._latest_folder is not None:
            self._open_path(self._latest_folder / "Analysis_Report.pdf", "the PDF report")

    def _open_folder(self):
        if self._latest_folder is not None:
            self._open_path(self._latest_folder, "the results folder")

    def _copy_summary(self):
        if self._latest_result is not None and self._settings is not None:
            QApplication.clipboard().setText(
                summary_text(self._latest_result, self._settings.experiment_type)
            )
            self._set_status("Analysis summary copied")

    def _maybe_finish_close(self):
        if self._closing and not any(
            worker is not None and worker.isRunning()
            for worker in (self.camera, self.analysis_thread, self.import_thread)
        ):
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        active = [
            worker
            for worker in (self.camera, self.analysis_thread, self.import_thread)
            if worker is not None and worker.isRunning()
        ]
        if active:
            event.ignore()
            self._closing = True
            self._set_controls()
            if self._camera_live():
                self.camera.stop_camera()
            self.statusBar().showMessage("Closing after the current operation finishes…")
            return
        self._graph_timer.stop()
        event.accept()


def main() -> None:
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Physics Motion Analyzer")
    app.setOrganizationName("Physics Motion Analyzer")
    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
