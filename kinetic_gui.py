import sys
import time
import os
import io
from collections import deque

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QTextEdit,
    QSpinBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QImage, QPixmap, QDesktopServices

from ultralytics import YOLO

try:
    from PySide6.QtPdfWidgets import QPdfView
    from PySide6.QtPdf import QPdfDocument
    PDF_WIDGETS_AVAILABLE = True
except Exception:
    PDF_WIDGETS_AVAILABLE = False

import kinetic_engine as ke

ENGINE_CONFIG = ke.CONFIG
OBJECT_PRESETS = ke.OBJECT_PRESETS
MotionSegmenter = ke.MotionSegmenter
smooth_series = ke.smooth_series
apply_segment_kinematics = ke.apply_segment_kinematics
compute_physics_stats = ke.compute_physics_stats
plot_trajectory = ke.plot_trajectory
plot_position_time = ke.plot_position_time
plot_velocity_time = ke.plot_velocity_time
plot_acceleration_time = ke.plot_acceleration_time
plot_energy_time = ke.plot_energy_time
plot_segmentation_debug = ke.plot_segmentation_debug
create_pdf_report = ke.create_pdf_report
draw_hud_block = ke.draw_hud_block


class VideoLabel(QLabel):


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #202020; border: 1px solid #444;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            win = self.window()
            if hasattr(win, "handle_video_click"):
                if hasattr(event, "position"):
                    pos = event.position()
                    x, y = int(pos.x()), int(pos.y())
                else:
                    pos = event.pos()
                    x, y = pos.x(), pos.y()
                win.handle_video_click(x, y)
        super().mousePressEvent(event)


class PdfViewerWindow(QMainWindow):

    def __init__(self, pdf_path: str, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.setWindowTitle("Analysis Report Viewer")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        if PDF_WIDGETS_AVAILABLE:
            self.doc = QPdfDocument(self)
            self.doc.load(pdf_path)

            self.view = QPdfView()
            self.view.setDocument(self.doc)
            self.view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
            layout.addWidget(self.view)
        else:
            info_label = QLabel(
                "Built-in PDF viewer is not available on this system right now.\n\n"
                "Click the button below to open the PDF in your normal PDF viewer."
            )
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_open_external = QPushButton("Open in system viewer")
        self.btn_open_external.clicked.connect(self.open_external)
        btn_row.addWidget(self.btn_open_external)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_close)

        layout.addLayout(btn_row)

        self.showMaximized()

    def open_external(self):
        if not os.path.exists(self.pdf_path):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.pdf_path))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(ENGINE_CONFIG.project_title)
        # Change display size if needed
        self.frame_width = 960
        self.frame_height = 540

        self.cap = None
        self.model = None
        self.timer = None

        self.state = "IDLE" 
        self.pixels_per_meter = None
        self.calib_points = []  
        self.data_log = [] 
        self.trail = deque(maxlen=ENGINE_CONFIG.trail_length)
        self.t0 = None
        self.selected_experiment_type = "General motion"
        # Change it later if needed
        self.graph_mode = "live_height"
        self.analysis_available = False   
        self.frame_counter = 0

        self.output_dir = ke.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.latest_pdf_path: str | None = None
        self.pdf_viewer: PdfViewerWindow | None = None


        self.graph_files = {
            "seg_debug": os.path.join(self.output_dir, "segmentation_debug.png"),
            "trajectory": os.path.join(self.output_dir, "trajectory.png"),
            "position": os.path.join(self.output_dir, "position_time.png"),
            "velocity": os.path.join(self.output_dir, "velocity_time.png"),
            "accel": os.path.join(self.output_dir, "acceleration_time.png"),
            "energy": os.path.join(self.output_dir, "energy_time.png"),
        }

        self._build_ui()
        self._init_camera_and_model()


    def get_unique_output_path(self, base_name: str) -> str:
        base, ext = os.path.splitext(base_name)
        candidate = os.path.join(self.output_dir, base_name)
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(self.output_dir, f"{base}_{counter}{ext}")
            counter += 1
        return candidate

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_layout = QVBoxLayout()

        setup_group = QGroupBox("Experiment Setup")
        setup_form = QFormLayout()

        self.name_edit = QLineEdit(ENGINE_CONFIG.student_name)
        self.title_edit = QLineEdit(ENGINE_CONFIG.project_title)

        self.experiment_combo = QComboBox()
        self.experiment_types = [
            "General motion",
            "Free-fall / bounce",
            "Lift and lower",
            "Horizontal toss",
            "Custom experiment",
        ]
        self.experiment_combo.addItems(self.experiment_types)

        self.object_combo = QComboBox()
        self.object_combo.addItem(OBJECT_PRESETS["1"]["name"], "1")
        self.object_combo.addItem(OBJECT_PRESETS["2"]["name"], "2")
        self.object_combo.addItem(OBJECT_PRESETS["3"]["name"], "3")
        self.object_combo.addItem(OBJECT_PRESETS["4"]["name"], "4")
        self.object_combo.addItem("Custom...", "5")

        self.custom_class_spin = QSpinBox()
        self.custom_class_spin.setRange(0, 100)
        self.custom_class_spin.setValue(32)

        self.custom_mass_spin = QDoubleSpinBox()
        self.custom_mass_spin.setRange(0.01, 200.0)
        self.custom_mass_spin.setDecimals(3)
        self.custom_mass_spin.setValue(0.20)

        self.custom_class_spin.setEnabled(False)
        self.custom_mass_spin.setEnabled(False)

        self.object_combo.currentIndexChanged.connect(self._on_object_preset_changed)

        setup_form.addRow("Student name:", self.name_edit)
        setup_form.addRow("Project title:", self.title_edit)
        setup_form.addRow("Experiment type:", self.experiment_combo)
        setup_form.addRow("Object preset:", self.object_combo)
        setup_form.addRow("Custom COCO id (if you know, if not ignore):", self.custom_class_spin)
        setup_form.addRow("Custom mass (kg):", self.custom_mass_spin)

        setup_group.setLayout(setup_form)

        self.btn_apply = QPushButton("Apply Settings")
        self.btn_apply.clicked.connect(self.apply_settings)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        self.btn_calibrate = QPushButton("Step 1 - Calibrate")
        self.btn_start = QPushButton("Step 2 - Start Tracking")
        self.btn_stop = QPushButton("Step 3 - Stop & Analyze Motion")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self.btn_calibrate.clicked.connect(self.on_calibrate)
        self.btn_start.clicked.connect(self.on_start_tracking)
        self.btn_stop.clicked.connect(self.on_stop_analyze)

        controls_layout.addWidget(self.btn_calibrate)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_group.setLayout(controls_layout)

        left_layout.addWidget(setup_group)
        left_layout.addWidget(self.btn_apply)
        left_layout.addWidget(controls_group)
        left_layout.addStretch()

        main_layout.addLayout(left_layout, 0)

        right_layout = QVBoxLayout()

        self.video_label = VideoLabel(self)
        self.video_label.setFixedSize(self.frame_width, self.frame_height)
        right_layout.addWidget(self.video_label)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(150)
        self.summary_text.setPlaceholderText(
            "After you run an experiment and press 'Stop & Analyze',\n"
            "a summary of the piecewise motion and the stats will come up here."
        )
        right_layout.addWidget(self.summary_text)

        self.btn_view_pdf = QPushButton("Open Report as a PDF")
        self.btn_view_pdf.setEnabled(False)
        self.btn_view_pdf.clicked.connect(self.on_view_pdf)
        right_layout.addWidget(self.btn_view_pdf)

        # Graph viewer group
        graph_group = QGroupBox("Graphs")
        graph_layout = QVBoxLayout()

        self.graph_combo = QComboBox()
        self.graph_combo.addItem("Live: Height vs Time", "live_height")
        self.graph_combo.addItem("Segmentation Debug (Y & v_y vs Time)", "seg_debug")
        self.graph_combo.addItem("Trajectory (Y vs X)", "trajectory")
        self.graph_combo.addItem("Vertical Position vs Time", "position")
        self.graph_combo.addItem("Velocity vs Time", "velocity")
        self.graph_combo.addItem("Acceleration vs Time", "accel")
        self.graph_combo.addItem("Energy vs Time", "energy")

        self.graph_combo.currentIndexChanged.connect(self.on_graph_selection_changed)

        self.graph_label = QLabel()
        self.graph_label.setAlignment(Qt.AlignCenter)
        self.graph_label.setMinimumHeight(220)
        self.graph_label.setStyleSheet("background-color: #101010; border: 1px solid #444;")
        self.graph_label.setText(
            "Select a graph to view.\n\n"
            "For live graphs, start tracking.\n"
            "For analysis graphs, run 'Stop & Analyze'."
        )

        graph_layout.addWidget(self.graph_combo)
        graph_layout.addWidget(self.graph_label)
        graph_group.setLayout(graph_layout)

        right_layout.addWidget(graph_group)

        main_layout.addLayout(right_layout, 1)

        self._on_object_preset_changed(self.object_combo.currentIndex())
        self.apply_settings()

        self.statusBar().showMessage("Ready - set up the experiment and click 'Calibrate' to start.")
        self.resize(1400, 800)


    def _init_camera_and_model(self):
        try:
            self.model = YOLO("yolov8x.pt") 
        except Exception as e:
            self.model = None
            self.statusBar().showMessage(f"Error loading YOLO model: {e}")
            return

        self.cap = cv2.VideoCapture(ENGINE_CONFIG.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not self.cap.isOpened():
            self.statusBar().showMessage("Error: couldn't open camera.")
            self.cap = None
            return

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # around 30 FPS, change if your cameras nice


    def _on_object_preset_changed(self, index: int):
        key = str(self.object_combo.itemData(index))
        if key == "5":  # Custom
            self.custom_class_spin.setEnabled(True)
            self.custom_mass_spin.setEnabled(True)
        else:
            self.custom_class_spin.setEnabled(False)
            self.custom_mass_spin.setEnabled(False)

    def apply_settings(self):
        name = self.name_edit.text().strip()
        title = self.title_edit.text().strip()

        if name:
            ENGINE_CONFIG.student_name = name
        if title:
            ENGINE_CONFIG.project_title = title

        self.setWindowTitle(ENGINE_CONFIG.project_title)

        self.selected_experiment_type = self.experiment_combo.currentText()

        key = str(self.object_combo.currentData())
        if key not in OBJECT_PRESETS:
            key = "1"
        preset = OBJECT_PRESETS[key]

        if key == "5":
            class_id = int(self.custom_class_spin.value())
            mass = float(self.custom_mass_spin.value())
            ENGINE_CONFIG.target_class_id = class_id
            ENGINE_CONFIG.mass_kg = mass
            ENGINE_CONFIG.object_name = f"Custom (class {class_id})"
        else:
            ENGINE_CONFIG.target_class_id = preset["class_id"]
            ENGINE_CONFIG.mass_kg = preset["mass"]
            ENGINE_CONFIG.object_name = preset["name"]

        self.statusBar().showMessage(
            f"Settings applied. Tracking: {ENGINE_CONFIG.object_name}, mass ≈ {ENGINE_CONFIG.mass_kg:.3f} kg."
        )

    #Calibratiuhn
    def handle_video_click(self, x: int, y: int):
        if self.state != "CALIBRATION":
            return

        if len(self.calib_points) < 2:
            self.calib_points.append((x, y))
            self.statusBar().showMessage(
                f"Calibration point {len(self.calib_points)} at ({x}, {y})."
            )
        else:
            self.calib_points = [(x, y)]
            self.statusBar().showMessage("Restarting calibration: first point set.")

    def on_calibrate(self):
        self.state = "CALIBRATION"
        self.calib_points = []
        self.pixels_per_meter = None
        self.data_log = []
        self.trail.clear()
        self.t0 = None
        self.analysis_available = False
        self.latest_pdf_path = None
        self.btn_view_pdf.setEnabled(False)

        self.btn_calibrate.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self.summary_text.clear()
        self.graph_label.setText(
            "Calibration mode.\n\n"
            "Click two points in the video that are one known length apart. e.g. a piece of paper.\n"
            "Then click 'Start Tracking'."
        )
        self.graph_label.setPixmap(QPixmap())
        self.statusBar().showMessage(
            "Calibration mode: click two points in the video that are one known length apart."
        )

    def on_start_tracking(self):
        if self.pixels_per_meter is None:
            self.statusBar().showMessage("Calibrate first: no scale defined yet.")
            return

        self.state = "TRACKING"
        self.data_log = []
        self.trail.clear()
        self.t0 = None
        self.analysis_available = False
        self.latest_pdf_path = None
        self.btn_view_pdf.setEnabled(False)

        self.btn_calibrate.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.summary_text.clear()
        self.graph_label.setText(
            "Live graph.\n\n"
            "Move the object in front of the camera.\n"
            "Select 'Live: Height vs Time' to see the curve appear."
        )
        self.graph_label.setPixmap(QPixmap())
        self.statusBar().showMessage(
            f"Tracking started for {ENGINE_CONFIG.object_name}. Move it in front of the camera."
        )

    def on_stop_analyze(self):
        if self.state != "TRACKING":
            self.statusBar().showMessage("Not currently tracking.")
            return

        self.state = "IDLE"

        self.btn_calibrate.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self.statusBar().showMessage("Tracking stopped. Running analysis...")
        self.run_analysis()

    def on_view_pdf(self):

        if not self.latest_pdf_path or not os.path.exists(self.latest_pdf_path):
            self.statusBar().showMessage("No PDF report available yet.")
            return

        self.pdf_viewer = PdfViewerWindow(self.latest_pdf_path, self)
        self.pdf_viewer.show()


    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.statusBar().showMessage("Camera frame lost.")
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        H, W, _ = frame.shape

        hud_lines = []
        self.frame_counter += 1

        if self.state == "CALIBRATION":
            hud_lines.append("CALIBRATION MODE")
            hud_lines.append(
                f"Reference length: {ENGINE_CONFIG.calibration_object_length_m:.3f} m"
            )
            hud_lines.append("Click two points that are one known length apart.")

            if len(self.calib_points) >= 1:
                cv2.circle(frame, self.calib_points[0], 5, (0, 0, 255), -1)

            if len(self.calib_points) == 2:
                p1, p2 = self.calib_points
                cv2.circle(frame, p2, 5, (0, 0, 255), -1)
                cv2.line(frame, p1, p2, (0, 255, 255), 2)

                dist_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                if dist_px > 5.0:
                    self.pixels_per_meter = dist_px / ENGINE_CONFIG.calibration_object_length_m
                    hud_lines.append(f"Scale: {self.pixels_per_meter:.1f} px/m")
                    hud_lines.append("Click 'Start Tracking' to begin.")
                    self.btn_start.setEnabled(True)

        elif self.state == "TRACKING":
            hud_lines.append("TRACKING MODE")
            hud_lines.append(f"Object: {ENGINE_CONFIG.object_name}")

            if self.pixels_per_meter is None:
                hud_lines.append("ERROR: No calibration scale. Please recalibrate.")
            elif self.model is not None:
                now = time.time()
                if self.t0 is None:
                    self.t0 = now
                t = now - self.t0

                detected = False
                last_center = None

                results = self.model(frame, stream=True, verbose=False)
                for r in results:
                    for box in r.boxes:
                        if box.conf is None:
                            continue
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if (
                            cls_id == ENGINE_CONFIG.target_class_id
                            and conf >= ENGINE_CONFIG.confidence_threshold
                        ):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            last_center = (cx, cy)

                            real_x = cx / self.pixels_per_meter
                            real_y = (H - cy) / self.pixels_per_meter  

                            self.data_log.append([t, real_x, real_y])
                            self.trail.appendleft((cx, cy))

                            hud_lines.append(f"Points: {len(self.data_log)}")
                            hud_lines.append(f"Height: {real_y:.3f} m")

                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (60, 230, 60),
                                2,
                            )
                            detected = True
                            break
                    if detected:
                        break

                if not detected:
                    hud_lines.append("Object not detected in this frame.")

                for i in range(1, len(self.trail)):
                    if self.trail[i - 1] is None or self.trail[i] is None:
                        continue
                    thickness = int(np.sqrt(len(self.trail) / (i + 1)) * 2)
                    cv2.line(frame, self.trail[i - 1], self.trail[i], (0, 0, 255), thickness)

                if len(self.data_log) >= 2 and last_center is not None:
                    t1, x1_m, y1_m = self.data_log[-2]
                    t2, x2_m, y2_m = self.data_log[-1]
                    dt = max(t2 - t1, 1e-6)
                    vx = (x2_m - x1_m) / dt
                    vy = (y2_m - y1_m) / dt
                    speed = float(np.sqrt(vx ** 2 + vy ** 2))

                    hud_lines.append(f"Approx speed: {speed:.3f} m/s")

                    scale = 25
                    if speed > 0:
                        vx_px = int((vx / speed) * scale)
                        vy_px = int((-vy / speed) * scale)
                    else:
                        vx_px = 0
                        vy_px = 0

                    x0, y0 = last_center
                    cv2.arrowedLine(
                        frame,
                        (x0, y0),
                        (x0 + vx_px, y0 + vy_px),
                        (255, 200, 0),
                        2,
                        tipLength=0.3,
                    )

                if self.graph_mode == "live_height" and self.frame_counter % 5 == 0:
                    self.update_live_graph()

        else:
            hud_lines.append("Live view (idle)")
            hud_lines.append("Set up your experiment and press the calibrate button to start.")

        if hud_lines:
            draw_hud_block(frame, hud_lines, (10, 10), width=420)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))


    def on_graph_selection_changed(self, index: int):
        mode = self.graph_combo.itemData(index)
        self.graph_mode = mode
        self.update_graph_view()

    def update_graph_view(self):

        if self.graph_mode == "live_height":
            self.update_live_graph()
        else:
            self.update_analysis_graph()

    def update_live_graph(self):

        if not self.data_log:
            self.graph_label.setPixmap(QPixmap())
            self.graph_label.setText(
                "No data yet.\n\nStart tracking and move the object\nto see the live Height vs Time graph."
            )
            return

        times = [row[0] for row in self.data_log]
        heights = [row[2] for row in self.data_log]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.plot(times, heights, linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (m)")
        ax.set_title("Live: Height vs Time")
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        img = QImage.fromData(buf.getvalue(), "PNG")
        if img.isNull():
            self.graph_label.setPixmap(QPixmap())
            self.graph_label.setText("Error rendering live graph.")
            return

        pix = QPixmap.fromImage(img)
        pix = pix.scaled(
            self.graph_label.width(),
            self.graph_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.graph_label.setPixmap(pix)
        self.graph_label.setText("")

    def update_analysis_graph(self):

        if not self.analysis_available:
            self.graph_label.setPixmap(QPixmap())
            self.graph_label.setText(
                "Analysis graph isn't available yet.\n\n"
                "Run 'Stop & Analyze' after recording an experiment."
            )
            return

        filename = self.graph_files.get(self.graph_mode)
        if not filename:
            self.graph_label.setPixmap(QPixmap())
            self.graph_label.setText("Unknown graph selection.")
            return

        if not os.path.exists(filename):
            self.graph_label.setPixmap(QPixmap())
            self.graph_label.setText(
                f"Graph file '{filename}' not found.\n\n"
                "Try running 'Stop & Analyze' again."
            )
            return

        img = QImage(filename)
        if img.isNull():
            self.graph_label.setPixmap(QPixmap())
            self.graph_label.setText(f"Could not load graph '{filename}'.")
            return

        pix = QPixmap.fromImage(img)
        pix = pix.scaled(
            self.graph_label.width(),
            self.graph_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.graph_label.setPixmap(pix)
        self.graph_label.setText("")


    def run_analysis(self):
        if len(self.data_log) < ENGINE_CONFIG.min_data_points:
            msg = "Not enough data points for analysis. Get more motion in or make it more clear"
            self.statusBar().showMessage(msg)
            self.summary_text.setPlainText(msg)
            self.analysis_available = False
            self.btn_view_pdf.setEnabled(False)
            self.update_graph_view()
            return

        df = pd.DataFrame(self.data_log, columns=["Time", "X", "Y"])
        df["Y_Smooth"] = smooth_series(df["Y"].values, ENGINE_CONFIG.smoothing_window)

        segmenter = MotionSegmenter(
            df["Time"].values,
            df["X"].values,
            df["Y"].values,
            df["Y_Smooth"].values,
            ENGINE_CONFIG,
        )
        segments = segmenter.analyze()

        if not segments:
            msg = "No usable motion segments found. Get more motion in or make it more clear."
            self.statusBar().showMessage(msg)
            self.summary_text.setPlainText(msg)
            self.analysis_available = False
            self.btn_view_pdf.setEnabled(False)
            self.update_graph_view()
            return

        df_kin = apply_segment_kinematics(df, segments)
        stats, df_full = compute_physics_stats(df_kin, segments)

        seg_debug_path = self.graph_files["seg_debug"]
        traj_path = self.graph_files["trajectory"]
        pos_path = self.graph_files["position"]
        vel_path = self.graph_files["velocity"]
        acc_path = self.graph_files["accel"]
        energy_path = self.graph_files["energy"]

        plot_segmentation_debug(df_full, segmenter, seg_debug_path)
        plot_trajectory(df_full, segments, traj_path)
        plot_position_time(df_full, segments, pos_path)
        plot_velocity_time(df_full, vel_path)
        plot_acceleration_time(df_full, acc_path)
        plot_energy_time(df_full, energy_path)

        csv_path = self.get_unique_output_path("analysis_data.csv")
        df_full.to_csv(csv_path, index=False)

        pdf_path = self.get_unique_output_path("Analysis_Report.pdf")
        figure_paths = {
            "segmentation_debug": seg_debug_path,
            "trajectory": traj_path,
            "position_time": pos_path,
            "velocity_time": vel_path,
            "acceleration_time": acc_path,
            "energy_time": energy_path,
        }
        create_pdf_report(segments, df_full, stats, output_pdf=pdf_path, figure_paths=figure_paths)
        self.latest_pdf_path = pdf_path

        self.analysis_available = True
        self.btn_view_pdf.setEnabled(True)

        lines = []
        lines.append(f"Experiment type: {self.selected_experiment_type}")
        lines.append(
            f"Tracked object: {ENGINE_CONFIG.object_name} "
            f"(mass ≈ {ENGINE_CONFIG.mass_kg:.3f} kg)"
        )
        lines.append("")
        lines.append(f"Number of segments: {len(segments)}")
        lines.append("")

        for seg in segments:
            lines.append(f"Segment {seg.index}: {seg.motion_type}")
            lines.append(f"  Time: {seg.t_start:.3f} s to {seg.t_end:.3f} s")
            lines.append(f"  x(t) = {seg.eq_x}")
            lines.append(f"  y(t) = {seg.eq_y}")
            lines.append(f"  R2_x = {seg.r2_x:.3f}, R2_y = {seg.r2_y:.3f}")
            if seg.gravity_est is not None:
                lines.append(f"  g estimate: {seg.gravity_est:.3f} m/s^2")
            lines.append("")

        lines.append("Global stats:")
        if stats.get("duration") is not None:
            lines.append(f"  Duration: {stats['duration']:.3f} s")
        if stats.get("max_height") is not None:
            lines.append(f"  Max height: {stats['max_height']:.3f} m")
        if stats.get("max_speed") is not None:
            lines.append(f"  Max modeled speed: {stats['max_speed']:.3f} m/s")

        g_mean = stats.get("g_mean")
        g_std = stats.get("g_std")
        if g_mean is not None:
            g_err = 100.0 * abs(g_mean - 9.81) / 9.81
            lines.append(
                f"  Global g estimate: {g_mean:.3f} ± {g_std:.3f} m/s^2 "
                f"(error {g_err:.2f} % vs 9.81 m/s^2)"
            )

        e_mean = stats.get("e_mean")
        e_std = stats.get("e_std")
        if e_mean is not None:
            lines.append(
                f"  Avg coefficient of restitution: {e_mean:.3f} "
                f"(std {e_std:.3f})"
            )
        else:
            lines.append("  No clear bounces detected; not reported.")

        lines.append("")
        lines.append("Files generated for this run:")
        lines.append(f"  - PDF report: {os.path.basename(pdf_path)} (in {self.output_dir})")
        lines.append(f"  - CSV data:   {os.path.basename(csv_path)} (in {self.output_dir})")
        lines.append(f"  - Graph PNGs (in {self.output_dir}):")
        lines.append("      trajectory.png, position_time.png, velocity_time.png,")
        lines.append("      acceleration_time.png, energy_time.png, segmentation_debug.png")

        self.summary_text.setPlainText("\n".join(lines))
        self.statusBar().showMessage("Analysis complete. PDF, CSV, and plots generated.")
        self.update_graph_view()

    def closeEvent(self, event):
        if self.timer is not None:
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
