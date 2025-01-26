import sys
import cv2
import os
from pathlib import Path
import numpy as np
import torch
import json
import logging
import datetime
import platform

from PySide6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QHBoxLayout,
    QFormLayout, QMessageBox, QFileDialog, QComboBox, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QSlider, QTextEdit, QSplitter, QSizePolicy, QScrollArea,
    QTabWidget, QMainWindow
)
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap

# Matplotlib / interactive canvas
import matplotlib
matplotlib.use("QtAgg")  # or "Qt5Agg", generally works with PySide6
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ultralytics import YOLO
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from io import BytesIO
import pandas as pd

from group_analyses import analyze_morhpometrics_for_group_root
from analysis_visuals import (
    plot_rainshadow_distributions,
    plot_regression_lines
)


# Configure logging
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')
# Empty the log file on start
open('debug_log.txt', 'w').close()

def convert_numpy_types(obj):
    """Convert NumPy types to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

def select_primary_detection(results):
    """
    From a YOLO `results` object, pick exactly one detection—
    the same logic you use in `correct_detections`—but without
    any warp/homography. Return a dictionary with e.g.:

        {
            'mask': <the raw mask as a 2D np.uint8 array>,
            'box': (x1, y1, x2, y2) or None,
            'class_id': ...
            'confidence': ...
            ...
        }

    or None if no detection found.
    """

    detections_list = []

    for result in results:
        # For each image in the batch (usually just one)
        if result.masks is not None and result.masks.data is not None:
            for idx, mask_tensor in enumerate(result.masks.data):
                # e.g. the raw mask
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                # Possibly we also grab the bounding box info
                boxes = result.boxes
                if boxes is not None and idx < len(boxes):
                    # each box has .xyxy, .conf, .cls, etc.
                    xyxy = boxes.xyxy[idx].cpu().numpy()
                    conf = boxes.conf[idx].cpu().item()
                    cls_id = boxes.cls[idx].cpu().item()
                else:
                    xyxy = None
                    conf = None
                    cls_id = None

                # If the mask is valid, store it
                if np.count_nonzero(mask_np) > 0:
                    detections_list.append({
                        'mask': mask_np,
                        'box': xyxy,
                        'class_id': cls_id,
                        'confidence': conf
                    })

    # If you always pick the first valid detection:
    if len(detections_list) > 0:
        return detections_list[0]

    return None


def smooth_closed_contour(contour_points, iterations=2):
    """
    Given a contour in shape (N,2) (a closed loop),
    run a simple moving-average filter 'iterations' times
    to reduce small zigzags. Returns a new (N,2) array.

    We treat the contour as closed, so it wraps around:
    point[-1] neighbors point[0], etc.
    """
    N = len(contour_points)
    pts = contour_points.copy()

    for _ in range(iterations):
        smoothed = pts.copy()
        for i in range(N):
            i_prev = (i - 1) % N
            i_next = (i + 1) % N
            smoothed[i] = (pts[i_prev] + pts[i] + pts[i_next]) / 3.0
        pts = smoothed

    return pts

def warp_points(points, H):
    """
    Given an Nx2 array of (x,y) in camera coords,
    apply the 3x3 homography H to produce an Nx2 array in the corrected domain.
    """
    if len(points) == 0:
        return np.empty((0,2), dtype=np.float32)
    # Convert to homogeneous
    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_homo = np.hstack([points, ones])  # shape Nx3

    # Warp
    warped = (H @ pts_homo.T).T  # Nx3

    # Divide by final row for homogeneous
    warped[:, 0] /= (warped[:, 2] + 1e-9)
    warped[:, 1] /= (warped[:, 2] + 1e-9)

    return warped[:, :2].astype(np.float32)


class PolarCanvas(FigureCanvas):
    """
    An interactive Matplotlib canvas for the polar plot.
    SHIFT-click removes nearest peak; normal click adds a new peak.
    """
    peaksChanged = Signal(np.ndarray)  # Emitted whenever peaks are updated

    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        super().__init__(fig)
        self.setParent(parent)

        self.angles = np.array([])
        self.distances = np.array([])
        self.peaks = np.array([])

        # Flip the polar axis so that "up" in the image is "up" in the plot
        self.ax.set_theta_direction(-1)    # Make angles go clockwise
        self.ax.set_theta_offset(0)  # Put 0° at the top

        self.ax.set_title("Interactive Polar Plot")
        self.mpl_connect('button_press_event', self.on_click)

    def set_data(self, angles, distances, peaks):
        self.angles = angles
        self.distances = distances
        self.peaks = peaks
        self.update_plot()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        angle_clicked = event.xdata
        if event.guiEvent.modifiers() & Qt.ShiftModifier:
            # SHIFT => remove nearest peak
            if self.peaks.size == 0:
                return
            peak_angles = self.angles[self.peaks.astype(int)]
            diffs = np.abs((peak_angles - angle_clicked + np.pi) % (2*np.pi) - np.pi)
            nearest_idx = np.argmin(diffs)
            self.peaks = np.delete(self.peaks, nearest_idx)
        else:
            # normal click => add
            angle_diffs = np.abs((self.angles - angle_clicked + np.pi) % (2*np.pi) - np.pi)
            new_idx = np.argmin(angle_diffs)
            if new_idx not in self.peaks:
                self.peaks = np.append(self.peaks, new_idx)

        self.update_plot()
        self.peaksChanged.emit(self.peaks)

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Interactive Polar Plot")

        # We must re-apply the direction + offset after clearing:
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_offset(0)

        # Draw the main distance profile as before
        if len(self.angles) > 0 and len(self.angles) == len(self.distances):
            self.ax.plot(self.angles, self.distances, label='Distance Profile')

        # --- Now draw each arm tip as a number ---
        if hasattr(self, 'arm_angles') and len(self.arm_angles) > 0:
            for angle, r, lbl, color in zip(self.arm_angles,
                                            self.arm_dists,
                                            self.arm_labels,
                                            self.arm_colors):
                self.ax.text(
                    angle,
                    r,
                    str(lbl),
                    color=color,
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
                self.ax.scatter(angle, r, color=color, s=20)

        self.draw()  # redraw canvas

    def set_arm_labels(self, angles, distances, labels, colors):
        """
        Tell the polar plot exactly which angles/distances to plot,
        and which number and color each tip should have.
        """
        self.arm_angles = np.array(angles)
        self.arm_dists = np.array(distances)
        self.arm_labels = labels  # list of integers
        self.arm_colors = colors  # list of e.g. 'red' or 'blue'
        self.update_plot()


class WebcamStream(QWidget):
    """
    The detection + morphometrics GUI, updated to:
      - use QSplitters for resizing
      - a Zoom slider for the camera feed
      - use an interactive PolarCanvas instead of QLabels for the polar plot
    """
    def __init__(self):
        super().__init__()

        # Load YOLO model
        path_model = os.path.join('..', '..', 'models', 'best.pt')
        self.yolo_model = YOLO(path_model)
        self.yolo_active = False

        # Determine OpenCV backend
        current_os = platform.system()
        if current_os == "Windows":
            backend = cv2.CAP_DSHOW
        elif current_os == "Darwin":
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_ANY

        self.cap = cv2.VideoCapture(0, backend)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Webcam Error", "Unable to access the webcam.")
            sys.exit()

        self.checkerboard_info = None
        self.corrected_checkerboard = None

        # For the camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.last_frame = None

        # For morphometrics
        self.current_measurement_folder = None
        self.angles_sorted = None
        self.distances_smoothed = None
        self.peaks = None
        self.zoom_factor = 1.0  # for the camera feed

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 8)  # optional

        self.create_ui_components()

    def create_ui_components(self):
        # -------------------- Left Panel --------------------

        # Root dir
        self.root_dir_button = QPushButton("Select Root Data Directory", self)
        self.root_dir_label = QLabel(os.path.join('..', '..', 'measurements'))

        # Form for checkerboard inputs
        form_layout = QFormLayout()
        self.rows_input = QLineEdit("8")
        self.cols_input = QLineEdit("10")
        self.square_size_input = QLineEdit("25")
        self.group_input = QLineEdit("lab")
        self.id_type_combo = QComboBox()
        self.id_type_combo.addItems(["knownID", "unknownID"])
        self.id_input = QLineEdit("inputIDcode")

        self.initials_input = QLineEdit()
        self.initials_input.setPlaceholderText("Enter your initials (e.g., ABC)")
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Enter any notes or comments here...")
        self.notes_input.setMaximumHeight(25)

        form_layout.addRow("Checkerboard Rows (Squares):", self.rows_input)
        form_layout.addRow("Checkerboard Columns (Squares):", self.cols_input)
        form_layout.addRow("Square Size (mm):", self.square_size_input)
        form_layout.addRow("Group Name:", self.group_input)
        form_layout.addRow("ID Type:", self.id_type_combo)
        form_layout.addRow("ID Value:", self.id_input)
        form_layout.addRow("User Initials:", self.initials_input)
        form_layout.addRow("User Notes:", self.notes_input)

        # Buttons
        self.start_button = QPushButton("Start Stream")
        self.stop_button = QPushButton("Stop Stream")
        self.detect_button = QPushButton("Detect Checkerboard")
        self.clear_button = QPushButton("Clear Checkerboard")
        self.start_yolo_button = QPushButton("Start Detections")
        self.stop_yolo_button = QPushButton("Stop Detections")
        self.save_detection_button = QPushButton("Get Detection")
        self.run_morphometrics_button = QPushButton("Run Morphometrics")
        self.save_numbering_button = QPushButton("Save Morphometrics")

        self.clear_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.stop_yolo_button.setEnabled(False)
        self.save_detection_button.setEnabled(False)
        self.run_morphometrics_button.setEnabled(False)
        self.save_numbering_button.setEnabled(False)

        # Sliders for morphometrics
        self.smoothing_label = QLabel("Smoothing Factor: 5")
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(1, 15)
        self.smoothing_slider.setValue(5)

        self.prominence_label = QLabel("Prominence Factor: 0.05")
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 100)
        self.prominence_slider.setValue(1)

        self.distance_label = QLabel("Distance Factor: 5")
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(0, 15)
        self.distance_slider.setValue(5)

        self.arm_rotation_label = QLabel("Arm Rotation: 0")
        self.arm_rotation_slider = QSlider(Qt.Horizontal)
        self.arm_rotation_slider.setRange(0, 24)
        self.arm_rotation_slider.setValue(0)

        # A Zoom slider for the camera feed
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)  # 50% to 200%
        self.zoom_slider.setValue(100)

        # Pack them in a layout
        sliders_container = QWidget()
        sliders_layout = QVBoxLayout(sliders_container)
        sliders_layout.addWidget(self.smoothing_label)
        sliders_layout.addWidget(self.smoothing_slider)
        sliders_layout.addSpacing(10)
        sliders_layout.addWidget(self.prominence_label)
        sliders_layout.addWidget(self.prominence_slider)
        sliders_layout.addSpacing(10)
        sliders_layout.addWidget(self.distance_label)
        sliders_layout.addWidget(self.distance_slider)
        sliders_layout.addSpacing(10)
        sliders_layout.addWidget(self.arm_rotation_label)
        sliders_layout.addWidget(self.arm_rotation_slider)
        sliders_layout.addSpacing(10)
        sliders_layout.addWidget(self.zoom_label)
        sliders_layout.addWidget(self.zoom_slider)
        sliders_layout.addStretch()

        sliders_scroll_area = QScrollArea()
        sliders_scroll_area.setWidgetResizable(True)
        sliders_scroll_area.setWidget(sliders_container)
        sliders_scroll_area.setFixedHeight(350)

        # Button row layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.clear_button)

        yolo_button_layout = QHBoxLayout()
        yolo_button_layout.addWidget(self.start_yolo_button)
        yolo_button_layout.addWidget(self.stop_yolo_button)
        yolo_button_layout.addWidget(self.save_detection_button)
        yolo_button_layout.addWidget(self.run_morphometrics_button)

        # Connect signals
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.detect_button.clicked.connect(self.detect_checkerboard)
        self.clear_button.clicked.connect(self.clear_checkerboard)
        self.start_yolo_button.clicked.connect(self.start_yolo)
        self.stop_yolo_button.clicked.connect(self.stop_yolo)
        self.save_detection_button.clicked.connect(self.save_corrected_detection)
        self.run_morphometrics_button.clicked.connect(self.run_morphometrics)
        self.save_numbering_button.clicked.connect(self.save_updated_morphometrics)

        self.id_type_combo.currentIndexChanged.connect(self.update_id_input)
        self.group_input.textChanged.connect(self.update_id_input)

        self.smoothing_slider.valueChanged.connect(self.on_smoothing_slider_change)
        self.prominence_slider.valueChanged.connect(self.on_prominence_slider_change)
        self.distance_slider.valueChanged.connect(self.on_distance_slider_change)
        self.arm_rotation_slider.valueChanged.connect(self.rotate_arm_numbering)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)

        left_panel_layout = QVBoxLayout()
        left_panel_layout.addWidget(self.root_dir_button)
        left_panel_layout.addWidget(self.root_dir_label)
        left_panel_layout.addLayout(form_layout)
        left_panel_layout.addLayout(button_layout)
        left_panel_layout.addLayout(yolo_button_layout)
        left_panel_layout.addWidget(sliders_scroll_area)
        left_panel_layout.addWidget(self.save_numbering_button)
        left_panel_layout.addStretch()

        # --------------- Right Panel: Split with camera feed, detection, polar canvas ---------------
        self.camera_label = QLabel("Webcam Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setMinimumSize(640, 480)

        # Detection figure is shown in self.result_label (like before)
        self.result_label = QLabel("Detection Plot")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_label.setMinimumSize(300, 300)

        # An interactive polar canvas
        self.polar_canvas = PolarCanvas()
        self.polar_canvas.peaksChanged.connect(self.on_peaks_changed)

        # We’ll create a splitter for the bottom half (detection vs polar)
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(self.result_label)
        bottom_splitter.addWidget(self.polar_canvas)

        # Another splitter for top (camera) vs bottom
        main_vertical_splitter = QSplitter(Qt.Vertical)
        main_vertical_splitter.addWidget(self.camera_label)
        main_vertical_splitter.addWidget(bottom_splitter)

        # Put that splitter in the right panel layout
        right_panel_layout = QVBoxLayout()
        right_panel_layout.addWidget(main_vertical_splitter)
        right_widget = QWidget()
        right_widget.setLayout(right_panel_layout)

        # The overall horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel_layout)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 4)
        main_splitter.setSizes([300, 900])

        # Final layout for this widget
        layout = QHBoxLayout(self)
        layout.addWidget(main_splitter)
        self.setLayout(layout)

        left_widget.setMaximumWidth(500)

        self.update_id_input()

    # ------------------- Zoom Slider Handler -------------------
    def on_zoom_slider_changed(self, value):
        self.zoom_factor = value / 100.0  # e.g. 0.5..2.0
        self.zoom_label.setText(f"Zoom: {value}%")
        # Re-draw the camera feed
        self.update_frame()

    # ------------------- Checkerboard and YOLO Buttons -------------------
    def update_id_input(self):
        id_type = self.id_type_combo.currentText()
        self.id_input.setEnabled(True)
        if id_type == "knownID":
            if not self.id_input.text() or self.id_input.text().startswith(self.group_input.text() + "_uID_"):
                self.id_input.setText("inputIDcode")
        else:
            root_dir = self.root_dir_label.text()
            group_name = self.group_input.text()
            unknown_id_dir = os.path.join(root_dir, group_name, "unknownID")
            os.makedirs(unknown_id_dir, exist_ok=True)
            existing_ids = [d for d in os.listdir(unknown_id_dir)
                            if os.path.isdir(os.path.join(unknown_id_dir, d))]
            pattern = f"{group_name}_uID_"
            max_n = max([int(d.replace(pattern, "")) for d in existing_ids
                         if d.startswith(pattern) and d.replace(pattern, "").isdigit()] + [0])
            default_id = f"{group_name}_uID_{max_n + 1}"
            if not self.id_input.text() or self.id_input.text() == "inputIDcode":
                self.id_input.setText(default_id)

    def start_stream(self):
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        logging.debug("Stream started.")

    def stop_stream(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        logging.debug("Stream stopped.")

    def detect_checkerboard(self):
        try:
            rows = int(self.rows_input.text())
            cols = int(self.cols_input.text())
            square_size = float(self.square_size_input.text())
            logging.debug(f"Checkerboard detection: rows={rows}, cols={cols}, size={square_size}mm")

            if rows <= 1 or cols <= 1:
                raise ValueError("Checkerboard dimensions must be > 1")

            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to read from webcam.")
                QMessageBox.warning(self, "Webcam Error", "Failed to capture frame.")
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            board_dims = (cols - 1, rows - 1)
            found, corners = cv2.findChessboardCorners(gray, board_dims, None)
            if found:
                self.corrected_checkerboard = frame.copy()
                logging.debug(f"Checkerboard corners found: {len(corners)}")

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                self.checkerboard_info = {
                    'dims': board_dims,
                    'corners': corners_refined,
                    'image_points': corners_refined.reshape(-1, 2),
                    'square_size': square_size
                }
                self.clear_button.setEnabled(True)
                QMessageBox.information(self, "Detection Successful", "Checkerboard detected!")
            else:
                logging.warning("Checkerboard not detected.")
                self.checkerboard_info = None
                self.clear_button.setEnabled(False)
                QMessageBox.warning(self, "Detection Failed", "Checkerboard not detected.")
        except ValueError as e:
            logging.error(f"Input error: {str(e)}")
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            self.corrected_checkerboard = None
            logging.exception("Unexpected error in detect_checkerboard.")
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")

    def clear_checkerboard(self):
        self.checkerboard_info = None
        self.corrected_checkerboard = None
        self.clear_button.setEnabled(False)
        logging.debug("Checkerboard info cleared.")

    def start_yolo(self):
        self.yolo_active = True
        self.start_yolo_button.setEnabled(False)
        self.stop_yolo_button.setEnabled(True)
        logging.debug("YOLO detection started.")

    def stop_yolo(self):
        self.yolo_active = False
        self.start_yolo_button.setEnabled(True)
        self.stop_yolo_button.setEnabled(False)
        self.save_detection_button.setEnabled(False)
        logging.debug("YOLO detection stopped.")

    def closeEvent(self, event):
        self.cap.release()
        logging.debug("Webcam released.")
        event.accept()

    # ------------------- CAMERA FEED (update_frame) -------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.last_frame = frame.copy()

        if self.yolo_active:
            results = self.yolo_model.predict(frame, verbose=False)
            if results and len(results) > 0:
                # Use the same selection logic:
                primary_det = select_primary_detection(results)
                if primary_det is not None:
                    # Mark that "Get Detection" is valid
                    self.save_detection_button.setEnabled(True)

                    # Draw the mask or box manually
                    mask_np = primary_det['mask']
                    if mask_np is not None:
                        # e.g. overlay the mask boundary in green
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

                    # or if you want a bounding box:
                    box = primary_det['box']
                    if box is not None:
                        x1, y1, x2, y2 = box.astype(int)
                        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Possibly print class/conf on top
                    cls_id = primary_det['class_id']
                    conf = primary_det['confidence']
                    if conf:
                        cv2.putText(
                            frame,
                            f"Cls={cls_id}, conf={conf:.2f}",
                            (x1, max(0, y1 - 5)),  # above top-left corner
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2
                        )
                else:
                    # No detection found
                    self.save_detection_button.setEnabled(False)
            else:
                self.save_detection_button.setEnabled(False)
        else:
            self.save_detection_button.setEnabled(False)

        # -- Draw the checkerboard corners exactly as before --
        if self.checkerboard_info is not None:
            overlay = frame.copy()
            cv2.drawChessboardCorners(
                overlay,
                self.checkerboard_info['dims'],
                self.checkerboard_info['corners'],
                True
            )
            # alpha-blend 'overlay' back onto 'frame' at some fraction
            alpha = 0.5  # e.g. 50% overlay
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Apply zoom factor
        h, w, ch = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

        # Zoom the dimension
        target_width = int(self.camera_label.width() * self.zoom_factor)
        target_height = int(self.camera_label.height() * self.zoom_factor)
        scaled_img = q_img.scaled(target_width, target_height, Qt.KeepAspectRatio)

        self.camera_label.setPixmap(QPixmap.fromImage(scaled_img))

    # ------------------- "Get Detection" & Checkerboard Correction -------------------
    def save_corrected_detection(self):
        if self.yolo_active and self.checkerboard_info is not None:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                # ... (Same code from your snippet) ...
                # see next lines for the full method
                # We'll just call the original logic below
            else:
                QMessageBox.warning(self, "Save Error", "Failed to read from webcam.")
        else:
            QMessageBox.warning(self, "Save Error", "YOLO is not active OR no checkerboard info.")

        # The rest of your original logic goes here—unchanged:
        # (for brevity, placed inline)
        # -------------------------------------------------------------------------
        if ret:
            raw_frame = frame.copy()
            root_dir = self.root_dir_label.text()
            group_name = self.group_input.text()
            id_type = self.id_type_combo.currentText()
            id_value = self.id_input.text().strip()

            if not id_value:
                QMessageBox.warning(self, "ID Error", "Please enter a valid ID value.")
                return

            # Build directory path
            id_folder = os.path.join(root_dir, group_name, id_type, id_value)
            os.makedirs(id_folder, exist_ok=True)

            # Measurement date
            measurement_date = datetime.datetime.now().strftime("%m_%d_%Y")
            date_folder = os.path.join(id_folder, measurement_date)
            os.makedirs(date_folder, exist_ok=True)

            # Next mFolder
            existing_mfolders = [
                d for d in os.listdir(date_folder)
                if os.path.isdir(os.path.join(date_folder, d)) and d.startswith("mFolder_")
            ]
            m_numbers = [
                int(d.replace("mFolder_", "")) for d in existing_mfolders
                if d.replace("mFolder_", "").isdigit()
            ]
            m_next = max(m_numbers) + 1 if m_numbers else 1
            measurement_folder = os.path.join(date_folder, f"mFolder_{m_next}")
            os.makedirs(measurement_folder, exist_ok=True)

            # Save raw frame
            raw_frame_path = os.path.join(measurement_folder, 'raw_frame.png')
            cv2.imwrite(raw_frame_path, raw_frame)

            # YOLO
            results = self.yolo_model.predict(frame, verbose=False)

            # Checkerboard correction
            corrected_detection = self.correct_detections(results)
            if corrected_detection:
                corrected_mask = corrected_detection['corrected_mask']
                corrected_object = corrected_detection['corrected_object']
                corrected_frame = corrected_detection['corrected_frame']

                mask_path = os.path.join(measurement_folder, 'corrected_mask.png')
                object_path = os.path.join(measurement_folder, 'corrected_object.png')
                json_path = os.path.join(measurement_folder, 'corrected_detection.json')

                cv2.imwrite(mask_path, corrected_mask * 255)
                cv2.imwrite(object_path, corrected_object)

                # Combine
                try:
                    if corrected_frame.shape != corrected_object.shape:
                        co_resized = cv2.resize(
                            corrected_object,
                            (corrected_frame.shape[1], corrected_frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        cm_resized = cv2.resize(
                            corrected_mask,
                            (corrected_frame.shape[1], corrected_frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        co_resized = corrected_object
                        cm_resized = corrected_mask

                    alpha_mask = cm_resized.astype(float) / 255.0
                    alpha_mask = np.stack([alpha_mask]*3, axis=2)
                    corrected_frame_float = corrected_frame.astype(float)
                    co_rgb = cv2.cvtColor(co_resized, cv2.COLOR_BGR2RGB).astype(float)

                    combined_image = (1.0 - alpha_mask) * corrected_frame_float + alpha_mask * co_rgb
                    combined_image = combined_image.astype(np.uint8)

                    combined_image_path = os.path.join(measurement_folder, 'checkerboard_with_object.png')
                    cv2.imwrite(combined_image_path, combined_image)
                except Exception as e:
                    logging.exception("Failed to combine images.")
                    QMessageBox.warning(self, "Combine Error", f"Failed to combine images: {str(e)}")

                class_id = corrected_detection['class_id']
                class_name = self.yolo_model.names[class_id]
                coordinate = corrected_detection['real_world_coordinate']
                homography_matrix = corrected_detection['homography_matrix']
                corrected_polygon = corrected_detection['corrected_polygon']
                mm_per_pixel = corrected_detection['mm_per_pixel']

                detection_info = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'real_world_coordinate': coordinate,
                    'homography_matrix': homography_matrix,
                    'corrected_polygon': corrected_polygon,
                    'mask_path': mask_path,
                    'object_path': object_path,
                    'raw_frame_path': raw_frame_path,
                    'mm_per_pixel': mm_per_pixel,
                    'combined_image_path': combined_image_path
                }
                info_converted = convert_numpy_types(detection_info)
                with open(json_path, 'w') as f:
                    json.dump(info_converted, f, indent=4)

                self.current_measurement_folder = measurement_folder
                self.run_morphometrics_button.setEnabled(True)

                # Display corrected object
                if corrected_object.size != 0:
                    obj_rgb_display = cv2.cvtColor(corrected_object, cv2.COLOR_BGR2RGB)
                    h2, w2, ch2 = obj_rgb_display.shape
                    bytes_per_line2 = ch2 * w2
                    q_img2 = QImage(obj_rgb_display.data, w2, h2, bytes_per_line2, QImage.Format_RGB888)
                    scaled_img2 = q_img2.scaled(
                        self.result_label.width(), self.result_label.height(),
                        Qt.KeepAspectRatio
                    )
                    self.result_label.setPixmap(QPixmap.fromImage(scaled_img2))
                else:
                    self.result_label.clear()

                QMessageBox.information(self, "Save Successful", f"Detection saved to {json_path}")
            else:
                QMessageBox.warning(self, "Save Error", "No detections found or corrected mask is empty.")

    def correct_detections(self, results):
        if not self.checkerboard_info:
            logging.warning("Checkerboard not detected.")
            QMessageBox.warning(self, "Correction Error", "Checkerboard not detected.")
            return None

        try:
            # 1) Compute homography as before
            square_size = self.checkerboard_info['square_size']
            img_pts = self.checkerboard_info['image_points'].reshape(-1, 2)
            board_dims = self.checkerboard_info['dims']
            obj_pts = np.zeros((board_dims[0] * board_dims[1], 2), np.float32)
            obj_pts[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
            obj_pts *= square_size

            H, status = cv2.findHomography(img_pts, obj_pts)
            if H is None:
                logging.error("Failed to compute homography matrix.")
                QMessageBox.warning(self, "Homography Error", "Failed to compute homography matrix.")
                return None

            max_x = int(obj_pts[:, 0].max()) + 10
            max_y = int(obj_pts[:, 1].max()) + 10
            corrected_frame = cv2.warpPerspective(self.last_frame, H, (max_x, max_y))

            # 2) For mm_per_pixel, same as before
            obj_pt1, obj_pt2 = obj_pts[0], obj_pts[1]
            real_distance_mm = np.linalg.norm(obj_pt1 - obj_pt2)
            img_pt1 = cv2.perspectiveTransform(
                np.array([[img_pts[0]]], dtype='float32'), H
            )[0][0]
            img_pt2 = cv2.perspectiveTransform(
                np.array([[img_pts[1]]], dtype='float32'), H
            )[0][0]
            pixel_distance = np.linalg.norm(img_pt1 - img_pt2)
            mm_per_pixel = real_distance_mm / pixel_distance

            # 3) Now detect in camera coords (no warp first!)
            detections_list = []
            for result in results:
                if result.masks is not None and result.masks.data is not None:
                    for idx, mask_tensor in enumerate(result.masks.data):
                        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)

                        # If you need to resize to match self.last_frame, do so
                        h_img, w_img = self.last_frame.shape[:2]
                        mask_cam = cv2.resize(mask_np, (w_img, h_img),
                                              interpolation=cv2.INTER_NEAREST)

                        # findContours in camera coordinates
                        contours, _ = cv2.findContours(mask_cam, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue

                        # pick the largest contour
                        c = max(contours, key=cv2.contourArea)
                        # (Optional) do approximate or smoothing
                        # c_approx = cv2.approxPolyDP(c, 2.0, closed=True)

                        area_pixels = cv2.contourArea(c)
                        if area_pixels < 10:  # or some threshold
                            continue

                        # 4) Warp the contour points to checkerboard space
                        # c is shape (N,1,2) -> (N,2)
                        c_reshaped = c.reshape(-1, 2).astype(np.float32)
                        c_warped = warp_points(c_reshaped, H)

                        # 5) Build a "corrected_mask" by filling that polygon
                        corrected_mask = np.zeros((max_y, max_x), dtype=np.uint8)
                        c_warped_int = np.round(c_warped).astype(np.int32)
                        cv2.fillPoly(corrected_mask, [c_warped_int], 255)

                        # create the corrected_object
                        corrected_object = cv2.bitwise_and(
                            corrected_frame, corrected_frame, mask=corrected_mask
                        )

                        # real-world center from the warped polygon, e.g. moments in corrected coords
                        M = cv2.moments(corrected_mask)
                        if M['m00'] != 0:
                            cx = M['m10'] / M['m00']
                            cy = M['m01'] / M['m00']
                            rw_coord = [cx * mm_per_pixel, cy * mm_per_pixel]
                        else:
                            rw_coord = [None, None]

                        # store everything
                        class_id = int(result.boxes.cls[idx].item())
                        detections_list.append({
                            'class_id': class_id,
                            'corrected_mask': (corrected_mask // 255).astype(np.uint8),
                            'corrected_object': corrected_object,
                            'corrected_polygon': c_warped_int.reshape(-1, 2).tolist(),
                            'real_world_coordinate': rw_coord
                        })

            if detections_list:
                # pick first or highest confidence
                detection = detections_list[0]
                detection['mm_per_pixel'] = mm_per_pixel
                detection['homography_matrix'] = H.tolist()
                detection['corrected_frame'] = corrected_frame
                return detection
            else:
                logging.warning("No detections found to correct.")
                return None

        except Exception as e:
            logging.exception("Error in correct_detections.")
            QMessageBox.critical(self, "Correction Error", f"An error occurred: {str(e)}")
            return None

    # ------------------- Morphometrics & Sliders for smoothing, etc. -------------------

    def on_smoothing_slider_change(self, value):
        self.smoothing_label.setText(f"Smoothing Factor: {value}")
        self.perform_morphometrics_analysis()

    def on_prominence_slider_change(self, value):
        prominence = value / 100.0
        self.prominence_label.setText(f"Prominence Factor: {prominence:.2f}")
        self.perform_morphometrics_analysis()

    def on_distance_slider_change(self, value):
        self.distance_label.setText(f"Distance Factor: {value}")
        self.perform_morphometrics_analysis()

    def rotate_arm_numbering(self):
        self.arm_rotation_label.setText(f"Arm Rotation: {self.arm_rotation_slider.value()}")
        self.update_arm_visualization()

    def find_arm_tips(self, contour, center, smoothing_factor, prominence_factor, distance_factor):
        shifted_contour = contour - center
        angles = np.arctan2(shifted_contour[:, 1], shifted_contour[:, 0])
        distances = np.hypot(shifted_contour[:, 0], shifted_contour[:, 1])

        sorted_indices = np.argsort(angles)
        angles_sorted = angles[sorted_indices]
        distances_sorted = distances[sorted_indices]
        distances_smoothed = uniform_filter1d(distances_sorted, size=smoothing_factor)

        def find_peaks_on_array(arr):
            return find_peaks(arr, prominence=prominence_factor * arr.max(), distance=distance_factor)[0]

        peaks1 = find_peaks_on_array(distances_smoothed)
        roll_amount = int(len(distances_smoothed) * (np.pi / 12) / (2 * np.pi))
        distances_rolled = np.roll(distances_smoothed, roll_amount)
        peaks2 = (find_peaks_on_array(distances_rolled) - roll_amount) % len(distances_smoothed)

        all_peaks = np.unique(np.concatenate([peaks1, peaks2]))
        sorted_peaks = sorted(all_peaks, key=lambda x: distances_smoothed[x], reverse=True)
        peaks = sorted_peaks[:24]

        arm_tips = shifted_contour[sorted_indices][peaks] + center
        arm_angles = angles_sorted[peaks]
        sorted_arms = sorted(zip(arm_tips, arm_angles), key=lambda x: x[1])
        sorted_arm_tips, sorted_arm_angles = zip(*sorted_arms)

        return (
            np.array(sorted_arm_tips),  # the sorted arm-tip coords
            angles_sorted,  # sorted angles
            distances_smoothed,  # smoothed distances
            peaks,  # indices of the final peaks
            sorted_indices,  # the array that sorted angles
            shifted_contour  # the local coords (contour - center)
        )

    def perform_morphometrics_analysis(self):
        if self.current_measurement_folder is None:
            return
        try:
            json_path = os.path.join(self.current_measurement_folder, 'corrected_detection.json')
            mask_path = os.path.join(self.current_measurement_folder, 'corrected_mask.png')
            object_path = os.path.join(self.current_measurement_folder, 'corrected_object.png')

            with open(json_path, 'r') as f:
                detection_info = json.load(f)
            corrected_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            corrected_object = cv2.imread(object_path)
            if corrected_mask is None or corrected_object is None:
                QMessageBox.warning(self, "Morphometrics Error", "Failed to load images.")
                return

            mm_per_pixel = detection_info.get('mm_per_pixel', None)
            if mm_per_pixel is None:
                QMessageBox.warning(self, "Morphometrics Error", "mm_per_pixel not found.")
                return

            contours, _ = cv2.findContours(corrected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                QMessageBox.warning(self, "Morphometrics Error", "No contours in mask.")
                return

            contour = max(contours, key=cv2.contourArea)
            area_pixels = cv2.contourArea(contour)
            area_mm2 = area_pixels * (mm_per_pixel ** 2)

            M = cv2.moments(contour)
            if M['m00'] == 0:
                QMessageBox.warning(self, "Morphometrics Error", "Cannot compute center.")
                return
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            center = np.array([cx, cy])

            smoothing_factor = self.smoothing_slider.value()
            prominence_factor = self.prominence_slider.value() / 100.0
            distance_factor = self.distance_slider.value()

            # Convert the (N,1,2) contour to (N,2):
            raw_points = contour.reshape(-1, 2)

            # --- Smooth it in-place ---
            smoothed_points = smooth_closed_contour(raw_points, iterations=2)

            # Then use 'smoothed_points' for the rest of your code:
            contour_points = smoothed_points
            if contour_points.ndim != 2:
                QMessageBox.warning(self, "Morphometrics Error", "Contour shape error.")
                return

            # (NEW) We unpack 5 returns:
            #   arm_tips, angles_sorted, distances_smoothed, peaks, sorted_indices, shifted_contour
            arm_tips, angles_sorted, distances_smoothed, peaks, sorted_indices, shifted_contour = self.find_arm_tips(
                contour_points, center, smoothing_factor, prominence_factor, distance_factor
            )

            num_arms = len(arm_tips)
            corrected_object_rgb = cv2.cvtColor(corrected_object, cv2.COLOR_BGR2RGB)

            # Build arm_data
            self.arm_data = []
            for i, tip in enumerate(arm_tips):
                x_vec = tip[0] - center[0]
                y_vec = tip[1] - center[1]
                length_px = np.hypot(x_vec, y_vec)
                length_mm = length_px * mm_per_pixel
                self.arm_data.append([i+1, x_vec, y_vec, length_mm])

            # (NEW) Also store the full sorted contour, in global coords, for manual peak editing
            # Because shifted_contour is in local coords => we add center back
            sorted_contour_global = shifted_contour[sorted_indices] + center
            self.sorted_contour_points = sorted_contour_global  # used in on_peaks_changed

            # Also store angles_sorted, distances_smoothed, etc.
            self.angles_sorted = angles_sorted
            self.distances_smoothed = distances_smoothed

            # We also want the mm_per_pixel easily accessible for on_peaks_changed
            # so store it in self.morphometrics_data
            self.morphometrics_data = {
                'area_mm2': area_mm2,
                'num_arms': num_arms,
                'arm_data': self.arm_data,
                'major_axis_mm': None,
                'minor_axis_mm': None,
                'contour_coordinates': contour_points.tolist(),
                'mm_per_pixel': mm_per_pixel  # so we can find length_mm later
            }

            # If ellipse possible
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (x0, y0), (axis_length1, axis_length2), angle = ellipse
                if axis_length1 >= axis_length2:
                    major_axis_length = axis_length1
                    minor_axis_length = axis_length2
                else:
                    major_axis_length = axis_length2
                    minor_axis_length = axis_length1
                    angle += 90
                major_axis_mm = major_axis_length * mm_per_pixel
                minor_axis_mm = minor_axis_length * mm_per_pixel
                self.morphometrics_data['major_axis_mm'] = major_axis_mm
                self.morphometrics_data['minor_axis_mm'] = minor_axis_mm
                self.ellipse_data = (x0, y0, major_axis_length, minor_axis_length, angle)
            else:
                self.ellipse_data = None

            # reset the arm rotation slider
            max_rotation = len(self.arm_data) - 1
            self.arm_rotation_slider.setRange(0, max_rotation)
            self.arm_rotation_slider.setValue(0)

            self.corrected_object_rgb = corrected_object_rgb
            self.center = center

            self.update_arm_visualization()

            # Update the polar plot with the default found peaks
            angles_normalized = np.mod(angles_sorted, 2*np.pi)
            self.polar_canvas.set_data(angles_normalized, distances_smoothed, np.array(peaks))

            # re-enable sliders & save
            self.smoothing_slider.setEnabled(True)
            self.prominence_slider.setEnabled(True)
            self.distance_slider.setEnabled(True)
            self.arm_rotation_slider.setEnabled(True)
            self.save_numbering_button.setEnabled(True)

        except Exception as e:
            logging.exception("Morphometrics Error.")
            QMessageBox.critical(self, "Morphometrics Error", f"Error: {str(e)}")

    def on_peaks_changed(self, new_peaks):
        if not hasattr(self, 'sorted_contour_points'):
            return
        if not hasattr(self, 'morphometrics_data'):
            return
        if 'mm_per_pixel' not in self.morphometrics_data:
            return

        center_x, center_y = self.center

        # 1) Collect all selected peak points in global coords
        peak_coords = []
        for idx_peak in new_peaks.astype(int):
            pt_global = self.sorted_contour_points[idx_peak]
            peak_coords.append(pt_global)
        peak_coords = np.array(peak_coords)

        if len(peak_coords) == 0:
            # Nothing selected
            self.arm_data = []
            self.update_arm_visualization()
            return

        # 2) Compute angles for each point w.r.t. center
        shifted = peak_coords - [center_x, center_y]
        angles = np.arctan2(shifted[:, 1], shifted[:, 0])

        # 3) Sort them by ascending angle
        sort_idx = np.argsort(angles)
        peak_coords_sorted = peak_coords[sort_idx]

        # 4) Rebuild arm_data
        mm_per_pixel = self.morphometrics_data['mm_per_pixel']
        new_arm_data = []
        for i, pt_global in enumerate(peak_coords_sorted):
            x_vec = pt_global[0] - center_x
            y_vec = pt_global[1] - center_y
            length_px = np.hypot(x_vec, y_vec)
            length_mm = length_px * mm_per_pixel
            # Arm number i+1 by default (in ascending angle order)
            new_arm_data.append([i + 1, x_vec, y_vec, length_mm])

        self.arm_data = new_arm_data
        self.update_arm_visualization()

    def update_arm_visualization(self):
        if not hasattr(self, 'arm_data') or not self.arm_data:
            return

        self.ax.clear()
        self.ax.axis('off')
        if hasattr(self, 'corrected_object_rgb') and self.corrected_object_rgb is not None:
            self.ax.imshow(self.corrected_object_rgb)
        else:
            return

        cx, cy = self.center
        rotation = self.arm_rotation_slider.value()
        num_arms = len(self.arm_data)
        new_order = [(i - rotation) % num_arms + 1 for i in range(num_arms)]

        self.ax.plot(cx, cy, 'yo', markersize=10, label='Center')

        for i, arm_info in enumerate(self.arm_data):
            arm_number, x_vec, y_vec, length_mm = arm_info
            tip_x = cx + x_vec
            tip_y = cy + y_vec
            new_num = new_order[i]
            color = 'red' if new_num == 1 else 'blue'
            self.ax.plot([cx, tip_x], [cy, tip_y], color=color, linewidth=2, label='_nolegend_')
            self.ax.plot(tip_x, tip_y, 'o', color=color, markersize=6, label='_nolegend_')

            text_x = (cx + tip_x) / 2
            text_y = (cy + tip_y) / 2
            self.ax.text(text_x, text_y, str(new_num),
                         color='white', fontweight='bold', ha='center', va='center',
                         bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))

        if hasattr(self, 'ellipse_data') and self.ellipse_data:
            x0, y0, major_len, minor_len, angle = self.ellipse_data
            ellipse_patch = Ellipse((x0, y0), major_len, minor_len, angle=angle,
                                    edgecolor='yellow', facecolor='none', linewidth=2, label='Body Ellipse')
            self.ax.add_patch(ellipse_patch)

        area_val = self.morphometrics_data.get("area_mm2", 0)
        num_arms_val = self.morphometrics_data.get("num_arms", 0)
        major_mm = self.morphometrics_data.get("major_axis_mm", 0)
        minor_mm = self.morphometrics_data.get("minor_axis_mm", 0)

        meas_text = (
            f'Area: {area_val:.2f} mm²\n'
            f'Number of Arms: {num_arms_val}\n'
            f'Major Axis: {major_mm}\n'
            f'Minor Axis: {minor_mm}'
        )
        props = dict(boxstyle='round', facecolor='black', alpha=0.5)
        self.ax.text(0.05, 0.95, meas_text, transform=self.ax.transAxes,
                     fontsize=12, verticalalignment='top', bbox=props, color='white')

        self.ax.legend(loc='upper right')
        self.ax.set_title(f"Arm Numbering (Arm 1 = position {rotation + 1})")

        # -- Build lists for the polar plot --
        polar_angles = []
        polar_dists = []
        polar_labels = []
        polar_colors = []

        center_x, center_y = self.center
        for i, arm_info in enumerate(self.arm_data):
            # arm_info = [original_arm_number, x_vec, y_vec, length_mm]
            x_vec = arm_info[1]
            y_vec = arm_info[2]
            tip_angle = np.arctan2(y_vec, x_vec)
            tip_dist = np.hypot(x_vec, y_vec)

            label_after_rotation = new_order[i]  # e.g. 1..N
            color = 'red' if label_after_rotation == 1 else 'blue'

            polar_angles.append(tip_angle)
            polar_dists.append(tip_dist)
            polar_labels.append(label_after_rotation)
            polar_colors.append(color)

        # Tell the polar plot to use these numbered labels instead of red dots
        self.polar_canvas.set_arm_labels(
            polar_angles,
            polar_dists,
            polar_labels,
            polar_colors
        )

        self.fig.canvas.draw()

        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        qimg = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(qimg)
        self.result_label.setPixmap(pixmap.scaled(
            self.result_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def save_updated_morphometrics(self):
        # from your snippet (unchanged)
        if not hasattr(self, 'arm_data') or not self.arm_data:
            QMessageBox.warning(self, "Save Error", "No arm data available to save.")
            return

        rotation = self.arm_rotation_slider.value()
        num_arms = len(self.arm_data)
        reordered_arm_data = self.arm_data[rotation:] + self.arm_data[:rotation]
        for i, arm in enumerate(reordered_arm_data):
            arm[0] = i+1

        self.morphometrics_data['arm_data'] = reordered_arm_data
        self.morphometrics_data['arm_rotation'] = rotation
        user_initials = self.initials_input.text().strip()
        user_notes = self.notes_input.toPlainText().strip()

        if not user_initials.isalpha() or len(user_initials) != 3:
            QMessageBox.warning(self, "Input Error", "Please enter exactly three letters for initials.")
            return

        self.morphometrics_data['user_initials'] = user_initials
        self.morphometrics_data['user_notes'] = user_notes
        self.morphometrics_data.pop('arm_lengths_mm', None)

        morphometrics_json_path = os.path.join(self.current_measurement_folder, 'morphometrics.json')
        with open(morphometrics_json_path, 'w') as f:
            json.dump(self.morphometrics_data, f, indent=4)

        QMessageBox.information(self, "Save Successful",
                                f"Updated morphometrics data saved to {morphometrics_json_path}")

    def run_morphometrics(self):
        if self.current_measurement_folder is None:
            QMessageBox.warning(self, "Morphometrics Error", "No measurement data available.")
            return
        try:
            self.perform_morphometrics_analysis()
            QMessageBox.information(self, "Morphometrics", "Analysis completed and data saved.")
        except Exception as e:
            logging.exception("Error running morphometrics.")
            QMessageBox.critical(self, "Morphometrics Error", f"Error: {str(e)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # If we have data, re-draw the feed or detection
        if hasattr(self, 'angles_sorted') and self.angles_sorted is not None:
            # Re-call the polar canvas update
            pass


class AnalysisTab(QWidget):
    """
    An "Analysis" tab that:
     1) Lets user select a "group root" directory (e.g., knownID).
     2) Runs the aggregator to produce a .pkl file.
     3) Loads that .pkl into a pandas DataFrame.
     4) Displays summary info in a text box and uses advanced visualization functions
        from analysis_visuals.py to show plots in a Matplotlib canvas.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout
        self.layout = QVBoxLayout(self)

        # ---------------------------------------------------------------------
        # 1) A row with buttons to select & run aggregator
        # ---------------------------------------------------------------------
        self.button_layout = QHBoxLayout()

        self.select_dir_button = QPushButton("Select Group Root Directory")
        self.select_dir_button.clicked.connect(self.select_group_directory)
        self.button_layout.addWidget(self.select_dir_button)

        self.run_aggregator_button = QPushButton("Run Aggregation -> Create .pkl")
        self.run_aggregator_button.clicked.connect(self.run_aggregation)
        # Disabled initially until the user picks a valid directory
        self.run_aggregator_button.setEnabled(False)
        self.button_layout.addWidget(self.run_aggregator_button)

        self.layout.addLayout(self.button_layout)

        # ---------------------------------------------------------------------
        # 2) A label to show which directory is selected
        # ---------------------------------------------------------------------
        self.dir_label = QLabel("No directory selected.")
        self.dir_label.setStyleSheet("QLabel { color: gray; }")
        self.layout.addWidget(self.dir_label)

        # ---------------------------------------------------------------------
        # 3) A text area for output logs
        # ---------------------------------------------------------------------
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text, stretch=1)

        # ---------------------------------------------------------------------
        # 4) A Matplotlib figure/canvas for plotting
        # ---------------------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas, stretch=2)

        # Keep track of aggregator results & DataFrame
        self.selected_dir_path = None
        self.pkl_path = None
        self.df = None

        # ---------------------------------------------------------------------
        # 5) A button to visualize data once loaded
        # ---------------------------------------------------------------------
        self.plot_button = QPushButton("Generate Advanced Plots")
        self.plot_button.clicked.connect(self.plot_data)
        self.plot_button.setEnabled(False)
        self.layout.addWidget(self.plot_button)

    # -------------------------------------------------------------------------
    # Step 1: Select the Group Root Directory
    # -------------------------------------------------------------------------
    def select_group_directory(self):
        """
        Open a dialog for selecting a directory that contains measurements,
        e.g. '../measurements/snack_pack/knownID'.
        """
        directory = QFileDialog.getExistingDirectory(
            self, "Select a Group Root Directory", os.getcwd()
        )
        if directory:
            self.selected_dir_path = Path(directory)
            self.dir_label.setText(f"Selected Directory: {self.selected_dir_path}")
            self.dir_label.setStyleSheet("QLabel { color: black; }")
            self.run_aggregator_button.setEnabled(True)
        else:
            self.output_text.append("No directory selected.")

    # -------------------------------------------------------------------------
    # Step 2: Run Aggregation + Create .pkl
    # -------------------------------------------------------------------------
    def run_aggregation(self):
        """
        Calls the aggregator function to produce a .pkl file, then loads it into self.df.
        """
        if not self.selected_dir_path or not self.selected_dir_path.exists():
            QMessageBox.warning(self, "Invalid Directory", "Please select a valid group directory first.")
            return

        try:
            self.output_text.append(f"Running aggregation on {self.selected_dir_path} ...\n")

            # Call your aggregator script
            pkl_path, aggregated_measures_pkl = analyze_morhpometrics_for_group_root(
                self.selected_dir_path, verbosity=1
            )

            self.pkl_path = pkl_path
            self.output_text.append(f"Created aggregated file:\n  {self.pkl_path}\n")

            # Now load it into a DataFrame
            self.load_aggregated_file_into_df()

        except Exception as e:
            QMessageBox.critical(self, "Aggregator Error", f"Failed to run aggregator:\n{str(e)}")

    # -------------------------------------------------------------------------
    # Step 3: Load the .pkl into a pandas DataFrame
    # -------------------------------------------------------------------------
    def load_aggregated_file_into_df(self):
        """
        Loads the newly created .pkl file into a pandas DataFrame,
        logs the results, and enables the plot button.
        """
        if not self.pkl_path:
            QMessageBox.warning(self, "No PKL File", "No aggregator .pkl file found.")
            return

        try:
            self.output_text.append(f"Loading data from {self.pkl_path} ...")

            # aggregator result is typically a nested dict: {date: {ID: metrics_dict}}
            data = pd.read_pickle(self.pkl_path)

            rows = []
            for date_key, id_dict in data.items():
                for star_id, metrics in id_dict.items():
                    row = {"measurement_date": date_key, "id": star_id}
                    if isinstance(metrics, dict):
                        for k, v in metrics.items():
                            row[k] = v
                    rows.append(row)

            self.df = pd.DataFrame(rows)

            # Show some info
            self.output_text.append(f"DataFrame shape: {self.df.shape}")
            self.output_text.append(f"Columns: {list(self.df.columns)}\n")
            self.output_text.append("Head:\n" + str(self.df.head()) + "\n")

            # Enable the plotting button
            self.plot_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load data:\n{str(e)}")

    # -------------------------------------------------------------------------
    # Step 4: Call Our "analysis_visuals" Functions to Plot
    # -------------------------------------------------------------------------
    def plot_data(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "No data loaded.")
            return

        # Clear old figure
        self.ax.clear()
        self.fig.clear()

        # Example: create a new figure with boxplots, etc. using your advanced function:
        fig2, axs = plot_rainshadow_distributions(self.df, columns=['mean_width', 'area_mm2'])

        # Replace the old figure with this new one
        self.canvas.figure = fig2

        # Force a fresh draw so boxplots appear right away
        self.canvas.draw()
        self.canvas.update()

        # If you do multiple plots/updates, you might also call self.repaint() as needed

        # Example: run regression lines
        import io, sys
        backup_stdout = sys.stdout
        sys.stdout = io.StringIO()
        plot_regression_lines(self.df, columns=['mean_width', 'area_mm2'])
        reg_output = sys.stdout.getvalue()
        sys.stdout = backup_stdout

        self.output_text.append(reg_output)

def format_single_row_data(row):
    """
    Build a nicely-formatted, multiline string for a single measurement row.
    """
    lines = []
    lines.append(f"Measurement Date: {row.get('measurement_date','?')}")
    lines.append(f"ID: {row.get('id','?')}")

    # tip-to-tip widths
    tip_list = row.get('tip_to_tip_widths', [])
    if tip_list:
        lines.append("Tip-to-tip widths (mm):")
        for w in tip_list:
            lines.append(f"  - {w:.2f}")
    else:
        lines.append("Tip-to-tip widths: (none)")

    # mean width
    mean_w = row.get('mean_width')
    if mean_w is not None:
        lines.append(f"Mean Width (mm): {mean_w:.2f}")

    # median width
    median_w = row.get('median_width')
    if median_w is not None:
        lines.append(f"Median Width (mm): {median_w:.2f}")

    # std width
    std_w = row.get('std_width')
    if std_w is not None:
        lines.append(f"Std Width (mm): {std_w:.2f}")

    # area
    area_val = row.get('area_mm2')
    if area_val is not None:
        lines.append(f"Area (mm²): {area_val:.2f}")

    # major/minor
    major_len = row.get('major_axis_length_mm')
    if major_len is not None:
        lines.append(f"Major Axis (mm): {major_len:.2f}")

    minor_len = row.get('minor_axis_length_mm')
    if minor_len is not None:
        lines.append(f"Minor Axis (mm): {minor_len:.2f}")

    return "\n".join(lines)


class IndividualAnalysisTab(QWidget):
    """
    A combined tab that:
      1) Lets user select a group root directory, run aggregator -> create DataFrame.
      2) Provides combos for ID + date.
      3) On ID selection => show per-ID distribution (across all dates) if >=2 rows.
         Otherwise, show textual info if only 1 row or none.
      4) On date selection => find the latest mFolder on disk for that ID+date,
         display images (raw_frame, corrected_object, corrected_mask, checkerboard_with_object),
         and also show the row’s metrics (area, mean_width, etc.) in the text area.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)

        # --------------------------------------------------------------------
        # Buttons: Select group root dir & run aggregator
        # --------------------------------------------------------------------
        button_layout = QHBoxLayout()
        self.select_dir_button = QPushButton("Select Group Root Directory")
        self.select_dir_button.clicked.connect(self.select_directory)
        button_layout.addWidget(self.select_dir_button)

        self.run_agg_button = QPushButton("Run Aggregation")
        self.run_agg_button.clicked.connect(self.run_aggregation)
        self.run_agg_button.setEnabled(False)
        button_layout.addWidget(self.run_agg_button)

        self.layout.addLayout(button_layout)

        self.dir_label = QLabel("No directory selected.")
        self.dir_label.setStyleSheet("QLabel { color: gray; }")
        self.layout.addWidget(self.dir_label)

        # --------------------------------------------------------------------
        # Text area for logs / info
        # --------------------------------------------------------------------
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        # --------------------------------------------------------------------
        # ID + Date combos
        # --------------------------------------------------------------------
        combo_layout = QHBoxLayout()

        combo_layout.addWidget(QLabel("Select ID:"))
        self.id_combo = QComboBox()
        self.id_combo.setEnabled(False)
        self.id_combo.currentIndexChanged.connect(self.on_id_changed)
        combo_layout.addWidget(self.id_combo)

        combo_layout.addWidget(QLabel("Select Date:"))
        self.date_combo = QComboBox()
        self.date_combo.setEnabled(False)
        self.date_combo.currentIndexChanged.connect(self.on_date_changed)
        combo_layout.addWidget(self.date_combo)

        self.layout.addLayout(combo_layout)

        # --------------------------------------------------------------------
        # Matplotlib figure for advanced plots
        # --------------------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas, stretch=1)

        # --------------------------------------------------------------------
        # Four image labels in a row
        # --------------------------------------------------------------------
        image_layout = QHBoxLayout()

        self.label_raw = QLabel("raw_frame.png")
        self.label_obj = QLabel("corrected_object.png")
        self.label_mask = QLabel("corrected_mask.png")
        self.label_checker = QLabel("checkerboard_with_object.png")

        for lbl in [self.label_raw, self.label_obj, self.label_mask, self.label_checker]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(220, 220)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        image_layout.addWidget(self.label_raw)
        image_layout.addWidget(self.label_obj)
        image_layout.addWidget(self.label_mask)
        image_layout.addWidget(self.label_checker)

        self.layout.addLayout(image_layout)

        # --------------------------------------------------------------------
        # Internals
        # --------------------------------------------------------------------
        self.selected_dir_path = None
        self.pkl_path = None
        self.df_all = None     # Entire aggregator DataFrame
        self.df_by_id = None   # Filtered to the chosen ID
        self.df_by_id_date = None  # Filtered to that ID+date

    # ------------------------------------------------------------------------
    # Step 1) Select group root directory
    # ------------------------------------------------------------------------
    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Group Root Directory", os.getcwd()
        )
        if directory:
            self.selected_dir_path = Path(directory)
            self.dir_label.setText(f"Selected Directory: {self.selected_dir_path}")
            self.dir_label.setStyleSheet("QLabel { color: black; }")
            self.run_agg_button.setEnabled(True)
        else:
            self.output_text.append("No directory selected.")

    # ------------------------------------------------------------------------
    # Step 2) Run aggregator => flatten => fill combos
    # ------------------------------------------------------------------------
    def run_aggregation(self):
        if not self.selected_dir_path:
            QMessageBox.warning(self, "No Directory", "Please select a directory first.")
            return

        try:
            self.output_text.append(f"Running aggregator on {self.selected_dir_path}...\n")
            pkl_path, agg_dict = analyze_morhpometrics_for_group_root(
                self.selected_dir_path, verbosity=0
            )
            self.pkl_path = pkl_path
            self.output_text.append(f"Aggregated file: {pkl_path}\n")

            self.df_all = self.flatten_aggregator_dict(agg_dict)
            self.output_text.append(f"DataFrame shape: {self.df_all.shape}\n")
            if not self.df_all.empty:
                self.output_text.append("Head:\n" + str(self.df_all.head()) + "\n")

            # Unique IDs
            unique_ids = sorted(self.df_all['id'].unique().tolist())
            self.id_combo.clear()
            for uid in unique_ids:
                self.id_combo.addItem(uid)
            self.id_combo.setEnabled(True)

            # Date combo waits until user picks ID
            self.date_combo.clear()
            self.date_combo.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Aggregator Error", str(e))

    def flatten_aggregator_dict(self, agg_dict):
        rows = []
        for date_key, id_dict in agg_dict.items():
            for star_id, metrics in id_dict.items():
                row = {"measurement_date": date_key, "id": star_id}
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        row[k] = v
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # Step 3) On ID selection => filter to that ID => do advanced plot
    # ------------------------------------------------------------------------
    def on_id_changed(self):
        if self.df_all is None or self.df_all.empty:
            return
        chosen_id = self.id_combo.currentText()
        if not chosen_id:
            return

        self.df_by_id = self.df_all[self.df_all['id'] == chosen_id].copy()
        n = len(self.df_by_id)
        self.output_text.append(f"\nSelected ID: {chosen_id} => {n} row(s) total.\n")

        # Fill date combo
        if n == 0:
            self.date_combo.clear()
            self.date_combo.setEnabled(False)
        else:
            dates = sorted(self.df_by_id['measurement_date'].unique().tolist())
            self.date_combo.clear()
            for d in dates:
                self.date_combo.addItem(d)
            self.date_combo.setEnabled(True)

        # Now do the advanced ID-based plot across all the individual's dates
        self.plot_individual_data()

    def plot_individual_data(self):
        """
        If the chosen ID has >=2 rows (i.e. multiple dates), we do a distribution or regression across time.
        Otherwise, we skip advanced plots and show a single-row summary.
        """
        self.ax.clear()
        self.fig.clear()

        if self.df_by_id is None or self.df_by_id.empty:
            self.output_text.append("No data for this ID.\n")
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Data for ID", ha='center', va='center')
            self.canvas.draw()
            return

        n = len(self.df_by_id)
        if n < 2:
            # Only 1 row => skip advanced plots
            row_data = self.df_by_id.iloc[0].to_dict()

            self.output_text.append(
                f"Only 1 measurement for ID={row_data.get('id','?')} => no advanced plot.\n"
            )

            # Show row data in a nice multi-line format
            row_str = format_single_row_data(row_data)
            self.output_text.append(row_str + "\n")

            ax = self.fig.add_subplot(111)
            ax.axis('off')
            ax.text(
                0.5, 0.5,
                f"Only 1 measurement for ID={row_data.get('id','?')}\nNo advanced plot",
                ha='center', va='center'
            )
            self.canvas.draw()
            return
        else:
            # We have multiple rows => do advanced distribution + regression
            import sys, io
            backup = sys.stdout
            sys.stdout = io.StringIO()

            fig2, axs = plot_rainshadow_distributions(
                self.df_by_id, columns=['mean_width', 'area_mm2']
            )

            reg_output = ""
            try:
                plot_regression_lines(self.df_by_id, columns=['mean_width', 'area_mm2'])
                reg_output = sys.stdout.getvalue()
            finally:
                sys.stdout = backup

            # Put the figure on the canvas
            self.canvas.figure = fig2
            self.canvas.draw()

            # Show regression results
            self.output_text.append(reg_output)

    # ------------------------------------------------------------------------
    # Step 4) On date selection => find last mFolder => show images + row metrics
    # ------------------------------------------------------------------------
    def on_date_changed(self):
        if self.df_by_id is None or self.df_by_id.empty:
            return
        chosen_date = self.date_combo.currentText()
        if not chosen_date:
            return

        # Filter to that single date
        self.df_by_id_date = self.df_by_id[self.df_by_id['measurement_date'] == chosen_date].copy()
        n = len(self.df_by_id_date)
        self.output_text.append(f"\nChosen date: {chosen_date} => {n} row(s) for ID.\n")

        # If no row => clear images
        if n == 0:
            self.clear_image_labels()
            self.output_text.append("No aggregator row for that date.\n")
            return

        # If multiple rows, we pick the last one. Adjust if needed.
        row = self.df_by_id_date.iloc[-1].to_dict()

        # Show textual data about this row
        self.show_row_data(row)

        # Now find the last mFolder on disk for that ID & date
        chosen_id = self.id_combo.currentText()
        try:
            mfolder = self.find_latest_mfolder(chosen_id, chosen_date)
            if not mfolder:
                self.output_text.append("No mFolder found on disk for that ID & date.\n")
                self.clear_image_labels()
                return
            self.output_text.append(f"Latest mFolder: {mfolder}\n")
            self.load_images_from_folder(mfolder)
        except Exception as e:
            QMessageBox.warning(self, "mFolder Error", str(e))

    def show_row_data(self, row):
        """
        Show relevant metrics in the text area.  Uses the format_single_row_data helper
        for a cleaner multi-line display.
        """
        self.output_text.append(
            f"\nMetrics for ID={row.get('id','?')}, date={row.get('measurement_date','?')}:\n"
        )
        nice_str = format_single_row_data(row)
        self.output_text.append(nice_str + "\n")

    def find_latest_mfolder(self, id_val, date_str):
        """
        e.g.   <root_dir>/<id_val>/<date_str>/mFolder_<N>
        find the highest N
        """
        base_dir = self.selected_dir_path / id_val / date_str
        if not base_dir.exists():
            return None
        subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("mFolder_")]
        if not subdirs:
            return None

        def get_m_num(dpath):
            name = dpath.name
            try:
                return int(name.replace("mFolder_", ""))
            except:
                return -1

        subdirs_sorted = sorted(subdirs, key=get_m_num)
        return subdirs_sorted[-1]  # last one is the highest

    def clear_image_labels(self):
        self.label_raw.setText("raw_frame.png")
        self.label_obj.setText("corrected_object.png")
        self.label_mask.setText("corrected_mask.png")
        self.label_checker.setText("checkerboard_with_object.png")

    def load_images_from_folder(self, folder):
        """
        Attempt to load raw_frame.png, corrected_object.png, corrected_mask.png,
        checkerboard_with_object.png from the chosen folder.
        """
        self.load_one_image(folder / "raw_frame.png", self.label_raw)
        self.load_one_image(folder / "corrected_object.png", self.label_obj)
        self.load_one_image(folder / "corrected_mask.png", self.label_mask)
        self.load_one_image(folder / "checkerboard_with_object.png", self.label_checker)

    def load_one_image(self, path, label):
        if path.exists():
            pix = QPixmap(str(path))
            if not pix.isNull():
                label.setPixmap(
                    pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                return
        label.setText(path.name + "\n(not found)")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Tab Morphometric Tool")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: detection
        self.detection_tab = WebcamStream()
        self.tab_widget.addTab(self.detection_tab, "Detection")

        # Tab 2: analysis
        self.analysis_tab = AnalysisTab()
        self.tab_widget.addTab(self.analysis_tab, "Analysis")

        # Tab 3: individual analysis
        self.individual_tab = IndividualAnalysisTab()
        self.tab_widget.addTab(self.individual_tab, "Individual Report")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
