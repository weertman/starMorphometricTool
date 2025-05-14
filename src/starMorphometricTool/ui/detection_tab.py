import os
import json
import logging
import datetime
import platform
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QFormLayout, QMessageBox, QScrollArea, QSlider, QTextEdit, QSplitter,
    QSizePolicy, QComboBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

# Import our custom modules
from utils.image_processing import smooth_closed_contour, warp_points
from utils.data_utils import convert_numpy_types
from detection.yolo_handler import load_yolo_model, select_primary_detection
from detection.checkerboard import find_checkerboard, compute_checkerboard_homography, calculate_mm_per_pixel
from morphometrics.analysis import find_arm_tips, calculate_morphometrics
from morphometrics.visualization import create_morphometrics_visualization, render_figure_to_pixmap
from ui.components.polar_canvas import PolarCanvas


class DetectionTab(QWidget):
    """
    Detection tab for morphometric analysis.
    Handles webcam capture, checkerboard calibration, YOLO detection,
    and morphometric measurement.
    """

    def __init__(self):
        super().__init__()

        # Load YOLO model
        self.yolo_model = load_yolo_model()
        self.yolo_active = False

        # Initialize webcam with appropriate backend
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
            raise RuntimeError("Failed to open webcam")

        # Initialize state variables
        self.checkerboard_info = None
        self.corrected_checkerboard = None
        self.current_measurement_folder = None
        self.angles_sorted = None
        self.distances_smoothed = None
        self.peaks = None
        self.zoom_factor = 1.0

        # Create matplotlib figure for visualization
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 8)

        # Setup timer for webcam feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.last_frame = None

        # Build UI components
        self.create_ui_components()

    def create_ui_components(self):
        """Create and arrange UI components"""
        # -------------------- Left Panel --------------------

        # Root dir selection
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

        # Action buttons
        self.start_button = QPushButton("Start Stream")
        self.stop_button = QPushButton("Stop Stream")
        self.detect_button = QPushButton("Detect Checkerboard")
        self.clear_button = QPushButton("Clear Checkerboard")
        self.start_yolo_button = QPushButton("Start Detections")
        self.stop_yolo_button = QPushButton("Stop Detections")
        self.save_detection_button = QPushButton("Get Detection")
        self.run_morphometrics_button = QPushButton("Run Morphometrics")
        self.save_numbering_button = QPushButton("Save Morphometrics")

        # Set initial button states
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

        # Zoom slider for camera feed
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)  # 50% to 200%
        self.zoom_slider.setValue(100)

        # Pack sliders in scrollable container
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

        # Organize buttons in layouts
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
        self.root_dir_button.clicked.connect(self.select_root_directory)
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

        # Assemble left panel
        left_panel_layout = QVBoxLayout()
        left_panel_layout.addWidget(self.root_dir_button)
        left_panel_layout.addWidget(self.root_dir_label)
        left_panel_layout.addLayout(form_layout)
        left_panel_layout.addLayout(button_layout)
        left_panel_layout.addLayout(yolo_button_layout)
        left_panel_layout.addWidget(sliders_scroll_area)
        left_panel_layout.addWidget(self.save_numbering_button)
        left_panel_layout.addStretch()

        # -------------------- Right Panel --------------------
        self.camera_label = QLabel("Webcam Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setMinimumSize(640, 480)

        self.result_label = QLabel("Detection Plot")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_label.setMinimumSize(300, 300)

        # Interactive polar canvas
        self.polar_canvas = PolarCanvas()
        self.polar_canvas.peaksChanged.connect(self.on_peaks_changed)

        # Create splitters for resizable panels
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(self.result_label)
        bottom_splitter.addWidget(self.polar_canvas)

        main_vertical_splitter = QSplitter(Qt.Vertical)
        main_vertical_splitter.addWidget(self.camera_label)
        main_vertical_splitter.addWidget(bottom_splitter)

        right_panel_layout = QVBoxLayout()
        right_panel_layout.addWidget(main_vertical_splitter)
        right_widget = QWidget()
        right_widget.setLayout(right_panel_layout)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel_layout)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 4)
        main_splitter.setSizes([300, 900])

        # Final layout
        layout = QHBoxLayout(self)
        layout.addWidget(main_splitter)
        self.setLayout(layout)

        # Set layout constraints
        left_widget.setMaximumWidth(500)

        # Initialize ID input
        self.update_id_input()

    def select_root_directory(self):
        """Open file dialog to select the root data directory"""
        pass  # Implement directory selection dialog

    # ------------------- Zoom Slider Handler -------------------
    def on_zoom_slider_changed(self, value):
        """Handle changes to the zoom slider"""
        self.zoom_factor = value / 100.0  # e.g. 0.5..2.0
        self.zoom_label.setText(f"Zoom: {value}%")
        # Re-draw the camera feed
        self.update_frame()

    # ------------------- ID and Input Handlers -------------------
    def update_id_input(self):
        """Update the ID input field based on selected ID type"""
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

    # ------------------- Webcam Stream Controls -------------------
    def start_stream(self):
        """Start the webcam stream"""
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        logging.debug("Stream started.")

    def stop_stream(self):
        """Stop the webcam stream"""
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        logging.debug("Stream stopped.")

    def update_frame(self):
        """Update the camera feed display"""
        ret, frame = self.cap.read()
        if not ret:
            return

        self.last_frame = frame.copy()

        # Add YOLO detection overlay if active
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
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

        # Draw checkerboard corners if detected
        if self.checkerboard_info is not None:
            overlay = frame.copy()
            cv2.drawChessboardCorners(
                overlay,
                self.checkerboard_info['dims'],
                self.checkerboard_info['corners'],
                True
            )
            # Alpha-blend overlay onto frame
            alpha = 0.5  # e.g. 50% overlay
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Convert to QPixmap and apply zoom
        h, w, ch = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

        # Apply zoom factor
        target_width = int(self.camera_label.width() * self.zoom_factor)
        target_height = int(self.camera_label.height() * self.zoom_factor)
        scaled_img = q_img.scaled(target_width, target_height, Qt.KeepAspectRatio)

        self.camera_label.setPixmap(QPixmap.fromImage(scaled_img))

    # ------------------- Checkerboard Detection -------------------
    def detect_checkerboard(self):
        """Detect checkerboard pattern in the current frame"""
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

            # Use the checkerboard detection function
            board_dims = (cols - 1, rows - 1)
            found, corners, corners_refined = find_checkerboard(frame, board_dims)

            if found:
                self.corrected_checkerboard = frame.copy()
                logging.debug(f"Checkerboard corners found: {len(corners)}")

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
        """Clear the checkerboard detection data"""
        self.checkerboard_info = None
        self.corrected_checkerboard = None
        self.clear_button.setEnabled(False)
        logging.debug("Checkerboard info cleared.")

    # ------------------- YOLO Detection Controls -------------------
    def start_yolo(self):
        """Start YOLO object detection"""
        self.yolo_active = True
        self.start_yolo_button.setEnabled(False)
        self.stop_yolo_button.setEnabled(True)
        logging.debug("YOLO detection started.")

    def stop_yolo(self):
        """Stop YOLO object detection"""
        self.yolo_active = False
        self.start_yolo_button.setEnabled(True)
        self.stop_yolo_button.setEnabled(False)
        self.save_detection_button.setEnabled(False)
        logging.debug("YOLO detection stopped.")

    # ------------------- Detection Saving -------------------
    def save_corrected_detection(self):
        """Save the corrected detection with checkerboard calibration"""
        if self.yolo_active and self.checkerboard_info is not None:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
            else:
                QMessageBox.warning(self, "Save Error", "Failed to read from webcam.")
                return
        else:
            QMessageBox.warning(self, "Save Error", "YOLO is not active OR no checkerboard info.")
            return

        # Process the detection
        if ret:
            raw_frame = frame.copy()
            root_dir = self.root_dir_label.text()
            group_name = self.group_input.text()
            id_type = self.id_type_combo.currentText()
            id_value = self.id_input.text().strip()

            if not id_value:
                QMessageBox.warning(self, "ID Error", "Please enter a valid ID value.")
                return

            # Build directory structure
            id_folder = os.path.join(root_dir, group_name, id_type, id_value)
            os.makedirs(id_folder, exist_ok=True)

            # Measurement date
            measurement_date = datetime.datetime.now().strftime("%m_%d_%Y")
            date_folder = os.path.join(id_folder, measurement_date)
            os.makedirs(date_folder, exist_ok=True)

            # Create next measurement folder
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

            # Run YOLO detection
            results = self.yolo_model.predict(frame, verbose=False)

            # Correct detections using checkerboard
            corrected_detection = self.correct_detections(results)
            if corrected_detection:
                # Save detection artifacts
                corrected_mask = corrected_detection['corrected_mask']
                corrected_object = corrected_detection['corrected_object']
                corrected_frame = corrected_detection['corrected_frame']

                mask_path = os.path.join(measurement_folder, 'corrected_mask.png')
                object_path = os.path.join(measurement_folder, 'corrected_object.png')
                json_path = os.path.join(measurement_folder, 'corrected_detection.json')

                cv2.imwrite(mask_path, corrected_mask * 255)
                cv2.imwrite(object_path, corrected_object)

                # Combine checkerboard with object
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
                    alpha_mask = np.stack([alpha_mask] * 3, axis=2)
                    corrected_frame_float = corrected_frame.astype(float)
                    co_rgb = cv2.cvtColor(co_resized, cv2.COLOR_BGR2RGB).astype(float)

                    combined_image = (1.0 - alpha_mask) * corrected_frame_float + alpha_mask * co_rgb
                    combined_image = combined_image.astype(np.uint8)

                    combined_image_path = os.path.join(measurement_folder, 'checkerboard_with_object.png')
                    cv2.imwrite(combined_image_path, combined_image)
                except Exception as e:
                    logging.exception("Failed to combine images.")
                    QMessageBox.warning(self, "Combine Error", f"Failed to combine images: {str(e)}")

                # Extract detection information
                class_id = corrected_detection['class_id']
                class_name = self.yolo_model.names[class_id]
                coordinate = corrected_detection['real_world_coordinate']
                homography_matrix = corrected_detection['homography_matrix']
                corrected_polygon = corrected_detection['corrected_polygon']
                mm_per_pixel = corrected_detection['mm_per_pixel']

                # Save detection info to JSON
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
        """
        Correct detections using checkerboard homography.

        Args:
            results: YOLO detection results

        Returns:
            Dictionary with corrected detection data or None if failed
        """
        if not self.checkerboard_info:
            logging.warning("Checkerboard not detected.")
            QMessageBox.warning(self, "Correction Error", "Checkerboard not detected.")
            return None

        try:
            # 1) Compute homography
            square_size = self.checkerboard_info['square_size']
            img_pts = self.checkerboard_info['image_points'].reshape(-1, 2)
            board_dims = self.checkerboard_info['dims']

            # Use helper functions from checkerboard.py
            H, obj_pts = compute_checkerboard_homography(img_pts, board_dims, square_size)

            if H is None:
                logging.error("Failed to compute homography matrix.")
                QMessageBox.warning(self, "Homography Error", "Failed to compute homography matrix.")
                return None

            max_x = int(obj_pts[:, 0].max()) + 10
            max_y = int(obj_pts[:, 1].max()) + 10
            corrected_frame = cv2.warpPerspective(self.last_frame, H, (max_x, max_y))

            # 2) Calculate mm per pixel
            mm_per_pixel = calculate_mm_per_pixel(H, img_pts, obj_pts)

            if mm_per_pixel is None:
                logging.error("Failed to calculate mm per pixel.")
                QMessageBox.warning(self, "Calibration Error", "Failed to calculate mm per pixel.")
                return None

            # 3) Process detections in camera coordinates
            detections_list = []
            for result in results:
                if result.masks is not None and result.masks.data is not None:
                    for idx, mask_tensor in enumerate(result.masks.data):
                        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)

                        # Resize if needed to match self.last_frame
                        h_img, w_img = self.last_frame.shape[:2]
                        mask_cam = cv2.resize(mask_np, (w_img, h_img),
                                              interpolation=cv2.INTER_NEAREST)

                        # Find contours in camera coordinates
                        contours, _ = cv2.findContours(mask_cam, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue

                        # Pick the largest contour
                        c = max(contours, key=cv2.contourArea)

                        area_pixels = cv2.contourArea(c)
                        if area_pixels < 10:  # Threshold to filter noise
                            continue

                        # 4) Warp contour points to checkerboard space
                        # c is shape (N,1,2) -> reshape to (N,2)
                        c_reshaped = c.reshape(-1, 2).astype(np.float32)
                        c_warped = warp_points(c_reshaped, H)

                        # 5) Build corrected mask by filling polygon
                        corrected_mask = np.zeros((max_y, max_x), dtype=np.uint8)
                        c_warped_int = np.round(c_warped).astype(np.int32)
                        cv2.fillPoly(corrected_mask, [c_warped_int], 255)

                        # Create corrected object
                        corrected_object = cv2.bitwise_and(
                            corrected_frame, corrected_frame, mask=corrected_mask
                        )

                        # Calculate real-world center coordinates
                        M = cv2.moments(corrected_mask)
                        if M['m00'] != 0:
                            cx = M['m10'] / M['m00']
                            cy = M['m01'] / M['m00']
                            rw_coord = [cx * mm_per_pixel, cy * mm_per_pixel]
                        else:
                            rw_coord = [None, None]

                        # Store detection info
                        class_id = int(result.boxes.cls[idx].item())
                        detections_list.append({
                            'class_id': class_id,
                            'corrected_mask': (corrected_mask // 255).astype(np.uint8),
                            'corrected_object': corrected_object,
                            'corrected_polygon': c_warped_int.reshape(-1, 2).tolist(),
                            'real_world_coordinate': rw_coord
                        })

            if detections_list:
                # Pick first detection (or highest confidence if implemented)
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

    # ------------------- Morphometrics Analysis -------------------
    def run_morphometrics(self):
        """Run morphometric analysis on the current detection"""
        if self.current_measurement_folder is None:
            QMessageBox.warning(self, "Morphometrics Error", "No measurement data available.")
            return
        try:
            self.perform_morphometrics_analysis()
            QMessageBox.information(self, "Morphometrics", "Analysis completed and data saved.")
        except Exception as e:
            logging.exception("Error running morphometrics.")
            QMessageBox.critical(self, "Morphometrics Error", f"Error: {str(e)}")

    def perform_morphometrics_analysis(self):
        """Perform morphometric analysis on the detected object"""
        if self.current_measurement_folder is None:
            return
        try:
            # Load detection data from files
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

            # Extract mm per pixel ratio
            mm_per_pixel = detection_info.get('mm_per_pixel', None)
            if mm_per_pixel is None:
                QMessageBox.warning(self, "Morphometrics Error", "mm_per_pixel not found.")
                return

            # Find contours in mask
            contours, _ = cv2.findContours(corrected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                QMessageBox.warning(self, "Morphometrics Error", "No contours in mask.")
                return

            # Use largest contour
            contour = max(contours, key=cv2.contourArea)

            # Calculate basic morphometrics
            area_pixels = cv2.contourArea(contour)
            area_mm2 = area_pixels * (mm_per_pixel ** 2)

            # Find center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                QMessageBox.warning(self, "Morphometrics Error", "Cannot compute center.")
                return
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            center = np.array([cx, cy])

            # Get slider settings
            smoothing_factor = self.smoothing_slider.value()
            prominence_factor = self.prominence_slider.value() / 100.0
            distance_factor = self.distance_slider.value()

            # Convert contour format and smooth
            raw_points = contour.reshape(-1, 2)
            smoothed_points = smooth_closed_contour(raw_points, iterations=2)
            contour_points = smoothed_points

            if contour_points.ndim != 2:
                QMessageBox.warning(self, "Morphometrics Error", "Contour shape error.")
                return

            # Find arm tips using our analysis function
            arm_tips, angles_sorted, distances_smoothed, peaks, sorted_indices, shifted_contour = find_arm_tips(
                contour_points, center, smoothing_factor, prominence_factor, distance_factor
            )

            # Convert to RGB for display
            num_arms = len(arm_tips)
            corrected_object_rgb = cv2.cvtColor(corrected_object, cv2.COLOR_BGR2RGB)

            # Build arm_data for visualization
            self.arm_data = []
            for i, tip in enumerate(arm_tips):
                x_vec = tip[0] - center[0]
                y_vec = tip[1] - center[1]
                length_px = np.hypot(x_vec, y_vec)
                length_mm = length_px * mm_per_pixel
                self.arm_data.append([i + 1, x_vec, y_vec, length_mm])

            # Store contour data for interactive editing
            sorted_contour_global = shifted_contour[sorted_indices] + center
            self.sorted_contour_points = sorted_contour_global

            # Store profile data
            self.angles_sorted = angles_sorted
            self.distances_smoothed = distances_smoothed

            # Try to fit ellipse
            if len(contour) >= 5:
                try:
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

                    self.ellipse_data = (x0, y0, major_axis_length, minor_axis_length, angle)
                except Exception as e:
                    logging.warning(f"Failed to fit ellipse: {e}")
                    self.ellipse_data = None
                    major_axis_mm = None
                    minor_axis_mm = None
            else:
                self.ellipse_data = None
                major_axis_mm = None
                minor_axis_mm = None

            # Store morphometrics data
            self.morphometrics_data = {
                'area_mm2': area_mm2,
                'num_arms': num_arms,
                'arm_data': self.arm_data,
                'major_axis_mm': major_axis_mm,
                'minor_axis_mm': minor_axis_mm,
                'contour_coordinates': contour_points.tolist(),
                'mm_per_pixel': mm_per_pixel
            }

            # Configure arm rotation slider
            max_rotation = len(self.arm_data) - 1
            self.arm_rotation_slider.setRange(0, max_rotation if max_rotation > 0 else 0)
            self.arm_rotation_slider.setValue(0)

            # Store for visualization
            self.corrected_object_rgb = corrected_object_rgb
            self.center = center

            # Update visualizations
            self.update_arm_visualization()

            # Update polar plot
            angles_normalized = np.mod(angles_sorted, 2 * np.pi)
            self.polar_canvas.set_data(angles_normalized, distances_smoothed, np.array(peaks))

            # Enable UI controls
            self.smoothing_slider.setEnabled(True)
            self.prominence_slider.setEnabled(True)
            self.distance_slider.setEnabled(True)
            self.arm_rotation_slider.setEnabled(True)
            self.save_numbering_button.setEnabled(True)

        except Exception as e:
            logging.exception("Morphometrics Error.")
            QMessageBox.critical(self, "Morphometrics Error", f"Error: {str(e)}")

    # ------------------- Morphometrics Slider Handlers -------------------
    def on_smoothing_slider_change(self, value):
        """Handle changes to smoothing slider"""
        self.smoothing_label.setText(f"Smoothing Factor: {value}")
        self.perform_morphometrics_analysis()

    def on_prominence_slider_change(self, value):
        """Handle changes to prominence slider"""
        prominence = value / 100.0
        self.prominence_label.setText(f"Prominence Factor: {prominence:.2f}")
        self.perform_morphometrics_analysis()

    def on_distance_slider_change(self, value):
        """Handle changes to distance slider"""
        self.distance_label.setText(f"Distance Factor: {value}")
        self.perform_morphometrics_analysis()

    def rotate_arm_numbering(self):
        """Rotate arm numbering based on slider value"""
        self.arm_rotation_label.setText(f"Arm Rotation: {self.arm_rotation_slider.value()}")
        self.update_arm_visualization()

    # ------------------- Interactive Peak Editing -------------------
    def on_peaks_changed(self, new_peaks):
        """
        Handle changes to peaks from interactive polar plot.

        Args:
            new_peaks: Array of new peak indices
        """
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

    # ------------------- Visualization -------------------
    def update_arm_visualization(self):
        """Update arm visualization with current data"""
        if not hasattr(self, 'arm_data') or not self.arm_data:
            return

        self.ax.clear()
        self.ax.axis('off')

        if hasattr(self, 'corrected_object_rgb') and self.corrected_object_rgb is not None:
            self.ax.imshow(self.corrected_object_rgb)
        else:
            return

        # Use visualization module to create plot and get polar data
        rotation = self.arm_rotation_slider.value()
        polar_angles, polar_dists, polar_labels, polar_colors = create_morphometrics_visualization(
            self.ax,
            self.corrected_object_rgb,
            self.center,
            self.arm_data,
            rotation,
            self.ellipse_data if hasattr(self, 'ellipse_data') else None,
            self.morphometrics_data
        )

        # Tell polar plot to use these numbered labels
        self.polar_canvas.set_arm_labels(
            polar_angles,
            polar_dists,
            polar_labels,
            polar_colors
        )

        # Render to pixmap
        self.fig.canvas.draw()
        pixmap = render_figure_to_pixmap(self.fig, self.result_label)
        self.result_label.setPixmap(pixmap)

    # ------------------- Save Data -------------------
    def save_updated_morphometrics(self):
        """Save the updated morphometrics data to disk"""
        if not hasattr(self, 'arm_data') or not self.arm_data:
            QMessageBox.warning(self, "Save Error", "No arm data available to save.")
            return

        # Apply rotation to arm numbering
        rotation = self.arm_rotation_slider.value()
        num_arms = len(self.arm_data)
        reordered_arm_data = self.arm_data[rotation:] + self.arm_data[:rotation]

        # Renumber arms
        for i, arm in enumerate(reordered_arm_data):
            arm[0] = i + 1

        # Update morphometrics data
        self.morphometrics_data['arm_data'] = reordered_arm_data
        self.morphometrics_data['arm_rotation'] = rotation

        # Get user metadata
        user_initials = self.initials_input.text().strip()
        user_notes = self.notes_input.toPlainText().strip()

        # Validate user initials
        if not user_initials.isalpha() or len(user_initials) != 3:
            QMessageBox.warning(self, "Input Error", "Please enter exactly three letters for initials.")
            return

        # Add metadata
        self.morphometrics_data['user_initials'] = user_initials
        self.morphometrics_data['user_notes'] = user_notes

        # Remove any obsolete keys
        self.morphometrics_data.pop('arm_lengths_mm', None)

        # Save to disk
        morphometrics_json_path = os.path.join(self.current_measurement_folder, 'morphometrics.json')
        with open(morphometrics_json_path, 'w') as f:
            json.dump(self.morphometrics_data, f, indent=4)

        QMessageBox.information(self, "Save Successful",
                                f"Updated morphometrics data saved to {morphometrics_json_path}")

    def closeEvent(self, event):
        """Clean up resources when closing"""
        self.cap.release()
        logging.debug("Webcam released.")
        event.accept()

    def resizeEvent(self, event):
        """Handle widget resize events"""
        super().resizeEvent(event)
        # If we have data, re-draw the detection visualization
        if hasattr(self, 'angles_sorted') and self.angles_sorted is not None:
            # Update visualization to fit new size
            pass