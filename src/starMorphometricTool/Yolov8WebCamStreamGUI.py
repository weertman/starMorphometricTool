import sys
import cv2
import os
import numpy as np
import torch
import json
import logging
import datetime
import platform
from PySide6.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, QHBoxLayout,
    QFormLayout, QMessageBox, QFileDialog, QComboBox, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QSlider, QTextEdit, QSplitter, QSizePolicy, QScrollArea
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from ultralytics import YOLO
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from io import BytesIO

# Configure logging
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
# Empty the log file
open('debug_log.txt', 'w').close()

def convert_numpy_types(obj):
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

class WebcamStream(QWidget):
    def __init__(self):
        super().__init__()

        # Load YOLOv8 model
        path_model = os.path.join('..', '..', 'models', 'yolov8-n-pretrained.pt')
        self.yolo_model = YOLO(path_model)
        self.yolo_active = False  # Flag to control YOLOv8 streaming

        # Determine the appropriate OpenCV backend based on the OS
        current_os = platform.system()
        if current_os == "Windows":
            backend = cv2.CAP_DSHOW
        elif current_os == "Darwin":  # macOS is identified as 'Darwin'
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_ANY  # Fallback to any available backend

        # Set up webcam
        self.cap = cv2.VideoCapture(0, backend)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "Webcam Error", "Unable to access the webcam.")
            sys.exit()

        # Initialize checkerboard_info
        self.checkerboard_info = None
        self.corrected_checkerboard = None

        # Create UI components
        self.create_ui_components()

        # Set up a QTimer to refresh the video stream
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Initialize last frame and measurement folder
        self.last_frame = None
        self.current_measurement_folder = None

        # Initialize variables to store plot data
        self.angles_sorted = None
        self.distances_smoothed = None
        self.peaks = None

    def create_ui_components(self):
        # === Left Panel Components ===

        # Root Directory Selection
        self.root_dir_button = QPushButton("Select Root Data Directory", self)
        self.root_dir_label = QLabel(self)
        self.root_dir_label.setText(os.path.join('..', '..', 'measurements'))

        # Form layout for inputs
        form_layout = QFormLayout()
        self.rows_input = QLineEdit(self)
        self.cols_input = QLineEdit(self)
        self.square_size_input = QLineEdit(self)
        self.group_input = QLineEdit(self)
        self.id_type_combo = QComboBox(self)
        self.id_input = QLineEdit(self)

        # New User Initials and Notes Fields
        self.initials_input = QLineEdit(self)
        self.initials_input.setPlaceholderText("Enter your initials (e.g., ABC)")

        self.notes_input = QTextEdit(self)
        self.notes_input.setPlaceholderText("Enter any notes or comments here...")
        self.notes_input.setMaximumHeight(25)  # Limit height to prevent excessive space usage

        # Set default values for inputs
        self.rows_input.setText("5")
        self.cols_input.setText("7")
        self.square_size_input.setText("24.6")
        self.group_input.setText("lab")
        self.id_type_combo.addItems(["knownID", "unknownID"])
        self.id_input.setText("inputIDcode")

        # Add inputs to form layout
        form_layout.addRow("Checkerboard Rows (Squares):", self.rows_input)
        form_layout.addRow("Checkerboard Columns (Squares):", self.cols_input)
        form_layout.addRow("Square Size (mm):", self.square_size_input)
        form_layout.addRow("Group Name:", self.group_input)
        form_layout.addRow("ID Type:", self.id_type_combo)
        form_layout.addRow("ID Value:", self.id_input)
        form_layout.addRow("User Initials:", self.initials_input)  # New Field
        form_layout.addRow("User Notes:", self.notes_input)  # New Field

        # Buttons
        self.start_button = QPushButton("Start Stream", self)
        self.stop_button = QPushButton("Stop Stream", self)
        self.detect_button = QPushButton("Detect Checkerboard", self)
        self.clear_button = QPushButton("Clear Checkerboard", self)
        self.start_yolo_button = QPushButton("Start Detections", self)
        self.stop_yolo_button = QPushButton("Stop Detections", self)
        self.save_detection_button = QPushButton("Get Detection", self)
        self.run_morphometrics_button = QPushButton("Run Morphometrics", self)
        self.save_numbering_button = QPushButton("Save Morphometrics", self)

        # Initially disable buttons that shouldn't be active yet
        self.clear_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.stop_yolo_button.setEnabled(False)
        self.save_detection_button.setEnabled(False)
        self.run_morphometrics_button.setEnabled(False)
        self.save_numbering_button.setEnabled(False)

        # === Sliders ===

        # Smoothing Factor Slider
        self.smoothing_label = QLabel("Smoothing Factor: 5", self)
        self.smoothing_slider = QSlider(Qt.Horizontal, self)
        self.smoothing_slider.setRange(1, 15)
        self.smoothing_slider.setValue(5)
        self.smoothing_slider.setTickPosition(QSlider.TicksBelow)
        self.smoothing_slider.setTickInterval(1)
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_slider_change)

        # Prominence Factor Slider
        self.prominence_label = QLabel("Prominence Factor: 0.05", self)
        self.prominence_slider = QSlider(Qt.Horizontal, self)
        self.prominence_slider.setRange(1, 100)  # Represents 0.01 to 1.00
        self.prominence_slider.setValue(1)  # Default 0.05
        self.prominence_slider.setTickPosition(QSlider.TicksBelow)
        self.prominence_slider.setTickInterval(5)
        self.prominence_slider.valueChanged.connect(self.on_prominence_slider_change)

        # Distance Factor Slider
        self.distance_label = QLabel("Distance Factor: 5", self)
        self.distance_slider = QSlider(Qt.Horizontal, self)
        self.distance_slider.setRange(0, 15)
        self.distance_slider.setValue(5)
        self.distance_slider.setTickPosition(QSlider.TicksBelow)
        self.distance_slider.setTickInterval(1)
        self.distance_slider.valueChanged.connect(self.on_distance_slider_change)

        # New Arm Rotation Slider
        self.arm_rotation_label = QLabel("Arm Rotation: 0", self)
        self.arm_rotation_slider = QSlider(Qt.Horizontal, self)
        self.arm_rotation_slider.setRange(0, 24)  # Default range, will be updated later
        self.arm_rotation_slider.setValue(0)
        self.arm_rotation_slider.setTickPosition(QSlider.TicksBelow)
        self.arm_rotation_slider.setTickInterval(1)
        self.arm_rotation_slider.valueChanged.connect(self.rotate_arm_numbering)

        # Arrange sliders vertically in a separate container widget
        sliders_container = QWidget()
        sliders_container_layout = QVBoxLayout()
        sliders_container_layout.addWidget(self.smoothing_label)
        sliders_container_layout.addWidget(self.smoothing_slider)
        sliders_container_layout.addSpacing(10)  # Add space between sliders
        sliders_container_layout.addWidget(self.prominence_label)
        sliders_container_layout.addWidget(self.prominence_slider)
        sliders_container_layout.addSpacing(10)
        sliders_container_layout.addWidget(self.distance_label)
        sliders_container_layout.addWidget(self.distance_slider)
        sliders_container_layout.addSpacing(10)
        sliders_container_layout.addWidget(self.arm_rotation_label)
        sliders_container_layout.addWidget(self.arm_rotation_slider)
        sliders_container_layout.addStretch()  # Pushes all widgets to the top
        sliders_container.setLayout(sliders_container_layout)

        # Create a scroll area for the sliders
        sliders_scroll_area = QScrollArea()
        sliders_scroll_area.setWidgetResizable(True)  # Ensures the scroll area resizes with the window
        sliders_scroll_area.setWidget(sliders_container)
        sliders_scroll_area.setFixedHeight(300)  # Set a fixed height to constrain the scroll area
        # Alternatively, you can set size policies or leave it flexible based on design preference

        # === Buttons Layout ===
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

        # Connect buttons to actions
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

        # === Left Panel Layout ===
        left_panel_layout = QVBoxLayout()
        left_panel_layout.addWidget(self.root_dir_button)
        left_panel_layout.addWidget(self.root_dir_label)
        left_panel_layout.addLayout(form_layout)
        left_panel_layout.addLayout(button_layout)
        left_panel_layout.addLayout(yolo_button_layout)
        left_panel_layout.addWidget(sliders_scroll_area)  # Replace sliders_layout with scroll area
        left_panel_layout.addWidget(self.save_numbering_button)
        left_panel_layout.addStretch()  # Pushes all widgets to the top

        # === Right Panel Components ===

        # Video and Result Labels
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        # New QLabel for polar plot
        self.polar_plot_label = QLabel(self)
        self.polar_plot_label.setAlignment(Qt.AlignCenter)

        # Set size policies and minimum sizes to expand appropriately
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.polar_plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Set minimum sizes
        self.label.setMinimumSize(640, 480)  # Webcam Display
        self.result_label.setMinimumSize(300,300)  # Detection Results
        self.polar_plot_label.setMinimumSize(300,300)  # Polar Plot

        # Layout for video, result, and polar plot with stretch factors
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.label, stretch=3)  # Webcam Display
        video_layout.addWidget(self.result_label, stretch=1)  # Detection Results
        video_layout.addWidget(self.polar_plot_label, stretch=1)  # Polar Plot
        video_layout.addStretch(1)  # Pushes all widgets to the top

        # === Right Panel Layout ===
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(video_layout)

        # === Splitter to Divide Left and Right Panels ===
        splitter = QSplitter(Qt.Horizontal)
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel_layout)
        splitter.addWidget(left_panel_widget)
        splitter.addWidget(right_panel_widget)
        splitter.setStretchFactor(0, 1)  # Left Panel
        splitter.setStretchFactor(1, 4)  # Right Panel (increased stretch)

        # Optionally, set initial splitter sizes to allocate more space to the Right Panel
        splitter.setSizes([300, 900])  # Adjust based on total window size

        # === Main Layout ===
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Update ID input based on initial selection
        self.update_id_input()

        # Disable sliders and save numbering button initially
        self.smoothing_slider.setEnabled(False)
        self.prominence_slider.setEnabled(False)
        self.distance_slider.setEnabled(False)
        self.arm_rotation_slider.setEnabled(False)
        self.save_numbering_button.setEnabled(False)

        # Optional: Reduce Left Panel's maximum width to ensure it doesn't take too much space
        left_panel_widget.setMaximumWidth(500)  # Adjust as needed

    # === Slider Change Handlers ===

    def on_smoothing_slider_change(self, value):
        self.smoothing_label.setText(f"Smoothing Factor: {value}")
        self.perform_morphometrics_analysis()

    def on_prominence_slider_change(self, value):
        # Map slider value (1-100) to prominence factor (0.01 - 1.00)
        prominence = value / 100.0
        self.prominence_label.setText(f"Prominence Factor: {prominence:.2f}")
        self.perform_morphometrics_analysis()

    def on_distance_slider_change(self, value):
        self.distance_label.setText(f"Distance Factor: {value}")
        self.perform_morphometrics_analysis()

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
            existing_ids = [d for d in os.listdir(unknown_id_dir) if os.path.isdir(os.path.join(unknown_id_dir, d))]
            pattern = f"{group_name}_uID_"
            max_n = max([int(d.replace(pattern, "")) for d in existing_ids if d.startswith(pattern) and d.replace(pattern, "").isdigit()] + [0])
            default_id = f"{group_name}_uID_{max_n + 1}"
            if not self.id_input.text() or self.id_input.text() == "inputIDcode":
                self.id_input.setText(default_id)

    def select_root_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Root Data Directory",
                                                    os.path.join('..', '..', 'measurements'))
        if dir_path:
            self.root_dir_label.setText(dir_path)
            self.update_id_input()

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

    def clear_checkerboard(self):
        self.checkerboard_info = None
        self.clear_button.setEnabled(False)
        self.corrected_checkerboard = None
        logging.debug("Checkerboard info cleared.")

    def start_yolo(self):
        self.yolo_active = True
        self.start_yolo_button.setEnabled(False)
        self.stop_yolo_button.setEnabled(True)
        logging.debug("YOLOv8 detection started.")

    def stop_yolo(self):
        self.yolo_active = False
        self.start_yolo_button.setEnabled(True)
        self.stop_yolo_button.setEnabled(False)
        self.save_detection_button.setEnabled(False)
        logging.debug("YOLOv8 detection stopped.")

    def closeEvent(self, event):
        self.cap.release()
        logging.debug("Webcam released.")
        event.accept()

    def detect_checkerboard(self):
        try:
            rows = int(self.rows_input.text())
            cols = int(self.cols_input.text())
            square_size = float(self.square_size_input.text())

            logging.debug(f"Attempting checkerboard detection with dimensions: rows={rows}, cols={cols}, square_size={square_size}mm")

            if rows <= 1 or cols <= 1:
                raise ValueError("Checkerboard dimensions must be greater than 1")

            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                board_dims = (cols - 1, rows - 1)  # Number of inner corners

                ret, corners = cv2.findChessboardCorners(gray, board_dims, None)

                if ret:
                    # Store the unmasked corrected checkerboard image
                    self.corrected_checkerboard = frame.copy()

                    logging.debug(f"Checkerboard detected. Number of corners found: {len(corners)}")
                    # Refine corner locations
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    # Store checkerboard information
                    self.checkerboard_info = {
                        'dims': board_dims,
                        'corners': corners_refined,
                        'image_points': corners_refined.reshape(-1, 2),
                        'square_size': square_size
                    }

                    self.clear_button.setEnabled(True)
                    QMessageBox.information(self, "Detection Successful", "Checkerboard detected successfully!")
                else:
                    logging.warning("Checkerboard not detected.")
                    self.checkerboard_info = None
                    self.clear_button.setEnabled(False)
                    QMessageBox.warning(self, "Detection Failed", "Checkerboard not detected. Try again.")
            else:
                logging.error("Failed to capture frame from webcam during checkerboard detection.")
                QMessageBox.warning(self, "Webcam Error", "Failed to capture frame from webcam.")

        except ValueError as e:
            logging.error(f"Input Error: {str(e)}")
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            self.corrected_checkerboard = None
            logging.exception("An unexpected error occurred during checkerboard detection.")
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")

    def save_corrected_checkerboard(self, filename="corrected_checkerboard.png"):
        if self.corrected_checkerboard is not None:
            cv2.imwrite(filename, self.corrected_checkerboard)
            return True
        return False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Keep a copy of the original frame
            self.last_frame = frame.copy()

            # Draw checkerboard if info is available
            if self.checkerboard_info is not None:
                cv2.drawChessboardCorners(frame, self.checkerboard_info['dims'],
                                          self.checkerboard_info['corners'], True)

            # Process frame with YOLOv8 if active
            if self.yolo_active:
                results = self.yolo_model.predict(frame, verbose=False)
                # Draw YOLOv8 detections on the frame if any detections are present
                if results and len(results) > 0:
                    annotated_frame = results[0].plot()
                    frame = annotated_frame
                    # Enable save detection button
                    self.save_detection_button.setEnabled(True)
                else:
                    self.save_detection_button.setEnabled(False)
            else:
                self.save_detection_button.setEnabled(False)

            # Convert the frame to RGB format (required for PySide6 display)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to QImage format
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Scale the image to the size of the QLabel
            scaled_img = q_img.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)

            # Set the QImage on the QLabel
            self.label.setPixmap(QPixmap.fromImage(scaled_img))

    def save_corrected_detection(self):
        if self.yolo_active and self.checkerboard_info is not None:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()

                # Save the raw frame
                raw_frame = frame.copy()
                # Ensure measurement folder is determined before inference
                root_dir = self.root_dir_label.text()
                group_name = self.group_input.text()
                id_type = self.id_type_combo.currentText()
                id_value = self.id_input.text().strip()

                if not id_value:
                    QMessageBox.warning(self, "ID Error", "Please enter a valid ID value.")
                    return

                # Build the directory path
                id_folder = os.path.join(root_dir, group_name, id_type, id_value)

                # Create ID directory if it doesn't exist
                os.makedirs(id_folder, exist_ok=True)

                # Measurement date
                measurement_date = datetime.datetime.now().strftime("%m_%d_%Y")
                date_folder = os.path.join(id_folder, measurement_date)

                # Create date directory if it doesn't exist
                os.makedirs(date_folder, exist_ok=True)

                # Measurement folder
                # Find existing measurement folders
                existing_mfolders = [d for d in os.listdir(date_folder) if
                                     os.path.isdir(os.path.join(date_folder, d)) and d.startswith("mFolder_")]
                m_numbers = [int(d.replace("mFolder_", "")) for d in existing_mfolders if
                             d.replace("mFolder_", "").isdigit()]
                m_next = max(m_numbers) + 1 if m_numbers else 1
                measurement_folder = os.path.join(date_folder, f"mFolder_{m_next}")

                # Create measurement folder
                os.makedirs(measurement_folder, exist_ok=True)

                # Save the raw frame
                raw_frame_path = os.path.join(measurement_folder, 'raw_frame.png')
                cv2.imwrite(raw_frame_path, raw_frame)  # Save the raw frame

                results = self.yolo_model.predict(frame, verbose=False)
                # Apply checkerboard correction
                corrected_detection = self.correct_detections(results)
                # Save corrected detection
                if corrected_detection:
                    # Save corrected mask and object images
                    corrected_mask = corrected_detection['corrected_mask']
                    corrected_object = corrected_detection['corrected_object']
                    corrected_frame = corrected_detection['corrected_frame']  # Retrieve the flattened checkerboard

                    mask_path = os.path.join(measurement_folder, 'corrected_mask.png')
                    object_path = os.path.join(measurement_folder, 'corrected_object.png')
                    json_path = os.path.join(measurement_folder, 'corrected_detection.json')

                    # Save images
                    cv2.imwrite(mask_path, corrected_mask * 255)  # Multiply by 255 to save as image
                    cv2.imwrite(object_path, corrected_object)
                    logging.debug(f"Saved corrected mask to {mask_path}")
                    logging.debug(f"Saved corrected object to {object_path}")

                    # Combine the flattened checkerboard with the detected object
                    try:
                        # Ensure both images are the same size
                        if corrected_frame.shape != corrected_object.shape:
                            # Resize corrected_object to match corrected_frame
                            corrected_object_resized = cv2.resize(corrected_object,
                                                                  (corrected_frame.shape[1], corrected_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                            corrected_mask_resized = cv2.resize(corrected_mask,
                                                                (corrected_frame.shape[1], corrected_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                            logging.debug("Resized corrected object and mask to match the flattened checkerboard.")
                        else:
                            corrected_object_resized = corrected_object
                            corrected_mask_resized = corrected_mask

                        # Create an alpha mask from the corrected mask
                        alpha_mask = corrected_mask_resized.astype(float) / 255.0
                        alpha_mask = np.stack([alpha_mask] * 3, axis=2)  # Make it 3-channel

                        # Convert images to float for blending
                        corrected_frame_float = corrected_frame.astype(float)
                        corrected_object_rgb = cv2.cvtColor(corrected_object_resized, cv2.COLOR_BGR2RGB).astype(float)

                        # Overlay the object onto the flattened checkerboard
                        combined_image = (1.0 - alpha_mask) * corrected_frame_float + alpha_mask * corrected_object_rgb

                        # Convert back to uint8
                        combined_image = combined_image.astype(np.uint8)

                        # Save the combined image
                        combined_image_path = os.path.join(measurement_folder, 'checkerboard_with_object.png')
                        cv2.imwrite(combined_image_path, combined_image)
                        logging.debug(f"Saved combined image to {combined_image_path}")
                    except Exception as e:
                        logging.exception("Failed to combine images.")
                        QMessageBox.warning(self, "Combine Error", f"Failed to combine images: {str(e)}")

                    # Update the detection_info dictionary with new paths
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
                        'combined_image_path': combined_image_path  # Add path to combined image
                    }

                    # Convert detection_info to native Python types
                    detection_info_converted = convert_numpy_types(detection_info)

                    # Save detection data to a file (e.g., JSON)
                    with open(json_path, 'w') as f:
                        json.dump(detection_info_converted, f, indent=4)
                    logging.debug(f"Saved detection data to {json_path}")
                    QMessageBox.information(self, "Save Successful", f"Detection saved to {json_path}")

                    # Store the measurement folder path for later use
                    self.current_measurement_folder = measurement_folder

                    # Enable the Run Morphometrics button
                    self.run_morphometrics_button.setEnabled(True)

                    # Display corrected object in result_label
                    if corrected_object.size != 0:
                        corrected_object_rgb_display = cv2.cvtColor(corrected_object, cv2.COLOR_BGR2RGB)
                        h, w, ch = corrected_object_rgb_display.shape
                        bytes_per_line = ch * w
                        q_img = QImage(corrected_object_rgb_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        scaled_img = q_img.scaled(self.result_label.width(), self.result_label.height(),
                                                  Qt.KeepAspectRatio)
                        self.result_label.setPixmap(QPixmap.fromImage(scaled_img))
                    else:
                        self.result_label.clear()
                        logging.warning("Corrected object is empty.")
                else:
                    logging.warning("No corrected detection to save.")
                    QMessageBox.warning(self, "Save Error", "No detections to save or corrected mask is empty.")
            else:
                logging.warning("YOLOv8 detection is not active or checkerboard not detected.")
                QMessageBox.warning(self, "Save Error", "YOLOv8 detection is not active or checkerboard not detected.")

    def find_arm_tips(self, contour, center, smoothing_factor, prominence_factor, distance_factor):
        """
        Improved Arm Tip Finding Algorithm with adjustable parameters and edge correction.

        Args:
            contour (np.array): Contour points of the sea star.
            center (np.array): Center coordinates of the sea star.
            smoothing_factor (int): The size parameter for the uniform filter.
            prominence_factor (float): The prominence factor for peak detection.
            distance_factor (int): The minimum distance between peaks.

        Returns:
            tuple: arm_tips (np.array), angles_sorted (np.array), distances_smoothed (np.array), peaks (np.array)
        """
        # Shift contour to have the center at the origin
        shifted_contour = contour - center

        # Convert to polar coordinates
        angles = np.arctan2(shifted_contour[:, 1], shifted_contour[:, 0])
        distances = np.hypot(shifted_contour[:, 0], shifted_contour[:, 1])

        # Sort by angle
        sorted_indices = np.argsort(angles)
        angles_sorted = angles[sorted_indices]
        distances_sorted = distances[sorted_indices]

        # Smooth distances to reduce noise
        distances_smoothed = uniform_filter1d(distances_sorted, size=smoothing_factor)

        # Function to find peaks on a given array
        def find_peaks_on_array(arr):
            return find_peaks(arr, prominence=prominence_factor * arr.max(), distance=distance_factor)[0]

        # Find peaks on the original array
        peaks1 = find_peaks_on_array(distances_smoothed)

        # Roll the array by 15 degrees (which is pi/12 radians)
        roll_amount = int(len(distances_smoothed) * (np.pi / 12) / (2 * np.pi))
        distances_rolled = np.roll(distances_smoothed, roll_amount)

        # Find peaks on the rolled array
        peaks2 = (find_peaks_on_array(distances_rolled) - roll_amount) % len(distances_smoothed)

        # Combine peaks from both arrays
        all_peaks = np.unique(np.concatenate([peaks1, peaks2]))

        # Sort peaks by their distance value to keep only the most prominent ones
        sorted_peaks = sorted(all_peaks, key=lambda x: distances_smoothed[x], reverse=True)

        # Keep only a reasonable number of peaks (e.g., maximum 7)
        peaks = sorted_peaks[:24]

        # Extract arm tips
        arm_tips = shifted_contour[sorted_indices][peaks] + center

        # Calculate angles for each arm tip
        arm_angles = angles_sorted[peaks]

        # Sort arm tips and angles together based on the angles
        sorted_arms = sorted(zip(arm_tips, arm_angles), key=lambda x: x[1])
        sorted_arm_tips, sorted_arm_angles = zip(*sorted_arms)

        return np.array(sorted_arm_tips), angles_sorted, distances_smoothed, peaks

    def perform_morphometrics_analysis(self):
        """
        Perform morphometric analysis based on the current measurement folder.
        Updates the visualization and polar plot.
        """
        if self.current_measurement_folder is None:
            return  # No data to process

        try:
            # Paths to the data
            json_path = os.path.join(self.current_measurement_folder, 'corrected_detection.json')
            mask_path = os.path.join(self.current_measurement_folder, 'corrected_mask.png')
            object_path = os.path.join(self.current_measurement_folder, 'corrected_object.png')

            # Load JSON data
            with open(json_path, 'r') as f:
                detection_info = json.load(f)
            logging.debug(f"Loaded detection info from {json_path}")

            # Load images
            corrected_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            corrected_object = cv2.imread(object_path)
            corrected_object_rgb = cv2.cvtColor(corrected_object, cv2.COLOR_BGR2RGB)

            if corrected_mask is None or corrected_object is None:
                logging.error("Failed to load images for morphometrics.")
                QMessageBox.warning(self, "Morphometrics Error", "Failed to load images for morphometrics.")
                return

            # Get mm_per_pixel scaling factor
            mm_per_pixel = detection_info.get('mm_per_pixel', None)
            if mm_per_pixel is None:
                logging.warning("mm per pixel value not found in detection info.")
                QMessageBox.warning(self, "Morphometrics Error", "mm per pixel value not found.")
                return

            # Find contours of the mask
            contours, _ = cv2.findContours(corrected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                logging.warning("No contours found in the corrected mask.")
                QMessageBox.warning(self, "Morphometrics Error", "No contours found in the corrected mask.")
                return

            # Assume the largest contour is the sea star
            contour = max(contours, key=cv2.contourArea)

            # Calculate area
            area_pixels = cv2.contourArea(contour)
            area_mm2 = area_pixels * (mm_per_pixel ** 2)

            # Find center of the sea star
            M = cv2.moments(contour)
            if M['m00'] == 0:
                logging.warning("Zero division error in moments calculation.")
                QMessageBox.warning(self, "Morphometrics Error", "Cannot compute center of the sea star.")
                return
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            center = np.array([cx, cy])

            # Get parameters from sliders
            smoothing_factor = self.smoothing_slider.value()
            prominence_factor = self.prominence_slider.value() / 100.0
            distance_factor = self.distance_slider.value()

            # Extract contour points
            contour_points = contour.squeeze()
            if contour_points.ndim != 2:
                logging.warning("Contour points have unexpected shape.")
                QMessageBox.warning(self, "Morphometrics Error", "Contour points have unexpected shape.")
                return

            # Find arm tips using the improved method
            arm_tips, angles_sorted, distances_smoothed, peaks = self.find_arm_tips(
                contour_points, center, smoothing_factor, prominence_factor, distance_factor)
            num_arms = len(arm_tips)

            # Measure arm lengths and create detailed arm data
            self.arm_data = []
            for i, tip in enumerate(arm_tips):
                x_vector = tip[0] - center[0]
                y_vector = tip[1] - center[1]
                length_pixels = np.hypot(x_vector, y_vector)
                length_mm = length_pixels * mm_per_pixel
                self.arm_data.append([i + 1, x_vector, y_vector, length_mm])

            # Store contour coordinates
            self.contour_coordinates = contour_points.tolist()

            # Measure major and minor body axes using ellipse fitting
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (x0, y0), (axis_length1, axis_length2), angle = ellipse
                # Ensure major_axis_length >= minor_axis_length
                if axis_length1 >= axis_length2:
                    major_axis_length = axis_length1
                    minor_axis_length = axis_length2
                else:
                    major_axis_length = axis_length2
                    minor_axis_length = axis_length1
                    angle += 90  # Adjust angle if axes are swapped
                major_axis_mm = major_axis_length * mm_per_pixel
                minor_axis_mm = minor_axis_length * mm_per_pixel
            else:
                major_axis_mm = minor_axis_mm = None
                x0 = y0 = major_axis_length = minor_axis_length = angle = None

            # Set up the rotation slider
            max_rotation = len(self.arm_data) - 1
            self.arm_rotation_slider.setRange(0, max_rotation)
            self.arm_rotation_slider.setValue(0)

            # Store necessary data for visualization updates
            self.corrected_object_rgb = corrected_object_rgb
            self.center = center
            self.ellipse_data = (x0, y0, major_axis_length, minor_axis_length, angle) if major_axis_mm else None

            # Initialize matplotlib figure and axis if not already done
            if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
                self.fig, self.ax = plt.subplots()
                self.fig.set_size_inches(8, 8)  # Adjust size as needed

            # Prepare morphometrics data
            self.morphometrics_data = {
                'area_mm2': area_mm2,
                'num_arms': num_arms,
                'arm_data': self.arm_data,
                'major_axis_mm': major_axis_mm,
                'minor_axis_mm': minor_axis_mm,
                'contour_coordinates': self.contour_coordinates
            }

            # Initial visualization
            self.update_arm_visualization()

            # Create and update polar plot
            self.update_polar_plot(angles_sorted, distances_smoothed, peaks)

            # Enable sliders and save button
            self.smoothing_slider.setEnabled(True)
            self.prominence_slider.setEnabled(True)
            self.distance_slider.setEnabled(True)
            self.arm_rotation_slider.setEnabled(True)
            self.save_numbering_button.setEnabled(True)

        except Exception as e:
            logging.exception("An error occurred during morphometrics.")
            QMessageBox.critical(self, "Morphometrics Error", f"An error occurred: {str(e)}")

    def update_arm_visualization(self):
        if not hasattr(self, 'arm_data') or not self.arm_data:
            return

        self.ax.clear()
        self.ax.imshow(self.corrected_object_rgb)
        self.ax.axis('off')

        rotation = self.arm_rotation_slider.value()
        num_arms = len(self.arm_data)

        # Calculate new arm numbering order
        new_order = [(i - rotation) % num_arms + 1 for i in range(num_arms)]

        # Draw center
        self.ax.plot(self.center[0], self.center[1], 'yo', markersize=10)

        # Draw arm tips and lines to center
        for i, arm_info in enumerate(self.arm_data):
            arm_number, x_vector, y_vector, length = arm_info
            tip_x = self.center[0] + x_vector
            tip_y = self.center[1] + y_vector

            new_number = new_order[i]
            color = 'red' if new_number == 1 else 'blue'

            self.ax.plot([self.center[0], tip_x], [self.center[1], tip_y], color=color, linewidth=2)
            self.ax.plot(tip_x, tip_y, 'bo', markersize=5)

            # Add arm number
            text_position = ((tip_x + self.center[0]) / 2, (tip_y + self.center[1]) / 2)
            self.ax.text(text_position[0], text_position[1], str(new_number),
                         color='white', fontweight='bold', ha='center', va='center',
                         bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))

        # Draw major and minor axes if available
        if self.ellipse_data:
            x0, y0, major_axis_length, minor_axis_length, angle = self.ellipse_data
            ellipse_patch = Ellipse((x0, y0), major_axis_length, minor_axis_length, angle=angle,
                                    edgecolor='yellow', facecolor='none', linewidth=2)
            self.ax.add_patch(ellipse_patch)

        # Display measurements as text
        measurements_text = (
            f'Area: {self.morphometrics_data["area_mm2"]:.2f} mm²\n'
            f'Number of Arms: {self.morphometrics_data["num_arms"]}\n'
            f'Major Axis: {self.morphometrics_data["major_axis_mm"]:.2f} mm\n'
            f'Minor Axis: {self.morphometrics_data["minor_axis_mm"]:.2f} mm'
        )
        props = dict(boxstyle='round', facecolor='black', alpha=0.5)
        self.ax.text(0.05, 0.95, measurements_text, transform=self.ax.transAxes, fontsize=12,
                     verticalalignment='top', bbox=props, color='white')

        # Update title with rotation information
        self.ax.set_title(f"Arm Numbering (Arm 1 Position: {rotation + 1})")

        self.fig.canvas.draw()

        # Convert the Matplotlib figure to a QPixmap and display it
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        qimg = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(qimg)
        self.result_label.setPixmap(
            pixmap.scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_polar_plot(self, angles_sorted, distances_smoothed, peaks):
        fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})

        # Normalize angles to [0, 2pi]
        angles_sorted_normalized = np.mod(angles_sorted, 2 * np.pi)

        # Plot the smoothed distances
        ax_polar.plot(angles_sorted_normalized, distances_smoothed, label='Distance Profile')

        # Mark the peaks
        ax_polar.plot(angles_sorted_normalized[peaks], distances_smoothed[peaks], 'ro', label='Arm Tips')

        ax_polar.set_title('Polar Plot of Sea Star Contour')
        ax_polar.legend()

        # Convert polar plot to QImage
        buf_polar = BytesIO()
        fig_polar.savefig(buf_polar, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig_polar)
        buf_polar.seek(0)
        qimg_polar = QImage()
        qimg_polar.loadFromData(buf_polar.getvalue(), 'PNG')

        # Display in polar_plot_label
        scaled_img_polar = qimg_polar.scaled(self.polar_plot_label.width(), self.polar_plot_label.height(),
                                             Qt.KeepAspectRatio)
        self.polar_plot_label.setPixmap(QPixmap.fromImage(scaled_img_polar))

        # Store data for resizing
        self.angles_sorted = angles_sorted
        self.distances_smoothed = distances_smoothed
        self.peaks = peaks

    def rotate_arm_numbering(self):
        self.update_arm_visualization()

    def save_updated_morphometrics(self):
        if not hasattr(self, 'arm_data') or not self.arm_data:
            QMessageBox.warning(self, "Save Error", "No arm data available to save.")
            return

        rotation = self.arm_rotation_slider.value()
        num_arms = len(self.arm_data)

        # Reorder arm data based on rotation
        reordered_arm_data = self.arm_data[rotation:] + self.arm_data[:rotation]

        # Update arm numbers after rotation
        for i, arm in enumerate(reordered_arm_data):
            arm[0] = i + 1

        self.morphometrics_data['arm_data'] = reordered_arm_data
        self.morphometrics_data['arm_rotation'] = rotation

        # Capture User Initials and Notes
        user_initials = self.initials_input.text().strip()
        user_notes = self.notes_input.toPlainText().strip()

        # Validate User Initials (Optional)
        if not user_initials.isalpha() or len(user_initials) != 3:
            QMessageBox.warning(self, "Input Error",
                                "Please enter exactly three alphabetic characters for User Initials.")
            return

        # Add to morphometrics_data
        self.morphometrics_data['user_initials'] = user_initials
        self.morphometrics_data['user_notes'] = user_notes

        # Remove the redundant 'arm_lengths_mm' if it exists
        self.morphometrics_data.pop('arm_lengths_mm', None)

        # Save updated morphometrics data to JSON
        morphometrics_json_path = os.path.join(self.current_measurement_folder, 'morphometrics.json')
        with open(morphometrics_json_path, 'w') as f:
            json.dump(self.morphometrics_data, f, indent=4)

        logging.debug(f"Saved updated morphometrics data to {morphometrics_json_path}")
        QMessageBox.information(self, "Save Successful",
                                f"Updated morphometrics data saved to {morphometrics_json_path}")

    def run_morphometrics(self):
        if self.current_measurement_folder is None:
            QMessageBox.warning(self, "Morphometrics Error", "No measurement data available.")
            return

        try:
            self.perform_morphometrics_analysis()
            QMessageBox.information(self, "Morphometrics", "Morphometric analysis completed and data saved.")
        except Exception as e:
            logging.exception("An error occurred during morphometrics.")
            QMessageBox.critical(self, "Morphometrics Error", f"An error occurred: {str(e)}")

    def correct_detections(self, results):
        if self.checkerboard_info is None:
            logging.warning("Checkerboard not detected. Cannot apply corrections.")
            QMessageBox.warning(self, "Correction Error", "Checkerboard not detected. Cannot apply corrections.")
            return None

        try:
            square_size = self.checkerboard_info['square_size']
            img_pts = self.checkerboard_info['image_points'].reshape(-1, 2)
            board_dims = self.checkerboard_info['dims']
            obj_pts = np.zeros((board_dims[0] * board_dims[1], 2), np.float32)
            obj_pts[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
            obj_pts *= square_size

            h_matrix, status = cv2.findHomography(img_pts, obj_pts)
            if h_matrix is None:
                logging.error("Failed to compute homography matrix.")
                QMessageBox.warning(self, "Homography Error", "Failed to compute homography matrix.")
                return None

            max_x = int(obj_pts[:, 0].max()) + 10
            max_y = int(obj_pts[:, 1].max()) + 10
            corrected_frame = cv2.warpPerspective(self.last_frame, h_matrix, (max_x, max_y))
            logging.debug(f"Corrected frame shape: {corrected_frame.shape}")

            obj_pt1, obj_pt2 = obj_pts[0], obj_pts[1]
            real_distance_mm = np.linalg.norm(obj_pt1 - obj_pt2)
            img_pt1 = cv2.perspectiveTransform(np.array([[img_pts[0]]], dtype='float32'), h_matrix)[0][0]
            img_pt2 = cv2.perspectiveTransform(np.array([[img_pts[1]]], dtype='float32'), h_matrix)[0][0]
            pixel_distance = np.linalg.norm(img_pt1 - img_pt2)
            mm_per_pixel = real_distance_mm / pixel_distance
            logging.debug(f"Calculated mm per pixel: {mm_per_pixel}")

            detections_list = []

            for result in results:
                if result.masks is not None and result.masks.data is not None:
                    for idx, mask in enumerate(result.masks.data):
                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        h_img, w_img = self.last_frame.shape[:2]
                        mask_resized = cv2.resize(mask_np, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                        corrected_mask = cv2.warpPerspective(mask_resized, h_matrix, (max_x, max_y))
                        non_zero_pixels = np.count_nonzero(corrected_mask)
                        if non_zero_pixels == 0:
                            continue
                        moments = cv2.moments(corrected_mask)
                        if moments['m00'] != 0:
                            cx = moments['m10'] / moments['m00']
                            cy = moments['m01'] / moments['m00']
                            real_world_coordinate = [cx * mm_per_pixel, cy * mm_per_pixel]
                        else:
                            real_world_coordinate = [None, None]
                        contours, _ = cv2.findContours(corrected_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        corrected_polygon = [cnt.reshape(-1, 2).tolist() for cnt in contours]
                        corrected_object = cv2.bitwise_and(corrected_frame, corrected_frame,
                                                           mask=corrected_mask.astype(np.uint8))

                        detections_list.append({
                            'result': result,
                            'corrected_mask': corrected_mask,
                            'corrected_object': corrected_object,
                            'class_id': int(result.boxes.cls[idx].item()),
                            'corrected_polygon': corrected_polygon,
                            'real_world_coordinate': real_world_coordinate,
                        })

            if detections_list:
                detection = detections_list[0]
                detection['mm_per_pixel'] = mm_per_pixel
                detection['homography_matrix'] = h_matrix.tolist()
                detection['corrected_frame'] = corrected_frame  # Add the corrected frame to detection
                return detection
            else:
                logging.warning("No detections found to correct.")
                return None

        except Exception as e:
            logging.exception("An error occurred during detection correction.")
            QMessageBox.critical(self, "Correction Error", f"An error occurred during correction: {str(e)}")
            return None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Check if morphometrics data exists
        if hasattr(self, 'angles_sorted') and self.angles_sorted is not None:
            # Redraw plots with current label sizes
            self.update_arm_visualization()
            self.update_polar_plot(self.angles_sorted, self.distances_smoothed, self.peaks)

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create the main window and show it
    window = WebcamStream()
    window.setWindowTitle("Webcam Stream with Checkerboard and YOLOv8 Detection in PySide6")
    window.setGeometry(100, 100, 1200, 800)
    window.show()

    sys.exit(app.exec())
