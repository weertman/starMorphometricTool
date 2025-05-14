from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QScrollArea, QSlider, QSplitter,
    QSizePolicy, QLineEdit, QFormLayout, QTextEdit, QSpacerItem
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

import os
import json
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.image_processing import smooth_closed_contour
from utils.data_utils import convert_numpy_types
from morphometrics.analysis import find_arm_tips
from morphometrics.visualization import create_morphometrics_visualization, render_figure_to_pixmap
from ui.components.polar_canvas import PolarCanvas


class RerunMorphometricsTab(QWidget):
    """
    A tab for re-running morphometrics on existing detections.
    Allows users to load any mFolder, visualize the detection,
    adjust morphometrics parameters, and save updated results.
    """

    def __init__(self):
        super().__init__()

        # Initialize state variables
        self.current_folder = None
        self.detection_info = None
        self.corrected_mask = None
        self.corrected_object = None
        self.corrected_object_rgb = None
        self.arm_data = None
        self.sorted_contour_points = None
        self.angles_sorted = None
        self.distances_smoothed = None
        self.center = None
        self.ellipse_data = None
        self.morphometrics_data = None

        # Create matplotlib figure for visualization
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 8)

        # Build UI components
        self.create_ui_components()

    def create_ui_components(self):
        """Create and arrange UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)

        # -------------------- Top Controls --------------------
        top_layout = QHBoxLayout()

        # Folder selection
        self.select_folder_button = QPushButton("Select mFolder")
        self.select_folder_button.clicked.connect(self.select_mfolder)
        top_layout.addWidget(self.select_folder_button)

        # Display current folder path
        self.folder_label = QLabel("No folder selected")
        top_layout.addWidget(self.folder_label, stretch=1)

        # Run morphometrics button
        self.run_button = QPushButton("Run Morphometrics")
        self.run_button.clicked.connect(self.run_morphometrics)
        self.run_button.setEnabled(False)
        top_layout.addWidget(self.run_button)

        main_layout.addLayout(top_layout)

        # -------------------- Sliders and Metadata --------------------
        middle_layout = QHBoxLayout()

        # Left side: Sliders for morphometrics parameters
        sliders_container = QWidget()
        sliders_layout = QVBoxLayout(sliders_container)

        # Smoothing slider
        self.smoothing_label = QLabel("Smoothing Factor: 5")
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(1, 15)
        self.smoothing_slider.setValue(5)
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_slider_change)
        sliders_layout.addWidget(self.smoothing_label)
        sliders_layout.addWidget(self.smoothing_slider)
        sliders_layout.addSpacing(10)

        # Prominence slider
        self.prominence_label = QLabel("Prominence Factor: 0.05")
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 100)
        self.prominence_slider.setValue(5)
        self.prominence_slider.valueChanged.connect(self.on_prominence_slider_change)
        sliders_layout.addWidget(self.prominence_label)
        sliders_layout.addWidget(self.prominence_slider)
        sliders_layout.addSpacing(10)

        # Distance slider
        self.distance_label = QLabel("Distance Factor: 5")
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(0, 15)
        self.distance_slider.setValue(5)
        self.distance_slider.valueChanged.connect(self.on_distance_slider_change)
        sliders_layout.addWidget(self.distance_label)
        sliders_layout.addWidget(self.distance_slider)
        sliders_layout.addSpacing(10)

        # Arm rotation slider
        self.arm_rotation_label = QLabel("Arm Rotation: 0")
        self.arm_rotation_slider = QSlider(Qt.Horizontal)
        self.arm_rotation_slider.setRange(0, 24)
        self.arm_rotation_slider.setValue(0)
        self.arm_rotation_slider.valueChanged.connect(self.rotate_arm_numbering)
        sliders_layout.addWidget(self.arm_rotation_label)
        sliders_layout.addWidget(self.arm_rotation_slider)

        # Wrap sliders in a scroll area
        sliders_scroll_area = QScrollArea()
        sliders_scroll_area.setWidgetResizable(True)
        sliders_scroll_area.setWidget(sliders_container)
        sliders_scroll_area.setFixedWidth(300)
        middle_layout.addWidget(sliders_scroll_area)

        # Right side: Metadata input form
        form_container = QWidget()
        form_layout = QFormLayout(form_container)

        self.initials_input = QLineEdit()
        self.initials_input.setPlaceholderText("Enter your initials (e.g., ABC)")
        form_layout.addRow("User Initials:", self.initials_input)

        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Enter any notes or comments here...")
        self.notes_input.setMaximumHeight(60)
        form_layout.addRow("User Notes:", self.notes_input)

        form_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.save_button = QPushButton("Save Morphometrics")
        self.save_button.clicked.connect(self.save_morphometrics)
        self.save_button.setEnabled(False)
        form_layout.addRow(self.save_button)

        middle_layout.addWidget(form_container)
        main_layout.addLayout(middle_layout)

        # -------------------- Visualization --------------------
        # Main splitter with detection plot and polar canvas
        main_splitter = QSplitter(Qt.Horizontal)

        # Left: Detection visualization
        self.result_label = QLabel("Detection Plot")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_label.setMinimumSize(400, 400)
        main_splitter.addWidget(self.result_label)

        # Right: Polar canvas
        self.polar_canvas = PolarCanvas()
        self.polar_canvas.peaksChanged.connect(self.on_peaks_changed)
        main_splitter.addWidget(self.polar_canvas)

        # Set initial sizes
        main_splitter.setSizes([500, 500])

        main_layout.addWidget(main_splitter, stretch=1)

        # Disable controls until a folder is loaded
        self.disable_controls()

    def select_mfolder(self):
        """Open file dialog to select an mFolder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select mFolder", os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if folder_path:
            self.current_folder = folder_path
            self.folder_label.setText(f"Selected: {os.path.basename(folder_path)}")

            # Check if this is a valid mFolder with required files
            if self.validate_mfolder(folder_path):
                self.run_button.setEnabled(True)

                # Try to load existing morphometrics if available
                morphometrics_path = os.path.join(folder_path, 'morphometrics.json')
                if os.path.exists(morphometrics_path):
                    try:
                        with open(morphometrics_path, 'r') as f:
                            self.morphometrics_data = json.load(f)
                        self.load_existing_morphometrics()
                    except Exception as e:
                        logging.error(f"Error loading existing morphometrics: {e}")
            else:
                self.run_button.setEnabled(False)
                QMessageBox.warning(
                    self, "Invalid Folder",
                    "The selected folder does not appear to be a valid mFolder with detection data."
                )

    def validate_mfolder(self, folder_path):
        """
        Check if the folder contains required detection files.

        Args:
            folder_path: Path to the folder to validate

        Returns:
            bool: True if folder contains required files, False otherwise
        """
        required_files = [
            'corrected_detection.json',
            'corrected_mask.png',
            'corrected_object.png'
        ]

        for file in required_files:
            if not os.path.exists(os.path.join(folder_path, file)):
                return False

        return True

    def load_existing_morphometrics(self):
        """Load and apply existing morphometrics data"""
        if not self.morphometrics_data:
            return

        # Load user data if available
        if 'user_initials' in self.morphometrics_data:
            self.initials_input.setText(self.morphometrics_data['user_initials'])

        if 'user_notes' in self.morphometrics_data:
            self.notes_input.setText(self.morphometrics_data['user_notes'])

        # Set sliders to saved values
        if 'arm_rotation' in self.morphometrics_data:
            rotation = self.morphometrics_data['arm_rotation']
            self.arm_rotation_slider.setValue(rotation)

        # Run morphometrics to load the rest of the data
        self.run_morphometrics()

    def run_morphometrics(self):
        """Load detection data and run morphometric analysis"""
        if not self.current_folder:
            return

        try:
            # Load detection data
            json_path = os.path.join(self.current_folder, 'corrected_detection.json')
            mask_path = os.path.join(self.current_folder, 'corrected_mask.png')
            object_path = os.path.join(self.current_folder, 'corrected_object.png')

            with open(json_path, 'r') as f:
                self.detection_info = json.load(f)

            self.corrected_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.corrected_object = cv2.imread(object_path)

            if self.corrected_mask is None or self.corrected_object is None:
                QMessageBox.warning(self, "Load Error", "Failed to load images.")
                return

            # Get mm_per_pixel from detection info
            mm_per_pixel = self.detection_info.get('mm_per_pixel', None)
            if mm_per_pixel is None:
                QMessageBox.warning(self, "Data Error", "mm_per_pixel not found in detection data.")
                return

            # Find contours in mask
            contours, _ = cv2.findContours(self.corrected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                QMessageBox.warning(self, "Contour Error", "No contours found in mask.")
                return

            # Use largest contour
            contour = max(contours, key=cv2.contourArea)
            area_pixels = cv2.contourArea(contour)
            area_mm2 = area_pixels * (mm_per_pixel ** 2)

            # Calculate center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                QMessageBox.warning(self, "Center Error", "Cannot compute center (zero moment).")
                return

            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            self.center = np.array([cx, cy])

            # Get morphometrics parameters from sliders
            smoothing_factor = self.smoothing_slider.value()
            prominence_factor = self.prominence_slider.value() / 100.0
            distance_factor = self.distance_slider.value()

            # Convert and smooth contour
            raw_points = contour.reshape(-1, 2)
            smoothed_points = smooth_closed_contour(raw_points, iterations=2)
            contour_points = smoothed_points

            if contour_points.ndim != 2:
                QMessageBox.warning(self, "Contour Error", "Invalid contour shape.")
                return

            # Find arm tips
            arm_tips, angles_sorted, distances_smoothed, peaks, sorted_indices, shifted_contour = find_arm_tips(
                contour_points, self.center, smoothing_factor, prominence_factor, distance_factor
            )

            # Convert image to RGB for display
            num_arms = len(arm_tips)
            self.corrected_object_rgb = cv2.cvtColor(self.corrected_object, cv2.COLOR_BGR2RGB)

            # Build arm_data for visualization
            self.arm_data = []
            for i, tip in enumerate(arm_tips):
                x_vec = tip[0] - self.center[0]
                y_vec = tip[1] - self.center[1]
                length_px = np.hypot(x_vec, y_vec)
                length_mm = length_px * mm_per_pixel
                self.arm_data.append([i + 1, x_vec, y_vec, length_mm])

            # Store contour data for interactive editing
            sorted_contour_global = shifted_contour[sorted_indices] + self.center
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

            # Create morphometrics data object
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

            # Update visualizations
            self.update_arm_visualization()

            # Update polar plot
            angles_normalized = np.mod(angles_sorted, 2 * np.pi)
            self.polar_canvas.set_data(angles_normalized, distances_smoothed, np.array(peaks))

            # Enable UI controls
            self.enable_controls()

        except Exception as e:
            logging.exception(f"Error in run_morphometrics: {e}")
            QMessageBox.critical(self, "Morphometrics Error", f"Error: {str(e)}")

    def update_arm_visualization(self):
        """Update the arm visualization plot"""
        if not hasattr(self, 'arm_data') or not self.arm_data:
            return

        self.ax.clear()
        self.ax.axis('off')

        if not hasattr(self, 'corrected_object_rgb') or self.corrected_object_rgb is None:
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

        # Update arm rotation slider range based on new number of arms
        num_arms = len(self.arm_data)
        if num_arms > 0:
            max_rotation = num_arms - 1
            # Keep current value if possible, otherwise reset to 0
            current_val = min(self.arm_rotation_slider.value(), max_rotation)
            self.arm_rotation_slider.setRange(0, max_rotation)
            self.arm_rotation_slider.setValue(current_val)
        else:
            self.arm_rotation_slider.setRange(0, 0)
            self.arm_rotation_slider.setValue(0)

        # Update the morphometrics data with new arm count
        if self.morphometrics_data:
            self.morphometrics_data['num_arms'] = num_arms

        self.update_arm_visualization()

    def on_smoothing_slider_change(self, value):
        """Handle changes to smoothing slider"""
        self.smoothing_label.setText(f"Smoothing Factor: {value}")
        self.run_morphometrics()

    def on_prominence_slider_change(self, value):
        """Handle changes to prominence slider"""
        prominence = value / 100.0
        self.prominence_label.setText(f"Prominence Factor: {prominence:.2f}")
        self.run_morphometrics()

    def on_distance_slider_change(self, value):
        """Handle changes to distance slider"""
        self.distance_label.setText(f"Distance Factor: {value}")
        self.run_morphometrics()

    def rotate_arm_numbering(self):
        """Rotate arm numbering based on slider value"""
        self.arm_rotation_label.setText(f"Arm Rotation: {self.arm_rotation_slider.value()}")
        self.update_arm_visualization()

    def save_morphometrics(self):
        """Save the updated morphometrics data to disk"""
        if not self.current_folder or not hasattr(self, 'arm_data') or not self.arm_data:
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
        morphometrics_json_path = os.path.join(self.current_folder, 'morphometrics.json')

        try:
            with open(morphometrics_json_path, 'w') as f:
                json.dump(self.morphometrics_data, f, indent=4)

            QMessageBox.information(
                self, "Save Successful",
                f"Updated morphometrics data saved to {os.path.basename(morphometrics_json_path)}"
            )
        except Exception as e:
            logging.exception(f"Error saving morphometrics: {e}")
            QMessageBox.critical(self, "Save Error", f"Error saving morphometrics: {str(e)}")

    def enable_controls(self):
        """Enable UI controls after loading data"""
        self.smoothing_slider.setEnabled(True)
        self.prominence_slider.setEnabled(True)
        self.distance_slider.setEnabled(True)
        self.arm_rotation_slider.setEnabled(True)
        self.save_button.setEnabled(True)

    def disable_controls(self):
        """Disable UI controls when no data is loaded"""
        self.smoothing_slider.setEnabled(False)
        self.prominence_slider.setEnabled(False)
        self.distance_slider.setEnabled(False)
        self.arm_rotation_slider.setEnabled(False)
        self.save_button.setEnabled(False)