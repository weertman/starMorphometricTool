import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from io import BytesIO
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2


def create_morphometrics_visualization(ax, corrected_object_rgb, center, arm_data,
                                       rotation, ellipse_data, morphometrics_data):
    """
    Create a visualization of the specimen with numbered arms.

    Args:
        ax: Matplotlib axis to draw on
        corrected_object_rgb: RGB image of the corrected object
        center: (x, y) center point
        arm_data: List of [arm_number, x_vec, y_vec, length_mm]
        rotation: Rotation value for arm numbering
        ellipse_data: Tuple of ellipse parameters (x0, y0, major_len, minor_len, angle)
        morphometrics_data: Dictionary of morphometric measurements

    Returns:
        arm_angles, arm_dists, arm_labels, arm_colors for polar plot
    """
    ax.clear()
    ax.axis('off')

    # Draw the specimen image
    ax.imshow(corrected_object_rgb)

    cx, cy = center
    num_arms = len(arm_data)
    new_order = [(i - rotation) % num_arms + 1 for i in range(num_arms)]

    # Mark the center
    ax.plot(cx, cy, 'yo', markersize=10, label='Center')

    # Polar plot data
    polar_angles = []
    polar_dists = []
    polar_labels = []
    polar_colors = []

    # Draw each arm
    for i, arm_info in enumerate(arm_data):
        arm_number, x_vec, y_vec, length_mm = arm_info
        tip_x = cx + x_vec
        tip_y = cy + y_vec
        new_num = new_order[i]
        color = 'red' if new_num == 1 else 'blue'

        # Line from center to tip
        ax.plot([cx, tip_x], [cy, tip_y], color=color, linewidth=2, label='_nolegend_')
        ax.plot(tip_x, tip_y, 'o', color=color, markersize=6, label='_nolegend_')

        # Number label
        text_x = (cx + tip_x) / 2
        text_y = (cy + tip_y) / 2
        ax.text(text_x, text_y, str(new_num),
                color='white', fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))

        # Store polar plot data
        tip_angle = np.arctan2(y_vec, x_vec)
        tip_dist = np.hypot(x_vec, y_vec)

        polar_angles.append(tip_angle)
        polar_dists.append(tip_dist)
        polar_labels.append(new_num)
        polar_colors.append(color)

    # Draw ellipse if available
    if ellipse_data:
        x0, y0, major_len, minor_len, angle = ellipse_data
        ellipse_patch = Ellipse((x0, y0), major_len, minor_len, angle=angle,
                                edgecolor='yellow', facecolor='none', linewidth=2,
                                label='Body Ellipse')
        ax.add_patch(ellipse_patch)

    # Show measurements as text overlay
    area_val = morphometrics_data.get("area_mm2", 0)
    num_arms_val = len(arm_data)
    major_mm = morphometrics_data.get("major_axis_mm", 0)
    minor_mm = morphometrics_data.get("minor_axis_mm", 0)

    meas_text = (
        f'Area: {area_val:.2f} mm²\n'
        f'Number of Arms: {num_arms_val}\n'
        f'Major Axis: {major_mm}\n'
        f'Minor Axis: {minor_mm}'
    )
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    ax.text(0.05, 0.95, meas_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props, color='white')

    ax.legend(loc='upper right')
    ax.set_title(f"Arm Numbering (Arm 1 = position {rotation + 1})")

    return polar_angles, polar_dists, polar_labels, polar_colors


def render_figure_to_pixmap(fig, target_widget):
    """
    Render a matplotlib figure to a QPixmap for display.

    Args:
        fig: Matplotlib figure to render
        target_widget: QWidget for determining target size

    Returns:
        QPixmap ready for display
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    qimg = QImage.fromData(buf.getvalue())
    pixmap = QPixmap.fromImage(qimg)

    return pixmap.scaled(
        target_widget.size(),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )