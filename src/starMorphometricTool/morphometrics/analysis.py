import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import cv2
import logging


def find_arm_tips(contour, center, smoothing_factor, prominence_factor, distance_factor):
    """
    Find arm tips from contour based on distance peaks.

    Args:
        contour: Numpy array of contour points, shape (N, 2)
        center: (x, y) center point
        smoothing_factor: Window size for smoothing the distance profile
        prominence_factor: Prominence factor for peak detection (0.0-1.0)
        distance_factor: Minimum distance between peaks

    Returns:
        Tuple of:
        - arm_tips: Array of arm tip coordinates
        - angles_sorted: Sorted angles
        - distances_smoothed: Smoothed distance profile
        - peaks: Indices of peaks
        - sorted_indices: Indices that sorted angles
        - shifted_contour: Contour shifted to center as origin
    """
    # Shift contour to center as origin
    shifted_contour = contour - center

    # Calculate polar coordinates
    angles = np.arctan2(shifted_contour[:, 1], shifted_contour[:, 0])
    distances = np.hypot(shifted_contour[:, 0], shifted_contour[:, 1])

    # Sort by angle
    sorted_indices = np.argsort(angles)
    angles_sorted = angles[sorted_indices]
    distances_sorted = distances[sorted_indices]

    # Smooth the distance profile
    distances_smoothed = uniform_filter1d(distances_sorted, size=smoothing_factor)

    # Find peaks in distance profile
    def find_peaks_on_array(arr):
        return find_peaks(arr, prominence=prominence_factor * arr.max(), distance=distance_factor)[0]

    # Try finding peaks in two offsets to improve detection
    peaks1 = find_peaks_on_array(distances_smoothed)
    roll_amount = int(len(distances_smoothed) * (np.pi / 12) / (2 * np.pi))
    distances_rolled = np.roll(distances_smoothed, roll_amount)
    peaks2 = (find_peaks_on_array(distances_rolled) - roll_amount) % len(distances_smoothed)

    # Combine peaks and select the most prominent ones
    all_peaks = np.unique(np.concatenate([peaks1, peaks2]))
    sorted_peaks = sorted(all_peaks, key=lambda x: distances_smoothed[x], reverse=True)
    peaks = sorted_peaks[:24]  # Limit to 24 arms maximum

    # Get arm tips in original coordinates
    arm_tips = shifted_contour[sorted_indices][peaks] + center
    arm_angles = angles_sorted[peaks]

    # Sort arm tips by angle for consistent numbering
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


def calculate_morphometrics(contour, mm_per_pixel):
    """
    Calculate morphometric measurements for a contour.

    Args:
        contour: Numpy array of contour points
        mm_per_pixel: Conversion factor from pixels to mm

    Returns:
        Dictionary of morphometric data
    """
    # Calculate area
    area_pixels = cv2.contourArea(contour)
    area_mm2 = area_pixels * (mm_per_pixel ** 2)

    # Find center
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    center = np.array([cx, cy])

    # Fit ellipse if possible
    ellipse_data = None
    major_axis_mm = None
    minor_axis_mm = None

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
            ellipse_data = (x0, y0, major_axis_length, minor_axis_length, angle)
        except Exception as e:
            logging.warning(f"Error fitting ellipse: {e}")

    # Compile morphometric data
    morphometrics = {
        'area_mm2': area_mm2,
        'center': center,
        'major_axis_mm': major_axis_mm,
        'minor_axis_mm': minor_axis_mm,
        'ellipse_data': ellipse_data,
        'contour_coordinates': contour.reshape(-1, 2).tolist(),
        'mm_per_pixel': mm_per_pixel
    }

    return morphometrics