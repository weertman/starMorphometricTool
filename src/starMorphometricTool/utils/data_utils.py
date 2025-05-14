import numpy as np
import json
import os
import logging
import math

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


def load_morphometrics(file_path):
    """
    Load morphometric data from a JSON file.

    Args:
        file_path (str): Path to the morphometrics.json file

    Returns:
        dict: The loaded morphometric data, or None if the file couldn't be loaded

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Morphometrics file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        return data

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing morphometrics JSON: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading morphometrics file: {e}")
        raise


def get_arm_lengths(morphometrics):
    """
    Extract just the arm lengths from morphometrics data.

    Args:
        morphometrics (dict): Morphometrics data

    Returns:
        list: List of arm lengths in mm
    """
    arm_data = morphometrics.get('arm_data', [])
    arm_lengths = [arm[3] for arm in arm_data]  # arm[3] is the length_mm value
    return arm_lengths


def calculate_arm_diameters(morphometrics_data):
    """
    Calculate arm diameters by finding directly opposed arm tips.
    For each arm, it finds the most directly opposite arm (closest to 180° away)
    and calculates the distance between their tips.

    Args:
        morphometrics_data (dict): Morphometrics data containing 'arm_data'

    Returns:
        list: List of tip-to-tip diameter lengths in mm

    Notes:
        - Arm data format is expected to be: [arm_number, x_vec, y_vec, length_mm]
        - Returns empty list if fewer than 2 arms
    """
    arm_data = morphometrics_data.get('arm_data', [])

    # Need at least 2 arms to calculate diameters
    if len(arm_data) < 2:
        return []

    # Calculate angle of each arm vector (in radians)
    arm_angles = []
    for arm in arm_data:
        arm_number, x_vec, y_vec, length_mm = arm
        angle = math.atan2(y_vec, x_vec)
        arm_angles.append((arm_number, angle, x_vec, y_vec, length_mm))

    diameters = []

    # For each arm, find the most opposite arm
    for i, (arm1_number, arm1_angle, x1, y1, length1) in enumerate(arm_angles):
        # Calculate opposite angle (add 180° = π radians)
        opposite_angle = (arm1_angle + math.pi) % (2 * math.pi)

        # Find the arm with angle closest to the opposite
        min_diff = float('inf')
        opposite_arm_idx = -1

        for j, (arm2_number, arm2_angle, _, _, _) in enumerate(arm_angles):
            if j == i:  # Skip self
                continue

            # Calculate angular difference (accounting for circular nature)
            diff = abs((arm2_angle - opposite_angle + math.pi) % (2 * math.pi) - math.pi)

            if diff < min_diff:
                min_diff = diff
                opposite_arm_idx = j

        if opposite_arm_idx != -1:
            _, _, x2, y2, _ = arm_angles[opposite_arm_idx]

            # Calculate diameter (tip-to-tip distance)
            diameter = math.sqrt((x1 + x2) ** 2 + (y1 + y2) ** 2)
            diameters.append(diameter)

    return diameters