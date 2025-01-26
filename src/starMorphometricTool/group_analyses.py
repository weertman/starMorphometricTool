from pathlib import Path
import json
import numpy as np
import pickle


##########################
# Verbosity Print Helper #
##########################

def _vprint(msg, verbosity):
    """Prints the message only if verbosity > 0."""
    if verbosity > 0:
        print(msg)


##########################
# Measurement Data Load  #
##########################

def get_mFolder_num(mFolder):
    mFolder_num = mFolder.name.split('_')[-1]
    return int(mFolder_num)


def load_measurement_data(measurement_folder):
    """
    Load both corrected_detection.json and morphometrics.json from a given folder.
    Return them as dictionaries (or None if not found).
    """
    detection_path = Path(measurement_folder) / 'corrected_detection.json'
    morpho_path = Path(measurement_folder) / 'morphometrics.json'

    detection_data = None
    morpho_data = None

    if detection_path.exists():
        with open(detection_path, 'r') as f:
            detection_data = json.load(f)

    if morpho_path.exists():
        with open(morpho_path, 'r') as f:
            morpho_data = json.load(f)

    return detection_data, morpho_data


##########################
# Arm-Based Computations #
##########################

def _angle_diff_rad(a, b):
    """
    Compute the minimal difference between angles a and b (both in radians),
    normalized to the range [-pi, pi].
    """
    d = a - b
    # wrap difference into [-pi, pi]
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


def compute_arm_angles(arm_data):
    """
    Given 'arm_data' (the list of [arm_number, x_vector, y_vector, length_mm]),
    return a list of angles in radians, where angle[i] = arctan2(y, x).
    """
    angles = []
    for entry in arm_data:
        # entry = [arm_number, x_vec, y_vec, length_mm]
        _, x_vec, y_vec, _ = entry
        angle = np.arctan2(y_vec, x_vec)  # Radians in [-pi, pi]
        angles.append(angle)
    return angles


def compute_tip_to_tip_widths(arm_data):
    """
    For each arm i, find the arm j whose angle is closest to angle[i] + pi.
    Then the tip-to-tip width = length_i + length_j.

    Returns:
        widths (list of float): The computed tip-to-tip widths (mm) for each arm.
    """
    angles = compute_arm_angles(arm_data)
    lengths = [entry[3] for entry in arm_data]  # length_mm is the 4th item

    n_arms = len(angles)
    widths = []

    for i in range(n_arms):
        angle_i = angles[i]
        length_i = lengths[i]

        target_angle = angle_i + np.pi
        best_j = None
        best_diff = float('inf')
        for j in range(n_arms):
            if j == i:
                continue
            diff = _angle_diff_rad(angles[j], target_angle)
            if diff < best_diff:
                best_diff = diff
                best_j = j

        if best_j is not None:
            width_ij = length_i + lengths[best_j]
            widths.append(width_ij)

    return widths


def analyze_body_widths(morpho_data):
    """
    Compute tip-to-tip widths and basic statistics:
        - 'tip_to_tip_widths': the list of widths (mm)
        - 'mean_width': average
        - 'median_width': median
        - 'std_width': standard deviation
    """
    if 'arm_data' not in morpho_data:
        raise ValueError("morpho_data has no 'arm_data' key.")

    arm_data = morpho_data['arm_data']
    if not arm_data:
        raise ValueError("No arm data found in 'arm_data'.")

    widths = compute_tip_to_tip_widths(arm_data)
    if len(widths) == 0:
        raise ValueError("No tip-to-tip widths were computed. Possibly only 1 arm?")

    mean_width = np.mean(widths)
    median_width = np.median(widths)
    std_width = np.std(widths)

    return {
        'tip_to_tip_widths': widths,
        'mean_width': mean_width,
        'median_width': median_width,
        'std_width': std_width,
    }


##########################
# PCA-based Body Axes    #
##########################

def get_arm_tips_array(morpho_data):
    """
    Convert morpho_data['arm_data'] into an N×2 array of arm tip coordinates.
    """
    if 'arm_data' not in morpho_data:
        raise ValueError("morpho_data has no 'arm_data' key.")
    arm_data = morpho_data['arm_data']
    if not arm_data:
        raise ValueError("No arm data in morpho_data['arm_data'].")

    tips = []
    for entry in arm_data:
        _, x_vec, y_vec, _ = entry
        tips.append([x_vec, y_vec])
    return np.array(tips)


def pca_axes_from_arm_tips(tips):
    """
    Compute major/minor axis lengths (bounding diameter) via PCA on arm-tip coords.
    """
    if tips.shape[1] != 2:
        raise ValueError("tips array must be N×2.")

    mean_xy = tips.mean(axis=0)
    centered = tips - mean_xy

    cov = np.cov(centered.T)  # 2×2 covariance
    eigenvals, eigenvecs = np.linalg.eig(cov)

    # Sort eigenvalues (and vectors) descending
    idx_sorted = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx_sorted]
    eigenvecs = eigenvecs[:, idx_sorted]

    axis1 = eigenvecs[:, 0]  # direction of largest spread
    axis2 = eigenvecs[:, 1]  # direction of smaller spread

    proj1 = centered @ axis1
    proj2 = centered @ axis2

    major_axis_length = proj1.max() - proj1.min()
    minor_axis_length = proj2.max() - proj2.min()

    return {
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "eigenvalues": eigenvals.tolist(),
        "eigenvectors": eigenvecs.tolist(),
        "mean_xy": mean_xy.tolist(),
    }


def analyze_body_axes_via_armtips(morpho_data):
    """
    PCA-based bounding approach to compute major/minor axis from arm tips.
    """
    tips = get_arm_tips_array(morpho_data)
    pca_result = pca_axes_from_arm_tips(tips)
    return {
        "major_axis_length_mm": pca_result["major_axis_length"],
        "minor_axis_length_mm": pca_result["minor_axis_length"],
        "eigenvalues": pca_result["eigenvalues"],
        "eigenvectors": pca_result["eigenvectors"],
        "mean_xy": pca_result["mean_xy"]
    }


##############################
# Main Analysis per Group    #
##############################

def analyze_morhpometrics_for_group_root(path_group_root, verbosity=1):
    """
    Main function to process an entire group root, analyzing the latest mFolder
    for each ID on each measurement date. Saves aggregated results in a pickle.

    Args:
        path_group_root (Path): Root directory for a single group (e.g. knownID).
        verbosity (int): 0 = silent, 1 = normal prints, 2+ = very verbose

    Returns:
        (pkl_path, aggregated_measures_pkl)
    """
    if not path_group_root.exists():
        raise FileNotFoundError(f"Path not found: {path_group_root}")

    # Grab all IDs under this group
    ids = [f for f in path_group_root.iterdir() if f.is_dir()]
    _vprint(f"Found {len(ids)} IDs", verbosity)

    # Identify distinct measurement dates across IDs
    measurement_dates = []
    for id_dir in ids:
        tmp_dates = [f for f in id_dir.iterdir() if f.is_dir()]
        for tmp_date in tmp_dates:
            if tmp_date.name not in measurement_dates:
                measurement_dates.append(tmp_date.name)

    measurement_dates_dict = {}
    for measurement_date in measurement_dates:
        _vprint(f"Processing measurement date: {measurement_date}", verbosity)
        for id_dir in ids:
            tmp_dates = [f for f in id_dir.iterdir() if f.is_dir()]
            for tmp_date in tmp_dates:
                if tmp_date.name == measurement_date:
                    if measurement_date not in measurement_dates_dict:
                        measurement_dates_dict[measurement_date] = []
                    mFolders = [f for f in tmp_date.iterdir() if f.is_dir()]
                    mFolders = sorted(mFolders, key=lambda x: get_mFolder_num(x))

                    # Check from newest to oldest mFolder
                    for mFolder in reversed(mFolders):
                        detection_data, morpho_data = load_measurement_data(mFolder)
                        if morpho_data is not None and detection_data is not None:
                            measurement_dates_dict[measurement_date].append(
                                {id_dir.name: [mFolder, detection_data, morpho_data]}
                            )
                            break

    # Now compute the analysis and aggregate
    aggregated_measures_pkl = {}
    for measurement_date, id_list in measurement_dates_dict.items():
        _vprint(f"\n== Measurement date: {measurement_date} ==", verbosity)
        aggregated_measures_pkl[measurement_date] = {}

        for id_dict in id_list:
            for id_name, data in id_dict.items():
                mfolder_path, detection_data, morpho_data = data
                _vprint(f"ID: {id_name}, mFolder: {mfolder_path.name}", verbosity)

                # Analyze tip-to-tip widths
                widths_info = {}
                try:
                    widths_info = analyze_body_widths(morpho_data)
                    if verbosity > 0:
                        _vprint(
                            f"\tTip-to-tip widths (mm): {widths_info['tip_to_tip_widths']}",
                            verbosity
                        )
                        _vprint(
                            f"\tMean width: {widths_info['mean_width']:.2f} mm | "
                            f"Median: {widths_info['median_width']:.2f} mm | "
                            f"Std: {widths_info['std_width']:.2f} mm",
                            verbosity
                        )
                        _vprint(f"\tArea mm^2: {morpho_data['area_mm2']}", verbosity)
                except ValueError as e:
                    _vprint(f"\t[WARNING] Could not compute widths: {e}", verbosity)

                # Analyze PCA-based major/minor axes
                axis_info = {}
                try:
                    axis_info = analyze_body_axes_via_armtips(morpho_data)
                    major_mm = axis_info["major_axis_length_mm"]
                    minor_mm = axis_info["minor_axis_length_mm"]
                    if verbosity > 0:
                        _vprint(
                            f"\tPCA-based major axis: {major_mm:.2f} mm | "
                            f"minor axis: {minor_mm:.2f} mm",
                            verbosity
                        )
                except ValueError as ve:
                    _vprint(f"\t[WARNING] PCA axis computation failed: {ve}", verbosity)
                    major_mm, minor_mm = (None, None)

                # Store results
                aggregated_measures_pkl[measurement_date][id_name] = {
                    'tip_to_tip_widths': widths_info.get('tip_to_tip_widths', []),
                    'mean_width': widths_info.get('mean_width'),
                    'median_width': widths_info.get('median_width'),
                    'std_width': widths_info.get('std_width'),
                    'area_mm2': morpho_data.get('area_mm2'),
                    'major_axis_length_mm': major_mm,
                    'minor_axis_length_mm': minor_mm
                }

    # Save results into a pickle
    pkl_path = path_group_root / 'aggregated_measures.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(aggregated_measures_pkl, f)

    return pkl_path, aggregated_measures_pkl


###################
# Script Entry    #
###################

if __name__ == '__main__':
    # Example usage:
    path_group_root = Path(r'../../measurements/snack_pack/knownID')
    #  verbosity=0  => no console output
    #  verbosity=1  => standard console prints
    #  verbosity=2+ => you can add more prints or debug info if you want
    pkl_path, aggregated_measures_pkl = analyze_morhpometrics_for_group_root(path_group_root, verbosity=1)
