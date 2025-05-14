import numpy as np
import cv2
import math

def normalize_corrected_objects(corrected_objects, morphometrics_list, target_angle=90):
    """
    Normalize corrected object images by:
    1. Rotating each image so arm 1 points in the same direction
    2. Cropping to the masked region
    3. Padding all images to the same dimensions

    Args:
        corrected_objects (list): List of corrected object images (RGB format)
        morphometrics_list (list): List of morphometrics data corresponding to each image
        target_angle (float): Target angle in degrees for arm 1 (default: 90° = straight up)

    Returns:
        list: List of normalized images with the same dimensions
    """
    normalized_images = []
    max_height, max_width = 0, 0

    for img, morpho in zip(corrected_objects, morphometrics_list):
        if img is None or len(img.shape) < 2:
            continue

        # Get the angle of arm 1
        arm_data = morpho.get('arm_data', [])
        if not arm_data:
            # If no arm data, just use the original image
            normalized_images.append(img)
            h, w = img.shape[:2]
            max_height = max(max_height, h)
            max_width = max(max_width, w)
            continue

        # Get the first arm's vector (arm 1)
        arm1 = arm_data[0]
        x_vec, y_vec = arm1[1], arm1[2]

        # Calculate current angle in degrees
        current_angle = math.degrees(math.atan2(y_vec, x_vec))

        # Calculate rotation angle to align arm 1 to target angle
        rotation_angle = target_angle - current_angle

        # Get image center for rotation
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Apply rotation
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        # Create mask of non-black pixels (assuming black background)
        mask = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Find bounding box of non-black region
        coords = cv2.findNonZero(mask)
        if coords is None:
            # If image is all black, use original
            normalized_images.append(img)
            max_height = max(max_height, h)
            max_width = max(max_width, w)
            continue

        x, y, w, h = cv2.boundingRect(coords)

        # Crop to bounding box
        cropped = rotated[y:y + h, x:x + w]

        # Update maximum dimensions
        max_height = max(max_height, h)
        max_width = max(max_width, w)

        normalized_images.append(cropped)

    # Pad all images to the same dimensions
    padded_images = []
    for img in normalized_images:
        h, w = img.shape[:2]

        # Calculate padding
        pad_top = (max_height - h) // 2
        pad_bottom = max_height - h - pad_top
        pad_left = (max_width - w) // 2
        pad_right = max_width - w - pad_left

        # Apply padding (black padding)
        padded = cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        padded_images.append(padded)

    return padded_images