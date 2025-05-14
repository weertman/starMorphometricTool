import cv2
import numpy as np
import logging


def find_checkerboard(frame, board_dims):
    """
    Detect a checkerboard in a frame.

    Args:
        frame: The image frame (BGR format)
        board_dims: Tuple of (cols-1, rows-1) for the internal corners of the board

    Returns:
        Tuple of (success, corners, refined_corners)
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_dims, None)

        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners, corners_refined
        else:
            return False, None, None

    except Exception as e:
        logging.exception(f"Error in find_checkerboard: {e}")
        return False, None, None


def compute_checkerboard_homography(image_points, board_dims, square_size):
    """
    Compute homography from checkerboard image points to real-world coordinates.

    Args:
        image_points: Array of checkerboard corner points in image
        board_dims: Tuple of (cols-1, rows-1) for the internal corners of the board
        square_size: Size of checkerboard squares in mm

    Returns:
        Tuple of (H, obj_pts) - homography matrix and object points
    """
    try:
        # Generate object points in real-world coordinates
        obj_pts = np.zeros((board_dims[0] * board_dims[1], 2), np.float32)
        obj_pts[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
        obj_pts *= square_size

        # Find homography
        H, status = cv2.findHomography(image_points.reshape(-1, 2), obj_pts)

        return H, obj_pts
    except Exception as e:
        logging.exception(f"Error in compute_checkerboard_homography: {e}")
        return None, None


def calculate_mm_per_pixel(H, img_pts, obj_pts):
    """
    Calculate the mm per pixel ratio from homography.

    Args:
        H: Homography matrix
        img_pts: Image points
        obj_pts: Object points in real-world coordinates

    Returns:
        mm_per_pixel: Real-world distance per pixel ratio
    """
    try:
        # Use the first two object points for calculation
        obj_pt1, obj_pt2 = obj_pts[0], obj_pts[1]
        real_distance_mm = np.linalg.norm(obj_pt1 - obj_pt2)

        # Transform the first two image points using homography
        img_pt1 = cv2.perspectiveTransform(
            np.array([[img_pts[0]]], dtype='float32'), H
        )[0][0]
        img_pt2 = cv2.perspectiveTransform(
            np.array([[img_pts[1]]], dtype='float32'), H
        )[0][0]

        pixel_distance = np.linalg.norm(img_pt1 - img_pt2)
        mm_per_pixel = real_distance_mm / pixel_distance

        return mm_per_pixel
    except Exception as e:
        logging.exception(f"Error in calculate_mm_per_pixel: {e}")
        return None