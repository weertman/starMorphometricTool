import numpy as np
import cv2
import math

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
