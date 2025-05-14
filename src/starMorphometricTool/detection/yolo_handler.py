"""
YOLO model handling and detection functions
"""
import numpy as np
import cv2
import os
import logging
from ultralytics import YOLO


def load_yolo_model(path=None):
    """Load and return the YOLO model from the specified path."""
    if path is None:
        path = os.path.join('..', '..', 'models', 'best.pt')

    try:
        model = YOLO(path)
        return model
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        raise


def select_primary_detection(results):
    """
    From a YOLO `results` object, pick exactly one detection—
    without any warp/homography. Return a dictionary with e.g.:

        {
            'mask': <the raw mask as a 2D np.uint8 array>,
            'box': (x1, y1, x2, y2) or None,
            'class_id': ...
            'confidence': ...
            ...
        }

    or None if no detection found.
    """
    detections_list = []

    for result in results:
        # For each image in the batch (usually just one)
        if result.masks is not None and result.masks.data is not None:
            for idx, mask_tensor in enumerate(result.masks.data):
                # e.g. the raw mask
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                # Possibly we also grab the bounding box info
                boxes = result.boxes
                if boxes is not None and idx < len(boxes):
                    # each box has .xyxy, .conf, .cls, etc.
                    xyxy = boxes.xyxy[idx].cpu().numpy()
                    conf = boxes.conf[idx].cpu().item()
                    cls_id = boxes.cls[idx].cpu().item()
                else:
                    xyxy = None
                    conf = None
                    cls_id = None

                # If the mask is valid, store it
                if np.count_nonzero(mask_np) > 0:
                    detections_list.append({
                        'mask': mask_np,
                        'box': xyxy,
                        'class_id': cls_id,
                        'confidence': conf
                    })

    # If you always pick the first valid detection:
    if len(detections_list) > 0:
        return detections_list[0]

    return None