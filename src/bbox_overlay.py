"""Bounding box overlay for detection results."""
import cv2
import numpy as np
def draw_bbox(image: np.ndarray, bbox, color=(255,0,0), thickness=2, label=None):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)
    if label:
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image
