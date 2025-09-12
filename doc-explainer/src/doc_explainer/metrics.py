import math
from typing import Tuple

# [x0, y0, x1, y1]
BBox = Tuple[float, float, float, float]


def bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def compute_iou(b1: BBox, b2: BBox) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    """
    try:
        x0 = max(b1[0], b2[0])
        y0 = max(b1[1], b2[1])
        x1 = min(b1[2], b2[2])
        y1 = min(b1[3], b2[3])
    except Exception as e:
        print(f"Error computing IoU: {e}")
        return 0.0

    inter_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union_area = bbox_area(b1) + bbox_area(b2) - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_iou_with_threshold(b1: BBox, b2: BBox, threshold: float = 0.5) -> bool:
    """
    Check if IoU between two bounding boxes exceeds a given threshold.
    """
    iou = compute_iou(b1, b2)
    return iou >= threshold


def compute_normalized_center_distance(b1: BBox, b2: BBox) -> float:
    """
    Compute the normalized Euclidean distance between the centers of two bboxes.
    Normalized by the diagonal length of the smallest box that encloses both.
    """
    try:
        cx1, cy1 = bbox_center(b1)
        cx2, cy2 = bbox_center(b2)
    except Exception as e:
        print(f"Error computing bbox centers: {e}")
        return 0.0

    distance = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    # Compute diagonal of the enclosing box
    x0 = min(b1[0], b2[0])
    y0 = min(b1[1], b2[1])
    x1 = max(b1[2], b2[2])
    y1 = max(b1[3], b2[3])
    diag = math.sqrt(((x1 - x0) ** 2 + (y1 - y0) ** 2))

    return distance / diag if diag > 0 else 0.0
