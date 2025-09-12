from doc_explainer.metrics import (
    bbox_area,
    bbox_center,
    compute_iou,
    compute_iou_with_threshold,
    compute_normalized_center_distance,
)


def test_bbox_area():
    bbox = [0.0, 0.0, 1.0, 1.0]
    assert bbox_area(bbox) == 1.0


def test_bbox_center():
    bbox = [0.0, 0.0, 1.0, 1.0]
    assert bbox_center(bbox) == (0.5, 0.5)


def test_compute_iou():
    bbox1 = [0.0, 0.0, 1.0, 1.0]
    bbox2 = [0.0, 0.0, 1.0, 1.0]
    assert compute_iou(bbox1, bbox2) == 1.0


def test_compute_iou_with_threshold():
    bbox1 = [0.0, 0.0, 1.0, 1.0]
    bbox2 = [0.0, 0.0, 1.0, 1.0]
    assert compute_iou_with_threshold(bbox1, bbox2) is True

    # Test with non-overlapping bboxes (should return False)
    bbox3 = [2.0, 2.0, 3.0, 3.0]
    assert compute_iou_with_threshold(bbox1, bbox3) is False


def test_compute_normalized_center_distance():
    bbox1 = [0.0, 0.0, 1.0, 1.0]
    bbox2 = [2.0, 2.0, 3.0, 3.0]
    assert abs(compute_normalized_center_distance(bbox1, bbox2) - 2 / 3) < 1e-10
