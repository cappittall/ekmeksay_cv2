import numpy as np

def bbox_overlaps(boxes1, boxes2):
    """
    Calculate IoUs between two sets of bounding boxes.

    Args:
        boxes1 (numpy.ndarray): Array of bounding boxes with shape (N, 4).
        boxes2 (numpy.ndarray): Array of bounding boxes with shape (M, 4).

    Returns:
        numpy.ndarray: IoU matrix with shape (N, M).
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    lt = np.maximum(boxes1[:, np.newaxis, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, np.newaxis, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clip(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N,]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M,]

    iou = inter / (area1[:, np.newaxis] + area2 - inter)

    return iou