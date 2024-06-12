from typing import Optional

import torch
from torchvision.ops import nms


def non_maximum_suppression(
    predictions: dict[str, torch.Tensor], iou_threshold: float = 0.5
) -> Optional[dict[str, torch.Tensor]]:
    """
    Apply Non-Maximum Suppression (NMS) on the predictions results.

    Args:
        predictions (dict[str, torch.Tensor]): Dictionary containing 'boxes', 'scores', and 'labels'.
        iou_threshold (float): Intersection over Union (IoU) threshold for NMS.

    Returns:
        Optional[dict[str, torch.Tensor]]: Dictionary containing the filtered 'boxes', 'scores', and 'labels' after NMS.
    """
    nms_indexes = nms(
        boxes=predictions["boxes"],
        scores=predictions["scores"],
        iou_threshold=iou_threshold,
    )

    return filter_prediction_indexes(predictions, nms_indexes)


def filter_prediction_indexes(
    predictions: dict[str, torch.Tensor], indexes: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Filters the predictions dictionary by the specified indexes.

    Args:
        predictions (dict[str, torch.Tensor]): Dictionary containing 'boxes', 'scores', and 'labels'.
        indexes (torch.Tensor): Tensor of indexes to filter the predictions dictionary.

    Returns:
        dict[str, torch.Tensor]: Filtered predictions dictionary containing 'boxes', 'scores', and 'labels'.
    """
    boxes = predictions["boxes"][indexes]
    scores = predictions["scores"][indexes]
    labels = predictions["labels"][indexes]

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }
