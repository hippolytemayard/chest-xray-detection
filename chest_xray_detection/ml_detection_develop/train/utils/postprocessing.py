from ast import Raise
from typing import Optional

import torch
from torchvision.ops import nms


def non_maximum_suppression(
    predictions: dict[str, torch.Tensor],
    iou_threshold: float = 0.5,
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

    predictions = filter_prediction_indexes(predictions, nms_indexes)

    return predictions


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


def filter_scores(predictions: dict[str, torch.Tensor], scores: int) -> dict[str, torch.Tensor]:
    """TODO: docstring

    Args:
        predictions (dict[str, torch.Tensor]): _description_
        scores (int): _description_

    Returns:
        dict[str, torch.Tensor]: _description_
    """

    if isinstance(predictions, list):
        predictions = predictions[0]

    elif isinstance(predictions, dict):
        pass

    else:
        raise TypeError("predictions must be a list or a dictionary")

    indexes = predictions["scores"] > scores

    boxes = predictions["boxes"][indexes]
    scores = predictions["scores"][indexes]
    labels = predictions["labels"][indexes]

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }
