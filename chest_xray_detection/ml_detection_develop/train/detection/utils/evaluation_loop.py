import logging
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from chest_xray_detection.ml_detection_develop.train.detection.utils.postprocessing import (
    non_maximum_suppression,
    filter_scores,
)


@torch.no_grad()
def evaluation_loop(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    metrics_collection: Optional[MetricCollection] = None,
    writer=None,
    device: str | torch.device = "cpu",
    apply_nms: bool = False,
    nms_iou_threshold: float = 0.1,
    apply_scores_filter: bool = False,
    scores_filter_threshold: float = 0.3,
) -> dict:
    """
    Model validation loop

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader providing the validation data.
        epoch (int): The current epoch number.
        metrics_collection (Optional[MetricCollection], optional): Collection
            of metrics to compute during validation. Defaults to None.
        writer (optional): Object for writing validation progress to a log.
            Defaults to None.
        device (str | torch.device, optional): Device on which to
            perform validation. Defaults to "cpu".

    Returns:
        Optional[dict]: A dictionary containing computed metrics,
            if metrics_collection is provided; otherwise, returns None.
    """
    model.eval()

    for batch_idx, (data, targets) in enumerate(loader):
        images = list(image.to(device) for image in data)

        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        with torch.inference_mode():
            predictions = model(images)

        if apply_scores_filter:
            predictions = [
                filter_scores(predictions=prediction, scores=scores_filter_threshold)
                for prediction in predictions
            ]

        if apply_nms:
            predictions = [
                non_maximum_suppression(predictions=prediction, iou_threshold=nms_iou_threshold)
                for prediction in predictions
            ]

        if metrics_collection is not None:
            metrics_collection.update(predictions, targets)

    if metrics_collection is not None:
        metrics = metrics_collection.compute()

        for k, v in metrics.items():
            if k == "classes":
                continue

            logging.info(f"Validation | {k} = {v.detach().cpu().item()}")

            if writer is not None:
                writer.add_scalar(f"Validation {k}", v.detach().cpu().item(), epoch)

        metrics_collection.reset()

        return metrics
