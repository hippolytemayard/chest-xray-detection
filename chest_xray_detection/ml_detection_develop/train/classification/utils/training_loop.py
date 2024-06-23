import logging
from typing import Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection


def training_loop(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    epoch: int,
    optimizer: Optimizer,
    label_weights: Optional[torch.Tensor] = None,
    with_logits: bool = False,
    metrics_collection: Optional[MetricCollection] = None,
    log_interval: int = 200,
    writer=None,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """
    Model training loop.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): DataLoader providing the training data.
        criterion (nn.modules.loss._Loss): Loss function.
        epoch (int): Current epoch number.
        optimizer (Optimizer): Optimizer used for training.
        label_weights (Optional[torch.Tensor], optional): Weights for each class label. Defaults to None.
        with_logits (bool, optional): Whether the model output includes logits. Defaults to False.
        metrics_collection (Optional[MetricCollection], optional): Collection of metrics to compute during training. Defaults to None.
        log_interval (int, optional): Interval for logging training progress. Defaults to 200.
        writer (optional): Object for writing training progress to a log. Defaults to None.
        device (Union[str, torch.device], optional): Device for training. Defaults to "cpu".
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
    """
    model.train()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device, torch.float)

        optimizer.zero_grad()

        output = model(data)
        prediction = torch.sigmoid(output).squeeze(1)

        loss = criterion(output, target)
        loss = (loss * label_weights).mean() if label_weights is not None else loss.mean()

        if writer is not None:
            writer.add_scalar(
                "Training Loss (batch)",
                loss,
                epoch * len(loader) + batch_idx,
            )

        if metrics_collection is not None:
            metrics_collection.update(prediction, target.long())

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            logging.info(
                "Train | Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.item(),
                )
            )

    if metrics_collection is not None:
        metrics = metrics_collection.compute()

        for k, v in metrics.items():
            logging.info(f"Training | {k} = {v.detach().cpu().item()}")

            if writer is not None:
                writer.add_scalar(f"Training {k}", v.detach().cpu().item(), epoch)

        metrics_collection.reset()
