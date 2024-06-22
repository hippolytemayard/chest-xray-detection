import logging
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm


@torch.no_grad()
def validation_loop(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    criterion: nn.modules.loss._Loss,
    label_weights: Optional[torch.Tensor] = None,
    with_logits: bool = False,
    metrics_collection: Optional[MetricCollection] = None,
    writer=None,
    device: Union[str, torch.device] = "cpu",
):
    """
    Model validation loop

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader providing the validation data.
        epoch (int): The current epoch number.
        criterion (nn.modules.loss._Loss): The loss function
        with_logits (bool, optional): Indicates whether the model output
            includes logits. Defaults to False.
        metrics_collection (Optional[MetricCollection], optional): Collection
            of metrics to compute during validation. Defaults to None.
        writer (optional): Object for writing validation progress to a log.
            Defaults to None.
        device (Union[str, torch.device], optional): Device on which to
            perform validation. Defaults to "cpu".

    Returns:
        Optional[dict]: A dictionary containing computed metrics,
            if metrics_collection is provided; otherwise, returns None.
    """
    model.eval()

    val_loss = 0

    for data, target in tqdm(loader):
        data = data.to(device)
        target = target.to(device, torch.float)

        output = model(data)
        prediction = torch.sigmoid(output).squeeze(1)

        # val_loss += (
        #    criterion(prediction, target).data.item()
        #    if not with_logits
        #    else criterion(output, target).data.item()
        # )

        loss = criterion(output, target)
        loss = (loss * label_weights).mean()

        val_loss += loss

        if metrics_collection is not None:
            metrics_collection.update(prediction, target.long())

    val_loss /= len(loader)

    logging.info(f"Validation | loss = {val_loss}")

    if writer is not None:
        writer.add_scalar("validation loss", val_loss, epoch)

    if metrics_collection is not None:
        metrics = metrics_collection.compute()

        for k, v in metrics.items():
            logging.info(f"Validation | {k} = {v.detach().cpu().item()}")

            if writer is not None:
                writer.add_scalar(f"Validation {k}", v.detach().cpu().item(), epoch)

        metrics_collection.reset()

        return metrics
