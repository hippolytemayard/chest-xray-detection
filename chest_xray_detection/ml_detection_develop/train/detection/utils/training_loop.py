import logging

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def training_loop(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    optimizer: Optimizer,
    writer=None,
    device: str | torch.device = "cpu",
) -> None:
    """
    Model training loop.

    Args:
        model (nn.Module): The model model to train.
        loader (DataLoader): The data loader providing the training data.
        criterion (nn.modules.loss._Loss): The loss function
        epoch (int): The current epoch number.
        optimizer (Optimizer): The optimizer used for training.
        with_logits (bool, optional): Indicates whether the model output
            includes logits. Defaults to False.
        metrics_collection (Optional[MetricCollection], optional): Collection
            of metrics to compute during training. Defaults to None.
        log_interval (int, optional): Interval for logging training progress.
            Defaults to 200.
        writer (optional): Object for writing training progress to a log.
            Defaults to None.
        device (str | torch.device, optional): Device on which to
            perform training. Defaults to "cpu".
    """
    model.train()

    for batch_idx, (data, targets) in enumerate(loader):
        images = list(image.to(device) for image in data)

        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        # if batch_idx == 0:
        #    print(targets)

        loss_dict = model(images, targets)
        loss_reduced = sum(loss for loss in loss_dict.values())

        loss_classifier = loss_dict.get("loss_classifier").item()
        loss_box_reg = loss_dict.get("loss_box_reg").item()
        loss_objectness = loss_dict.get("loss_objectness").item()
        loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg").item()

        optimizer.zero_grad()

        logging.info(
            f"Epoch:{epoch}  [{batch_idx}/{len(loader)}]  loss:{loss_reduced}]"
            f"  loss_classifier:{loss_classifier}  loss_box_reg:{loss_box_reg}"
            f"  loss_objectness:{loss_objectness}  loss_rpn_box_reg:{loss_rpn_box_reg}"
        )

        if writer is not None:

            step = epoch * len(loader) + batch_idx
            writer.add_scalar("Loss_train", loss_reduced.item(), step)
            writer.add_scalar("Classifier loss_train", loss_classifier, step)
            writer.add_scalar("Box reg loss_train", loss_box_reg, step)
            writer.add_scalar("Objectness loss_train", loss_objectness, step)
            writer.add_scalar("Rpn box reg loss_train", loss_rpn_box_reg, step)

        loss_reduced.backward()
        optimizer.step()
