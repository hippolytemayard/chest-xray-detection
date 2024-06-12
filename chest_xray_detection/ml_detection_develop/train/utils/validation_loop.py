import torch
from torch import nn
from torch.utils.data import DataLoader


def validation_loop(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    writer=None,
    device: str | torch.device = "cpu",
) -> None:

    model.train()

    for batch_idx, (data, targets) in enumerate(loader):
        images = list(image.to(device) for image in data)

        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        with torch.inference_mode():
            loss_dict = model(images, targets)

        loss_reduced = sum(loss for loss in loss_dict.values())

        loss_classifier = loss_dict.get("loss_classifier").item()
        loss_box_reg = loss_dict.get("loss_box_reg").item()
        loss_objectness = loss_dict.get("loss_objectness").item()
        loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg").item()

        if writer is not None:

            step = epoch * len(loader) + batch_idx

            writer.add_scalar("Loss_val", loss_reduced.item(), step)
            writer.add_scalar("Classifier loss_val", loss_classifier, step)
            writer.add_scalar("Box reg loss_val", loss_box_reg, step)
            writer.add_scalar("Objectness loss_val", loss_objectness, step)
            writer.add_scalar("Rpn box reg loss_val", loss_rpn_box_reg, step)
