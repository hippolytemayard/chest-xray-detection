import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from chest_xray_detection.ml_detection_develop.configs.settings import ANNOTATION_PATH
from chest_xray_detection.ml_detection_develop.dataset.dataloader import get_train_val_dataloaders
from chest_xray_detection.ml_detection_develop.dataset.transforms.detection.utils import (
    instantiate_transforms_from_config,
)
from chest_xray_detection.ml_detection_develop.metrics.utils import instantiate_metrics_from_config
from chest_xray_detection.ml_detection_develop.models.detection.faster_rcnn import get_faster_rcnn
from chest_xray_detection.ml_detection_develop.optimizer import optimizer_dict
from chest_xray_detection.ml_detection_develop.train.detection.utils.evaluation_loop import (
    evaluation_loop,
)
from chest_xray_detection.ml_detection_develop.train.detection.utils.training_loop import (
    training_loop,
)
from chest_xray_detection.ml_detection_develop.train.detection.utils.validation_loop import (
    validation_loop,
)
from chest_xray_detection.ml_detection_develop.utils.files import load_yaml, make_exists


# TODO : modify if false
def stratified_split_train_model_from_config(
    config: OmegaConf,
    device: torch.device,
):
    """
    Train a model based on the provided configuration.

    Args:
        config (OmegaConf): Experiment configuration.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
    """

    writer = (
        SummaryWriter(config.TRAINING.TENSORBOARD_DIR)
        if config.TRAINING.ENABLE_TENSORBOARD
        else None
    )

    num_classes = config.TRAINING.DATASET.NUM_CLASSES if not config.TRAINING.MERGE_CLASSES else 2
    logging.info(f"Number of classes : {num_classes}")

    model = get_faster_rcnn(
        pretrained=config.TRAINING.PRETRAINED,
        num_classes=num_classes,
        backbone=config.TRAINING.BACKBONE,
        trainable_backbone_layers=5,
    ).to(device)

    logging.info(f"Backbone : {config.TRAINING.BACKBONE}")

    if False:
        logging.info(f"Load state dict")
        path_model = Path("experiments/experiment_300/saved_models") / "best_model.pt"
        model_state_dict = torch.load(path_model)
        model.load_state_dict(model_state_dict["model"])
        logging.info(f"Loading {path_model}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"The model has {trainable_params} trainable parameters")

    train_transforms = instantiate_transforms_from_config(
        transform_config=config.TRAINING.DATASET.TRANSFORMS.TRAINING
    )
    val_transforms = instantiate_transforms_from_config(
        transform_config=config.TRAINING.DATASET.TRANSFORMS.VALIDATION
    )

    (train_loader, val_loader, distribution) = get_train_val_dataloaders(
        annotation_filepath=ANNOTATION_PATH,
        batch_size=config.TRAINING.BATCH_SIZE,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        val_size=config.TRAINING.DATASET.VALIDATION_SPLIT,
        merge_classes=config.TRAINING.MERGE_CLASSES,
    )

    for k, v in distribution.items():
        logging.info(f"Distribution of {k} set {v}")

    optimizer = optimizer_dict[config.TRAINING.OPTIMIZER](
        params=model.parameters(), lr=config.TRAINING.LEARNING_RATE
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    metrics_collection = instantiate_metrics_from_config(
        metrics_config=config.VALIDATION.METRICS
    ).to(device)

    best_map = -1

    for epoch in range(1, config.TRAINING.EPOCHS + 1):

        logging.info(f"EPOCH {epoch}")

        training_loop(
            model=model,
            loader=train_loader,
            epoch=epoch,
            optimizer=optimizer,
            writer=writer,
            device=device,
        )

        lr_scheduler.step()

        validation_loop(
            model=model,
            loader=val_loader,
            epoch=epoch,
            writer=writer,
            device=device,
        )

        dict_metrics = evaluation_loop(
            model=model,
            loader=val_loader,
            epoch=epoch,
            metrics_collection=metrics_collection,
            writer=writer,
            device=device,
        )

        save_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
        }
        torch.save(save_dict, Path(config.TRAINING.PATH_MODEL) / "checkpoint.pt")

        if best_map < dict_metrics["map_50"]:
            logging.info(
                f"Validation | model improved from {best_map} to {dict_metrics['map_50']} | saving model"
            )
            best_map = dict_metrics["map_50"]
            save_dict["val_metrics"] = best_map

            torch.save(save_dict, Path(config.TRAINING.PATH_MODEL) / "best_model.pt")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using configuration file")
    parser.add_argument(
        "--config",
        type=str,
        default="chest_xray_detection/configs/training/stratified_split/training_faster_rcnn.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    make_exists(config.EXPERIMENT_FOLDER)
    make_exists(config.ROOT_EXPERIMENT)
    make_exists(config.TRAINING.PATH_MODEL)
    make_exists(config.TRAINING.TENSORBOARD_DIR)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(config.TRAINING.PATH_LOGS),
            logging.StreamHandler(),
        ],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"device : {device}")

    stratified_split_train_model_from_config(config=config, device=device)
