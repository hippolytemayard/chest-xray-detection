import argparse
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from chest_xray_detection.ml_detection_develop.dataset.dataloader import (
    get_single_dataloader,
)
from chest_xray_detection.ml_detection_develop.dataset.transforms.detection.utils import (
    instantiate_transforms_from_config,
)
from chest_xray_detection.ml_detection_develop.metrics.utils import (
    instantiate_metrics_from_config,
)
from chest_xray_detection.ml_detection_develop.models.faster_rcnn import (
    get_faster_rcnn,
)
from chest_xray_detection.ml_detection_develop.optimizer import optimizer_dict
from chest_xray_detection.ml_detection_develop.train.utils.training_loop import (
    training_loop,
)
from chest_xray_detection.ml_detection_develop.train.utils.evaluation_loop import (
    evaluation_loop,
)
from chest_xray_detection.ml_detection_develop.utils.files import (
    load_yaml,
    make_exists,
)


def cross_validation_training_from_config(
    config: OmegaConf,
    device: Union[str, torch.device],
) -> None:
    """
    Perform cross-validation training based on the provided configuration.

    Args:
        config (OmegaConf): Configuration object containing training
            parameters.
        device (Union[str, torch.device]): Device to use for training
            (e.g., 'cpu' or 'cuda').

    Returns:
        None
    """

    path_train_label = Path(config.TRAINING.DATASET.PATH_LABELS)
    images_list = Path(config.TRAINING.DATASET.IMAGES_DIR).glob("*.jpg")
    images_list = sorted(list(images_list))

    labels = read_txt_object(
        path_train_label,
    )
    labels = [int(label) for label in labels]

    train_transforms = instantiate_transforms_from_config(
        transform_config=config.TRAINING.DATASET.TRANSFORMS.TRAINING
    )
    val_transforms = instantiate_transforms_from_config(
        transform_config=config.TRAINING.DATASET.TRANSFORMS.VALIDATION
    )

    skf = StratifiedKFold(n_splits=config.TRAINING.DATASET.KFOLD)

    logging.info(f"Starting {config.TRAINING.DATASET.KFOLD}-Fold training")

    for fold_id, (train_index, val_index) in enumerate(skf.split(images_list, labels)):

        train_images, val_images = (
            np.asarray(images_list)[train_index].tolist(),
            np.asarray(images_list)[val_index].tolist(),
        )
        train_labels, val_labels = (
            np.asarray(labels)[train_index].tolist(),
            np.asarray(labels)[val_index].tolist(),
        )

        train_labels_distribution = {
            0: len(train_labels) - sum(train_labels),
            1: sum(train_labels),
        }

        logging.info(f"Train dataset classes distribution: {train_labels_distribution}")

        train_loader = get_single_dataloader(
            images_list=train_images,
            labels=train_labels,
            transform=train_transforms,
            batch_size=config.TRAINING.BATCH_SIZE,
        )
        val_loader = get_single_dataloader(
            images_list=val_images,
            labels=val_labels,
            transform=val_transforms,
            batch_size=config.TRAINING.BATCH_SIZE,
        )

        logging.info(f"Training Fold {fold_id}")

        train_one_fold(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            train_labels_distribution=train_labels_distribution,
            fold_id=fold_id,
            device=device,
        )


def train_one_fold(
    config: OmegaConf,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels_distribution: dict[int, int],
    fold_id: int,
    device: Union[str, torch.device],
) -> None:
    """
    Perform training for a single fold.

    Args:
        config (OmegaConf): Configuration object containing training parameters.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        train_labels_distribution (Dict[int, int]): Distribution of training labels.
        fold_id (int): Fold ID.
        device (torch.device): Device to use for training.

    Returns:
        None
    """

    writer = (
        SummaryWriter(config.TRAINING.TENSORBOARD_DIR)
        if config.TRAINING.ENABLE_TENSORBOARD
        else None
    )

    model = get_faster_rcnn(
        pretrained=config.TRAINING.PRETRAINED,
        num_classes=config.TRAINING.DATASET.NUM_CLASSES,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"The model has {trainable_params} trainable parameters")

    optimizer = optimizer_dict[config.TRAINING.OPTIMIZER](
        params=model.parameters(), lr=config.TRAINING.LEARNING_RATE
    )

    metrics_collection = instantiate_metrics_from_config(
        metrics_config=config.VALIDATION.METRICS
    ).to(device)

    best_hter = float("inf")

    for epoch in range(1, config.TRAINING.EPOCHS + 1):

        logging.info(f"EPOCH {epoch}")
        training_loop(
            model=model,
            loader=train_loader,
            criterion=criterion,
            with_logits=config.TRAINING.WITH_LOGITS,
            metrics_collection=metrics_collection.clone(),
            epoch=epoch,
            optimizer=optimizer,
            writer=writer,
            device=device,
        )

        dict_metrics = evaluation_loop(
            model=model,
            loader=val_loader,
            epoch=epoch,
            criterion=criterion,
            with_logits=config.TRAINING.WITH_LOGITS,
            metrics_collection=metrics_collection.clone(),
            writer=writer,
            device=device,
        )

        if hter < best_hter:
            logging.info(f"Validation | model improved from {best_hter} to {hter} | saving model")
            best_hter = hter
            save_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "val_metrics": {"far": far, "frr": frr, "hter": hter},
            }
            torch.save(
                save_dict,
                Path(config.TRAINING.PATH_MODEL) / f"best_model_fold{fold_id}.pt",
            )

            best_metrics = dict_metrics

    logging.info(f"best metrics : {best_metrics}")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model using configuration file")
    parser.add_argument(
        "--config",
        type=str,
        default="chest_xray_detection/configs/training/training_faster_rcnn.yaml",
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

    cross_validation_training_from_config(
        config=config,
        device=device,
    )
