import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from chest_xray_detection.ml_detection_develop.dataset.classification.dataloader import (
    get_train_val_dataloaders,
)
from chest_xray_detection.ml_detection_develop.dataset.transforms.utils import (
    instantiate_transforms_from_config,
)
from chest_xray_detection.ml_detection_develop.losses import losses_dict
from chest_xray_detection.ml_detection_develop.metrics.classification.utils import (
    instantiate_metrics_from_config,
)
from chest_xray_detection.ml_detection_develop.models.classification.models import (
    get_mobilenet_v2_architecture,
)
from chest_xray_detection.ml_detection_develop.optimizer import optimizer_dict
from chest_xray_detection.ml_detection_develop.train.classification.utils.training_loop import (
    training_loop,
)
from chest_xray_detection.ml_detection_develop.train.classification.utils.validation_loop import (
    validation_loop,
)
from chest_xray_detection.ml_detection_develop.utils.files import load_yaml, make_exists


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

    train_transforms = instantiate_transforms_from_config(
        transform_config=config.TRAINING.DATASET.TRANSFORMS.TRAINING, task="classification"
    )
    val_transforms = instantiate_transforms_from_config(
        transform_config=config.TRAINING.DATASET.TRANSFORMS.VALIDATION, task="classification"
    )

    train_loader, val_loader, label_distribution = get_train_val_dataloaders(
        images_path=config.TRAINING.DATASET.IMAGES_PATH,
        annotation_filepath=config.TRAINING.DATASET.PATH_LABELS,
        annotation_box_filepath=config.TRAINING.DATASET.PATH_BOXES,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=config.TRAINING.BATCH_SIZE,
    )

    logging.info(f"Train dataset classes distribution: {label_distribution}")

    label_weights = torch.tensor(list(label_distribution["training"].values())).to(device=device)
    label_weights = label_weights.sum() / label_weights

    model = get_mobilenet_v2_architecture(
        architecture=config.TRAINING.BACKBONE,
        n_classes=len(label_distribution["training"].keys()),
        fine_tune=config.TRAINING.FINE_TUNE,
        pretrained=config.TRAINING.PRETRAINED,
    ).to(device)

    logging.info(f"Dataset has {len(label_distribution['training'].keys())} classes")

    logging.info(f"Load state dict")
    path_model = "/home/ubuntu/code/chest-xray-detection/experiments_classification/experiment_1/saved_models/best_model.pt"
    model_state_dict = torch.load(path_model)
    model.load_state_dict(model_state_dict["model"])
    logging.info(f"Loading {path_model}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"The model has {trainable_params} trainable parameters")

    criterion = losses_dict[config.TRAINING.LOSS](reduction="none")

    optimizer = optimizer_dict[config.TRAINING.OPTIMIZER](
        params=model.parameters(), lr=config.TRAINING.LEARNING_RATE
    )
    optimizer.load_state_dict(model_state_dict["opt"])
    logging.info(f"Loading optimizer {path_model}")

    metrics_collection = instantiate_metrics_from_config(
        metrics_config=config.VALIDATION.METRICS
    ).to(device)

    best_auc = -1

    for epoch in range(1, config.TRAINING.EPOCHS + 1):

        logging.info(f"EPOCH {epoch}")
        training_loop(
            model=model,
            loader=train_loader,
            criterion=criterion,
            label_weights=label_weights,
            with_logits=config.TRAINING.WITH_LOGITS,
            metrics_collection=metrics_collection,
            epoch=epoch,
            optimizer=optimizer,
            writer=writer,
            device=device,
            log_interval=30,
        )

        dict_metrics = validation_loop(
            model=model,
            loader=val_loader,
            epoch=epoch,
            criterion=criterion,
            label_weights=label_weights,
            with_logits=config.TRAINING.WITH_LOGITS,
            metrics_collection=metrics_collection,
            writer=writer,
            device=device,
        )

        auc = dict_metrics[config.VALIDATION.SAVING_METRIC]

        if auc > best_auc:
            logging.info(f"Validation | model improved from {best_auc} to {auc} | saving model")
            best_auc = auc
            save_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "val_metrics": {
                    "auc": auc,
                    "accuracy": dict_metrics["MultilabelAccuracy"],
                    "precision": dict_metrics["MultilabelPrecision"],
                    "recall": dict_metrics["MultilabelRecall"],
                },
            }
            torch.save(save_dict, Path(config.TRAINING.PATH_MODEL) / "best_model.pt")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using configuration file")
    parser.add_argument(
        "--config",
        type=str,
        default="chest_xray_detection/ml_detection_develop/configs/training/classification/training_mobilenet_v2.yaml",
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
