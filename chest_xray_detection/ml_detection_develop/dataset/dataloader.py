from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose

from chest_xray_detection.ml_detection_develop.configs.settings import (
    DATA_PATH,
    LABEL_MAPPING_DICT,
)
from chest_xray_detection.ml_detection_develop.dataset.detection_dataset import (
    DetectionDataset,
)


def get_train_val_dataloaders(
    annotation_filepath: str | Path,
    batch_size: int,
    train_transforms: Optional[Compose],
    val_transforms: Optional[Compose],
    val_size: float = 0.25,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Get training and validation data loaders from a stratified train/val split.

    Args:
        annotation_filepath (str | Path): Path to the CSV file containing annotations.
        batch_size (int): Batch size for DataLoader.
        train_transforms (Optional[Compose]): Transformations to apply to training data.
        val_transforms (Optional[Compose]): Transformations to apply to validation data.
        val_size (float, optional): Percentage of data to use for validation. Defaults to 0.25.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.

    Returns:
        tuple[DataLoader, DataLoader, dict]: Tuple containing train DataLoader, validation DataLoader,
                                             and dictionary representing the distribution of labels in the training set.
    """
    df_annotation = pd.read_csv(annotation_filepath)
    list_images = [
        DATA_PATH / i for i in df_annotation["Image Index"].unique().tolist()
    ]
    labels = [
        df_annotation[df_annotation["Image Index"] == idx]["Finding Label"].values
        for idx in df_annotation["Image Index"].unique()
    ]

    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(labels)

    train_images, val_images, train_labels, val_labels = train_test_split(
        list_images,
        one_hot_labels,
        test_size=val_size,
        random_state=42,
        stratify=None,
    )

    train_loader = get_single_dataloader(
        images_list=train_images,
        annotation_filepath=annotation_filepath,
        transforms=train_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_loader = get_single_dataloader(
        images_list=val_images,
        annotation_filepath=annotation_filepath,
        transforms=val_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    label_distribution = {
        phase: {
            LABEL_MAPPING_DICT.mapping.decoding[idx + 1]: sum(labels[:, idx])
            / len(labels)
            for idx in range(labels.shape[1])
        }
        for phase, labels in zip(["training", "validation"], [train_labels, val_labels])
    }

    return train_loader, val_loader, label_distribution


def get_single_dataloader(
    images_list: list[Path],
    annotation_filepath: str | Path,
    transforms: Optional[Compose],
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """
    Get a single DataLoader from image paths and annotations file.

    Args:
        images_list (list[Path]): List of image file paths.
        annotation_filepath (str | Path): Path to the CSV file containing annotations.
        transforms (Optional[Compose]): Transformations to apply to data.
        batch_size (int): Batch size for DataLoader.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = DetectionDataset(
        images_list=images_list,
        annotation_filepath=annotation_filepath,
        transforms=transforms,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    return dataloader
