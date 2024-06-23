from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from chest_xray_detection.ml_detection_develop.dataset.classification.classification_dataset import (
    ClassificationDataset,
)


def get_train_val_dataloaders(
    images_path: Union[str, Path],
    annotation_filepath: Union[str, Path],
    annotation_box_filepath: Union[str, Path],
    batch_size: int,
    train_transforms: transforms.Compose,
    val_transforms: transforms.Compose,
    val_size: float = 0.25,
) -> tuple[DataLoader, DataLoader, dict[int, float]]:
    """
    Function to get training and validation data loaders from a stratified
    train/val split.

    Args:
        images_path (Union[str, Path]): Path to the directory containing images.
        annotation_filepath (Union[str, Path]): Path to the CSV file containing image annotations.
        annotation_box_filepath (Union[str, Path]): Path to the CSV file containing image box annotations.
        batch_size (int): Batch size for DataLoader.
        train_transforms (transforms.Compose): Transformations to apply to training data.
        val_transforms (transforms.Compose): Transformations to apply to validation data.
        val_size (float, optional): Percentage of data to use for validation. Defaults to 0.25.

    Returns:
        tuple[DataLoader, DataLoader, dict[int, float]]: Tuple containing
            train DataLoader, validation DataLoader, and dictionary
            representing the distribution of labels in the training set.
    """
    df_annotation = pd.read_csv(annotation_filepath)
    box_annotation = pd.read_csv(annotation_box_filepath)

    images_list = df_annotation["Image Index"].to_list()

    discarded_images = box_annotation["Image Index"].to_list()
    images_list = [image for image in images_list if image not in discarded_images]

    labels = (
        df_annotation[df_annotation["Image Index"].isin(images_list)]["Finding Labels"]
        .str.get_dummies(sep="|")
        .to_numpy()
    )

    train_images, val_images, train_labels, val_labels = train_test_split(
        images_list,
        labels,
        test_size=val_size,
        random_state=42,
        stratify=None,
    )

    train_loader = get_single_dataloader(
        images_path=images_path,
        images_list=train_images,
        annotation_filepath=annotation_filepath,
        transform=train_transforms,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = get_single_dataloader(
        images_path=images_path,
        images_list=val_images,
        annotation_filepath=annotation_filepath,
        transform=val_transforms,
        batch_size=batch_size,
        shuffle=False,
    )

    label_distribution = {
        phase: {idx + 1: sum(labels[:, idx]) / len(labels) for idx in range(labels.shape[1])}
        for phase, labels in zip(["training", "validation"], [train_labels, val_labels])
    }

    return train_loader, val_loader, label_distribution


def get_single_dataloader(
    images_path: Union[str, Path],
    images_list: list,
    annotation_filepath: Union[str, Path],
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """
    Function to get a single DataLoader from image path and labels files.

    Args:
        images_path (Union[str, Path]): Path to the directory containing images.
        images_list (list): List of image file paths.
        annotation_filepath (Union[str, Path]): Path to the CSV file containing image annotations.
        transform (transforms.Compose): Transformations to apply to data.
        batch_size (int): Batch size for DataLoader.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = ClassificationDataset(
        images_path=images_path,
        images_list=images_list,
        annotation_filepath=annotation_filepath,
        transforms=transform,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return dataloader
