from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import Compose

from chest_xray_detection.ml_detection_develop.configs.settings import (
    LABEL_MAPPING_DICT,
)


class DetectionDataset(Dataset):
    """
    Custom dataset class for object detection tasks.
    """

    def __init__(
        self,
        images_list: list[Path],
        annotation_filepath: str | Path,
        transforms: Optional[Compose] = None,
        merge_classes: bool = False,
    ) -> None:
        """
        Initialize the DetectionDataset.

        Args:
            images_list (list[Path]): List of image file paths.
            annotation_filepath (str | Path): Path to the CSV file containing annotations.
            transforms (Optional[Compose]): Transformations to be applied to the images.
        """
        self.images_list = images_list
        self.df_annotation = pd.read_csv(annotation_filepath)
        self.transforms = transforms
        self.merge_classes = merge_classes
        self.classes = [class_ for class_ in LABEL_MAPPING_DICT.mapping.encoding.keys()]

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.images_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Retrieve the image and its corresponding target at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: tuple containing the image tensor and the target dictionary.
        """
        image_path = self.images_list[idx]
        image = self._load_image(image_path)
        boxes, labels = self._load_target(image_path.name, image)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """
        Load and preprocess the image.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image = read_image(str(image_path))

        if image.shape[0] == 4:
            image = image[:3, :, :]

        if image.shape[0] == 3:
            image = F.rgb_to_grayscale(image)

        image = image.float() / 255.0

        return tv_tensors.Image(image)

    def _load_target(
        self, image_name: str, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess the target data for a given image.

        Args:
            image_name (str): Name of the image file.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple containing the bounding boxes and labels tensors.
        """
        boxes_df = self.df_annotation[self.df_annotation["Image Index"] == image_name][
            ["Bbox [x", "y", "w", "h]"]
        ]
        boxes = self._process_boxes(boxes=boxes_df.values, image=image)

        labels_df = self.df_annotation[self.df_annotation["Image Index"] == image_name][
            ["Finding Label"]
        ]
        labels = self._process_labels(labels=labels_df.values.squeeze(-1))

        return boxes, labels

    def _process_boxes(self, boxes: np.ndarray, image: torch.Tensor) -> torch.Tensor:
        """
        Process the bounding boxes.

        Args:
            boxes (np.ndarray): Array of bounding boxes.

        Returns:
            torch.Tensor: Tensor of bounding boxes.
        """
        boxes = boxes.astype(np.int16)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        boxes = torch.from_numpy(boxes).float()
        return tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=image.shape[-2:],
        )

    def _process_labels(self, labels: np.ndarray) -> torch.Tensor:
        """
        Process the labels.

        Args:
            labels (np.ndarray): Array of label strings.

        Returns:
            torch.Tensor: Tensor of label integers.
        """
        if not self.merge_classes:
            labels = [LABEL_MAPPING_DICT.mapping.encoding[label] for label in labels]
        else:
            labels = [1 for _ in labels]

        return torch.as_tensor(labels, dtype=torch.int64)
