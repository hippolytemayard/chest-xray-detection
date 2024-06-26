from pathlib import Path
from typing import Union, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose


class ClassificationDataset(Dataset):
    """
    Custom dataset class for classification tasks.
    """

    def __init__(
        self,
        images_path: Union[str, Path],
        images_list: List[str],
        annotation_filepath: Union[str, Path],
        transforms: Union[Compose, None] = None,
    ) -> None:
        """
        Constructor for the ClassificationDataset class.

        Args:
            images_path (Union[str, Path]): Path to the directory containing images.
            images_list (List[str]): List of file names of the images.
            annotation_filepath (Union[str, Path]): Path to the CSV file containing image annotations.
            transforms (Union[Compose, None], optional): Optional transformations to apply to images. Defaults to None.
        """
        self.images_path = Path(images_path)
        self.images_list = images_list
        self.df_annotation = pd.read_csv(annotation_filepath)
        self.one_hot_matrix = self.df_annotation["Finding Labels"].str.get_dummies(sep="|")
        self.transforms = transforms
        self.class_list = self.one_hot_matrix.columns

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.images_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an image and its corresponding label from the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image tensor and its corresponding label.
        """
        image_path = self.images_path / self.images_list[idx]
        image = read_image(str(image_path)).to(torch.float) / 255.0

        # Ensure the image has 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3, :, :]

        if self.transforms is not None:
            image = self.transforms(image)

        label_index = self.df_annotation.index[self.df_annotation["Image Index"] == image_path.name]
        label = self.one_hot_matrix.loc[label_index].to_numpy()
        label = torch.from_numpy(label).squeeze(0)

        return image, label
