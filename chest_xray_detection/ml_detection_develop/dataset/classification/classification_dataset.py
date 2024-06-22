from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose


# TODO : clean code
class ClassificationDataset(Dataset):
    """
    Custom dataset class for classification tasks.
    """

    def __init__(
        self,
        images_path: str | Path,
        images_list: list,
        annotation_filepath: str | Path,
        transforms: Union[Compose, None] = None,
    ) -> None:
        """
        Constructor for the ClassificationDataset class.

        Args:
            images_list (list[str]): List of file paths to images.
            images_labels (Union[list[int],None]): List of corresponding
                labels for images. if None inference mode.
            transforms (Union[Compose, None], optional): Optional
                transformations to apply to images. Defaults to None.
            inference (bool): Inference mode dataset.
        """
        self.images_path = Path(images_path)
        self.images_list = images_list
        self.df_annotation = pd.read_csv(annotation_filepath)
        self.one_hot_matrix = self.df_annotation["Finding Labels"].str.get_dummies(sep="|")
        self.transforms = transforms

        self.class_list = self.one_hot_matrix.columns

    def __len__(self) -> int:
        return len(self.images_list)

    def _encode_label(self, label, classes_list: list):

        target = torch.zeros(len(classes_list))
        for l in label:
            idx = classes_list.index(l)
            target[idx] = 1
        return target

    def __getitem__(self, idx) -> Union[tuple[torch.Tensor, int], torch.Tensor]:
        """
        Retrieves an image and its corresponding label from the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple[torch.Tensor, int]: Tuple containing the image tensor and
                its corresponding label or only tensor if inference mode.
        """
        image_path = self.images_path / self.images_list[idx]

        image = read_image(str(image_path))
        image = image.to(torch.float) / 255.0

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        elif image.shape[0] == 4:
            image = image[:3, :, :]

        if self.transforms is not None:
            image = self.transforms(image)

        label_index = self.df_annotation.index[self.df_annotation["Image Index"] == image_path.name]
        label = self.one_hot_matrix.loc[label_index].to_numpy()
        label = torch.from_numpy(label).squeeze(0)

        # label = self._encode_label(label, classes_list=list(classes_list))

        return image, label
