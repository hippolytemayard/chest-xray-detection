import pytest
from pathlib import Path
import torch
from torchvision.transforms import Compose
from chest_xray_detection.ml_detection_develop.dataset.detection.detection_dataset import (
    DetectionDataset,
)

ANNOTATION_FILE = Path("./tests/unit/data/box_annotation.csv")
IMAGE_DIR = Path("./tests/unit/data/images/")


@pytest.fixture(scope="module")
def create_detection_dataset():
    dataset = DetectionDataset(
        images_list=list(IMAGE_DIR.glob("*.png")),
        annotation_filepath=ANNOTATION_FILE,
        transforms=None,
        merge_classes=False,
    )
    return dataset


def test_detection_dataset_length(create_detection_dataset):
    dataset = create_detection_dataset
    assert len(dataset) == 4


def test_detection_dataset_item(create_detection_dataset):
    dataset = create_detection_dataset
    image, target = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, dict)
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
    # Checking the box
    assert torch.all(
        torch.eq(target["boxes"].data, torch.tensor([[601, 595, 816, 775]], dtype=torch.float32))
    ).item()
    # Checking label "Pneumonia"
    assert torch.all(target["labels"] == torch.tensor([7], dtype=torch.int64))
