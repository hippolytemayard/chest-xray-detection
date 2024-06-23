import os

import pandas as pd
import pytest
from torch.utils.data import DataLoader

from chest_xray_detection.ml_detection_develop.dataset.detection.dataloader import (
    get_train_val_dataloaders,
)

ANNOTATION_FILEPATH = "test_annotations.csv"


@pytest.fixture(scope="module")
def create_mock_annotation_file():
    data = {
        "Image Index": [f"image_{i}.png" for i in range(5)],
        "Bbox [x": [50, 30, 70, 20, 60],
        "y": [60, 40, 80, 30, 70],
        "w": [100, 60, 120, 50, 110],
        "h": [200, 80, 140, 70, 150],
        "Finding Label": ["Pneumonia", "No Finding", "Pneumonia", "No Finding", "Pneumonia"],
    }
    df = pd.DataFrame(data)
    df.to_csv(ANNOTATION_FILEPATH, index=False)
    yield
    os.remove(ANNOTATION_FILEPATH)


def test_get_train_val_dataloaders(create_mock_annotation_file):
    train_loader, val_loader, label_distribution = get_train_val_dataloaders(
        annotation_filepath=ANNOTATION_FILEPATH,
        batch_size=2,
        train_transforms=None,
        val_transforms=None,
        val_size=0.4,
        num_workers=0,
        merge_classes=False,
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(label_distribution, dict)
    assert "training" in label_distribution
    assert "validation" in label_distribution
    assert len(train_loader.dataset) + len(val_loader.dataset) == 5


def test_label_distribution(create_mock_annotation_file):
    _, _, label_distribution = get_train_val_dataloaders(
        annotation_filepath=ANNOTATION_FILEPATH,
        batch_size=2,
        train_transforms=None,
        val_transforms=None,
        val_size=0.4,
        num_workers=0,
        merge_classes=False,
    )

    assert isinstance(label_distribution, dict)
    assert "training" in label_distribution
    assert "validation" in label_distribution
    assert isinstance(label_distribution["training"], dict)
    assert isinstance(label_distribution["validation"], dict)
