import pandas as pd
import pytest
from torch.utils.data import DataLoader
from torchvision import transforms

from chest_xray_detection.ml_detection_develop.dataset.classification.dataloader import (
    get_single_dataloader,
    get_train_val_dataloaders,
)


@pytest.fixture
def setup_files(tmpdir):
    # Create temporary CSV files with sample data for testing
    annotation_filepath = tmpdir.join("annotations.csv")
    annotation_box_filepath = tmpdir.join("annotations_box.csv")

    df_annotations = pd.DataFrame(
        {
            "Image Index": ["img1.png", "img2.png", "img3.png", "img4.png"],
            "Finding Labels": ["A|B", "B", "C", "A|C"],
        }
    )
    df_annotations.to_csv(annotation_filepath, index=False)

    df_annotations_box = pd.DataFrame({"Image Index": ["img4.png"]})
    df_annotations_box.to_csv(annotation_box_filepath, index=False)

    images_path = tmpdir.mkdir("images")
    for img in ["img1.png", "img2.png", "img3.png", "img4.png"]:
        images_path.join(img).write("fake image content")

    return images_path, annotation_filepath, annotation_box_filepath


def test_get_train_val_dataloaders(setup_files):
    images_path, annotation_filepath, annotation_box_filepath = setup_files

    train_transforms = transforms.Compose([transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.ToTensor()])

    train_loader, val_loader, label_distribution = get_train_val_dataloaders(
        images_path=images_path,
        annotation_filepath=annotation_filepath,
        annotation_box_filepath=annotation_box_filepath,
        batch_size=2,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        val_size=0.5,
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    assert "training" in label_distribution
    assert "validation" in label_distribution

    assert len(train_loader) > 0
    assert len(val_loader) > 0

    for phase in ["training", "validation"]:
        assert all(isinstance(key, int) for key in label_distribution[phase].keys())
        assert all(isinstance(value, float) for value in label_distribution[phase].values())


def test_get_single_dataloader(setup_files):
    images_path, annotation_filepath, _ = setup_files

    transform = transforms.Compose([transforms.ToTensor()])

    dataloader = get_single_dataloader(
        images_path=images_path,
        images_list=["img1.png", "img2.png"],
        annotation_filepath=annotation_filepath,
        transform=transform,
        batch_size=2,
        shuffle=True,
    )

    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) > 0
