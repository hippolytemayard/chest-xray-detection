import logging
from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN

from chest_xray_detection.ml_detection_develop.models.detection.faster_rcnn import (
    get_faster_rcnn,
    get_faster_rcnn_mobilenet_backbone,
    get_faster_rcnn_resnet50_backbone,
)


def test_get_faster_rcnn_resnet50_backbone():
    num_classes = 9
    pretrained = True
    trainable_backbone_layers = 2
    box_score_thresh = 0.05

    model = get_faster_rcnn_resnet50_backbone(
        num_classes=num_classes,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=box_score_thresh,
    )

    assert isinstance(model, FasterRCNN)
    assert model.backbone.__class__.__name__ == "BackboneWithFPN"
    assert model.roi_heads.box_predictor.cls_score.in_features == 1024


def test_get_faster_rcnn_mobilenet_backbone():
    num_classes = 15
    pretrained = True
    box_score_thresh = 0.5

    model = get_faster_rcnn_mobilenet_backbone(
        num_classes=num_classes,
        pretrained=pretrained,
        box_score_thresh=box_score_thresh,
    )

    print(model.backbone[-1])

    assert isinstance(model, FasterRCNN)
    assert model.backbone.__class__.__name__ == "Sequential"
    assert model.backbone.out_channels == 1280


def test_get_faster_rcnn():
    num_classes = 9
    backbone = "resnet50"
    pretrained = True
    trainable_backbone_layers = 2
    box_score_thresh = 0.05

    model = get_faster_rcnn(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=box_score_thresh,
    )

    assert isinstance(model, FasterRCNN)
    assert model.backbone.__class__.__name__ == "BackboneWithFPN"
    assert model.roi_heads.box_predictor.cls_score.in_features == 1024


def test_get_faster_rcnn_with_loading():
    num_classes = 9
    backbone = "resnet50"
    pretrained = True
    trainable_backbone_layers = 2
    box_score_thresh = 0.05

    model = get_faster_rcnn(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=box_score_thresh,
    )

    logging.info(f"Load state dict")
    path_model = Path("experiments/experiment_300/saved_models") / "best_model.pt"
    model_state_dict = torch.load(path_model)
    model.load_state_dict(model_state_dict["model"])
    logging.info(f"Loading {path_model}")

    assert model is not None
    assert isinstance(model, FasterRCNN)


if __name__ == "__main__":
    import pytest

    pytest.main()
