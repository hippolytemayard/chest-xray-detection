import logging

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FastRCNNPredictor,
)
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import ResNet50_Weights


def get_faster_rcnn(
    num_classes: int,
    backbone: str = "resnet50",
    pretrained: bool = True,
    trainable_backbone_layers: int = 2,
    box_score_thresh: float = 0.05,
) -> FasterRCNN:
    """
    Creates a Faster R-CNN model with the specified backbone architecture.

    Args:
        num_classes (int): Number of classes for the object detection model.
        backbone (str): The backbone architecture to use ('resnet50' or 'mobilenet').
        pretrained (bool): If True, use pretrained weights for the model and backbone.
        trainable_backbone_layers (int): Number of trainable layers in the backbone.
        box_score_thresh (float): Minimum score for the predicted boxes to be considered.

    Returns:
        FasterRCNN: A Faster R-CNN model with the specified backbone.

    Raises:
        ValueError: If an unsupported backbone architecture is specified.
    """
    if backbone == "resnet50":
        model = get_faster_rcnn_resnet50_backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            box_score_thresh=box_score_thresh,
        )
    elif backbone == "mobilenet":
        model = get_faster_rcnn_mobilenet_backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            box_score_thresh=box_score_thresh,
        )
    else:
        raise ValueError("Unsupported Backbone architecture.")

    return model


def get_faster_rcnn_resnet50_backbone(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 2,
    box_score_thresh: float = 0.05,
) -> FasterRCNN:
    """
    Creates a Faster R-CNN model with a ResNet-50-FPN backbone.

    Args:
        num_classes (int): Number of classes for the object detection model.
        pretrained (bool): If True, use pretrained weights for the model and backbone.
        trainable_backbone_layers (int): Number of trainable layers in the backbone.
        box_score_thresh (float): Minimum score for the predicted boxes to be considered.

    Returns:
        FasterRCNN: A Faster R-CNN model with a ResNet-50-FPN backbone.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None

    logging.info(f"weights : {weights}")
    logging.info(f"weights_backbone : {weights_backbone}")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=box_score_thresh,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_faster_rcnn_mobilenet_backbone(
    num_classes: int,
    pretrained: bool = True,
    box_score_thresh: float = 0.5,
) -> FasterRCNN:
    """
    Creates a Faster R-CNN model with a MobileNetV2 backbone.

    Args:
        num_classes (int): Number of classes for the object detection model.
        pretrained (bool): If True, use pretrained weights for the model and backbone.
        box_score_thresh (float): Minimum score for the predicted boxes to be considered.

    Returns:
        FasterRCNN: A Faster R-CNN model with a MobileNetV2 backbone.
    """
    weights = "DEFAULT" if pretrained else None

    backbone = torchvision.models.mobilenet_v2(weights=weights).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_score_thresh=box_score_thresh,
    )

    return model
