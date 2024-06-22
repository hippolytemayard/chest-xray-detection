from torch import nn
from torchvision.models import mobilenet_v2, MobileNetV2


def get_mobilenet_v2_architecture(
    architecture: str = "mobilenet_v2",
    n_classes: int = 1,
    fine_tune: bool = True,
    pretrained: bool = True,
) -> MobileNetV2:
    """
    Get a ResNet architecture with specified parameters.

    Args:
        architecture (str): ResNet architecture name.
            ("resnet18", "resnet34", or "resnet50")
        n_classes (int): Number of output classes.
        fine_tune (bool): Whether to fine-tune the model.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        ResNet: ResNet model with specified architecture and modifications.
    """
    if architecture == "mobilenet_v2":
        model = mobilenet_v2(weights="DEFAULT" if pretrained else None, progress=True)
        # backbone = torchvision.models.mobilenet_v2(weights=weights)
    else:
        raise ValueError(
            "Unsupported ResNet architecture. "
            "Please choose from 'resnet18', 'resnet34', or 'resnet50'."
        )

    for param in model.parameters():
        param.requires_grad = True

    # if fine_tune:
    #    for param in model.layer4.parameters():
    #        param.requires_grad = True
    #

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, n_classes),
    )

    return model
