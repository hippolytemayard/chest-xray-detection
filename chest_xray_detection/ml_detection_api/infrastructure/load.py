import io

import numpy as np
from cv2 import COLOR_BGR2GRAY, IMREAD_GRAYSCALE, cvtColor, imdecode, imread
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from chest_xray_detection.ml_detection_api.configs.settings import logging


def load_image(image, filename: str | None) -> np.ndarray:
    if isinstance(image, bytes):
        if filename is not None and any(filename.lower().endswith(ext) for ext in ["png", "jpg"]):
            image = load_image_from_bytes(image)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError(f"Unrecognized image type {type(image).__name__}!")
    return image


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)

    return image_array


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image = pil_to_tensor(pic=image)

    return image


if __name__ == "__main__":

    path = "/home/ubuntu/data/images/00010366_000.png"
    with open(path, "rb") as file:
        image_bytes = file.read()
    image_array = load_image_from_bytes(image_bytes)
    print(image_array.shape)
