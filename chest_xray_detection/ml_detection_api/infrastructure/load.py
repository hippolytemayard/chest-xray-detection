import io
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from bytes and convert it to a NumPy array.

    Args:
        image_bytes (bytes): Bytes representing the image.

    Returns:
        np.ndarray: NumPy array representing the image.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = pil_to_tensor(pic=image)
    return image_tensor


if __name__ == "__main__":
    path = "/home/ubuntu/data/images/00010366_000.png"
    with open(path, "rb") as file:
        image_bytes = file.read()
    image_tensor = load_image_from_bytes(image_bytes)
    print(image_tensor.shape)
