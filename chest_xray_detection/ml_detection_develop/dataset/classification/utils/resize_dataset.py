import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

from chest_xray_detection.ml_detection_develop.configs.settings import DATA_PATH
from chest_xray_detection.ml_detection_develop.utils.files import make_exists


def resize_single_image(filename, output_folder, size):
    try:
        if filename.suffix in [".jpg", ".png"]:
            with Image.open(filename) as image:
                image = image.resize(size)
                output_path = output_folder / filename.name
                image.save(output_path)
        return f"Processed {filename}"
    except Exception as e:
        return f"Failed to process {filename}: {e}"


def resize_image(
    input_folder: str | Path, output_folder: str | Path, size: tuple = (224, 224)
) -> None:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    make_exists(output_folder)

    images_list = list(input_folder.iterdir())

    with ThreadPoolExecutor() as executor:
        # Utilisation de ThreadPoolExecutor pour redimensionner les images en parallèle
        results = list(
            tqdm(
                executor.map(
                    resize_single_image,
                    images_list,
                    [input_folder] * len(images_list),
                    [output_folder] * len(images_list),
                    [size] * len(images_list),
                ),
                total=len(images_list),
            )
        )

    # Affichez les résultats
    for result in results:
        print(result)


if __name__ == "__main__":
    input_folder = DATA_PATH
    output_folder = DATA_PATH.parent / "images_resized"

    resize_image(
        input_folder=input_folder,
        output_folder=output_folder,
        size=(224, 224),
    )
