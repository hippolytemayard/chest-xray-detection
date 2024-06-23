from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from chest_xray_detection.ml_detection_develop.configs.settings import DATA_PATH


def visualize_ground_truth(image_path: str, csv_path: str) -> None:
    """
    Visualizes the ground truth bounding boxes on the image.

    Args:
        image_path (str): Path to the image file.
        csv_path (str): Path to the CSV file containing bounding box coordinates.

    CSV file format:
        The CSV file should contain the following columns: ['xmin', 'ymin', 'xmax', 'ymax', 'label']
    """
    image_path = Path(DATA_PATH / image_path)
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")

    df_annotation = pd.read_csv(csv_path)

    boxes_df = df_annotation[df_annotation["Image Index"] == image_path.name][
        ["Bbox [x", "y", "w", "h]"]
    ]
    boxes = boxes_df.values.astype(np.int16)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    labels = df_annotation[df_annotation["Image Index"] == image_path.name][
        ["Finding Label"]
    ].values.squeeze(-1)

    for row, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax, label = (
            row[0],
            row[1],
            row[2],
            row[3],
            label,
        )

        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        plt.text(
            xmin, ymin, label, fontsize=12, color="white", bbox=dict(facecolor="red", alpha=0.5)
        )

    plt.show()


# Example usage
# image_path = "00000032_037.png"
# csv_path = "/home/ubuntu/data/BBox_List_2017.csv"
# visualize_ground_truth(image_path, csv_path)
