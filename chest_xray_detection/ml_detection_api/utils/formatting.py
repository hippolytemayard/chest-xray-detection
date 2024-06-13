from typing import Literal, Optional

from chest_xray_detection.ml_detection_api.utils.geometry import (
    get_box_object,
    get_coords_from_polygon,
)
from chest_xray_detection.ml_detection_api.utils.objects.base_objects import BBoxPrediction, Coord2D
from chest_xray_detection.ml_detection_api.utils.objects.prediction import ObjectDetectionFormat


def convert_to_api_format(
    output: ObjectDetectionFormat, classes_list: list, model_name: Literal["multiclass_detection"]
) -> list[BBoxPrediction]:

    detections_list = []
    for box_, class_, score_ in zip(
        output.boxes.tolist(), output.labels.tolist(), output.scores.tolist()
    ):
        detection = format_detection(
            model_name=model_name,
            classes_list=classes_list,
            box=box_,
            classe=class_,
            score=score_,
        )
        detections_list.append(detection)

    return detections_list


def format_detection(
    model_name: Literal["multiclass_detection"],
    classes_list: list[str],
    box: list[float],
    classe: Optional[str] = None,
    score: float = None,
):

    label = None
    classe_int = None

    if classe is not None:
        classe_int = int(classe)
        label = classes_list[classe_int]
        label = str(label)

    if model_name == "multiclass_detection":
        return BBoxPrediction(
            detection_patology=label,
            detection_boxes=[box[0], box[1], box[2], box[3]],
            detection_scores=score,
            detection_classes=classe_int,
            detection_poly=get_coords_from_polygon(
                get_box_object([box[0], box[1], box[2], box[3]])
            ),
        )

    else:
        raise ValueError(f"Model {model_name} is not supported.")
