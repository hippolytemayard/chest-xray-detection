from typing import Optional
from chest_xray_detection.ml_detection_api.domain.wrappers.multiclass_detection_wrapper import (
    MultiClassDetectionWrapper,
)
from chest_xray_detection.ml_detection_api.configs.settings import logging
from chest_xray_detection.ml_detection_api.infrastructure.load import load_image_from_bytes
from chest_xray_detection.ml_detection_api.utils.objects.base_objects import BBoxPrediction


def run_xray_detection(
    image: Optional[bytes],
    detection_model: Optional[MultiClassDetectionWrapper],
    debug: bool = False,
) -> list[BBoxPrediction]:

    image = load_image_from_bytes(image_bytes=image)
    detected_pathology = detection_model.__call__(image=image)

    if debug:
        logging.info("Detected pathologies")
        for detection in detected_pathology:
            logging.info(f">> {detection}")

    return detected_pathology
