from chest_xray_detection.ml_detection_api.configs.configs import INFERENCE_CONFIG
from chest_xray_detection.ml_detection_api.configs.settings import logging
from chest_xray_detection.ml_detection_api.domain.wrappers.multiclass_detection_wrapper import (
    MultiClassDetectionWrapper,
)

MULTI_DETECTION_MODEL_CONFIG = INFERENCE_CONFIG.MODELS.MULTICLASS_DETECTION

logging.info(f"Loading models")
try:
    multi_class_detection_model = MultiClassDetectionWrapper.load(
        config=MULTI_DETECTION_MODEL_CONFIG
    )
    logging.info(f"Models loaded")

except Exception as e:
    logging.info(f"Error initializing models")
    raise e
