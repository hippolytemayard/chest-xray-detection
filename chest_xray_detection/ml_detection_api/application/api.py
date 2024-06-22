__version__ = "0.0.1"

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from numpy import imag

from chest_xray_detection.ml_detection_api.configs.settings import SERVER_NAME, logging
from chest_xray_detection.ml_detection_api.application.load_models import (
    multi_class_detection_model,
)
from chest_xray_detection.ml_detection_api.domain.inference.inference_pipeline import (
    run_xray_detection,
)
from chest_xray_detection.ml_detection_api.utils.objects.base_objects import BBoxPrediction

logging.info(f"-----> Running Server {SERVER_NAME} ...")

app = FastAPI(
    title=SERVER_NAME,
    description="",
    version=__version__,
)


# Healthcheck endpoint
@app.get("/")
def health_check():
    return "API is ready!"


EXCLUDE_VERIFICATION_PATHS = ["/", "/api/"]  # TODO: remove this when fixed


@app.post("/api/pathology-detection", response_model=list[BBoxPrediction])
def chest_xray_analysis(file: UploadFile = File(...)) -> list[BBoxPrediction]:
    print(file)
    image = file.file.read()

    return run_xray_detection(image=image, detection_model=multi_class_detection_model, debug=True)