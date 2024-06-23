import pytest
from fastapi.testclient import TestClient

from chest_xray_detection.ml_detection_api.application.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "API is ready!"


def test_chest_xray_analysis(client):

    image_path = "tests/unit/data/images/00000032_037.png"
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = client.post("/api/pathology-detection", files=files)

    assert response.status_code == 200
    predictions = response.json()
    assert isinstance(predictions, list)

    for prediction in predictions:
        assert "detection_classes" in prediction
        assert "detection_boxes" in prediction
