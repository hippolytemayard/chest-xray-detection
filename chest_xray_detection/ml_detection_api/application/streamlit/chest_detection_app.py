import argparse
import io

import requests
import streamlit as st
from PIL import Image

from chest_xray_detection.ml_detection_api.configs.configs import INFERENCE_CONFIG
from chest_xray_detection.ml_detection_api.utils.visualization.plot import plot_json_predictions

DETECTION_ENDPOINT = "/api/pathology-detection"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Streamlit server for Chest X-ray Detection"
    )
    parser.add_argument(
        "-u", "--url", type=str, default="http://127.0.0.1:8000/", help="URL of the detection API"
    )
    args = parser.parse_args()
    return args


def get_elements_api_call(api_url, files):
    response = requests.post(url=f"{api_url}/{DETECTION_ENDPOINT}", files={"file": files[0]})
    return response.json()


def main():
    args = parse_args()

    st.set_page_config(
        page_title="Chest X-ray Object Detection",
        page_icon=":hospital:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Chest X-ray Pathology Detection 🩻")
    st.markdown(
        """
        This application allows you to upload a chest X-ray image and perform object detection to identify potential pathologies.
        The detected pathologies will be highlighted on the image.
        """
    )

    uploaded_file = st.file_uploader("Choose a Chest X-ray Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.markdown(f"**File uploaded:** {uploaded_file.name}")

        bytes_data = uploaded_file.getvalue()
        files = [("file", bytes_data)]

        st.markdown("#### Processing the image and detecting pathologies...")
        detections = get_elements_api_call(args.url, files)

        image = Image.open(io.BytesIO(bytes_data))

        detection_color = INFERENCE_CONFIG.MODELS.MULTICLASS_DETECTION.PATHOLOGY_COLORS
        plotly_chart = plot_json_predictions(
            image=image, detections=detections, detection_color=detection_color
        )

        st.plotly_chart(plotly_chart, use_container_width=True)


if __name__ == "__main__":
    main()
