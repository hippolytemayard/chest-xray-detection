import argparse
import io

import requests
import streamlit as st
from PIL import Image

from chest_xray_detection.ml_detection_api.configs.configs import INFERENCE_CONFIG
from chest_xray_detection.ml_detection_api.utils.visualization.plot import plot_json_predictions

DETECTION_ENDPOINT = "/api/pathology-detection"


def parse_args():
    parser = argparse.ArgumentParser(description="launch a streamlit server")
    parser.add_argument("-u", "--url", type=str, default="http://127.0.0.1:8000/")

    args = parser.parse_args()
    return args


def get_elements_api_call(_args, _files):
    return requests.post(url=f"{_args.url}/{DETECTION_ENDPOINT}", files={"file": _files[0]}).json()


def main():
    args = parse_args()

    st.set_page_config(
        page_title="Chest X-ray Object Detection",
        page_icon=":hospital:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    uploaded_file = st.file_uploader("Choose a chest X-Ray")
    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        bytes_data = uploaded_file.getvalue()
        files = [("files", bytes_data)]

        detections = get_elements_api_call(args, files)

        image = Image.open(io.BytesIO(bytes_data))

        detection_color = INFERENCE_CONFIG.MODELS.MULTICLASS_DETECTION.PATHOLOGY_COLORS
        plotly_chart = plot_json_predictions(
            image=image, detections=detections, detection_color=detection_color
        )
        st.plotly_chart(plotly_chart, use_container_width=True)


if __name__ == "__main__":

    main()
