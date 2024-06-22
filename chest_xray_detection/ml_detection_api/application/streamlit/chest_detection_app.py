import argparse
import csv
import io
import os
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

from chest_xray_detection.ml_detection_api.utils.visualization.plot import plot_json_predictions
from chest_xray_detection.ml_detection_api.configs.configs import INFERENCE_CONFIG

DETECTION_ENDPOINT = "/api/pathology-detection"


def parse_args():
    parser = argparse.ArgumentParser(description="launch a streamlit server")
    parser.add_argument("-u", "--url", type=str, default="http://127.0.0.1:8000/")
    parser.add_argument("-d", "--datapath", type=str, default="/tmp/streamlit")
    args = parser.parse_args()
    return args


# @st.cache_resource
def get_elements_api_call(_args, _files):

    return requests.post(url=f"{_args.url}/{DETECTION_ENDPOINT}", files={"file": _files[0]}).json()


def main():
    args = parse_args()
    home_path = os.getenv("HOME")
    csv_path = f"{home_path}{args.datapath}"
    Path(csv_path).mkdir(parents=True, exist_ok=True)
    st.set_page_config(page_title="Allisone Intra Sandbox", page_icon="🤖", layout="wide")
    mode = st.select_slider("Mode:", ["intra", "pano"])
    uploaded_file = st.file_uploader("Choose an intra oral")
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


def on_click(ok: str, uploaded_file_name: str, datapath: str):
    with open(f"{datapath}/res.csv", "a") as fd:
        writer = csv.writer(fd)
        writer.writerow([uploaded_file_name, ok, time.time()])


if __name__ == "__main__":
    if "thresholded_elements" not in st.session_state:
        st.session_state["thresholded_elements"] = []
    main()
