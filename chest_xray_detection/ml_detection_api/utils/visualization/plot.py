from typing import Optional

import numpy as np
import plotly.express as px

from chest_xray_detection.ml_detection_api.utils.objects.base_objects import BBoxPrediction


def plot_predictions(
    image: np.ndarray,
    detections: list[BBoxPrediction],
    detection_color: Optional[dict[str, str]] = None,
    title: str = "",
) -> px.imshow:
    fig = px.imshow(
        image,
        binary_string=True,
        title=title,
    )
    for detection in detections:
        if detection.detection_scores is None or detection.detection_poly is None:
            continue
        polygon_array = np.asarray([(coords.x, coords.y) for coords in detection.detection_poly])
        text = f"{detection.detection_patology}<br>Score: {np.round(detection.detection_scores, decimals=2)}"
        if detection.detection_patology in detection_color:
            color_contour = detection_color[detection.detection_patology]
        else:
            color_contour = (
                "rgba(0, 102, 255,1)"
                if int(detection.detection_patology) % 2
                else "rgba(255, 80, 80,1)"
            )
        color_fill = color_contour[:-2] + ".1)"

        centroid_x = np.mean(polygon_array[:, 0])
        centroid_y = np.mean(polygon_array[:, 1])

        fig.add_scatter(
            x=polygon_array[:, 0],
            y=polygon_array[:, 1],
            mode="lines",
            line=dict(dash="dot", width=2, color=color_contour),
            hoverinfo="skip",
            fill="tozeroy",
            fillcolor=color_fill,
            name=detection.detection_patology,
        )

        fig.add_scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode="text",
            text=[text],
            showlegend=False,
            hoverinfo="skip",
            textposition="middle center",
            textfont=dict(color=color_contour),
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        showlegend=True,
        autosize=True,
        height=650,
    )

    return fig


def plot_json_predictions(
    image: np.ndarray,
    detections: list[BBoxPrediction],
    detection_color: Optional[dict[str, str]] = None,
    title: str = "",
) -> px.imshow:
    fig = px.imshow(
        image,
        binary_string=True,
        title=title,
    )
    for detection in detections:
        if detection["detection_scores"] is None or detection["detection_poly"] is None:
            continue
        polygon_array = np.asarray(
            [(coords["x"], coords["y"]) for coords in detection["detection_poly"]]
        )
        text = f"{detection['detection_patology']}<br>Score: {np.round(detection['detection_scores'], decimals=2)}"
        if detection["detection_patology"] in detection_color:
            color_contour = detection_color[detection["detection_patology"]]
        else:
            color_contour = (
                "rgba(0, 102, 255,1)"
                if int(detection["detection_patology"]) % 2
                else "rgba(255, 80, 80,1)"
            )
        color_fill = color_contour[:-2] + ".1)"

        centroid_x = np.mean(polygon_array[:, 0])
        centroid_y = np.mean(polygon_array[:, 1])

        fig.add_scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode="text",
            text=[text],
            showlegend=False,
            hoverinfo="skip",
            textposition="middle center",
            textfont=dict(color=color_contour),
        )

        fig.add_scatter(
            x=polygon_array[:, 0],
            y=polygon_array[:, 1],
            mode="lines",
            line=dict(dash="dot", width=2, color=color_contour),
            hoverinfo="text",
            fill="tozeroy",
            fillcolor=color_fill,
            text=text,
            name=detection["detection_patology"],
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        showlegend=True,
        autosize=True,
        height=650,
    )

    return fig


def plot_prediction_(
    image,
    detections,
    detection_color: dict = None,
    title="",
):
    fig = px.imshow(
        image,
        binary_string=True,
        title=title,
    )
    for detection in detections:
        if not detection.detection_scores or not detection.detection_poly:
            continue
        polygon_array = np.asarray([(coords.x, coords.y) for coords in detection.detection_poly])
        text = f"Label: {detection.detection_patology}\nProba: {np.round(detection.detection_scores, decimals=2)}"
        if detection.detection_patology in detection_color:
            color_contour = detection_color[detection.detection_patology]
        else:
            color_contour = (
                "rgba(0, 102, 255,1)"
                if int(detection.detection_patology) % 2
                else "rgba(255, 80, 80,1)"
            )
        color_fill = color_contour[:-2] + ".1)"
        fig.add_scatter(
            x=polygon_array[:, 0],
            y=polygon_array[:, 1],
            mode="lines",
            line=dict(dash="dot", width=2, color=color_contour),
            hoverinfo="text",
            fill="tozeroy",
            fillcolor=color_fill,
            text=text,
            name=detection.detection_patology,
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        showlegend=True,
        autosize=True,
        height=650,
    )

    return fig
