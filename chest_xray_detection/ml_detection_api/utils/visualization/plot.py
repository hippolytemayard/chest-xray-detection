import numpy as np
import plotly.express as px


def plot_predictions(
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
        text = f"Label: {detection.detection_patology}<br>Proba: {np.round(detection.detection_scores, decimals=4)}"
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
