import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field
from shapely.geometry import Polygon
from typing_extensions import Self

from chest_xray_detection.ml_detection_api.utils.objects.custom_typing import (
    get_json_schema_compatible_custom_typing,
)

# from __future__ import annotations


Polygon_ = get_json_schema_compatible_custom_typing(Polygon)


class Coord2D(BaseModel):
    x: float
    y: float

    def round_coordinates(self) -> Self:
        self.x = float(round(self.x))
        self.y = float(round(self.y))
        return self

    def resize(self, factor: float) -> None:
        self.x = self.x * factor
        self.y = self.y * factor


class BaseObject(BaseModel):
    detection_patology: str
    detection_boxes: list[float] = []
    detection_poly: list[Coord2D] = []
    detection_scores: Optional[float] = None
    detection_classes: Optional[int] = None


class BBoxPrediction(BaseObject):
    polygon: Optional[Polygon_] = Field(default=None, exclude=True)
    # polygon = Field(default=None, exclude=True)

    # def resize(self, factor: float) -> None:
    #    super().resize(factor=factor)
    #    self.polygon = resize_polygon(polygon=self.polygon, factor=factor)


@dataclass
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    def get_all_points(self) -> list[Coord2D]:
        """Get box's four points coordinates in the following order:
        top-left, top-right, bottom-right, bottom-left"""
        return [
            Coord2D(x=self.xmin, y=self.ymin),
            Coord2D(x=self.xmax, y=self.ymin),
            Coord2D(x=self.xmax, y=self.ymax),
            Coord2D(x=self.xmin, y=self.ymax),
        ]

    def tolist(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def to_polygon(self) -> Polygon:
        return Polygon([(point.x, point.y) for point in self.get_all_points()])

    def get_xywh_format(self) -> dict[str, float]:
        return {"x": self.xmin, "y": self.ymin, "h": self.height, "w": self.width}

    def add_margin(self, margin: float) -> Self:
        self.xmin -= margin
        self.ymin -= margin
        self.xmax += margin
        self.ymax += margin

        return self
