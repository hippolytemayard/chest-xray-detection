from shapely.geometry import Polygon, box
from chest_xray_detection.ml_detection_api.utils.objects.base_objects import Coord2D
from typing import List


def get_coords_from_polygon(polygon: Polygon) -> List[Coord2D]:
    """
    Extracts coordinates from a shapely Polygon and converts them to a list of Coord2D objects.

    Args:
        polygon (Polygon): A shapely Polygon object.

    Returns:
        List[Coord2D]: A list of Coord2D objects representing the coordinates of the polygon.
    """
    x, y = polygon.exterior.coords.xy
    return [Coord2D(x=point_x, y=point_y) for point_x, point_y in zip(x, y)]


def get_box_object(coords_list: List[float]) -> Polygon:
    """
    Creates a Polygon box from a list of coordinates.

    Args:
        coords_list (List[float]): A list of coordinates in the format [minx, miny, maxx, maxy].

    Returns:
        Polygon: A shapely Polygon object representing the box.
    """
    return box(*coords_list)
