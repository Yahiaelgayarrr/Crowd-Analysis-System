"""
Zone manager for manual Labelme polygon zones.

This module loads Labelme JSON zone annotations and provides utilities to:
- read polygon zones
- calculate polygon area
- check whether FIDTM points fall inside zones
- compute zone count and pixel-based density per frame

The density here is pixel-based:
    zone_density_pixel = zone_count / zone_area_pixels

It is NOT persons per square meter unless the camera is physically calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Zone:
    """
    Represents one manually annotated scene zone.

    Attributes:
        name: Full zone name from Labelme.
        short_id: Short display ID used in visualizations.
        polygon: Polygon points as float32 array.
        polygon_int: Polygon points as int32 array for drawing.
        area_pixels: Polygon area in pixels.
    """

    name: str
    short_id: str
    polygon: np.ndarray
    polygon_int: np.ndarray
    area_pixels: float


DEFAULT_ZONE_SHORT_IDS = {
    "crosswalk_main": "CW1",
    "crosswalk_left": "CW2",
    "crosswalk_top": "CW3",
    "crosswalk_bottom": "CW4",
    "sidewalk_top": "SW1",
    "sidewalk_right": "SW2",
    "sidewalk_bottom": "SW3",
    "sidewalk_left": "SW4",
}


def short_zone_name(zone_name: str) -> str:
    """
    Return a compact zone ID for display.
    """
    return DEFAULT_ZONE_SHORT_IDS.get(zone_name, zone_name)


def load_labelme_zones(json_path: str | Path) -> list[Zone]:
    """
    Load zones from a Labelme JSON file.

    Args:
        json_path: Path to Labelme JSON.

    Returns:
        List of Zone objects.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Zone JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "shapes" not in data:
        raise ValueError(f"Invalid Labelme JSON. Missing 'shapes': {json_path}")

    zones: list[Zone] = []

    for idx, shape in enumerate(data["shapes"]):
        label = shape.get("label", f"zone_{idx}")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")

        if shape_type not in ["polygon", None]:
            print(f"Skipping non-polygon shape: {label} | type={shape_type}")
            continue

        if len(points) < 3:
            print(f"Skipping zone with fewer than 3 points: {label}")
            continue

        polygon = np.array(points, dtype=np.float32)
        polygon_int = polygon.astype(np.int32)
        area = float(abs(cv2.contourArea(polygon_int)))

        if area <= 0:
            print(f"Skipping zero-area zone: {label}")
            continue

        zones.append(
            Zone(
                name=label,
                short_id=short_zone_name(label),
                polygon=polygon,
                polygon_int=polygon_int,
                area_pixels=area,
            )
        )

    if not zones:
        raise ValueError(f"No valid polygon zones found in: {json_path}")

    return zones


def point_in_zone(x: float, y: float, zone: Zone) -> bool:
    """
    Check whether a point is inside a zone polygon.

    Args:
        x: Point x coordinate.
        y: Point y coordinate.
        zone: Zone object.

    Returns:
        True if the point lies inside or on the polygon boundary.
    """
    result = cv2.pointPolygonTest(
        zone.polygon.astype(np.float32),
        (float(x), float(y)),
        False,
    )

    return result >= 0


def assign_points_to_zones(
    points: Iterable[tuple[float, float, float]],
    zones: list[Zone],
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Assign FIDTM points to zones.

    Args:
        points: Iterable of points in format (x, y, score).
        zones: List of Zone objects.

    Returns:
        Dictionary:
            {
                zone_name: [(x, y, score), ...]
            }
    """
    zone_points = {zone.name: [] for zone in zones}

    for x, y, score in points:
        for zone in zones:
            if point_in_zone(x, y, zone):
                zone_points[zone.name].append((x, y, score))

    return zone_points


def compute_zone_stats(
    frame_id: int,
    timestamp_sec: float,
    points: Iterable[tuple[float, float, float]],
    zones: list[Zone],
    risk_classifier,
    inference_time_sec: float | None = None,
    inference_fps: float | None = None,
) -> list[dict]:
    """
    Compute count, density, and risk for every zone.

    Args:
        frame_id: Current frame index.
        timestamp_sec: Current video timestamp in seconds.
        points: FIDTM points, each as (x, y, score).
        zones: List of Zone objects.
        risk_classifier: Function accepting zone_count and returning
            either risk_level or (risk_level, color).
        inference_time_sec: Optional processing time per frame.
        inference_fps: Optional processing FPS.

    Returns:
        List of dictionaries, one row per zone.
    """
    points = list(points)
    zone_points = assign_points_to_zones(points, zones)

    rows: list[dict] = []

    for zone in zones:
        pts = zone_points[zone.name]
        zone_count = len(pts)
        zone_density = zone_count / (zone.area_pixels + 1e-9)

        risk_result = risk_classifier(zone_count)

        if isinstance(risk_result, tuple):
            risk_level = risk_result[0]
        else:
            risk_level = risk_result

        row = {
            "frame_id": frame_id,
            "timestamp_sec": timestamp_sec,
            "zone_name": zone.name,
            "zone_short_id": zone.short_id,
            "zone_count": zone_count,
            "zone_density_pixel": zone_density,
            "zone_area_pixels": zone.area_pixels,
            "risk_level": risk_level,
        }

        if inference_time_sec is not None:
            row["inference_time_sec"] = inference_time_sec

        if inference_fps is not None:
            row["inference_fps"] = inference_fps

        rows.append(row)

    return rows


def summarize_zones(zones: list[Zone]) -> list[dict]:
    """
    Create a compact summary of all zones.
    """
    return [
        {
            "zone_name": zone.name,
            "zone_short_id": zone.short_id,
            "zone_area_pixels": zone.area_pixels,
            "num_polygon_points": int(len(zone.polygon)),
        }
        for zone in zones
    ]


def save_zone_summary(zones: list[Zone], output_path: str | Path) -> None:
    """
    Save zone summary as JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_zones(zones)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def draw_zone_verification(
    frame_bgr: np.ndarray,
    zones: list[Zone],
    show_large_labels: bool = False,
) -> np.ndarray:
    """
    Draw zones over an image for verification.

    Args:
        frame_bgr: Input frame in BGR format.
        zones: List of Zone objects.
        show_large_labels: If True, draw full zone names and area.
            If False, draw only small short IDs.

    Returns:
        Annotated frame.
    """
    out = frame_bgr.copy()

    for zone in zones:
        polygon = zone.polygon_int

        cv2.polylines(
            out,
            [polygon],
            isClosed=True,
            color=(0, 255, 255),
            thickness=3,
        )

        fill = out.copy()
        cv2.fillPoly(fill, [polygon], color=(0, 255, 255))
        out = cv2.addWeighted(fill, 0.15, out, 0.85, 0)

        cx = int(np.mean(polygon[:, 0]))
        cy = int(np.mean(polygon[:, 1]))

        if show_large_labels:
            text = f"{zone.name} | area={int(zone.area_pixels)}"
            cv2.rectangle(out, (cx - 10, cy - 30), (cx + 420, cy + 8), (0, 0, 0), -1)
            cv2.putText(
                out,
                text,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(out, (cx - 22, cy - 18), (cx + 22, cy + 12), (0, 0, 0), -1)
            cv2.putText(
                out,
                zone.short_id,
                (cx - 16, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return out