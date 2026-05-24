"""
Drawing utilities for FIDTM localization and zone-risk visualization.

This module creates the main visual outputs:
- FIDTM localization + total count
- clean zone risk overlay
- compact top-left zone summary panel
- 2x2 side-by-side output frame

The zone-risk visualization intentionally avoids large labels inside every zone.
Instead, it uses:
- thin colored zone outlines
- small short IDs such as CW1, SW2
- compact information panel at the top-left
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from src.pipeline.risk_classifier import risk_to_color_bgr


Point = tuple[float, float, float]


def draw_fidtm_points(
    frame_bgr: np.ndarray,
    points: Iterable[Point],
    count: int,
    fps_text: str | None = None,
) -> np.ndarray:
    """
    Draw FIDTM localized points and total count.

    Args:
        frame_bgr: Original BGR frame.
        points: Iterable of points as (x, y, score).
        count: Total estimated count.
        fps_text: Optional FPS text.

    Returns:
        Annotated BGR frame.
    """
    out = frame_bgr.copy()

    for x, y, score in points:
        xi = int(round(x))
        yi = int(round(y))

        cv2.circle(out, (xi, yi), 4, (0, 255, 0), -1)
        cv2.circle(out, (xi, yi), 7, (0, 0, 0), 1)

    cv2.rectangle(out, (20, 20), (460, 100), (0, 0, 0), -1)

    cv2.putText(
        out,
        f"FIDTM Count: {count}",
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if fps_text:
        cv2.putText(
            out,
            fps_text,
            (35, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_zone_outlines(
    frame_bgr: np.ndarray,
    zone_stats: list[dict],
    thickness: int = 3,
    draw_short_ids: bool = True,
) -> np.ndarray:
    """
    Draw colored zone outlines using each zone's risk level.

    Args:
        frame_bgr: Input frame.
        zone_stats: List of per-zone dictionaries. Expected keys:
            polygon_int, risk_level, zone_short_id, zone_count
        thickness: Outline thickness.
        draw_short_ids: Whether to draw compact zone IDs.

    Returns:
        Annotated frame.
    """
    out = frame_bgr.copy()

    for zone in zone_stats:
        polygon = zone["polygon_int"]
        risk_level = zone["risk_level"]
        short_id = zone.get("zone_short_id", zone.get("short_id", zone.get("zone_name", "")))
        color = risk_to_color_bgr(risk_level)

        cv2.polylines(
            out,
            [polygon],
            isClosed=True,
            color=color,
            thickness=thickness,
        )

        if draw_short_ids:
            cx = int(np.mean(polygon[:, 0]))
            cy = int(np.mean(polygon[:, 1]))

            cv2.rectangle(
                out,
                (cx - 24, cy - 20),
                (cx + 28, cy + 12),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                out,
                str(short_id),
                (cx - 18, cy + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return out


def draw_compact_zone_panel(
    frame_bgr: np.ndarray,
    zone_stats: list[dict],
    total_count: int,
    panel_position: tuple[int, int] = (15, 15),
) -> np.ndarray:
    """
    Draw compact zone statistics panel.

    Args:
        frame_bgr: Input frame.
        zone_stats: List of zone-stat dictionaries.
        total_count: Total frame count.
        panel_position: Top-left corner of panel.

    Returns:
        Annotated frame.
    """
    out = frame_bgr.copy()

    x1, y1 = panel_position

    panel_width = 540
    panel_height = 260

    x2 = x1 + panel_width
    y2 = y1 + panel_height

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)

    if zone_stats:
        highest = max(zone_stats, key=lambda z: z["zone_count"])
    else:
        highest = {
            "zone_short_id": "-",
            "risk_level": "LOW",
            "zone_count": 0,
        }

    highest_color = risk_to_color_bgr(highest["risk_level"])

    cv2.putText(
        out,
        f"Total Count: {total_count}",
        (x1 + 15, y1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"Highest Zone: {highest.get('zone_short_id', '-')} | {highest['risk_level']}",
        (x1 + 15, y1 + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        highest_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "Zones:",
        (x1 + 15, y1 + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    sorted_stats = sorted(
        zone_stats,
        key=lambda z: str(z.get("zone_short_id", z.get("zone_name", ""))),
    )

    start_y = y1 + 128

    for i, zone in enumerate(sorted_stats):
        col_x = x1 + 15 if i < 4 else x1 + 275
        row_y = start_y + (i % 4) * 27

        risk_level = zone["risk_level"]
        color = risk_to_color_bgr(risk_level)

        short_id = zone.get("zone_short_id", zone.get("short_id", zone["zone_name"]))
        zone_count = int(zone["zone_count"])

        line = f"{short_id}: {zone_count} | {risk_level}"

        cv2.putText(
            out,
            line,
            (col_x, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_zone_risk_clean(
    frame_bgr: np.ndarray,
    points: Iterable[Point],
    zone_stats: list[dict],
    total_count: int,
    draw_points: bool = True,
    draw_panel: bool = True,
) -> np.ndarray:
    """
    Draw clean zone risk visualization.

    Args:
        frame_bgr: Original frame.
        points: FIDTM points.
        zone_stats: Zone stats generated per frame.
        total_count: Total FIDTM count.
        draw_points: Whether to draw green localization dots.
        draw_panel: Whether to draw compact top-left panel.

    Returns:
        Visualization frame.
    """
    out = frame_bgr.copy()

    if draw_points:
        for x, y, score in points:
            xi = int(round(x))
            yi = int(round(y))
            cv2.circle(out, (xi, yi), 3, (0, 255, 0), -1)

    out = draw_zone_outlines(
        out,
        zone_stats=zone_stats,
        thickness=3,
        draw_short_ids=True,
    )

    if draw_panel:
        out = draw_compact_zone_panel(
            out,
            zone_stats=zone_stats,
            total_count=total_count,
        )

    return out


def make_side_by_side_2x2(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    img4: np.ndarray,
) -> np.ndarray:
    """
    Create 2x2 side-by-side visualization.

    Args:
        img1: Top-left image.
        img2: Top-right image.
        img3: Bottom-left image.
        img4: Bottom-right image.

    Returns:
        Combined image.
    """
    height, width = img1.shape[:2]

    images = []

    for img in [img1, img2, img3, img4]:
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        images.append(img)

    top = np.hstack([images[0], images[1]])
    bottom = np.hstack([images[2], images[3]])

    return np.vstack([top, bottom])


def draw_frame_title(
    frame_bgr: np.ndarray,
    title: str,
    position: tuple[int, int] = (20, 40),
) -> np.ndarray:
    """
    Draw a small title box on a frame.
    """
    out = frame_bgr.copy()
    x, y = position

    box_width = min(max(280, len(title) * 16), out.shape[1] - x - 20)
    box_height = 46

    cv2.rectangle(out, (x, y - 30), (x + box_width, y + box_height - 30), (0, 0, 0), -1)

    cv2.putText(
        out,
        title,
        (x + 12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out