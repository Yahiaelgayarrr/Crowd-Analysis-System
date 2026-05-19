import cv2
import numpy as np


def make_smooth_heatmap_from_points(frame_shape, points, sigma=35):
    h, w = frame_shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for x, y, score in points:
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h:
            heat[yi, xi] += 1.0

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if heat.max() > 0:
        heat = heat / heat.max()

    heat_uint8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    return heat_color, heat_uint8


def draw_heatmap_overlay(frame_bgr, points, count, sigma=35, draw_points=True):
    heat_color, _ = make_smooth_heatmap_from_points(frame_bgr.shape, points, sigma=sigma)

    overlay = cv2.addWeighted(frame_bgr, 0.60, heat_color, 0.40, 0)

    if draw_points:
        for x, y, score in points:
            cv2.circle(overlay, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.circle(overlay, (int(x), int(y)), 7, (0, 0, 0), 1)

    cv2.rectangle(overlay, (20, 20), (650, 100), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"FIDTM Heatmap + Localization | Count: {count}",
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


def draw_heatmap_only(frame_bgr, points, count, sigma=35, draw_points=True):
    heat_color, _ = make_smooth_heatmap_from_points(frame_bgr.shape, points, sigma=sigma)
    only = heat_color.copy()

    if draw_points:
        for x, y, score in points:
            cv2.circle(only, (int(x), int(y)), 3, (255, 255, 255), -1)
            cv2.circle(only, (int(x), int(y)), 6, (0, 0, 0), 1)

    cv2.rectangle(only, (20, 20), (650, 100), (0, 0, 0), -1)
    cv2.putText(
        only,
        f"FIDTM Heatmap Only + Points | Count: {count}",
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return only