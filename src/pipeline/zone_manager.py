import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


class ZoneManager:
    """
    Loads polygon zones from Labelme JSON or simple JSON format.
    Counts FIDTM head points inside each zone.
    """

    def __init__(self, zones_json_path: str | Path):
        self.zones_json_path = Path(zones_json_path)
        self.zones = self._load_zones(self.zones_json_path)

    def _load_zones(self, path: Path) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        zones = []

        # Labelme format
        if "shapes" in data:
            for shape in data["shapes"]:
                label = shape["label"]
                pts = np.array(shape["points"], dtype=np.int32)

                zones.append({
                    "name": label,
                    "polygon": pts,
                    "area_px": float(cv2.contourArea(pts)),
                })

        # Simple format: {"zone_1": [[x,y], ...]}
        else:
            for label, points in data.items():
                pts = np.array(points, dtype=np.int32)

                zones.append({
                    "name": label,
                    "polygon": pts,
                    "area_px": float(cv2.contourArea(pts)),
                })

        if not zones:
            raise ValueError(f"No zones found in {path}")

        return zones

    def count_points_in_zones(self, points: List[Point]) -> List[Dict]:
        results = []

        for zone in self.zones:
            count = 0
            polygon = zone["polygon"]

            for x, y in points:
                inside = cv2.pointPolygonTest(
                    polygon,
                    (float(x), float(y)),
                    False,
                )

                if inside >= 0:
                    count += 1

            area_px = max(zone["area_px"], 1.0)
            density = count / area_px

            results.append({
                "zone": zone["name"],
                "count": count,
                "area_px": area_px,
                "density": density,
            })

        return results

    def draw_zones(self, frame, zone_results: List[Dict]):
        vis = frame.copy()
        result_map = {r["zone"]: r for r in zone_results}

        for zone in self.zones:
            name = zone["name"]
            polygon = zone["polygon"]
            result = result_map.get(name, {})

            risk = result.get("risk", "UNKNOWN")
            count = result.get("count", 0)

            color = self._risk_color(risk)

            cv2.polylines(
                vis,
                [polygon],
                True,
                color,
                3,
            )

            cx = int(polygon[:, 0].mean())
            cy = int(polygon[:, 1].mean())

            cv2.putText(
                vis,
                f"{name}: {count} | {risk}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        return vis

    @staticmethod
    def _risk_color(risk: str):
        if risk == "LOW":
            return (0, 255, 0)
        if risk == "MEDIUM":
            return (0, 255, 255)
        if risk == "HIGH":
            return (0, 128, 255)
        if risk == "CRITICAL":
            return (0, 0, 255)

        return (255, 255, 255)