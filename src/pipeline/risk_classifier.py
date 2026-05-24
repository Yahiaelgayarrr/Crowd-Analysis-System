"""
Rule-based risk classifier for zone-level crowd monitoring.

Current prototype rule:
    LOW       : zone_count < 8
    MEDIUM    : 8 <= zone_count < 18
    HIGH      : 18 <= zone_count < 30
    CRITICAL  : zone_count >= 30

Important:
    These thresholds are prototype thresholds based on image-space zone counts.
    They are not physically calibrated safety thresholds.

Later, this can be upgraded to include:
    - pixel-based density
    - temporal density growth
    - optical-flow motion magnitude
    - directional instability
    - anomaly score
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


RISK_COLORS_BGR = {
    "LOW": (0, 255, 0),
    "MEDIUM": (0, 255, 255),
    "HIGH": (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}

RISK_COLORS_HEX = {
    "LOW": "#22c55e",
    "MEDIUM": "#f59e0b",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}


@dataclass
class RiskThresholds:
    """
    Threshold configuration for count-based risk classification.

    Attributes:
        low_max: Counts below this value are LOW.
        medium_max: Counts below this value are MEDIUM.
        high_max: Counts below this value are HIGH.
        Counts equal to or above high_max are CRITICAL.
    """

    low_max: int = 8
    medium_max: int = 18
    high_max: int = 30


def load_thresholds(path: str | Path | None = None) -> RiskThresholds:
    """
    Load risk thresholds from JSON if available.

    Supported JSON formats:
        {
            "low_max": 8,
            "medium_max": 18,
            "high_max": 30
        }

    Or:
        {
            "count_thresholds": {
                "low_max": 8,
                "medium_max": 18,
                "high_max": 30
            }
        }

    If the file is missing or invalid, default thresholds are used.
    """
    if path is None:
        return RiskThresholds()

    path = Path(path)

    if not path.exists():
        print(f"Risk threshold file not found, using defaults: {path}")
        return RiskThresholds()

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "count_thresholds" in data:
            data = data["count_thresholds"]

        return RiskThresholds(
            low_max=int(data.get("low_max", 8)),
            medium_max=int(data.get("medium_max", 18)),
            high_max=int(data.get("high_max", 30)),
        )

    except Exception as exc:
        print(f"Could not load risk thresholds from {path}: {exc}")
        print("Using default thresholds.")
        return RiskThresholds()


class RiskClassifier:
    """
    Count-based zone risk classifier.
    """

    def __init__(self, thresholds: RiskThresholds | None = None):
        self.thresholds = thresholds or RiskThresholds()

    def classify(self, zone_count: int | float) -> str:
        """
        Classify risk level from zone count.

        Args:
            zone_count: Number of FIDTM points inside the zone.

        Returns:
            Risk level string: LOW, MEDIUM, HIGH, or CRITICAL.
        """
        count = float(zone_count)

        if count < self.thresholds.low_max:
            return "LOW"

        if count < self.thresholds.medium_max:
            return "MEDIUM"

        if count < self.thresholds.high_max:
            return "HIGH"

        return "CRITICAL"

    def classify_with_color(self, zone_count: int | float) -> tuple[str, tuple[int, int, int]]:
        """
        Classify risk and return BGR drawing color.
        """
        risk = self.classify(zone_count)
        return risk, RISK_COLORS_BGR[risk]


def classify_risk(zone_count: int | float) -> str:
    """
    Convenience function using default thresholds.
    """
    return RiskClassifier().classify(zone_count)


def classify_risk_with_color(zone_count: int | float) -> tuple[str, tuple[int, int, int]]:
    """
    Convenience function using default thresholds and returning BGR color.
    """
    return RiskClassifier().classify_with_color(zone_count)


def risk_to_color_bgr(risk_level: str) -> tuple[int, int, int]:
    """
    Convert risk label to BGR color for OpenCV drawing.
    """
    return RISK_COLORS_BGR.get(risk_level.upper(), (255, 255, 255))


def risk_to_color_hex(risk_level: str) -> str:
    """
    Convert risk label to HEX color for dashboards/plots.
    """
    return RISK_COLORS_HEX.get(risk_level.upper(), "#e2e8f0")