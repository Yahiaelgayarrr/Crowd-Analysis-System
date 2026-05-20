from typing import Dict


class RiskClassifier:
    """
    Simple zone risk classifier based on people count.

    Later we can improve this using:
    - density
    - movement speed
    - congestion duration
    - zone type
    """

    def __init__(
        self,
        low_threshold: int = 8,
        medium_threshold: int = 18,
        high_threshold: int = 30,
    ):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

    def classify(self, count: int, density: float = 0.0) -> str:
        if count < self.low_threshold:
            return "LOW"

        if count < self.medium_threshold:
            return "MEDIUM"

        if count < self.high_threshold:
            return "HIGH"

        return "CRITICAL"

    def classify_zone_result(self, zone_result: Dict) -> Dict:
        count = int(zone_result["count"])
        density = float(zone_result["density"])

        zone_result["risk"] = self.classify(count, density)

        return zone_result