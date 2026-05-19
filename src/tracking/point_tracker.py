class SimplePointTracker:
    """
    Placeholder tracker for next phase.
    Later we will replace/extend this with persistent ID tracking.
    """

    def __init__(self):
        self.next_id = 1

    def update(self, points):
        tracked = []

        for x, y, score in points:
            tracked.append({
                "track_id": self.next_id,
                "x": x,
                "y": y,
                "score": score,
            })
            self.next_id += 1

        return tracked