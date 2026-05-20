from pathlib import Path
import sys
import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.zone_manager import ZoneManager
from src.pipeline.risk_classifier import RiskClassifier


VIDEO_PATH = PROJECT_ROOT / "data/raw/test4.mp4"

ZONE_JSON = PROJECT_ROOT / "config/zones_config.json"

OUTPUT_VIDEO = PROJECT_ROOT / "results/videos/zone_output.mp4"

OUTPUT_CSV = PROJECT_ROOT / "results/benchmark/zone_stats.csv"

OUTPUT_FRAME = PROJECT_ROOT / "results/visualizations/zone_midframe.jpg"


class FakeFIDTM:
    """
    temporary placeholder

    later replace by your actual FIDTM class
    """

    def predict(self, frame):

        return {
            "points":[]
        }


def draw_points(frame, points):

    vis=frame.copy()

    for x,y in points:

        cv2.circle(
            vis,
            (int(x),int(y)),
            4,
            (0,255,0),
            -1
        )

        cv2.circle(
            vis,
            (int(x),int(y)),
            8,
            (0,0,0),
            1
        )

    return vis


def main():

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cap=cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    fps=cap.get(
        cv2.CAP_PROP_FPS
    )

    total=int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    w=int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    h=int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    writer=cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w,h)
    )

    zone_manager=ZoneManager(
        ZONE_JSON
    )

    risk_classifier=RiskClassifier()

    fidtm=FakeFIDTM()

    rows=[]

    middle=total//2

    for frame_idx in tqdm(range(total)):

        ret,frame=cap.read()

        if not ret:
            break

        pred=fidtm.predict(
            frame
        )

        points=pred["points"]

        results=zone_manager.count_points_in_zones(
            points
        )

        results=[

            risk_classifier.classify_zone_result(
                r
            )

            for r in results

        ]

        vis=draw_points(
            frame,
            points
        )

        vis=zone_manager.draw_zones(
            vis,
            results
        )

        writer.write(
            vis
        )

        if frame_idx==middle:

            cv2.imwrite(
                str(
                    OUTPUT_FRAME
                ),
                vis
            )

        for r in results:

            rows.append({

                "frame_idx":frame_idx,
                "zone":r["zone"],
                "count":r["count"],
                "density":r["density"],
                "risk":r["risk"]

            })

    cap.release()

    writer.release()

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        "Saved:",
        OUTPUT_VIDEO
    )

    print(
        "Saved:",
        OUTPUT_CSV
    )


if __name__=="__main__":

    main()