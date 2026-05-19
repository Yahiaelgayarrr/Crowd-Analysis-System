import cv2
import numpy as np


class OpticalFlowHelper:
    """
    Sparse Lucas-Kanade optical flow prediction
    for track position propagation.
    """

    def __init__(self):

        self.prev_gray=None

        self.lk_params=dict(
            winSize=(21,21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )


    def predict(
        self,
        frame,
        track_points
    ):

        gray=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if (
            self.prev_gray is None
            or
            len(track_points)==0
        ):

            self.prev_gray=gray.copy()
            return track_points


        p0=np.array(
            track_points,
            dtype=np.float32
        ).reshape(-1,1,2)


        p1,st,err=cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            p0,
            None,
            **self.lk_params
        )

        self.prev_gray=gray.copy()

        if p1 is None:

            return track_points


        predicted=[]

        for i,(p,s) in enumerate(
            zip(p1,st)
        ):

            if s[0]==1:

                predicted.append(
                    (
                        float(p[0][0]),
                        float(p[0][1])
                    )
                )

            else:

                predicted.append(
                    track_points[i]
                )

        return predicted