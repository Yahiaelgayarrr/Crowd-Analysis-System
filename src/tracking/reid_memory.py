import cv2
import numpy as np


class ReIDMemory:

    def __init__(self):

        self.feature_bank={}
        self.lost_tracks={}


    def extract_feature(
        self,
        frame,
        x,
        y,
        crop_size=20
    ):

        h,w=frame.shape[:2]

        x1=max(0,int(x-crop_size))
        y1=max(0,int(y-crop_size))

        x2=min(
            w,
            int(x+crop_size)
        )

        y2=min(
            h,
            int(y+crop_size)
        )


        crop=frame[
            y1:y2,
            x1:x2
        ]


        if crop.size==0:

            return None


        hsv=cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV
        )


        hist=cv2.calcHist(
            [hsv],
            [0,1],
            None,
            [16,16],
            [0,180,0,256]
        )


        cv2.normalize(
            hist,
            hist
        )

        return hist.flatten()


    def compare(
        self,
        f1,
        f2
    ):

        if (
            f1 is None
            or
            f2 is None
        ):

            return 0

        return cv2.compareHist(
            f1.astype(np.float32),
            f2.astype(np.float32),
            cv2.HISTCMP_CORREL
        )