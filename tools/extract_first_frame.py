import cv2
from pathlib import Path


VIDEO_PATH = Path(r"data\raw\test4.mp4")
OUTPUT_PATH = Path(r"data\raw\zone_frame.jpg")


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read first frame")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), frame)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()