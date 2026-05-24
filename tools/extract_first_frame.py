"""
Extract a representative frame from a video for manual zone annotation.

This script is the first step of the crowd-analysis pipeline.

Pipeline step:
    input video
        -> extract frame
        -> save image
        -> open image in Labelme
        -> draw zone polygons
        -> save Labelme JSON

Example:
    python tools/extract_first_frame.py ^
        --video "data/raw/final full 5 mins.mp4" ^
        --output "data/raw/zone_frame.jpg" ^
        --mode middle

Then open Labelme:
    labelme data/raw/zone_frame.jpg

If labelme command does not work:
    python -m labelme data/raw/zone_frame.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    """
    Resolve a path relative to the project root if it is not absolute.
    """
    path = Path(path_str)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_video_info(video_path: Path) -> dict:
    """
    Read basic video metadata.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration_sec = total_frames / fps if fps and fps > 0 else 0.0

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def choose_frame_index(total_frames: int, mode: str, frame_index: int | None) -> int:
    """
    Choose which frame should be extracted.
    """
    if total_frames <= 0:
        raise ValueError("Video has no readable frames.")

    if mode == "custom":
        if frame_index is None:
            raise ValueError("--frame-index is required when --mode custom is used.")

        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError(
                f"frame-index must be between 0 and {total_frames - 1}, got {frame_index}"
            )

        return frame_index

    if mode == "first":
        return 0

    if mode == "middle":
        return total_frames // 2

    if mode == "quarter":
        return total_frames // 4

    if mode == "three_quarter":
        return int(total_frames * 0.75)

    raise ValueError(f"Unsupported mode: {mode}")


def extract_frame(video_path: Path, output_path: Path, frame_index: int) -> None:
    """
    Extract a single frame from a video and save it as an image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

    saved = cv2.imwrite(str(output_path), frame)

    if not saved:
        raise RuntimeError(f"Could not save frame to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a representative frame from a video for Labelme zone annotation."
    )

    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/final full 5 mins.mp4",
        help="Input video path.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/zone_frame.jpg",
        help="Output frame image path.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="middle",
        choices=["first", "quarter", "middle", "three_quarter", "custom"],
        help="Which frame to extract.",
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Custom frame index. Required only when --mode custom.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    video_path = resolve_path(args.video)
    output_path = resolve_path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)

    frame_index = choose_frame_index(
        total_frames=info["total_frames"],
        mode=args.mode,
        frame_index=args.frame_index,
    )

    extract_frame(
        video_path=video_path,
        output_path=output_path,
        frame_index=frame_index,
    )

    print("\n✅ Frame extracted successfully")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output frame: {output_path}")
    print(f"Selected frame index: {frame_index}")
    print("")
    print("Video information:")
    print(f"- Resolution: {info['width']} x {info['height']}")
    print(f"- FPS: {info['fps']:.2f}")
    print(f"- Total frames: {info['total_frames']}")
    print(f"- Duration: {info['duration_sec']:.2f} seconds")
    print("")
    print("Next step:")
    print(f"labelme {output_path}")
    print("")
    print("If labelme command does not work:")
    print(f"python -m labelme {output_path}")


if __name__ == "__main__":
    main()