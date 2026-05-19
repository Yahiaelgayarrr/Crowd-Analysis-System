from pathlib import Path
import argparse
import cv2
import pandas as pd
from tqdm import tqdm

from src.models.fidtm_inference import FIDTMInference
from src.visualization.draw_utils import draw_fidtm_points
from src.visualization.heatmap_utils import draw_heatmap_overlay, draw_heatmap_only


def run_count_localization_video(model, video_path, output_video, output_csv, midframe_path=None):
    video_path = Path(video_path)
    output_video = Path(output_video)
    output_csv = Path(output_csv)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    rows = []
    mid_idx = total_frames // 2

    for frame_idx in tqdm(range(total_frames), desc="FIDTM localization"):
        ret, frame = cap.read()
        if not ret:
            break

        pred = model.predict_frame(frame)

        vis = draw_fidtm_points(
            frame,
            pred["points"],
            pred["count"],
            fps_text=f"Inference FPS: {pred['fps']:.2f}",
        )

        writer.write(vis)

        if midframe_path and frame_idx == mid_idx:
            Path(midframe_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(midframe_path), vis)

        rows.append({
            "frame_idx": frame_idx,
            "count": pred["count"],
            "num_points": len(pred["points"]),
            "inference_time_sec": pred["time_sec"],
            "inference_fps": pred["fps"],
        })

    cap.release()
    writer.release()

    pd.DataFrame(rows).to_csv(output_csv, index=False)


def run_heatmap_video(model, video_path, output_overlay, output_only, output_csv, midframe_path=None):
    video_path = Path(video_path)
    output_overlay = Path(output_overlay)
    output_only = Path(output_only)
    output_csv = Path(output_csv)

    output_overlay.parent.mkdir(parents=True, exist_ok=True)
    output_only.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    overlay_writer = cv2.VideoWriter(
        str(output_overlay),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    only_writer = cv2.VideoWriter(
        str(output_only),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    rows = []
    mid_idx = total_frames // 2

    for frame_idx in tqdm(range(total_frames), desc="FIDTM heatmap"):
        ret, frame = cap.read()
        if not ret:
            break

        pred = model.predict_frame(frame)

        overlay = draw_heatmap_overlay(frame, pred["points"], pred["count"])
        only = draw_heatmap_only(frame, pred["points"], pred["count"])

        overlay_writer.write(overlay)
        only_writer.write(only)

        if midframe_path and frame_idx == mid_idx:
            Path(midframe_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(midframe_path), overlay)

        rows.append({
            "frame_idx": frame_idx,
            "count": pred["count"],
            "num_points": len(pred["points"]),
            "inference_time_sec": pred["time_sec"],
            "inference_fps": pred["fps"],
        })

    cap.release()
    overlay_writer.release()
    only_writer.release()

    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo", required=True, help="Path to cloned FIDTM repo")
    parser.add_argument("--weights", required=True, help="Path to FIDTM checkpoint")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--mode", choices=["localization", "heatmap"], required=True)

    parser.add_argument("--output-video", default="results/videos/output.mp4")
    parser.add_argument("--output-overlay", default="results/videos/heatmap_overlay.mp4")
    parser.add_argument("--output-only", default="results/videos/heatmap_only.mp4")
    parser.add_argument("--output-csv", default="results/benchmark/output_counts.csv")
    parser.add_argument("--midframe", default=None)

    args = parser.parse_args()

    model = FIDTMInference(
        repo_path=args.repo,
        weight_path=args.weights,
    )

    if args.mode == "localization":
        run_count_localization_video(
            model=model,
            video_path=args.video,
            output_video=args.output_video,
            output_csv=args.output_csv,
            midframe_path=args.midframe,
        )

    elif args.mode == "heatmap":
        run_heatmap_video(
            model=model,
            video_path=args.video,
            output_overlay=args.output_overlay,
            output_only=args.output_only,
            output_csv=args.output_csv,
            midframe_path=args.midframe,
        )


if __name__ == "__main__":
    main()