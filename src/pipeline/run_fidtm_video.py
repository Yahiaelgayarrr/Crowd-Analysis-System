"""
Run full FIDTM video pipeline.

This script reproduces the main Kaggle FIDTM pipeline in a clean VSCode file.

Pipeline:
    input video
        -> FIDTM head localization/counting
        -> smooth heatmap visualization
        -> zone count/density/risk analysis
        -> output videos
        -> output CSV files

Heavy step:
    This script requires the external FIDTM repo and checkpoint.
    It is expected to run on Kaggle/GPU or a machine with CUDA.

Example:
    python src/pipeline/run_fidtm_video.py ^
        --video "data/raw/final full 5 mins.mp4" ^
        --zones "config/zones_config.json" ^
        --fidtm-repo "external/FIDTM" ^
        --checkpoint "models/fidtm/model_best_jhu.pth" ^
        --output-root "results" ^
        --prefix "FULL"

Kaggle-style example:
    python src/pipeline/run_fidtm_video.py \
        --video "/kaggle/input/.../video.mp4" \
        --zones "/kaggle/working/crowd_outputs/zone_annotation/zones_config_complete_demo.json" \
        --fidtm-repo "/kaggle/working/repos/FIDTM" \
        --checkpoint "/kaggle/input/.../model_best_jhu.pth" \
        --output-root "/kaggle/working/crowd_outputs" \
        --prefix "FULL"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
from tqdm import tqdm

from src.models.fidtm_inference import FIDTMInferencer
from src.pipeline.zone_manager import load_labelme_zones, compute_zone_stats
from src.pipeline.risk_classifier import RiskClassifier, load_thresholds
from src.visualization.draw_utils import (
    draw_fidtm_points,
    draw_zone_risk_clean,
    make_side_by_side_2x2,
)
from src.visualization.heatmap_utils import (
    draw_heatmap_overlay,
    draw_heatmap_only,
)





def resolve_path(path_str: str) -> Path:
    """
    Resolve relative paths from project root.
    """
    path = Path(path_str)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def make_output_dirs(output_root: Path) -> dict[str, Path]:
    """
    Create standard output directories.
    """
    dirs = {
        "videos": output_root / "videos",
        "benchmark": output_root / "benchmark",
        "visualizations": output_root / "visualizations",
        "heatmaps": output_root / "heatmaps",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_video_info(video_path: Path) -> dict:
    """
    Read video metadata.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration_sec = total_frames / fps if fps > 0 else 0.0

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def create_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """
    Create an OpenCV MP4 writer.

    Note:
        OpenCV mp4v output may not always play directly in browsers.
        For dashboard playback, videos can later be converted to H.264.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")

    return writer


def enrich_zone_stats_for_drawing(zone_rows: list[dict], zones) -> list[dict]:
    """
    Add polygon arrays to zone rows so draw_utils can render polygons.
    """
    zone_map = {zone.name: zone for zone in zones}
    enriched = []

    for row in zone_rows:
        zone = zone_map[row["zone_name"]]
        item = dict(row)
        item["polygon_int"] = zone.polygon_int
        enriched.append(item)

    return enriched


def run_pipeline(
    video_path: Path,
    zones_path: Path,
    fidtm_repo: Path,
    checkpoint_path: Path,
    output_root: Path,
    prefix: str = "FULL",
    risk_thresholds_path: Path | None = None,
    heatmap_sigma: float = 35.0,
    max_frames: int | None = None,
    save_side_by_side: bool = False,
) -> None:
    """
    Run the full FIDTM video-processing pipeline.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not zones_path.exists():
        raise FileNotFoundError(f"Zones JSON not found: {zones_path}")

    if not fidtm_repo.exists():
        raise FileNotFoundError(f"FIDTM repo not found: {fidtm_repo}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dirs = make_output_dirs(output_root)

    print("\n✅ Loading zones")
    zones = load_labelme_zones(zones_path)

    for zone in zones:
        print(f"- {zone.name} ({zone.short_id}) | area={zone.area_pixels:.2f}")

    print("\n✅ Loading FIDTM model")
    inferencer = FIDTMInferencer(
        repo_dir=fidtm_repo,
        checkpoint_path=checkpoint_path,
    )

    thresholds = load_thresholds(risk_thresholds_path)
    risk_classifier = RiskClassifier(thresholds)

    video_info = get_video_info(video_path)

    fps = video_info["fps"]
    total_frames = video_info["total_frames"]
    width = video_info["width"]
    height = video_info["height"]

    if max_frames is not None:
        total_to_process = min(total_frames, max_frames)
    else:
        total_to_process = total_frames

    print("\n✅ Video information")
    print(f"Path: {video_path}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Frames to process: {total_to_process}")

    # Output file paths
    local_video_path = output_dirs["videos"] / f"{prefix}_01_shinjuku_fidtm_localization_count.mp4"
    heat_overlay_video_path = output_dirs["videos"] / f"{prefix}_02_shinjuku_fidtm_heatmap_overlay_points.mp4"
    heat_only_video_path = output_dirs["videos"] / f"{prefix}_03_shinjuku_fidtm_heatmap_only_points.mp4"
    zone_video_path = output_dirs["videos"] / f"{prefix}_04_shinjuku_fidtm_zone_density_risk.mp4"
    side_by_side_video_path = output_dirs["videos"] / f"{prefix}_05_shinjuku_side_by_side_outputs.mp4"

    frame_csv_path = output_dirs["benchmark"] / f"{prefix}_01_shinjuku_frame_counts.csv"
    zone_csv_path = output_dirs["benchmark"] / f"{prefix}_02_shinjuku_zone_density_risk.csv"
    midframe_path = output_dirs["visualizations"] / f"{prefix}_shinjuku_side_by_side_midframe.jpg"

    # Video writers
    writer_local = create_video_writer(local_video_path, fps, (width, height))
    writer_heat_overlay = create_video_writer(heat_overlay_video_path, fps, (width, height))
    writer_heat_only = create_video_writer(heat_only_video_path, fps, (width, height))
    writer_zone = create_video_writer(zone_video_path, fps, (width, height))

    writer_side = None
    if save_side_by_side:
        writer_side = create_video_writer(side_by_side_video_path, fps, (width * 2, height * 2))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_rows: list[dict] = []
    zone_rows_all: list[dict] = []

    midframe_saved = False
    mid_target = total_to_process // 2

    pbar = tqdm(total=total_to_process, desc="Processing FIDTM video")

    for frame_id in range(total_to_process):
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        timestamp_sec = frame_id / fps

        output = inferencer.predict_frame(frame)

        points = output.points
        total_count = output.count

        frame_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_sec": timestamp_sec,
                "total_count": total_count,
                "inference_time_sec": output.inference_time_sec,
                "inference_fps": output.inference_fps,
            }
        )

        zone_rows = compute_zone_stats(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            points=points,
            zones=zones,
            risk_classifier=risk_classifier.classify,
            inference_time_sec=output.inference_time_sec,
            inference_fps=output.inference_fps,
        )

        zone_rows_all.extend(zone_rows)

        zone_stats_for_drawing = enrich_zone_stats_for_drawing(zone_rows, zones)

        fps_text = f"Inference FPS: {output.inference_fps:.2f}"

        img_local = draw_fidtm_points(
            frame_bgr=frame,
            points=points,
            count=total_count,
            fps_text=fps_text,
        )

        img_heat_overlay = draw_heatmap_overlay(
            frame_bgr=frame,
            points=points,
            count=total_count,
            sigma=heatmap_sigma,
            draw_points=True,
        )

        img_heat_only = draw_heatmap_only(
            frame_bgr=frame,
            points=points,
            count=total_count,
            sigma=heatmap_sigma,
            draw_points=True,
        )

        img_zone = draw_zone_risk_clean(
            frame_bgr=frame,
            points=points,
            zone_stats=zone_stats_for_drawing,
            total_count=total_count,
            draw_points=True,
            draw_panel=True,
        )

        writer_local.write(img_local)
        writer_heat_overlay.write(img_heat_overlay)
        writer_heat_only.write(img_heat_only)
        writer_zone.write(img_zone)

        side = make_side_by_side_2x2(
            img_local,
            img_heat_overlay,
            img_heat_only,
            img_zone,
        )

        if writer_side is not None:
            writer_side.write(side)

        if not midframe_saved and frame_id >= mid_target:
            cv2.imwrite(str(midframe_path), side)
            midframe_saved = True

        pbar.update(1)

    pbar.close()
    cap.release()

    writer_local.release()
    writer_heat_overlay.release()
    writer_heat_only.release()
    writer_zone.release()

    if writer_side is not None:
        writer_side.release()

    df_frame = pd.DataFrame(frame_rows)
    df_zone = pd.DataFrame(zone_rows_all)

    df_frame.to_csv(frame_csv_path, index=False)
    df_zone.to_csv(zone_csv_path, index=False)

    print("\n✅ Full FIDTM video pipeline finished")
    print("=" * 70)
    print("Saved videos:")
    print(f"- {local_video_path}")
    print(f"- {heat_overlay_video_path}")
    print(f"- {heat_only_video_path}")
    print(f"- {zone_video_path}")

    if save_side_by_side:
        print(f"- {side_by_side_video_path}")

    print("\nSaved CSVs:")
    print(f"- {frame_csv_path}")
    print(f"- {zone_csv_path}")

    print("\nSaved midframe:")
    print(f"- {midframe_path}")

    if len(df_frame) > 0:
        print("\nQuick summary:")
        print(f"- Frames processed: {len(df_frame)}")
        print(f"- Average total count: {df_frame['total_count'].mean():.2f}")
        print(f"- Median total count: {df_frame['total_count'].median():.2f}")
        print(f"- Max total count: {int(df_frame['total_count'].max())}")
        print(f"- Average inference FPS: {df_frame['inference_fps'].mean():.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full FIDTM crowd video pipeline.")

    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/final full 5 mins.mp4",
        help="Input video path.",
    )

    parser.add_argument(
        "--zones",
        type=str,
        default="config/zones_config.json",
        help="Labelme zone JSON path.",
    )

    parser.add_argument(
        "--fidtm-repo",
        type=str,
        required=True,
        help="External FIDTM repository path.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="FIDTM checkpoint .pth path.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="Output root folder.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="FULL",
        help="Output filename prefix.",
    )

    parser.add_argument(
        "--risk-thresholds",
        type=str,
        default="config/risk_thresholds.json",
        help="Risk threshold JSON path.",
    )

    parser.add_argument(
        "--heatmap-sigma",
        type=float,
        default=35.0,
        help="Gaussian sigma for smooth heatmaps.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frames for testing.",
    )

    parser.add_argument(
        "--save-side-by-side",
        action="store_true",
        help="Also save full 2x2 side-by-side video. This can be very large.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    risk_thresholds_path = resolve_path(args.risk_thresholds)

    run_pipeline(
        video_path=resolve_path(args.video),
        zones_path=resolve_path(args.zones),
        fidtm_repo=resolve_path(args.fidtm_repo),
        checkpoint_path=resolve_path(args.checkpoint),
        output_root=resolve_path(args.output_root),
        prefix=args.prefix,
        risk_thresholds_path=risk_thresholds_path,
        heatmap_sigma=args.heatmap_sigma,
        max_frames=args.max_frames,
        save_side_by_side=args.save_side_by_side,
    )


if __name__ == "__main__":
    main()