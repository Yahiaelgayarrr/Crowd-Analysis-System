"""
Convert OpenCV MP4 videos into browser-friendly H.264 MP4 files.

Why:
    OpenCV often writes MP4 videos using the mp4v codec.
    Streamlit/browser video playback is more reliable with H.264 + yuv420p.

Input:
    results/videos/

Output:
    results/videos_dashboard/

This script does not modify the original videos.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "results" / "videos_dashboard"

VIDEO_NAMES = [
    "FULL_01_shinjuku_fidtm_localization_count.mp4",
    "FULL_02_shinjuku_fidtm_heatmap_overlay_points.mp4",
    "FULL_03_shinjuku_fidtm_heatmap_only_points.mp4",
    "FULL_04_shinjuku_fidtm_zone_density_risk.mp4",
]


def find_ffmpeg() -> str:
    """
    Find ffmpeg from system PATH or imageio-ffmpeg package.
    """
    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install it with:\n"
            "python -m pip install imageio-ffmpeg"
        ) from exc


def run_command(command: list[str]) -> None:
    """
    Run a command and stream output.
    """
    print("\nRunning command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))
    print("-" * 80)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed with return code {return_code}")


def convert_video(ffmpeg: str, input_path: Path, output_path: Path) -> None:
    """
    Convert one video to browser-friendly H.264.

    Notes:
    - libx264 creates H.264 video
    - yuv420p improves browser compatibility
    - faststart helps web playback start faster
    - scale uses even dimensions because H.264 prefers even width/height
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]

    run_command(command)


def main() -> None:
    print("=" * 80)
    print("Dashboard video conversion")
    print("=" * 80)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input video folder not found: {INPUT_DIR}")

    ffmpeg = find_ffmpeg()
    print(f"Using FFmpeg: {ffmpeg}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    converted = []
    missing = []

    for video_name in VIDEO_NAMES:
        input_path = INPUT_DIR / video_name
        output_path = OUTPUT_DIR / video_name

        if not input_path.exists():
            missing.append(input_path)
            print(f"\n⚠️ Missing input video: {input_path}")
            continue

        print("\n" + "=" * 80)
        print(f"Converting: {video_name}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print("=" * 80)

        convert_video(ffmpeg, input_path, output_path)
        converted.append(output_path)

    print("\n" + "=" * 80)
    print("Conversion finished")
    print("=" * 80)

    print("\nConverted videos:")
    for path in converted:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"- {path} ({size_mb:.2f} MB)")

    if missing:
        print("\nMissing videos:")
        for path in missing:
            print(f"- {path}")

    print("\nNext command:")
    print("streamlit cache clear")
    print("streamlit run src\\dashboard\\app.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        sys.exit(1)