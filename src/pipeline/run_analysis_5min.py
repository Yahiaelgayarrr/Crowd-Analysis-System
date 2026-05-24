"""
Run refined 5-minute Shinjuku CSV analysis locally.

This script reproduces the refined Kaggle analysis using only the two saved CSV files:

Input:
    results/benchmark/FULL_01_shinjuku_frame_counts.csv
    results/benchmark/FULL_02_shinjuku_zone_density_risk.csv

Output:
    results/analysis_5min_refined/
        tables/
        figures/
        insights/

This script does not require GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def prepare_output_dirs(output_dir: Path, clean: bool = True) -> dict[str, Path]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    insights_dir = output_dir / "insights"
    zip_dir = output_dir / "zip_backup"

    for folder in [tables_dir, figures_dir, insights_dir, zip_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return {
        "tables": tables_dir,
        "figures": figures_dir,
        "insights": insights_dir,
        "zip": zip_dir,
    }


def estimate_fps(frame_df: pd.DataFrame) -> float:
    timestamp_diff = frame_df["timestamp_sec"].diff().dropna()
    median_dt = timestamp_diff[timestamp_diff > 0].median()

    if pd.notna(median_dt) and median_dt > 0:
        return float(1.0 / median_dt)

    return 30.0


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_global_count(frame_df: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(frame_df["timestamp_sec"], frame_df["total_count"], label="Total Count")

    if "rolling_mean_count_5s" in frame_df.columns:
        plt.plot(
            frame_df["timestamp_sec"],
            frame_df["rolling_mean_count_5s"],
            label="5s Rolling Mean",
        )

    plt.title("Global Crowd Count Over Time — 5 Minute Video")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Total Count")
    plt.grid(True)
    plt.legend()
    save_figure(figures_dir / "global_count_over_time_5min_refined.png")


def plot_global_change(frame_df: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(frame_df["timestamp_sec"], frame_df["total_count_change"])
    plt.title("Global Count Rate of Change Proxy — 5 Minute Video")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Count Change Between Frames")
    plt.grid(True)
    save_figure(figures_dir / "global_count_change_over_time_5min_refined.png")


def plot_inference_fps(frame_df: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(frame_df["timestamp_sec"], frame_df["inference_fps"])
    plt.title("Full Pipeline FPS Over Time — 5 Minute Video")
    plt.xlabel("Time (seconds)")
    plt.ylabel("FPS")
    plt.grid(True)
    save_figure(figures_dir / "inference_fps_over_time_5min_refined.png")


def plot_barh(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    output: Path,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    plot_data = df.sort_values(x_col, ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_data[y_col], plot_data[x_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Zone")
    plt.grid(axis="x", alpha=0.4)
    save_figure(output)


def plot_risk_percentage(risk_percent: pd.DataFrame, figures_dir: Path) -> None:
    percent_cols = ["LOW_percent", "MEDIUM_percent", "HIGH_percent", "CRITICAL_percent"]

    if risk_percent.empty:
        return

    plt.figure(figsize=(16, 8))
    bottom = np.zeros(len(risk_percent))

    for col in percent_cols:
        if col in risk_percent.columns:
            values = risk_percent[col].values
            plt.bar(
                risk_percent["zone_name"],
                values,
                bottom=bottom,
                label=col.replace("_percent", ""),
            )
            bottom += values

    plt.title("Risk Level Percentage by Zone — 5 Minute Video")
    plt.xlabel("Zone")
    plt.ylabel("Percentage of Frames (%)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.4)
    plt.legend()
    save_figure(figures_dir / "risk_percentage_by_zone_5min_refined.png")


def plot_density_trend(trend_summary: pd.DataFrame, figures_dir: Path) -> None:
    percent_cols = ["ACCUMULATING_percent", "DISPERSING_percent", "STABLE_percent"]

    if trend_summary.empty:
        return

    plt.figure(figsize=(16, 8))
    bottom = np.zeros(len(trend_summary))

    for col in percent_cols:
        if col in trend_summary.columns:
            values = trend_summary[col].values
            plt.bar(
                trend_summary["zone_name"],
                values,
                bottom=bottom,
                label=col.replace("_percent", ""),
            )
            bottom += values

    plt.title("Density Trend Percentage by Zone — 5 Minute Video")
    plt.xlabel("Zone")
    plt.ylabel("Percentage of Frames (%)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.4)
    plt.legend()
    save_figure(figures_dir / "density_trend_percentage_by_zone_5min_refined.png")


def plot_correlation_heatmap(corr: pd.DataFrame, figures_dir: Path) -> None:
    if corr.empty:
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Zone Count Correlation Matrix — 5 Minute Video")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    save_figure(figures_dir / "zone_count_correlation_heatmap_5min_refined.png")


def plot_entropy(entropy_df: pd.DataFrame, figures_dir: Path) -> None:
    if entropy_df.empty:
        return

    plt.figure(figsize=(14, 7))
    plt.plot(entropy_df["timestamp_sec"], entropy_df["normalized_entropy"])
    plt.title("Crowd Distribution Entropy Over Time — 5 Minute Video")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Entropy")
    plt.grid(True)
    save_figure(figures_dir / "crowd_entropy_over_time_5min_refined.png")


def plot_risk_transition_matrix(matrix: pd.DataFrame, figures_dir: Path) -> None:
    if matrix.empty:
        return

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix.values, aspect="auto")
    plt.colorbar(label="Transition Count")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.title("Risk Transition Matrix — 5 Minute Video")

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            plt.text(j, i, str(int(matrix.values[i, j])), ha="center", va="center")

    save_figure(figures_dir / "risk_transition_matrix_5min_refined.png")


def make_empty_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)
    return df


def run_analysis(
    frame_csv: Path,
    zone_csv: Path,
    output_dir: Path,
    clean_output: bool = True,
) -> None:
    dirs = prepare_output_dirs(output_dir, clean=clean_output)

    tables_dir = dirs["tables"]
    figures_dir = dirs["figures"]
    insights_dir = dirs["insights"]
    zip_dir = dirs["zip"]

    if not frame_csv.exists():
        raise FileNotFoundError(f"Frame CSV not found: {frame_csv}")

    if not zone_csv.exists():
        raise FileNotFoundError(f"Zone CSV not found: {zone_csv}")

    print("Loading CSV files...")
    frame_df = pd.read_csv(frame_csv)
    zone_df = pd.read_csv(zone_csv)

    frame_df = frame_df.sort_values("timestamp_sec").reset_index(drop=True)
    zone_df = zone_df.sort_values(["zone_name", "timestamp_sec"]).reset_index(drop=True)

    estimated_fps = estimate_fps(frame_df)
    rolling_window_frames = max(3, int(estimated_fps * 5))
    video_duration_sec = float(frame_df["timestamp_sec"].max() - frame_df["timestamp_sec"].min())
    processed_frames = len(frame_df)

    print(f"Estimated FPS: {estimated_fps:.2f}")
    print(f"Duration: {video_duration_sec:.2f}s")
    print(f"Frames: {processed_frames}")

    # ------------------------------------------------------------------
    # 1. Global temporal analysis
    # ------------------------------------------------------------------
    frame_df["total_count_change"] = frame_df["total_count"].diff().fillna(0)
    frame_df["total_count_change_abs"] = frame_df["total_count_change"].abs()
    frame_df["rolling_mean_count_5s"] = frame_df["total_count"].rolling(
        rolling_window_frames,
        min_periods=1,
    ).mean()
    frame_df["rolling_mean_count_change_5s"] = frame_df["total_count_change"].rolling(
        rolling_window_frames,
        min_periods=1,
    ).mean()

    global_summary = pd.DataFrame(
        [
            {
                "video_duration_sec": video_duration_sec,
                "processed_frames": processed_frames,
                "estimated_video_fps": estimated_fps,
                "mean_total_count": frame_df["total_count"].mean(),
                "median_total_count": frame_df["total_count"].median(),
                "min_total_count": frame_df["total_count"].min(),
                "max_total_count": frame_df["total_count"].max(),
                "std_total_count": frame_df["total_count"].std(),
                "mean_abs_count_change": frame_df["total_count_change_abs"].mean(),
                "max_positive_count_change": frame_df["total_count_change"].max(),
                "max_negative_count_change": frame_df["total_count_change"].min(),
                "mean_full_pipeline_fps": frame_df["inference_fps"].mean(),
                "median_full_pipeline_fps": frame_df["inference_fps"].median(),
                "mean_processing_time_sec": frame_df["inference_time_sec"].mean(),
                "median_processing_time_sec": frame_df["inference_time_sec"].median(),
            }
        ]
    )

    global_summary.to_csv(tables_dir / "global_count_summary_5min_refined.csv", index=False)
    frame_df.to_csv(tables_dir / "global_count_change_features_5min_refined.csv", index=False)

    top_10_peak_frames = frame_df.sort_values("total_count", ascending=False).head(10).copy()
    top_10_peak_frames.to_csv(tables_dir / "top_10_peak_crowd_frames_5min_refined.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Zone count and density analysis
    # ------------------------------------------------------------------
    zone_summary = zone_df.groupby(["zone_name", "zone_short_id"]).agg(
        frames_observed=("frame_id", "count"),
        mean_count=("zone_count", "mean"),
        median_count=("zone_count", "median"),
        min_count=("zone_count", "min"),
        max_count=("zone_count", "max"),
        std_count=("zone_count", "std"),
        mean_density=("zone_density_pixel", "mean"),
        median_density=("zone_density_pixel", "median"),
        min_density=("zone_density_pixel", "min"),
        max_density=("zone_density_pixel", "max"),
        std_density=("zone_density_pixel", "std"),
        zone_area_pixels=("zone_area_pixels", "first"),
    ).reset_index()

    zone_summary["count_variability"] = zone_summary["std_count"] / (zone_summary["mean_count"] + 1e-9)
    zone_summary["density_variability"] = zone_summary["std_density"] / (zone_summary["mean_density"] + 1e-12)
    zone_summary = zone_summary.sort_values("mean_density", ascending=False)

    zone_summary.to_csv(tables_dir / "zone_count_density_summary_5min_refined.csv", index=False)
    zone_summary.sort_values("mean_count", ascending=False).to_csv(
        tables_dir / "zone_count_ranking_5min_refined.csv",
        index=False,
    )
    zone_summary.sort_values("mean_density", ascending=False).to_csv(
        tables_dir / "zone_density_ranking_5min_refined.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. Risk duration and risk percentage
    # ------------------------------------------------------------------
    risk_counts = pd.crosstab(zone_df["zone_name"], zone_df["risk_level"])

    for level in RISK_LEVELS:
        if level not in risk_counts.columns:
            risk_counts[level] = 0

    risk_counts = risk_counts[RISK_LEVELS].reset_index()

    risk_duration = risk_counts.copy()

    for level in RISK_LEVELS:
        risk_duration[level + "_frames"] = risk_duration[level]
        risk_duration[level + "_seconds"] = risk_duration[level] / estimated_fps

    risk_duration["total_frames"] = risk_duration[RISK_LEVELS].sum(axis=1)
    risk_duration["total_seconds"] = risk_duration["total_frames"] / estimated_fps
    risk_duration["high_or_critical_frames"] = risk_duration["HIGH"] + risk_duration["CRITICAL"]
    risk_duration["high_or_critical_seconds"] = risk_duration["high_or_critical_frames"] / estimated_fps
    risk_duration["high_or_critical_percent"] = (
        risk_duration["high_or_critical_frames"]
        / risk_duration["total_frames"].replace(0, np.nan)
        * 100
    ).fillna(0)

    risk_duration = risk_duration.sort_values("high_or_critical_seconds", ascending=False)
    risk_duration.to_csv(tables_dir / "risk_duration_summary_5min_refined.csv", index=False)

    risk_percent = risk_counts.copy()
    risk_percent_total = risk_percent[RISK_LEVELS].sum(axis=1).replace(0, np.nan)

    for level in RISK_LEVELS:
        risk_percent[level + "_percent"] = (risk_percent[level] / risk_percent_total * 100).fillna(0)

    risk_percent.to_csv(tables_dir / "risk_percentage_summary_5min_refined.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Correlation and lead-lag
    # ------------------------------------------------------------------
    zone_count_wide = zone_df.pivot_table(
        index="frame_id",
        columns="zone_name",
        values="zone_count",
        aggfunc="mean",
    ).fillna(0)

    zone_density_wide = zone_df.pivot_table(
        index="frame_id",
        columns="zone_name",
        values="zone_density_pixel",
        aggfunc="mean",
    ).fillna(0)

    zone_count_corr = zone_count_wide.corr()
    zone_density_corr = zone_density_wide.corr()

    zone_count_corr.to_csv(tables_dir / "zone_count_correlation_matrix_5min_refined.csv")
    zone_density_corr.to_csv(tables_dir / "zone_density_correlation_matrix_5min_refined.csv")

    max_lag_seconds = 5
    max_lag_frames = int(max_lag_seconds * estimated_fps)
    lead_lag_rows = []
    zones = list(zone_count_wide.columns)

    for zone_a in zones:
        for zone_b in zones:
            if zone_a == zone_b:
                continue

            series_a = zone_count_wide[zone_a]
            series_b = zone_count_wide[zone_b]

            best_corr = -999.0
            best_lag = 0

            for lag in range(-max_lag_frames, max_lag_frames + 1):
                shifted_b = series_b.shift(lag)
                corr = series_a.corr(shifted_b)

                if pd.notna(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag

            if best_lag > 0:
                interpretation = f"{zone_a} tends to lead {zone_b} by {abs(best_lag / estimated_fps):.2f}s"
            elif best_lag < 0:
                interpretation = f"{zone_b} tends to lead {zone_a} by {abs(best_lag / estimated_fps):.2f}s"
            else:
                interpretation = f"{zone_a} and {zone_b} peak at approximately the same time"

            lead_lag_rows.append(
                {
                    "zone_a": zone_a,
                    "zone_b": zone_b,
                    "best_lag_frames": best_lag,
                    "best_lag_seconds": best_lag / estimated_fps,
                    "best_correlation": best_corr,
                    "interpretation": interpretation,
                }
            )

    lead_lag_df = pd.DataFrame(lead_lag_rows).sort_values("best_correlation", ascending=False)
    lead_lag_df.to_csv(tables_dir / "zone_lead_lag_summary_5min_refined.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Temporal density trends
    # ------------------------------------------------------------------
    temporal_df = zone_df.copy()
    temporal_df = temporal_df.sort_values(["zone_name", "timestamp_sec"]).reset_index(drop=True)

    temporal_df["count_change"] = temporal_df.groupby("zone_name")["zone_count"].diff().fillna(0)
    temporal_df["density_change"] = temporal_df.groupby("zone_name")["zone_density_pixel"].diff().fillna(0)

    density_std_by_zone = temporal_df.groupby("zone_name")["zone_density_pixel"].transform("std").fillna(0)
    density_threshold = (density_std_by_zone * 0.20).clip(lower=1e-8)

    temporal_df["density_trend"] = np.select(
        [
            temporal_df["density_change"] > density_threshold,
            temporal_df["density_change"] < -density_threshold,
        ],
        [
            "ACCUMULATING",
            "DISPERSING",
        ],
        default="STABLE",
    )

    count_std_by_zone = temporal_df.groupby("zone_name")["zone_count"].transform("std").fillna(0)
    count_threshold = (count_std_by_zone * 0.20).clip(lower=1.0)

    temporal_df["count_trend"] = np.select(
        [
            temporal_df["count_change"] > count_threshold,
            temporal_df["count_change"] < -count_threshold,
        ],
        [
            "ACCUMULATING",
            "DISPERSING",
        ],
        default="STABLE",
    )

    temporal_df.to_csv(tables_dir / "temporal_density_features_5min_refined.csv", index=False)

    trend_summary = pd.crosstab(temporal_df["zone_name"], temporal_df["density_trend"]).reset_index()

    for col in ["ACCUMULATING", "DISPERSING", "STABLE"]:
        if col not in trend_summary.columns:
            trend_summary[col] = 0

    trend_summary["total_frames"] = trend_summary[["ACCUMULATING", "DISPERSING", "STABLE"]].sum(axis=1)

    for col in ["ACCUMULATING", "DISPERSING", "STABLE"]:
        trend_summary[col + "_percent"] = (
            trend_summary[col]
            / trend_summary["total_frames"].replace(0, np.nan)
            * 100
        ).fillna(0)

    trend_summary = trend_summary.sort_values("ACCUMULATING_percent", ascending=False)
    trend_summary.to_csv(tables_dir / "density_trend_summary_5min_refined.csv", index=False)

    rolling_df = temporal_df.copy()
    rolling_df["rolling_mean_count_5s"] = rolling_df.groupby("zone_name")["zone_count"].transform(
        lambda s: s.rolling(rolling_window_frames, min_periods=1).mean()
    )
    rolling_df["rolling_mean_density_5s"] = rolling_df.groupby("zone_name")["zone_density_pixel"].transform(
        lambda s: s.rolling(rolling_window_frames, min_periods=1).mean()
    )
    rolling_df["rolling_mean_density_change_5s"] = rolling_df.groupby("zone_name")["density_change"].transform(
        lambda s: s.rolling(rolling_window_frames, min_periods=1).mean()
    )

    rolling_df.to_csv(tables_dir / "rolling_behavior_features_5s_5min_refined.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Refined sudden spike detection
    # ------------------------------------------------------------------
    spike_window_frames = 30
    min_current_count = 20
    min_absolute_increase = 10
    min_percent_increase = 50.0
    min_event_gap_seconds = 3.0
    min_event_gap_frames = int(min_event_gap_seconds * estimated_fps)

    raw_spike_rows = []

    for zone_name, group in zone_df.groupby("zone_name"):
        group = group.sort_values("frame_id").copy()

        group["count_past"] = group["zone_count"].shift(spike_window_frames)
        group["density_past"] = group["zone_density_pixel"].shift(spike_window_frames)
        group["absolute_count_increase"] = group["zone_count"] - group["count_past"]

        group["count_percent_change"] = (
            group["absolute_count_increase"]
            / group["count_past"].replace(0, np.nan)
            * 100
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        group["density_percent_change"] = (
            (group["zone_density_pixel"] - group["density_past"])
            / group["density_past"].replace(0, np.nan)
            * 100
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        candidates = group[
            (group["zone_count"] >= min_current_count)
            & (group["absolute_count_increase"] >= min_absolute_increase)
            & (group["count_percent_change"] >= min_percent_increase)
        ].copy()

        for _, row in candidates.iterrows():
            raw_spike_rows.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "zone_name": zone_name,
                    "zone_short_id": row.get("zone_short_id", zone_name),
                    "zone_count": float(row["zone_count"]),
                    "count_past_30f": float(row["count_past"]),
                    "absolute_count_increase": float(row["absolute_count_increase"]),
                    "count_percent_change_30f": float(row["count_percent_change"]),
                    "zone_density_pixel": float(row["zone_density_pixel"]),
                    "density_percent_change_30f": float(row["density_percent_change"]),
                    "risk_level": row["risk_level"],
                }
            )

    raw_spike_columns = [
        "frame_id",
        "timestamp_sec",
        "zone_name",
        "zone_short_id",
        "zone_count",
        "count_past_30f",
        "absolute_count_increase",
        "count_percent_change_30f",
        "zone_density_pixel",
        "density_percent_change_30f",
        "risk_level",
    ]

    raw_spikes = pd.DataFrame(raw_spike_rows, columns=raw_spike_columns)
    raw_spikes.to_csv(tables_dir / "raw_sudden_spike_candidates_5min_refined.csv", index=False)

    refined_spike_events_rows = []

    if not raw_spikes.empty:
        for zone_name, group in raw_spikes.groupby("zone_name"):
            group = group.sort_values("frame_id").copy()

            current_event = []
            last_frame = None

            for _, row in group.iterrows():
                if last_frame is None or (row["frame_id"] - last_frame) <= min_event_gap_frames:
                    current_event.append(row)
                else:
                    event_df = pd.DataFrame(current_event)
                    peak = event_df.sort_values("absolute_count_increase", ascending=False).iloc[0]

                    refined_spike_events_rows.append(
                        {
                            "zone_name": zone_name,
                            "zone_short_id": peak["zone_short_id"],
                            "event_start_frame": int(event_df["frame_id"].min()),
                            "event_end_frame": int(event_df["frame_id"].max()),
                            "event_start_sec": float(event_df["timestamp_sec"].min()),
                            "event_end_sec": float(event_df["timestamp_sec"].max()),
                            "event_duration_sec": float(
                                event_df["timestamp_sec"].max() - event_df["timestamp_sec"].min()
                            ),
                            "peak_frame_id": int(peak["frame_id"]),
                            "peak_timestamp_sec": float(peak["timestamp_sec"]),
                            "peak_zone_count": float(peak["zone_count"]),
                            "peak_absolute_increase": float(peak["absolute_count_increase"]),
                            "peak_percent_increase": float(peak["count_percent_change_30f"]),
                            "peak_risk_level": peak["risk_level"],
                        }
                    )

                    current_event = [row]

                last_frame = row["frame_id"]

            if current_event:
                event_df = pd.DataFrame(current_event)
                peak = event_df.sort_values("absolute_count_increase", ascending=False).iloc[0]

                refined_spike_events_rows.append(
                    {
                        "zone_name": zone_name,
                        "zone_short_id": peak["zone_short_id"],
                        "event_start_frame": int(event_df["frame_id"].min()),
                        "event_end_frame": int(event_df["frame_id"].max()),
                        "event_start_sec": float(event_df["timestamp_sec"].min()),
                        "event_end_sec": float(event_df["timestamp_sec"].max()),
                        "event_duration_sec": float(
                            event_df["timestamp_sec"].max() - event_df["timestamp_sec"].min()
                        ),
                        "peak_frame_id": int(peak["frame_id"]),
                        "peak_timestamp_sec": float(peak["timestamp_sec"]),
                        "peak_zone_count": float(peak["zone_count"]),
                        "peak_absolute_increase": float(peak["absolute_count_increase"]),
                        "peak_percent_increase": float(peak["count_percent_change_30f"]),
                        "peak_risk_level": peak["risk_level"],
                    }
                )

    refined_spike_columns = [
        "zone_name",
        "zone_short_id",
        "event_start_frame",
        "event_end_frame",
        "event_start_sec",
        "event_end_sec",
        "event_duration_sec",
        "peak_frame_id",
        "peak_timestamp_sec",
        "peak_zone_count",
        "peak_absolute_increase",
        "peak_percent_increase",
        "peak_risk_level",
    ]

    refined_spike_events = pd.DataFrame(refined_spike_events_rows, columns=refined_spike_columns)
    refined_spike_events.to_csv(tables_dir / "refined_sudden_spike_events_5min.csv", index=False)

    if not refined_spike_events.empty:
        spike_summary = refined_spike_events.groupby("zone_name").agg(
            refined_spike_events=("zone_name", "count"),
            max_peak_zone_count=("peak_zone_count", "max"),
            max_absolute_increase=("peak_absolute_increase", "max"),
            max_percent_increase=("peak_percent_increase", "max"),
        ).reset_index()
    else:
        spike_summary = pd.DataFrame(
            columns=[
                "zone_name",
                "refined_spike_events",
                "max_peak_zone_count",
                "max_absolute_increase",
                "max_percent_increase",
            ]
        )

    spike_summary.to_csv(tables_dir / "refined_sudden_spike_summary_5min.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Z-score anomaly summary
    # ------------------------------------------------------------------
    anomaly_df = temporal_df.copy()

    def add_anomaly_scores(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("timestamp_sec").copy()

        rolling_mean = group["zone_density_pixel"].rolling(rolling_window_frames, min_periods=5).mean()
        rolling_std = group["zone_density_pixel"].rolling(rolling_window_frames, min_periods=5).std()

        group["density_rolling_mean_5s"] = rolling_mean
        group["density_rolling_std_5s"] = rolling_std

        group["density_z_score"] = (
            (group["zone_density_pixel"] - rolling_mean)
            / rolling_std.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        group["density_spike_anomaly"] = (group["density_z_score"] > 2.5).astype(int)

        dc_mean = group["density_change"].rolling(rolling_window_frames, min_periods=5).mean()
        dc_std = group["density_change"].rolling(rolling_window_frames, min_periods=5).std()

        group["density_change_z_score"] = (
            (group["density_change"] - dc_mean)
            / dc_std.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        group["accumulation_anomaly"] = (group["density_change_z_score"] > 2.5).astype(int)
        group["csv_anomaly_score"] = (
            0.60 * group["density_spike_anomaly"]
            + 0.40 * group["accumulation_anomaly"]
        )

        return group

    anomaly_parts = []

    for zone_name, group in anomaly_df.groupby("zone_name"):
        scored_group = add_anomaly_scores(group)
        scored_group["zone_name"] = zone_name
        anomaly_parts.append(scored_group)

    if anomaly_parts:
        anomaly_df = pd.concat(anomaly_parts, ignore_index=True)
    else:
        anomaly_df = pd.DataFrame(columns=list(temporal_df.columns))

    anomaly_df.to_csv(tables_dir / "csv_based_anomaly_features_5min_refined.csv", index=False)

    if not anomaly_df.empty and "zone_name" in anomaly_df.columns:
        anomaly_summary = anomaly_df.groupby("zone_name").agg(
            density_spike_anomalies=("density_spike_anomaly", "sum"),
            accumulation_anomalies=("accumulation_anomaly", "sum"),
            mean_density_z_score=("density_z_score", "mean"),
            max_density_z_score=("density_z_score", "max"),
            mean_density_change_z_score=("density_change_z_score", "mean"),
            max_density_change_z_score=("density_change_z_score", "max"),
        ).reset_index()
    else:
        anomaly_summary = pd.DataFrame(
            columns=[
                "zone_name",
                "density_spike_anomalies",
                "accumulation_anomalies",
                "mean_density_z_score",
                "max_density_z_score",
                "mean_density_change_z_score",
                "max_density_change_z_score",
            ]
        )

    anomaly_summary.to_csv(tables_dir / "csv_based_anomaly_summary_5min_refined.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Refined multi-zone alerts
    # ------------------------------------------------------------------
    min_high_critical_zones = 3
    min_alert_duration_sec = 3.0
    median_global_count = frame_df["total_count"].median()

    zone_alert_df = zone_df.copy()
    zone_alert_df["is_high_or_critical"] = zone_alert_df["risk_level"].isin(["HIGH", "CRITICAL"]).astype(int)

    multi_zone_timeline = zone_alert_df.groupby("frame_id").agg(
        timestamp_sec=("timestamp_sec", "first"),
        high_or_critical_zone_count=("is_high_or_critical", "sum"),
        total_zones=("zone_name", "nunique"),
    ).reset_index()

    zones_high = (
        zone_alert_df[zone_alert_df["is_high_or_critical"] == 1]
        .groupby("frame_id")["zone_name"]
        .apply(lambda x: ", ".join(sorted(x.unique())))
        .reset_index()
        .rename(columns={"zone_name": "zones_high_or_critical"})
    )

    multi_zone_timeline = multi_zone_timeline.merge(zones_high, on="frame_id", how="left")
    multi_zone_timeline["zones_high_or_critical"] = multi_zone_timeline["zones_high_or_critical"].fillna("")

    multi_zone_timeline = multi_zone_timeline.merge(
        frame_df[["frame_id", "total_count"]],
        on="frame_id",
        how="left",
    )

    multi_zone_timeline["raw_multi_zone_alert"] = (
        (multi_zone_timeline["high_or_critical_zone_count"] >= min_high_critical_zones)
        & (multi_zone_timeline["total_count"] >= median_global_count)
    ).astype(int)

    alert_events = []
    current_event = []

    for _, row in multi_zone_timeline.sort_values("frame_id").iterrows():
        if row["raw_multi_zone_alert"] == 1:
            current_event.append(row)
        else:
            if current_event:
                event_df = pd.DataFrame(current_event)
                duration = event_df["timestamp_sec"].max() - event_df["timestamp_sec"].min()

                if duration >= min_alert_duration_sec:
                    peak = event_df.sort_values(
                        ["high_or_critical_zone_count", "total_count"],
                        ascending=False,
                    ).iloc[0]

                    alert_events.append(
                        {
                            "event_start_frame": int(event_df["frame_id"].min()),
                            "event_end_frame": int(event_df["frame_id"].max()),
                            "event_start_sec": float(event_df["timestamp_sec"].min()),
                            "event_end_sec": float(event_df["timestamp_sec"].max()),
                            "event_duration_sec": float(duration),
                            "peak_frame_id": int(peak["frame_id"]),
                            "peak_timestamp_sec": float(peak["timestamp_sec"]),
                            "peak_high_or_critical_zone_count": int(peak["high_or_critical_zone_count"]),
                            "peak_total_count": float(peak["total_count"]),
                            "zones_high_or_critical_at_peak": peak["zones_high_or_critical"],
                        }
                    )

                current_event = []

    if current_event:
        event_df = pd.DataFrame(current_event)
        duration = event_df["timestamp_sec"].max() - event_df["timestamp_sec"].min()

        if duration >= min_alert_duration_sec:
            peak = event_df.sort_values(
                ["high_or_critical_zone_count", "total_count"],
                ascending=False,
            ).iloc[0]

            alert_events.append(
                {
                    "event_start_frame": int(event_df["frame_id"].min()),
                    "event_end_frame": int(event_df["frame_id"].max()),
                    "event_start_sec": float(event_df["timestamp_sec"].min()),
                    "event_end_sec": float(event_df["timestamp_sec"].max()),
                    "event_duration_sec": float(duration),
                    "peak_frame_id": int(peak["frame_id"]),
                    "peak_timestamp_sec": float(peak["timestamp_sec"]),
                    "peak_high_or_critical_zone_count": int(peak["high_or_critical_zone_count"]),
                    "peak_total_count": float(peak["total_count"]),
                    "zones_high_or_critical_at_peak": peak["zones_high_or_critical"],
                }
            )

    refined_multi_zone_alert_columns = [
        "event_start_frame",
        "event_end_frame",
        "event_start_sec",
        "event_end_sec",
        "event_duration_sec",
        "peak_frame_id",
        "peak_timestamp_sec",
        "peak_high_or_critical_zone_count",
        "peak_total_count",
        "zones_high_or_critical_at_peak",
    ]

    refined_multi_zone_alerts = pd.DataFrame(alert_events, columns=refined_multi_zone_alert_columns)

    multi_zone_timeline.to_csv(tables_dir / "multi_zone_risk_timeline_5min_refined.csv", index=False)
    refined_multi_zone_alerts.to_csv(tables_dir / "refined_multi_zone_alert_events_5min.csv", index=False)

    # ------------------------------------------------------------------
    # 9. Entropy
    # ------------------------------------------------------------------
    entropy_rows = []

    for frame_id, group in zone_df.groupby("frame_id"):
        counts = group["zone_count"].values.astype(float)
        timestamp = group["timestamp_sec"].iloc[0]
        total = counts.sum()

        if total <= 0:
            entropy = 0.0
            normalized_entropy = 0.0
        else:
            p = counts / total
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))
            normalized_entropy = entropy / np.log(len(counts)) if len(counts) > 1 else 0.0

        entropy_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_sec": timestamp,
                "zone_total_count": total,
                "crowd_distribution_entropy": entropy,
                "normalized_entropy": normalized_entropy,
            }
        )

    entropy_df = pd.DataFrame(entropy_rows).sort_values("timestamp_sec")

    entropy_summary = pd.DataFrame(
        [
            {
                "mean_entropy": entropy_df["crowd_distribution_entropy"].mean(),
                "median_entropy": entropy_df["crowd_distribution_entropy"].median(),
                "min_entropy": entropy_df["crowd_distribution_entropy"].min(),
                "max_entropy": entropy_df["crowd_distribution_entropy"].max(),
                "mean_normalized_entropy": entropy_df["normalized_entropy"].mean(),
                "median_normalized_entropy": entropy_df["normalized_entropy"].median(),
                "min_normalized_entropy": entropy_df["normalized_entropy"].min(),
                "max_normalized_entropy": entropy_df["normalized_entropy"].max(),
            }
        ]
    )

    entropy_df.to_csv(tables_dir / "crowd_entropy_over_time_5min_refined.csv", index=False)
    entropy_summary.to_csv(tables_dir / "crowd_entropy_summary_5min_refined.csv", index=False)

    # ------------------------------------------------------------------
    # 10. Risk transition matrix
    # ------------------------------------------------------------------
    transition_rows = []

    for zone_name, group in zone_df.groupby("zone_name"):
        group = group.sort_values("frame_id").copy()
        current_levels = group["risk_level"].values[:-1]
        next_levels = group["risk_level"].values[1:]

        for current, nxt in zip(current_levels, next_levels):
            transition_rows.append(
                {
                    "zone_name": zone_name,
                    "from_risk": current,
                    "to_risk": nxt,
                }
            )

    risk_transition_df = pd.DataFrame(transition_rows)

    risk_transition_matrix = pd.crosstab(
        risk_transition_df["from_risk"],
        risk_transition_df["to_risk"],
    )

    for level in RISK_LEVELS:
        if level not in risk_transition_matrix.index:
            risk_transition_matrix.loc[level] = 0
        if level not in risk_transition_matrix.columns:
            risk_transition_matrix[level] = 0

    risk_transition_matrix = risk_transition_matrix.loc[RISK_LEVELS, RISK_LEVELS]
    risk_transition_matrix.to_csv(tables_dir / "risk_transition_matrix_5min_refined.csv")

    risk_transition_by_zone = (
        risk_transition_df.groupby(["zone_name", "from_risk", "to_risk"])
        .size()
        .reset_index(name="transition_count")
        .sort_values(["zone_name", "transition_count"], ascending=[True, False])
    )

    risk_transition_by_zone.to_csv(tables_dir / "risk_transition_by_zone_5min_refined.csv", index=False)

    # ------------------------------------------------------------------
    # 11. Figures
    # ------------------------------------------------------------------
    print("Saving figures...")

    plot_global_count(frame_df, figures_dir)
    plot_global_change(frame_df, figures_dir)
    plot_inference_fps(frame_df, figures_dir)

    plot_barh(
        zone_summary,
        "mean_count",
        "zone_name",
        "Mean Count by Zone — 5 Minute Video",
        "Mean Zone Count",
        figures_dir / "mean_count_by_zone_5min_refined.png",
    )

    plot_barh(
        zone_summary,
        "max_count",
        "zone_name",
        "Maximum Count by Zone — 5 Minute Video",
        "Max Zone Count",
        figures_dir / "max_count_by_zone_5min_refined.png",
    )

    plot_barh(
        zone_summary,
        "mean_density",
        "zone_name",
        "Mean Pixel-Based Density by Zone — 5 Minute Video",
        "Mean Pixel-Based Density",
        figures_dir / "mean_density_by_zone_5min_refined.png",
    )

    plot_barh(
        zone_summary,
        "max_density",
        "zone_name",
        "Maximum Pixel-Based Density by Zone — 5 Minute Video",
        "Max Pixel-Based Density",
        figures_dir / "max_density_by_zone_5min_refined.png",
    )

    plot_risk_percentage(risk_percent, figures_dir)

    plot_barh(
        risk_duration,
        "high_or_critical_seconds",
        "zone_name",
        "HIGH/CRITICAL Risk Duration by Zone — 5 Minute Video",
        "Duration (seconds)",
        figures_dir / "high_critical_risk_duration_by_zone_5min_refined.png",
    )

    plot_density_trend(trend_summary, figures_dir)
    plot_correlation_heatmap(zone_count_corr, figures_dir)
    plot_entropy(entropy_df, figures_dir)
    plot_risk_transition_matrix(risk_transition_matrix, figures_dir)

    if not spike_summary.empty:
        plot_barh(
            spike_summary,
            "refined_spike_events",
            "zone_name",
            "Refined Sudden Spike Events by Zone — 5 Minute Video",
            "Number of Refined Spike Events",
            figures_dir / "refined_spike_events_by_zone_5min.png",
        )

    if not refined_multi_zone_alerts.empty:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(refined_multi_zone_alerts)), refined_multi_zone_alerts["event_duration_sec"])
        plt.title("Refined Multi-Zone Alert Event Durations — 5 Minute Video")
        plt.xlabel("Alert Event Index")
        plt.ylabel("Duration (seconds)")
        plt.grid(axis="y", alpha=0.4)
        save_figure(figures_dir / "refined_multi_zone_alert_durations_5min.png")

    # ------------------------------------------------------------------
    # 12. Insights
    # ------------------------------------------------------------------
    highest_mean_count = zone_summary.sort_values("mean_count", ascending=False).iloc[0]
    highest_max_count = zone_summary.sort_values("max_count", ascending=False).iloc[0]
    highest_mean_density = zone_summary.sort_values("mean_density", ascending=False).iloc[0]
    highest_max_density = zone_summary.sort_values("max_density", ascending=False).iloc[0]
    highest_risk_duration = risk_duration.sort_values("high_or_critical_seconds", ascending=False).iloc[0]
    highest_accumulating = trend_summary.sort_values("ACCUMULATING_percent", ascending=False).iloc[0]
    peak_frame = top_10_peak_frames.iloc[0]
    top_corr = lead_lag_df.iloc[0]
    highest_entropy_frame = entropy_df.sort_values("normalized_entropy", ascending=False).iloc[0]
    lowest_entropy_frame = entropy_df.sort_values("normalized_entropy", ascending=True).iloc[0]

    if not spike_summary.empty:
        highest_spike_zone = spike_summary.sort_values("refined_spike_events", ascending=False).iloc[0]
        highest_spike_zone_name = highest_spike_zone["zone_name"]
        highest_spike_count = int(highest_spike_zone["refined_spike_events"])
    else:
        highest_spike_zone_name = "None"
        highest_spike_count = 0

    refined_multi_zone_alert_total_duration = (
        refined_multi_zone_alerts["event_duration_sec"].sum()
        if not refined_multi_zone_alerts.empty
        else 0
    )

    refined_multi_zone_alert_percent = (
        refined_multi_zone_alert_total_duration / video_duration_sec * 100
        if video_duration_sec > 0
        else 0
    )

    insights = [
        "REFINED 5-MINUTE SHINJUKU CROWD ANALYSIS INSIGHTS",
        "=" * 70,
        "",
        "Dataset Overview",
        f"- Video duration: {video_duration_sec:.2f} seconds.",
        f"- Processed frames: {processed_frames}.",
        f"- Estimated video FPS: {estimated_fps:.2f}.",
        f"- Number of zones: {zone_df['zone_name'].nunique()}.",
        "",
        "1. Global Crowd Timeline",
        f"- Average total count: {global_summary.loc[0, 'mean_total_count']:.2f}.",
        f"- Median total count: {global_summary.loc[0, 'median_total_count']:.2f}.",
        f"- Maximum total count: {global_summary.loc[0, 'max_total_count']:.0f}.",
        f"- Peak moment occurred at {peak_frame['timestamp_sec']:.2f} seconds, frame {int(peak_frame['frame_id'])}.",
        f"- Standard deviation of total count: {global_summary.loc[0, 'std_total_count']:.2f}.",
        "",
        "2. Full Pipeline Performance",
        f"- Average full-pipeline FPS: {global_summary.loc[0, 'mean_full_pipeline_fps']:.2f}.",
        f"- Average processing time per frame: {global_summary.loc[0, 'mean_processing_time_sec']:.4f} seconds.",
        "- This is the full offline pipeline speed, including inference, point extraction, zone assignment, visualization, and logging.",
        "",
        "3. Zone Hotspot Ranking",
        f"- Highest average count zone: {highest_mean_count['zone_name']} with mean count {highest_mean_count['mean_count']:.2f}.",
        f"- Highest peak count zone: {highest_max_count['zone_name']} with max count {highest_max_count['max_count']:.0f}.",
        f"- Highest average density zone: {highest_mean_density['zone_name']} with mean density {highest_mean_density['mean_density']:.8f}.",
        f"- Highest maximum density zone: {highest_max_density['zone_name']} with max density {highest_max_density['max_density']:.8f}.",
        "- Density is pixel-based and relative to polygon area, not real-world persons per square meter.",
        "",
        "4. Risk Time Distribution",
        f"- Longest HIGH/CRITICAL risk duration zone: {highest_risk_duration['zone_name']}.",
        f"- HIGH/CRITICAL duration: {highest_risk_duration['high_or_critical_seconds']:.2f} seconds.",
        f"- HIGH/CRITICAL percentage: {highest_risk_duration['high_or_critical_percent']:.2f}% of frames for that zone.",
        "",
        "5. Temporal Density Trend",
        f"- Highest accumulation percentage zone: {highest_accumulating['zone_name']}.",
        f"- Accumulating frames percentage: {highest_accumulating['ACCUMULATING_percent']:.2f}%.",
        "- This indicates which zone most frequently increased in density over time.",
        "",
        "6. Zone Correlation and Lead-Lag",
        f"- Strongest lead-lag relationship: {top_corr['zone_a']} and {top_corr['zone_b']} with correlation {top_corr['best_correlation']:.3f}.",
        f"- Interpretation: {top_corr['interpretation']}.",
        "- This helps identify zones that fill together or sequentially.",
        "",
        "7. Refined Anomaly Detection",
        "- The sudden spike rule was refined to avoid counting tiny percentage changes as anomalies.",
        f"- Refined sudden spike events detected: {len(refined_spike_events)}.",
        f"- Zone with most refined spike events: {highest_spike_zone_name} with {highest_spike_count} events.",
        "- Refined spike rule: current count >= 20, absolute increase >= 10, percent increase >= 50%, compared to 30 frames earlier.",
        f"- Refined multi-zone alert events detected: {len(refined_multi_zone_alerts)}.",
        f"- Total refined multi-zone alert duration: {refined_multi_zone_alert_total_duration:.2f} seconds.",
        f"- Refined multi-zone alert percentage of video: {refined_multi_zone_alert_percent:.2f}%.",
        "- Refined multi-zone rule: at least 3 HIGH/CRITICAL zones, total count above median, and duration of at least 3 seconds.",
        "",
        "8. Statistical Insights",
        f"- Mean normalized crowd distribution entropy: {entropy_summary.loc[0, 'mean_normalized_entropy']:.4f}.",
        f"- Highest entropy occurred at {highest_entropy_frame['timestamp_sec']:.2f} seconds, meaning the crowd was most evenly spread across zones.",
        f"- Lowest entropy occurred at {lowest_entropy_frame['timestamp_sec']:.2f} seconds, meaning the crowd was most concentrated in fewer zones.",
        "- The risk transition matrix shows how zone risk states evolve between consecutive frames.",
        "",
        "Conclusion",
        "- The refined 5-minute analysis provides reliable evidence for global crowd trends, zone hotspots, density behavior, risk duration, and statistical crowd distribution.",
        "- The anomaly section is now more selective and thesis-safe compared with the initial over-sensitive rule.",
        "- Motion-based stagnation or congestion detection is intentionally left for the optical-flow stage.",
    ]

    insights_path = insights_dir / "automatic_analysis_insights_5min_refined.txt"
    insights_path.write_text("\n".join(insights), encoding="utf-8")

    summary_json = {
        "dataset_overview": {
            "video_duration_sec": video_duration_sec,
            "processed_frames": processed_frames,
            "estimated_video_fps": estimated_fps,
            "num_zones": int(zone_df["zone_name"].nunique()),
        },
        "global_summary": global_summary.to_dict(orient="records")[0],
        "highest_mean_count_zone": highest_mean_count.to_dict(),
        "highest_max_count_zone": highest_max_count.to_dict(),
        "highest_mean_density_zone": highest_mean_density.to_dict(),
        "highest_max_density_zone": highest_max_density.to_dict(),
        "highest_high_or_critical_duration_zone": highest_risk_duration.to_dict(),
        "highest_accumulating_zone": highest_accumulating.to_dict(),
        "top_peak_frame": peak_frame.to_dict(),
        "top_lead_lag_relationship": top_corr.to_dict(),
        "refined_anomaly_summary": {
            "refined_spike_events_total": int(len(refined_spike_events)),
            "highest_spike_zone": highest_spike_zone_name,
            "highest_spike_zone_event_count": highest_spike_count,
            "refined_multi_zone_alert_events_total": int(len(refined_multi_zone_alerts)),
            "refined_multi_zone_alert_total_duration_sec": float(refined_multi_zone_alert_total_duration),
            "refined_multi_zone_alert_percent": float(refined_multi_zone_alert_percent),
        },
        "entropy_summary": entropy_summary.to_dict(orient="records")[0],
    }

    with (insights_dir / "analysis_summary_5min_refined.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)

    # ------------------------------------------------------------------
    # 13. ZIP backup
    # ------------------------------------------------------------------
    zip_path = zip_dir / "analysis_5min_refined_outputs.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.rglob("*"):
            if file.is_file() and file != zip_path:
                zipf.write(file, file.relative_to(output_dir))

    print("\n✅ Refined 5-minute analysis complete")
    print("=" * 70)
    print(f"Output folder: {output_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Insights: {insights_dir}")
    print(f"ZIP backup: {zip_path}")
    print("")
    print("Main insights:")
    print("\n".join(insights))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run refined 5-minute Shinjuku CSV analysis.")

    parser.add_argument(
        "--frame-csv",
        type=str,
        default="results/benchmark/FULL_01_shinjuku_frame_counts.csv",
        help="Frame-level CSV path.",
    )

    parser.add_argument(
        "--zone-csv",
        type=str,
        default="results/benchmark/FULL_02_shinjuku_zone_density_risk.csv",
        help="Zone-level CSV path.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis_5min_refined",
        help="Output analysis folder.",
    )

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the old analysis folder before writing.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_analysis(
        frame_csv=resolve_path(args.frame_csv),
        zone_csv=resolve_path(args.zone_csv),
        output_dir=resolve_path(args.output_dir),
        clean_output=not args.no_clean,
    )


if __name__ == "__main__":
    main()