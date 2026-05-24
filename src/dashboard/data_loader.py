"""
Dashboard data loader.

This module loads the real saved outputs from the project:
- frame-level CSV
- zone-level CSV
- refined analysis tables
- refined analysis insights
- rendered videos
- midframe preview image

The dashboard does not rerun FIDTM.
It reads the outputs already produced by the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_DIR = RESULTS_DIR / "benchmark"
VIDEOS_DIR = RESULTS_DIR / "videos"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

ANALYSIS_DIR = RESULTS_DIR / "analysis_5min_refined"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
INSIGHTS_DIR = ANALYSIS_DIR / "insights"

FRAME_CSV = BENCHMARK_DIR / "FULL_01_shinjuku_frame_counts.csv"
ZONE_CSV = BENCHMARK_DIR / "FULL_02_shinjuku_zone_density_risk.csv"

MIDFRAME_IMAGE = VISUALIZATIONS_DIR / "FULL_shinjuku_side_by_side_midframe.jpg"

RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

ZONE_SHORT_IDS = {
    "crosswalk_main": "CW1",
    "crosswalk_left": "CW2",
    "crosswalk_top": "CW3",
    "crosswalk_bottom": "CW4",
    "sidewalk_top": "SW1",
    "sidewalk_right": "SW2",
    "sidewalk_bottom": "SW3",
    "sidewalk_left": "SW4",
}


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not file_exists(path):
        raise FileNotFoundError(f"Missing CSV file: {path}")
    return pd.read_csv(path)


def safe_read_text(path: Path) -> str:
    if not file_exists(path):
        return f"Missing text file: {path}"
    return path.read_text(encoding="utf-8")


def safe_read_json(path: Path) -> dict[str, Any]:
    if not file_exists(path):
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def short_zone_name(zone_name: str) -> str:
    return ZONE_SHORT_IDS.get(zone_name, zone_name)


def format_seconds(seconds: float) -> str:
    if pd.isna(seconds):
        return "0:00"

    minutes = int(seconds // 60)
    sec = int(seconds % 60)

    return f"{minutes}:{sec:02d}"


def estimate_video_fps(frame_df: pd.DataFrame) -> float:
    diffs = frame_df["timestamp_sec"].diff().dropna()
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 30.0

    median_dt = diffs.median()

    if pd.isna(median_dt) or median_dt <= 0:
        return 30.0

    return float(1.0 / median_dt)


def get_video_files() -> dict[str, Path]:
    """
    Return available rendered video outputs.
    """
    videos = {
        "Localization + Count": VIDEOS_DIR / "FULL_01_shinjuku_fidtm_localization_count.mp4",
        "Heatmap Overlay": VIDEOS_DIR / "FULL_02_shinjuku_fidtm_heatmap_overlay_points.mp4",
        "Heatmap Only": VIDEOS_DIR / "FULL_03_shinjuku_fidtm_heatmap_only_points.mp4",
        "Zone Density + Risk": VIDEOS_DIR / "FULL_04_shinjuku_fidtm_zone_density_risk.mp4",
    }

    return {name: path for name, path in videos.items() if file_exists(path)}


def load_frame_data() -> pd.DataFrame:
    """
    Load frame-level CSV and add derived dashboard columns.
    """
    df = safe_read_csv(FRAME_CSV)
    df = df.sort_values("timestamp_sec").reset_index(drop=True)

    required = [
        "frame_id",
        "timestamp_sec",
        "total_count",
        "inference_time_sec",
        "inference_fps",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Frame CSV missing required columns: {missing}")

    estimated_fps = estimate_video_fps(df)
    rolling_window = max(3, int(estimated_fps * 5))

    df["total_count_change"] = df["total_count"].diff().fillna(0)
    df["total_count_change_abs"] = df["total_count_change"].abs()

    df["rolling_mean_count_5s"] = df["total_count"].rolling(
        rolling_window,
        min_periods=1,
    ).mean()

    df["rolling_mean_count_change_5s"] = df["total_count_change"].rolling(
        rolling_window,
        min_periods=1,
    ).mean()

    df["time_label"] = df["timestamp_sec"].apply(format_seconds)

    return df


def load_zone_data() -> pd.DataFrame:
    """
    Load zone-level CSV and add derived dashboard columns.
    """
    df = safe_read_csv(ZONE_CSV)
    df = df.sort_values(["zone_name", "timestamp_sec"]).reset_index(drop=True)

    required = [
        "frame_id",
        "timestamp_sec",
        "zone_name",
        "zone_short_id",
        "zone_count",
        "zone_density_pixel",
        "zone_area_pixels",
        "risk_level",
        "inference_time_sec",
        "inference_fps",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Zone CSV missing required columns: {missing}")

    df["zone_short_id"] = df["zone_name"].map(short_zone_name).fillna(df["zone_short_id"])

    df["risk_level_text"] = df["risk_level"].astype(str)
    df["risk_level"] = pd.Categorical(
        df["risk_level_text"],
        categories=RISK_ORDER,
        ordered=True,
    )

    df["is_high_or_critical"] = df["risk_level_text"].isin(["HIGH", "CRITICAL"])

    df["count_change"] = df.groupby("zone_name")["zone_count"].diff().fillna(0)
    df["density_change"] = df.groupby("zone_name")["zone_density_pixel"].diff().fillna(0)

    density_std = df.groupby("zone_name")["zone_density_pixel"].transform("std").fillna(0)
    trend_threshold = (density_std * 0.20).clip(lower=1e-8)

    df["density_trend"] = np.select(
        [
            df["density_change"] > trend_threshold,
            df["density_change"] < -trend_threshold,
        ],
        [
            "ACCUMULATING",
            "DISPERSING",
        ],
        default="STABLE",
    )

    df["time_label"] = df["timestamp_sec"].apply(format_seconds)

    return df


def load_analysis_tables() -> dict[str, pd.DataFrame]:
    """
    Load refined analysis tables from results/analysis_5min_refined/tables.
    """
    table_paths = {
        "global_summary": TABLES_DIR / "global_count_summary_5min_refined.csv",
        "global_change": TABLES_DIR / "global_count_change_features_5min_refined.csv",
        "top_peaks": TABLES_DIR / "top_10_peak_crowd_frames_5min_refined.csv",
        "zone_summary": TABLES_DIR / "zone_count_density_summary_5min_refined.csv",
        "zone_count_ranking": TABLES_DIR / "zone_count_ranking_5min_refined.csv",
        "zone_density_ranking": TABLES_DIR / "zone_density_ranking_5min_refined.csv",
        "risk_duration": TABLES_DIR / "risk_duration_summary_5min_refined.csv",
        "risk_percentage": TABLES_DIR / "risk_percentage_summary_5min_refined.csv",
        "temporal_density": TABLES_DIR / "temporal_density_features_5min_refined.csv",
        "rolling_behavior": TABLES_DIR / "rolling_behavior_features_5s_5min_refined.csv",
        "density_trend": TABLES_DIR / "density_trend_summary_5min_refined.csv",
        "zone_count_correlation": TABLES_DIR / "zone_count_correlation_matrix_5min_refined.csv",
        "zone_density_correlation": TABLES_DIR / "zone_density_correlation_matrix_5min_refined.csv",
        "lead_lag": TABLES_DIR / "zone_lead_lag_summary_5min_refined.csv",
        "raw_spike_candidates": TABLES_DIR / "raw_sudden_spike_candidates_5min_refined.csv",
        "spike_events": TABLES_DIR / "refined_sudden_spike_events_5min.csv",
        "spike_summary": TABLES_DIR / "refined_sudden_spike_summary_5min.csv",
        "csv_anomaly_features": TABLES_DIR / "csv_based_anomaly_features_5min_refined.csv",
        "csv_anomaly_summary": TABLES_DIR / "csv_based_anomaly_summary_5min_refined.csv",
        "multi_zone_timeline": TABLES_DIR / "multi_zone_risk_timeline_5min_refined.csv",
        "multi_zone_alerts": TABLES_DIR / "refined_multi_zone_alert_events_5min.csv",
        "entropy_over_time": TABLES_DIR / "crowd_entropy_over_time_5min_refined.csv",
        "entropy_summary": TABLES_DIR / "crowd_entropy_summary_5min_refined.csv",
        "risk_transition_matrix": TABLES_DIR / "risk_transition_matrix_5min_refined.csv",
        "risk_transition_by_zone": TABLES_DIR / "risk_transition_by_zone_5min_refined.csv",
    }

    tables: dict[str, pd.DataFrame] = {}

    for name, path in table_paths.items():
        if file_exists(path):
            try:
                tables[name] = pd.read_csv(path)
            except Exception:
                tables[name] = pd.DataFrame()

    return tables


def load_analysis_insights() -> dict[str, Any]:
    """
    Load refined text and JSON insights.
    """
    text = safe_read_text(INSIGHTS_DIR / "automatic_analysis_insights_5min_refined.txt")
    data = safe_read_json(INSIGHTS_DIR / "analysis_summary_5min_refined.json")

    return {
        "text": text,
        "json": data,
    }


def get_available_zones(zone_df: pd.DataFrame) -> list[str]:
    return sorted(zone_df["zone_name"].dropna().unique().tolist())


def compute_global_kpis(
    frame_df: pd.DataFrame,
    zone_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """
    Compute the top KPI cards.
    """
    estimated_fps = estimate_video_fps(frame_df)

    duration_sec = float(frame_df["timestamp_sec"].max() - frame_df["timestamp_sec"].min())
    processed_frames = int(len(frame_df))

    avg_count = float(frame_df["total_count"].mean())
    median_count = float(frame_df["total_count"].median())
    max_count = int(frame_df["total_count"].max())

    peak_row = frame_df.loc[frame_df["total_count"].idxmax()]

    avg_pipeline_fps = float(frame_df["inference_fps"].mean())
    avg_processing_time = float(frame_df["inference_time_sec"].mean())

    risk_duration = tables.get("risk_duration", pd.DataFrame())
    spike_events = tables.get("spike_events", pd.DataFrame())
    multi_zone_alerts = tables.get("multi_zone_alerts", pd.DataFrame())

    if not risk_duration.empty and "high_or_critical_seconds" in risk_duration.columns:
        hotspot_row = risk_duration.sort_values(
            "high_or_critical_seconds",
            ascending=False,
        ).iloc[0]

        hotspot_zone = str(hotspot_row["zone_name"])
        hotspot_pct = float(hotspot_row.get("high_or_critical_percent", 0))
    else:
        zone_high = (
            zone_df.groupby("zone_name")["is_high_or_critical"]
            .mean()
            .sort_values(ascending=False)
        )

        hotspot_zone = str(zone_high.index[0])
        hotspot_pct = float(zone_high.iloc[0] * 100)

    anomaly_count = int(len(spike_events) + len(multi_zone_alerts))

    return {
        "duration_sec": duration_sec,
        "duration_label": format_seconds(duration_sec),
        "processed_frames": processed_frames,
        "estimated_fps": estimated_fps,
        "avg_count": avg_count,
        "median_count": median_count,
        "max_count": max_count,
        "peak_frame_id": int(peak_row["frame_id"]),
        "peak_timestamp_sec": float(peak_row["timestamp_sec"]),
        "peak_time_label": format_seconds(float(peak_row["timestamp_sec"])),
        "avg_pipeline_fps": avg_pipeline_fps,
        "avg_processing_time": avg_processing_time,
        "hotspot_zone": hotspot_zone,
        "hotspot_short": short_zone_name(hotspot_zone),
        "hotspot_high_critical_pct": hotspot_pct,
        "anomaly_count": anomaly_count,
    }


def compute_zone_kpis(
    zone_df: pd.DataFrame,
    zone_name: str,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """
    Compute useful KPIs for the selected zone.
    """
    zdf = zone_df[zone_df["zone_name"] == zone_name].copy()

    if zdf.empty:
        return {}

    zdf_sorted = zdf.sort_values("timestamp_sec")

    avg_count = float(zdf["zone_count"].mean())
    median_count = float(zdf["zone_count"].median())
    max_count = int(zdf["zone_count"].max())
    peak_row = zdf.loc[zdf["zone_count"].idxmax()]

    mean_density = float(zdf["zone_density_pixel"].mean())
    max_density = float(zdf["zone_density_pixel"].max())

    high_critical_pct = float(zdf["is_high_or_critical"].mean() * 100)

    current_row = zdf_sorted.iloc[-1]
    current_count = int(current_row["zone_count"])
    current_density = float(current_row["zone_density_pixel"])
    current_risk = str(current_row["risk_level_text"])

    risk_duration = tables.get("risk_duration", pd.DataFrame())
    density_trend = tables.get("density_trend", pd.DataFrame())
    spike_events = tables.get("spike_events", pd.DataFrame())

    high_critical_seconds = None

    if not risk_duration.empty and zone_name in risk_duration["zone_name"].values:
        row = risk_duration[risk_duration["zone_name"] == zone_name].iloc[0]
        high_critical_seconds = float(row.get("high_or_critical_seconds", 0))

    trend_summary = {
        "ACCUMULATING_percent": 0.0,
        "DISPERSING_percent": 0.0,
        "STABLE_percent": 0.0,
    }

    if not density_trend.empty and zone_name in density_trend["zone_name"].values:
        row = density_trend[density_trend["zone_name"] == zone_name].iloc[0]

        for key in trend_summary:
            trend_summary[key] = float(row.get(key, 0))

    zone_spike_events = pd.DataFrame()

    if not spike_events.empty and "zone_name" in spike_events.columns:
        zone_spike_events = spike_events[spike_events["zone_name"] == zone_name]

    return {
        "zone_name": zone_name,
        "zone_short_id": short_zone_name(zone_name),
        "avg_count": avg_count,
        "median_count": median_count,
        "max_count": max_count,
        "peak_timestamp_sec": float(peak_row["timestamp_sec"]),
        "peak_time_label": format_seconds(float(peak_row["timestamp_sec"])),
        "mean_density": mean_density,
        "max_density": max_density,
        "high_critical_pct": high_critical_pct,
        "current_count": current_count,
        "current_density": current_density,
        "current_risk": current_risk,
        "high_critical_seconds": high_critical_seconds,
        "trend_summary": trend_summary,
        "spike_event_count": int(len(zone_spike_events)),
    }


def build_zone_insight_text(zone_kpis: dict[str, Any]) -> str:
    """
    Build a readable selected-zone insight paragraph.
    """
    if not zone_kpis:
        return "No zone information available."

    high_seconds = zone_kpis.get("high_critical_seconds")

    if high_seconds is None:
        risk_text = f"{zone_kpis['high_critical_pct']:.2f}% of frames were HIGH/CRITICAL."
    else:
        risk_text = (
            f"{zone_kpis['high_critical_pct']:.2f}% of frames were HIGH/CRITICAL, "
            f"equal to about {high_seconds:.2f} seconds."
        )

    trends = zone_kpis["trend_summary"]

    return (
        f"{zone_kpis['zone_name']} ({zone_kpis['zone_short_id']}) had an average count of "
        f"{zone_kpis['avg_count']:.2f}, a peak count of {zone_kpis['max_count']} at "
        f"{zone_kpis['peak_time_label']}, and a mean pixel-based density of "
        f"{zone_kpis['mean_density']:.8f}. {risk_text} "
        f"Density behavior was {trends['ACCUMULATING_percent']:.2f}% accumulating, "
        f"{trends['DISPERSING_percent']:.2f}% dispersing, and "
        f"{trends['STABLE_percent']:.2f}% stable. "
        f"Refined spike events in this zone: {zone_kpis['spike_event_count']}."
    )


def build_global_summary_text(global_kpis: dict[str, Any]) -> str:
    """
    Build a readable global summary.
    """
    return (
        f"The 5-minute Shinjuku experiment processed {global_kpis['processed_frames']:,} frames "
        f"over {global_kpis['duration_label']}. The average crowd count was "
        f"{global_kpis['avg_count']:.2f}, with a peak of {global_kpis['max_count']} at "
        f"{global_kpis['peak_time_label']}. The most persistent high-risk hotspot was "
        f"{global_kpis['hotspot_zone']} ({global_kpis['hotspot_short']}), with "
        f"{global_kpis['hotspot_high_critical_pct']:.2f}% HIGH/CRITICAL risk time. "
        f"The full offline pipeline ran at an average of {global_kpis['avg_pipeline_fps']:.2f} FPS."
    )


def load_dashboard_data() -> dict[str, Any]:
    """
    Load everything required by the Streamlit dashboard.
    """
    frame_df = load_frame_data()
    zone_df = load_zone_data()
    tables = load_analysis_tables()
    insights = load_analysis_insights()
    videos = get_video_files()

    global_kpis = compute_global_kpis(frame_df, zone_df, tables)

    return {
        "frame_df": frame_df,
        "zone_df": zone_df,
        "tables": tables,
        "insights": insights,
        "videos": videos,
        "midframe": MIDFRAME_IMAGE,
        "global_kpis": global_kpis,
        "global_summary_text": build_global_summary_text(global_kpis),
        "available_zones": get_available_zones(zone_df),
        "project_root": PROJECT_ROOT,
    }


if __name__ == "__main__":
    data = load_dashboard_data()

    print("✅ Dashboard data loaded successfully")
    print(f"Frame rows: {len(data['frame_df'])}")
    print(f"Zone rows: {len(data['zone_df'])}")
    print(f"Videos found: {list(data['videos'].keys())}")
    print(f"Available zones: {data['available_zones']}")
    print(f"Global KPIs: {data['global_kpis']}")