from __future__ import annotations

"""
Data tools for the Crowd Monitoring AI Agent.

This file gives the agent access to real CSV outputs without sending full CSVs
directly to the LLM.

The agent flow is:
User question
→ Python extracts relevant facts from CSVs
→ compact factual context is sent to LLM
→ LLM explains the answer naturally
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import numpy as np
import pandas as pd


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class LoadedCrowdData:
    frame_df: pd.DataFrame
    zone_df: pd.DataFrame
    zone_summary: pd.DataFrame
    project_root: Path
    analysis_tables_dir: Path
    analysis_insights_dir: Path


# ============================================================
# PATHS
# ============================================================

def resolve_project_root(project_root: Optional[str | Path] = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()

    return Path(__file__).resolve().parents[2]


def get_default_paths(project_root: Optional[str | Path] = None) -> Dict[str, Path]:
    root = resolve_project_root(project_root)

    return {
        "project_root": root,
        "frame_csv": root / "results" / "benchmark" / "FULL_01_shinjuku_frame_counts.csv",
        "zone_csv": root / "results" / "benchmark" / "FULL_02_shinjuku_zone_density_risk.csv",
        "analysis_tables_dir": root / "results" / "analysis_5min_refined" / "tables",
        "analysis_insights_dir": root / "results" / "analysis_5min_refined" / "insights",
    }


# ============================================================
# FORMAT HELPERS
# ============================================================

def fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def fmt_float(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def fmt_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.1f}"


def fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}%"


def fmt_density_score(value: Any) -> str:
    """
    Convert tiny pixel density into readable score:
    density_score = pixel_density × 10,000
    """
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 10000:.2f}"


def fmt_seconds(seconds: Any) -> str:
    if seconds is None or pd.isna(seconds):
        return "-"

    seconds = float(seconds)
    minutes = int(seconds // 60)
    sec = int(round(seconds % 60))

    if sec == 60:
        minutes += 1
        sec = 0

    return f"{minutes}:{sec:02d}"


def compact_text(text: str, max_chars: int = 900) -> str:
    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_col(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
    label: str = "column",
) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if required:
        raise ValueError(
            f"Could not find {label}. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_risk(value: Any) -> str:
    value = str(value).strip().upper()

    if value in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        return value

    if value in {"MED", "MODERATE"}:
        return "MEDIUM"

    if value in {"CRIT"}:
        return "CRITICAL"

    return value if value else "UNKNOWN"


def risk_score(value: str) -> int:
    value = normalize_risk(value)

    return {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
        "CRITICAL": 3,
    }.get(value, -1)


def normalize_zone_name(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


# ============================================================
# TIME PARSING
# ============================================================

def parse_time_reference(question: str) -> Optional[float]:
    """
    Parse time references from questions.

    Supports:
    - "at 1:00"
    - "at minute 1"
    - "at 60 seconds"
    - "around 2:30"
    - "1 min 30 sec"
    """
    q = question.lower()

    # 1:00, 02:30, 4:05
    match = re.search(r"\b(?:at|around|near|minute|min|time)?\s*(\d{1,2})\s*:\s*(\d{2})\b", q)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return float(minutes * 60 + seconds)

    # at minute 1 / minute 2
    match = re.search(r"\b(?:at|around|near)?\s*minute\s+(\d+(?:\.\d+)?)\b", q)
    if match:
        return float(match.group(1)) * 60.0

    # 1 min 30 sec
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:min|minute|minutes)\s*(\d+(?:\.\d+)?)?\s*(?:sec|second|seconds)?", q)
    if match:
        minutes = float(match.group(1))
        seconds = float(match.group(2)) if match.group(2) else 0.0
        return minutes * 60.0 + seconds

    # 60 seconds / 60 sec
    match = re.search(r"\b(?:at|around|near)?\s*(\d+(?:\.\d+)?)\s*(?:sec|second|seconds)\b", q)
    if match:
        return float(match.group(1))

    return None


# ============================================================
# LOAD DATA
# ============================================================

def load_frame_csv(frame_csv: Path) -> pd.DataFrame:
    if not frame_csv.exists():
        raise FileNotFoundError(f"Frame CSV not found: {frame_csv}")

    df = pd.read_csv(frame_csv)

    frame_time_col = find_col(df, ["timestamp_sec", "timestamp", "time_sec"], required=False)
    frame_id_col = find_col(df, ["frame_id", "frame", "frame_idx"], required=False)
    total_count_col = find_col(df, ["total_count", "count", "global_count"], label="total count column")
    fps_col = find_col(df, ["fps", "inference_fps", "pipeline_fps"], required=False)

    if frame_time_col is None:
        df["timestamp_sec"] = np.arange(len(df)) / 30.0
    else:
        df["timestamp_sec"] = pd.to_numeric(df[frame_time_col], errors="coerce").fillna(0)

    if frame_id_col is None:
        df["frame_id"] = np.arange(len(df))
    else:
        df["frame_id"] = df[frame_id_col]

    df["total_count"] = pd.to_numeric(df[total_count_col], errors="coerce").fillna(0)

    if fps_col is None:
        df["fps"] = np.nan
    else:
        df["fps"] = pd.to_numeric(df[fps_col], errors="coerce")

    return df.sort_values("timestamp_sec").reset_index(drop=True)


def load_zone_csv(zone_csv: Path) -> pd.DataFrame:
    if not zone_csv.exists():
        raise FileNotFoundError(f"Zone CSV not found: {zone_csv}")

    df = pd.read_csv(zone_csv)

    zone_name_col = find_col(df, ["zone_name", "zone", "name"], label="zone name column")
    zone_count_col = find_col(df, ["zone_count", "count", "count_in_zone"], label="zone count column")
    density_col = find_col(
        df,
        ["zone_density_pixel", "density", "zone_density", "zone_density_pixel_cleaned", "density_pixel"],
        label="density column",
    )
    risk_col = find_col(df, ["risk_level_text", "risk_level", "risk", "zone_risk"], label="risk column")
    ts_col = find_col(df, ["timestamp_sec", "timestamp", "time_sec"], label="timestamp column")
    zid_col = find_col(df, ["zone_short_id", "zone_id", "zone_code", "zone_label"], required=False)

    df["zone_name"] = df[zone_name_col].astype(str)
    df["zone_name_normalized"] = df["zone_name"].map(normalize_zone_name)
    df["zone_count"] = pd.to_numeric(df[zone_count_col], errors="coerce").fillna(0)
    df["density"] = pd.to_numeric(df[density_col], errors="coerce").fillna(0)
    df["risk_level"] = df[risk_col].map(normalize_risk)
    df["timestamp_sec"] = pd.to_numeric(df[ts_col], errors="coerce").fillna(0)

    if zid_col is None:
        df["zone_id"] = ""
    else:
        df["zone_id"] = df[zid_col].astype(str)

    fallback_ids = {
        "crosswalk_main": "CW1",
        "crosswalk_left": "CW2",
        "crosswalk_top": "CW3",
        "crosswalk_bottom": "CW4",
        "sidewalk_top": "SW1",
        "sidewalk_right": "SW2",
        "sidewalk_bottom": "SW3",
        "sidewalk_left": "SW4",
    }

    df["zone_id"] = df.apply(
        lambda row: row["zone_id"]
        if str(row["zone_id"]).strip() not in ["", "nan", "None"]
        else fallback_ids.get(row["zone_name"], row["zone_name"]),
        axis=1,
    )

    return df.sort_values(["zone_name", "timestamp_sec"]).reset_index(drop=True)


def compute_zone_summary(frame_df: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    duration_sec = float(frame_df["timestamp_sec"].max())
    reference_time = duration_sec / 2.0

    reference_rows = []
    for zone_name, group in zone_df.groupby("zone_name"):
        idx = (group["timestamp_sec"] - reference_time).abs().idxmin()
        reference_rows.append(zone_df.loc[idx])

    reference_df = pd.DataFrame(reference_rows)[
        ["zone_name", "zone_count", "density", "risk_level", "timestamp_sec"]
    ].rename(
        columns={
            "zone_count": "reference_count",
            "density": "reference_density",
            "risk_level": "reference_risk",
            "timestamp_sec": "reference_time_sec",
        }
    )

    summary = (
        zone_df.groupby("zone_name")
        .agg(
            zone_id=("zone_id", "first"),
            avg_count=("zone_count", "mean"),
            median_count=("zone_count", "median"),
            peak_count=("zone_count", "max"),
            min_count=("zone_count", "min"),
            std_count=("zone_count", "std"),
            mean_density=("density", "mean"),
            max_density=("density", "max"),
            median_density=("density", "median"),
        )
        .reset_index()
    )

    risk_pct = (
        zone_df.assign(is_hc=zone_df["risk_level"].isin(["HIGH", "CRITICAL"]).astype(int))
        .groupby("zone_name")["is_hc"]
        .mean()
        .mul(100)
        .reset_index(name="high_critical_pct")
    )

    summary = summary.merge(risk_pct, on="zone_name", how="left")

    risk_dist = zone_df.groupby(["zone_name", "risk_level"]).size().reset_index(name="n")
    risk_dist["pct"] = risk_dist["n"] / risk_dist.groupby("zone_name")["n"].transform("sum") * 100

    risk_pivot = risk_dist.pivot_table(
        index="zone_name",
        columns="risk_level",
        values="pct",
        fill_value=0,
    ).reset_index()

    for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        if risk not in risk_pivot.columns:
            risk_pivot[risk] = 0.0

    risk_pivot = risk_pivot.rename(
        columns={
            "LOW": "low_pct",
            "MEDIUM": "medium_pct",
            "HIGH": "high_pct",
            "CRITICAL": "critical_pct",
        }
    )

    summary = summary.merge(risk_pivot, on="zone_name", how="left")

    risk_mode = (
        zone_df.groupby("zone_name")["risk_level"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
        .reset_index(name="dominant_risk")
    )

    summary = summary.merge(risk_mode, on="zone_name", how="left")

    peak_idx = zone_df.groupby("zone_name")["zone_count"].idxmax()
    peak_times = zone_df.loc[peak_idx, ["zone_name", "timestamp_sec"]].rename(
        columns={"timestamp_sec": "peak_time_sec"}
    )

    summary = summary.merge(peak_times, on="zone_name", how="left")

    temp = zone_df.copy()

    temp["prev_count_30"] = temp.groupby("zone_name")["zone_count"].shift(30)
    temp["delta_30"] = temp["zone_count"] - temp["prev_count_30"]
    temp["pct_30"] = np.where(
        temp["prev_count_30"] > 0,
        temp["delta_30"] / temp["prev_count_30"] * 100,
        0,
    )

    temp["spike_flag"] = (
        (temp["zone_count"] >= 20)
        & (temp["delta_30"] >= 10)
        & (temp["pct_30"] >= 50)
    ).astype(int)

    spikes = temp.groupby("zone_name")["spike_flag"].sum().reset_index(name="spike_events")
    summary = summary.merge(spikes, on="zone_name", how="left")

    temp["density_prev_5s"] = temp.groupby("zone_name")["density"].shift(150)
    temp["density_delta_5s"] = temp["density"] - temp["density_prev_5s"]

    trend = temp.groupby("zone_name")["density_delta_5s"].mean().reset_index(name="density_trend_score")
    summary = summary.merge(trend, on="zone_name", how="left")

    summary = summary.merge(reference_df, on="zone_name", how="left")

    fill_cols = [
        "high_critical_pct",
        "low_pct",
        "medium_pct",
        "high_pct",
        "critical_pct",
        "spike_events",
        "density_trend_score",
    ]

    for col in fill_cols:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0)

    return summary


def load_crowd_data(project_root: Optional[str | Path] = None) -> LoadedCrowdData:
    paths = get_default_paths(project_root)

    frame_df = load_frame_csv(paths["frame_csv"])
    zone_df = load_zone_csv(paths["zone_csv"])
    zone_summary = compute_zone_summary(frame_df, zone_df)

    return LoadedCrowdData(
        frame_df=frame_df,
        zone_df=zone_df,
        zone_summary=zone_summary,
        project_root=paths["project_root"],
        analysis_tables_dir=paths["analysis_tables_dir"],
        analysis_insights_dir=paths["analysis_insights_dir"],
    )


# ============================================================
# ZONE HELPERS
# ============================================================

def available_zones(data: LoadedCrowdData) -> List[str]:
    return sorted(data.zone_df["zone_name"].unique().tolist())


def resolve_zone_name(data: LoadedCrowdData, text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    text_norm = normalize_zone_name(text)

    for zone in available_zones(data):
        if normalize_zone_name(zone) == text_norm:
            return zone

    for zone in available_zones(data):
        zone_norm = normalize_zone_name(zone)
        if zone_norm in text_norm or text_norm in zone_norm:
            return zone

    for _, row in data.zone_summary.iterrows():
        zid = str(row.get("zone_id", "")).lower().strip()
        if zid and zid in text_norm:
            return str(row["zone_name"])

    return None


def extract_zone_from_question(data: LoadedCrowdData, question: str) -> Optional[str]:
    q = question.lower()
    q_norm = normalize_zone_name(question)

    for zone in available_zones(data):
        if normalize_zone_name(zone) in q_norm:
            return zone

    for zone in available_zones(data):
        if zone.replace("_", " ").lower() in q:
            return zone

    for _, row in data.zone_summary.iterrows():
        zid = str(row.get("zone_id", "")).lower().strip()
        if zid and re.search(rf"\b{re.escape(zid)}\b", q):
            return str(row["zone_name"])

    return None


def extract_all_zones_from_question(data: LoadedCrowdData, question: str) -> List[str]:
    q = question.lower()
    q_norm = normalize_zone_name(question)
    zones = []

    for zone in available_zones(data):
        if normalize_zone_name(zone) in q_norm or zone.replace("_", " ").lower() in q:
            zones.append(zone)

    for _, row in data.zone_summary.iterrows():
        zid = str(row.get("zone_id", "")).lower().strip()
        if zid and re.search(rf"\b{re.escape(zid)}\b", q):
            zone = str(row["zone_name"])
            if zone not in zones:
                zones.append(zone)

    return zones


# ============================================================
# FACT TOOLS
# ============================================================

def get_global_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    frame_df = data.frame_df
    zone_df = data.zone_df

    duration_sec = float(frame_df["timestamp_sec"].max())
    processed_frames = int(len(frame_df))
    video_fps = processed_frames / duration_sec if duration_sec > 0 else np.nan

    peak_idx = frame_df["total_count"].idxmax()
    peak_row = frame_df.loc[peak_idx]

    avg_pipeline_fps = (
        float(frame_df["fps"].dropna().mean())
        if frame_df["fps"].notna().any()
        else np.nan
    )

    return {
        "duration_sec": duration_sec,
        "duration_label": fmt_seconds(duration_sec),
        "processed_frames": processed_frames,
        "estimated_video_fps": video_fps,
        "average_count": float(frame_df["total_count"].mean()),
        "median_count": float(frame_df["total_count"].median()),
        "maximum_count": int(frame_df["total_count"].max()),
        "minimum_count": int(frame_df["total_count"].min()),
        "std_count": float(frame_df["total_count"].std()),
        "peak_frame_id": int(peak_row["frame_id"]),
        "peak_time_sec": float(peak_row["timestamp_sec"]),
        "peak_time_label": fmt_seconds(float(peak_row["timestamp_sec"])),
        "average_pipeline_fps": avg_pipeline_fps,
        "number_of_zones": int(zone_df["zone_name"].nunique()),
    }


def get_zone_summary(data: LoadedCrowdData, zone_name: str) -> Dict[str, Any]:
    resolved = resolve_zone_name(data, zone_name)

    if resolved is None:
        return {
            "error": f"Zone '{zone_name}' was not found.",
            "available_zones": available_zones(data),
        }

    row = data.zone_summary[data.zone_summary["zone_name"] == resolved].iloc[0]

    return {
        "zone_name": resolved,
        "zone_id": str(row.get("zone_id", "")),
        "average_count": float(row["avg_count"]),
        "median_count": float(row["median_count"]),
        "peak_count": int(row["peak_count"]),
        "peak_time_sec": float(row["peak_time_sec"]),
        "peak_time_label": fmt_seconds(float(row["peak_time_sec"])),
        "mean_density_pixel": float(row["mean_density"]),
        "mean_density_score_x10000": float(row["mean_density"]) * 10000,
        "max_density_pixel": float(row["max_density"]),
        "high_critical_pct": float(row["high_critical_pct"]),
        "low_pct": float(row.get("low_pct", 0)),
        "medium_pct": float(row.get("medium_pct", 0)),
        "high_pct": float(row.get("high_pct", 0)),
        "critical_pct": float(row.get("critical_pct", 0)),
        "dominant_risk": str(row.get("dominant_risk", "UNKNOWN")),
        "reference_count": int(row.get("reference_count", 0)),
        "reference_time_sec": float(row.get("reference_time_sec", 0)),
        "reference_time_label": fmt_seconds(float(row.get("reference_time_sec", 0))),
        "reference_risk": str(row.get("reference_risk", "UNKNOWN")),
        "spike_events": int(row.get("spike_events", 0)),
        "density_trend_score_x10000": float(row.get("density_trend_score", 0)) * 10000,
    }


def get_all_zone_rankings(data: LoadedCrowdData) -> Dict[str, Any]:
    summary = data.zone_summary.copy()

    highest_avg = summary.sort_values("avg_count", ascending=False).iloc[0]
    highest_peak = summary.sort_values("peak_count", ascending=False).iloc[0]
    highest_density = summary.sort_values("mean_density", ascending=False).iloc[0]
    most_risky = summary.sort_values("high_critical_pct", ascending=False).iloc[0]
    most_spikes = summary.sort_values("spike_events", ascending=False).iloc[0]

    return {
        "highest_average_count_zone": {
            "zone_name": highest_avg["zone_name"],
            "average_count": float(highest_avg["avg_count"]),
        },
        "highest_peak_count_zone": {
            "zone_name": highest_peak["zone_name"],
            "peak_count": int(highest_peak["peak_count"]),
            "peak_time_label": fmt_seconds(float(highest_peak["peak_time_sec"])),
        },
        "highest_mean_density_zone": {
            "zone_name": highest_density["zone_name"],
            "mean_density_score_x10000": float(highest_density["mean_density"]) * 10000,
        },
        "most_risky_zone": {
            "zone_name": most_risky["zone_name"],
            "high_critical_pct": float(most_risky["high_critical_pct"]),
            "dominant_risk": str(most_risky["dominant_risk"]),
        },
        "most_spike_events_zone": {
            "zone_name": most_spikes["zone_name"],
            "spike_events": int(most_spikes["spike_events"]),
        },
    }


def get_peak_moment_context(data: LoadedCrowdData, window_sec: float = 5.0) -> Dict[str, Any]:
    frame_df = data.frame_df
    zone_df = data.zone_df

    peak_idx = frame_df["total_count"].idxmax()
    peak_row = frame_df.loc[peak_idx]
    peak_time = float(peak_row["timestamp_sec"])

    nearby_frames = frame_df[
        (frame_df["timestamp_sec"] >= peak_time - window_sec)
        & (frame_df["timestamp_sec"] <= peak_time + window_sec)
    ]

    zone_rows = []
    for zone_name, group in zone_df.groupby("zone_name"):
        idx = (group["timestamp_sec"] - peak_time).abs().idxmin()
        row = group.loc[idx]

        zone_rows.append(
            {
                "zone_name": zone_name,
                "zone_id": str(row.get("zone_id", "")),
                "zone_count": int(row["zone_count"]),
                "density_score_x10000": float(row["density"]) * 10000,
                "risk_level": str(row["risk_level"]),
            }
        )

    zone_rows = sorted(
        zone_rows,
        key=lambda x: (risk_score(x["risk_level"]), x["zone_count"], x["density_score_x10000"]),
        reverse=True,
    )

    return {
        "peak_total_count": int(peak_row["total_count"]),
        "peak_frame_id": int(peak_row["frame_id"]),
        "peak_time_sec": peak_time,
        "peak_time_label": fmt_seconds(peak_time),
        "average_count_in_peak_window": float(nearby_frames["total_count"].mean()),
        "zones_ranked_at_peak": zone_rows,
        "high_or_critical_zone_count_at_peak": sum(
            1 for row in zone_rows if row["risk_level"] in {"HIGH", "CRITICAL"}
        ),
    }


def get_zone_status_at_time(data: LoadedCrowdData, timestamp_sec: float) -> Dict[str, Any]:
    frame_df = data.frame_df
    zone_df = data.zone_df

    duration = float(frame_df["timestamp_sec"].max())
    timestamp_sec = max(0.0, min(float(timestamp_sec), duration))

    frame_idx = (frame_df["timestamp_sec"] - timestamp_sec).abs().idxmin()
    frame_row = frame_df.loc[frame_idx]
    actual_time = float(frame_row["timestamp_sec"])

    rows = []
    for zone_name, group in zone_df.groupby("zone_name"):
        idx = (group["timestamp_sec"] - actual_time).abs().idxmin()
        row = group.loc[idx]

        rows.append(
            {
                "zone_name": zone_name,
                "zone_id": str(row.get("zone_id", "")),
                "count": int(row["zone_count"]),
                "density_pixel": float(row["density"]),
                "density_score_x10000": float(row["density"]) * 10000,
                "risk_level": str(row["risk_level"]),
                "risk_score": risk_score(str(row["risk_level"])),
            }
        )

    rows_by_risk = sorted(
        rows,
        key=lambda x: (x["risk_score"], x["count"], x["density_score_x10000"]),
        reverse=True,
    )

    rows_by_count = sorted(rows, key=lambda x: x["count"], reverse=True)

    return {
        "requested_time_sec": timestamp_sec,
        "nearest_time_sec": actual_time,
        "nearest_time_label": fmt_seconds(actual_time),
        "frame_id": int(frame_row["frame_id"]),
        "total_count": int(frame_row["total_count"]),
        "zones_ranked_by_risk_at_time": rows_by_risk,
        "zones_ranked_by_count_at_time": rows_by_count,
        "high_or_critical_zones_at_time": [
            row for row in rows_by_risk if row["risk_level"] in {"HIGH", "CRITICAL"}
        ],
    }


def get_context_around_time(data: LoadedCrowdData, timestamp_sec: float, window_sec: float = 5.0) -> Dict[str, Any]:
    frame_df = data.frame_df

    duration = float(frame_df["timestamp_sec"].max())
    timestamp_sec = max(0.0, min(float(timestamp_sec), duration))

    window = frame_df[
        (frame_df["timestamp_sec"] >= timestamp_sec - window_sec)
        & (frame_df["timestamp_sec"] <= timestamp_sec + window_sec)
    ]

    status = get_zone_status_at_time(data, timestamp_sec)

    return {
        "time_status": status,
        "window_sec": window_sec,
        "average_total_count_in_window": float(window["total_count"].mean()) if not window.empty else None,
        "minimum_total_count_in_window": int(window["total_count"].min()) if not window.empty else None,
        "maximum_total_count_in_window": int(window["total_count"].max()) if not window.empty else None,
    }


def get_risk_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    zone_df = data.zone_df
    summary = data.zone_summary

    total_zone_frames = len(zone_df)
    high_critical_frames = int(zone_df["risk_level"].isin(["HIGH", "CRITICAL"]).sum())

    risk_distribution = (
        zone_df["risk_level"]
        .value_counts(normalize=True)
        .mul(100)
        .to_dict()
    )

    most_risky = summary.sort_values("high_critical_pct", ascending=False).iloc[0]
    least_risky = summary.sort_values("high_critical_pct", ascending=True).iloc[0]

    return {
        "total_zone_frames": total_zone_frames,
        "high_critical_frames": high_critical_frames,
        "high_critical_pct_overall": high_critical_frames / total_zone_frames * 100 if total_zone_frames else 0,
        "risk_distribution_pct": {str(k): float(v) for k, v in risk_distribution.items()},
        "most_risky_zone": {
            "zone_name": most_risky["zone_name"],
            "high_critical_pct": float(most_risky["high_critical_pct"]),
            "dominant_risk": str(most_risky["dominant_risk"]),
        },
        "least_risky_zone": {
            "zone_name": least_risky["zone_name"],
            "high_critical_pct": float(least_risky["high_critical_pct"]),
            "dominant_risk": str(least_risky["dominant_risk"]),
        },
    }


def get_anomaly_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    summary = data.zone_summary.copy()

    total_spikes = int(summary["spike_events"].sum())
    top_zone = summary.sort_values("spike_events", ascending=False).iloc[0]

    return {
        "computed_total_refined_spike_events": total_spikes,
        "top_spike_zone": {
            "zone_name": top_zone["zone_name"],
            "spike_events": int(top_zone["spike_events"]),
        },
        "refined_spike_rule": (
            "current count >= 20, absolute increase >= 10, percent increase >= 50%, "
            "compared to 30 frames earlier"
        ),
        "interpretation": (
            "Spike events represent sudden changes in estimated zone count. "
            "They are analysis events, not confirmed real-world incidents."
        ),
    }


def get_temporal_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    frame_df = data.frame_df.copy()

    frame_df["count_change"] = frame_df["total_count"].diff().fillna(0)
    frame_df["abs_count_change"] = frame_df["count_change"].abs()
    frame_df["rolling_abs_change_5s"] = frame_df["abs_count_change"].rolling(150, min_periods=1).mean()

    idx = frame_df["rolling_abs_change_5s"].idxmax()
    row = frame_df.loc[idx]

    return {
        "global_summary": get_global_summary(data),
        "peak_context": get_peak_moment_context(data),
        "average_abs_count_change": float(frame_df["abs_count_change"].mean()),
        "strongest_count_change_time_sec": float(row["timestamp_sec"]),
        "strongest_count_change_time_label": fmt_seconds(float(row["timestamp_sec"])),
        "strongest_count_change_value": float(row["rolling_abs_change_5s"]),
    }


def get_spatial_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    return {
        "zone_rankings": get_all_zone_rankings(data),
        "risk_summary": get_risk_summary(data),
        "number_of_zones": int(data.zone_df["zone_name"].nunique()),
    }


def get_statistical_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    pivot = data.zone_df.pivot_table(
        index="timestamp_sec",
        columns="zone_name",
        values="zone_count",
        aggfunc="sum",
    ).fillna(0)

    corr = pivot.corr()
    pairs = []

    cols = list(corr.columns)

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            pairs.append(
                {
                    "zone_a": a,
                    "zone_b": b,
                    "correlation": float(corr.loc[a, b]),
                }
            )

    strongest = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[0] if pairs else None

    total = pivot.sum(axis=1).replace(0, np.nan)
    prob = pivot.div(total, axis=0).fillna(0)
    entropy = -(prob * np.log(prob.replace(0, np.nan))).sum(axis=1).fillna(0)

    max_entropy = np.log(len(pivot.columns)) if len(pivot.columns) > 1 else 1
    norm_entropy = entropy / max_entropy

    return {
        "mean_normalized_entropy": float(norm_entropy.mean()),
        "highest_entropy_time_sec": float(norm_entropy.idxmax()),
        "highest_entropy_time_label": fmt_seconds(float(norm_entropy.idxmax())),
        "lowest_entropy_time_sec": float(norm_entropy.idxmin()),
        "lowest_entropy_time_label": fmt_seconds(float(norm_entropy.idxmin())),
        "strongest_zone_correlation": strongest,
    }


def compare_zones(data: LoadedCrowdData, zone_a: str, zone_b: str) -> Dict[str, Any]:
    a = get_zone_summary(data, zone_a)
    b = get_zone_summary(data, zone_b)

    if "error" in a:
        return a

    if "error" in b:
        return b

    return {
        "zone_a": a,
        "zone_b": b,
        "comparison": {
            "higher_average_count": a["zone_name"] if a["average_count"] >= b["average_count"] else b["zone_name"],
            "higher_peak_count": a["zone_name"] if a["peak_count"] >= b["peak_count"] else b["zone_name"],
            "higher_density": a["zone_name"] if a["mean_density_score_x10000"] >= b["mean_density_score_x10000"] else b["zone_name"],
            "higher_risk": a["zone_name"] if a["high_critical_pct"] >= b["high_critical_pct"] else b["zone_name"],
            "more_spikes": a["zone_name"] if a["spike_events"] >= b["spike_events"] else b["zone_name"],
        },
    }


# ============================================================
# CONTEXT ROUTER
# ============================================================

def facts_to_text(title: str, facts: Any, max_depth: int = 4) -> str:
    def render(value: Any, depth: int = 0) -> List[str]:
        if depth > max_depth:
            return [compact_text(str(value), 250)]

        lines = []

        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{k}:")
                    lines.extend([f"  {line}" for line in render(v, depth + 1)])
                else:
                    lines.append(f"{k}: {v}")

        elif isinstance(value, list):
            for idx, item in enumerate(value[:12], start=1):
                if isinstance(item, dict):
                    lines.append(f"{idx}.")
                    lines.extend([f"  {line}" for line in render(item, depth + 1)])
                else:
                    lines.append(f"{idx}. {item}")

            if len(value) > 12:
                lines.append(f"... {len(value) - 12} more items")

        else:
            lines.append(str(value))

        return lines

    return f"## {title}\n" + "\n".join(render(facts))


def build_context_for_question(
    data: LoadedCrowdData,
    question: str,
    selected_zone: Optional[str] = None,
) -> Dict[str, Any]:
    q = question.lower()
    context: Dict[str, Any] = {}

    time_ref = parse_time_reference(question)
    zone_from_question = extract_zone_from_question(data, question)
    zones_mentioned = extract_all_zones_from_question(data, question)
    zone_to_use = zone_from_question or selected_zone

    # Always include global summary.
    context["global_summary"] = get_global_summary(data)

    # Identity/capability questions.
    if any(k in q for k in ["what are you", "who are you", "your role", "what can you do", "how do you work"]):
        context["agent_capability"] = {
            "role": "AI Insights Assistant for the crowd monitoring dashboard",
            "data_sources": [
                "global frame-count CSV",
                "zone density/risk CSV",
                "computed temporal, spatial, anomaly, and statistical summaries",
            ],
            "method": (
                "The assistant uses Python tools to retrieve exact facts from saved analysis outputs, "
                "then explains them in natural language."
            ),
            "limitation": (
                "It does not directly watch the video and does not make certified safety decisions."
            ),
        }
        return context

    # Time-specific questions.
    if time_ref is not None:
        context["time_specific_status"] = get_zone_status_at_time(data, time_ref)
        context["context_around_time"] = get_context_around_time(data, time_ref, window_sec=5.0)

    # Zone-specific questions.
    if zone_to_use:
        context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)

    # Comparison questions.
    if len(zones_mentioned) >= 2:
        context["zone_comparison"] = compare_zones(data, zones_mentioned[0], zones_mentioned[1])

    # Peak.
    if any(k in q for k in ["peak", "highest", "maximum", "max", "busiest"]):
        context["peak_moment_context"] = get_peak_moment_context(data)

    # Risk.
    if any(k in q for k in ["risk", "critical", "high", "danger", "unsafe", "crowded"]):
        context["risk_summary"] = get_risk_summary(data)
        context["zone_rankings"] = get_all_zone_rankings(data)

    # Anomaly.
    if any(k in q for k in ["anomaly", "anomalies", "spike", "sudden", "alert", "alerts"]):
        context["anomaly_summary"] = get_anomaly_summary(data)

    # Temporal.
    if any(k in q for k in ["time", "timeline", "temporal", "trend", "change", "increase", "decrease", "flow"]):
        context["temporal_summary"] = get_temporal_summary(data)

    # Spatial.
    if any(k in q for k in ["spatial", "hotspot", "where", "area", "region", "zone", "zones"]):
        context["spatial_summary"] = get_spatial_summary(data)

    # Statistical.
    if any(k in q for k in ["correlation", "entropy", "statistical", "distribution", "spread", "together"]):
        context["statistical_summary"] = get_statistical_summary(data)

    # Broad question fallback.
    if len(context) <= 1:
        context["zone_rankings"] = get_all_zone_rankings(data)
        context["risk_summary"] = get_risk_summary(data)
        context["anomaly_summary"] = get_anomaly_summary(data)

        if zone_to_use:
            context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)

    return context


def build_context_text_for_question(
    data: LoadedCrowdData,
    question: str,
    selected_zone: Optional[str] = None,
) -> str:
    context = build_context_for_question(data, question, selected_zone)

    sections = []

    for title, facts in context.items():
        sections.append(facts_to_text(title, facts))

    sections.append(
        "## Interpretation constraints\n"
        "- Density is pixel-based relative to manually drawn polygon area, not real-world persons per square meter.\n"
        "- Risk is a rule-based prototype label for analysis, not a certified safety threshold.\n"
        "- This dashboard currently uses saved/offline outputs, not direct live CCTV interpretation.\n"
        "- The assistant does not directly watch the video; it answers from structured CSV outputs and computed summaries.\n"
        "- Motion/stagnation requires optical flow or tracking and should not be claimed unless provided.\n"
        "- Use only the facts above. Do not invent unsupported numbers."
    )

    return "\n\n".join(sections)


# ============================================================
# RULE-BASED FALLBACK
# ============================================================

def answer_rule_based(
    data: LoadedCrowdData,
    question: str,
    selected_zone: Optional[str] = None,
) -> str:
    q = question.lower()

    if any(k in q for k in ["what are you", "who are you", "your role", "what can you do"]):
        return (
            "I am the AI Insights Assistant for this crowd monitoring dashboard. "
            "I answer questions using the saved CSV outputs and computed analysis summaries. "
            "I do not directly watch the video or make certified safety decisions."
        )

    time_ref = parse_time_reference(question)

    if time_ref is not None:
        status = get_zone_status_at_time(data, time_ref)
        risky = status["zones_ranked_by_risk_at_time"][:3]

        lines = [
            f"At **{status['nearest_time_label']}**, the total estimated count was **{fmt_int(status['total_count'])}**.",
            "The top zones by rule-based risk/count at that time were:",
        ]

        for row in risky:
            lines.append(
                f"- **{row['zone_name']} ({row['zone_id']})**: risk **{row['risk_level']}**, "
                f"count **{fmt_int(row['count'])}**, density score **{fmt_float(row['density_score_x10000'], 2)}**."
            )

        lines.append("Risk is rule-based and density is pixel-based, not real-world persons/m².")

        return "\n".join(lines)

    zone = extract_zone_from_question(data, question) or selected_zone

    if zone:
        z = get_zone_summary(data, zone)

        if "error" not in z:
            return (
                f"**{z['zone_name']} ({z['zone_id']})** had average count **{fmt_count(z['average_count'])}**, "
                f"peak count **{fmt_int(z['peak_count'])}** at **{z['peak_time_label']}**, "
                f"density score **{fmt_float(z['mean_density_score_x10000'], 2)}**, and "
                f"**{fmt_pct(z['high_critical_pct'])}** HIGH/CRITICAL frames. "
                f"The fixed middle-frame count was **{fmt_int(z['reference_count'])}** at **{z['reference_time_label']}**."
            )

    if any(k in q for k in ["risk", "risky", "riskiest", "critical"]):
        risk = get_risk_summary(data)
        z = risk["most_risky_zone"]

        return (
            f"The most risky zone is **{z['zone_name']}** with **{fmt_pct(z['high_critical_pct'])}** "
            f"HIGH/CRITICAL frames and dominant risk **{z['dominant_risk']}**. "
            f"This is a rule-based prototype risk label."
        )

    if any(k in q for k in ["peak", "highest", "maximum", "busiest"]):
        peak = get_peak_moment_context(data)
        return (
            f"The peak crowd moment occurred at **{peak['peak_time_label']}** "
            f"with total count **{fmt_int(peak['peak_total_count'])}**."
        )

    if any(k in q for k in ["anomaly", "spike", "alert"]):
        anomaly = get_anomaly_summary(data)
        top = anomaly["top_spike_zone"]

        return (
            f"The refined detector found **{fmt_int(anomaly['computed_total_refined_spike_events'])}** spike events. "
            f"The zone with the most spike events was **{top['zone_name']}** with **{fmt_int(top['spike_events'])}** events."
        )

    global_summary = get_global_summary(data)
    rankings = get_all_zone_rankings(data)

    return (
        f"The experiment processed **{fmt_int(global_summary['processed_frames'])}** frames over "
        f"**{global_summary['duration_label']}**. Average count was **{fmt_count(global_summary['average_count'])}** "
        f"and peak count was **{fmt_int(global_summary['maximum_count'])}** at **{global_summary['peak_time_label']}**. "
        f"The main hotspot was **{rankings['highest_average_count_zone']['zone_name']}**."
    )


__all__ = [
    "LoadedCrowdData",
    "load_crowd_data",
    "available_zones",
    "resolve_zone_name",
    "extract_zone_from_question",
    "extract_all_zones_from_question",
    "parse_time_reference",
    "get_global_summary",
    "get_zone_summary",
    "get_all_zone_rankings",
    "get_peak_moment_context",
    "get_zone_status_at_time",
    "get_context_around_time",
    "get_risk_summary",
    "get_anomaly_summary",
    "get_temporal_summary",
    "get_spatial_summary",
    "get_statistical_summary",
    "compare_zones",
    "build_context_for_question",
    "build_context_text_for_question",
    "answer_rule_based",
    "fmt_int",
    "fmt_float",
    "fmt_count",
    "fmt_pct",
    "fmt_density_score",
    "fmt_seconds",
]