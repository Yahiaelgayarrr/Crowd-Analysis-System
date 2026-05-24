"""
Compact professional Streamlit dashboard for the crowd analysis system.

This dashboard loads real saved outputs from:
- results/benchmark/FULL_01_shinjuku_frame_counts.csv
- results/benchmark/FULL_02_shinjuku_zone_density_risk.csv
- results/videos/
- results/analysis_5min_refined/tables/

This version focuses on:
- cleaner alignment
- compact dashboard layout
- no raw CSV tables in the main view
- reliable native Streamlit zone cards
- one analysis category visible at a time
- professional dashboard styling
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data_loader import (
    load_dashboard_data,
    compute_zone_kpis,
    build_zone_insight_text,
)
from src.dashboard import charts


st.set_page_config(
    page_title="Crowd Monitoring Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="Loading crowd monitoring dashboard data...")
def cached_dashboard_data():
    return load_dashboard_data()


def html(content: str) -> None:
    st.markdown(dedent(content).strip(), unsafe_allow_html=True)


def inject_css() -> None:
    html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(37,99,235,0.18), transparent 28%),
                radial-gradient(circle at 82% 4%, rgba(56,189,248,0.10), transparent 28%),
                linear-gradient(135deg, #050914 0%, #07101d 48%, #050914 100%);
            color: #e5e7eb;
        }

        .block-container {
            max-width: 1580px;
            padding-top: 0.7rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
            padding-bottom: 1.2rem;
        }

        [data-testid="stHeader"] {
            background: rgba(5,9,20,0.86);
            backdrop-filter: blur(14px);
        }

        .stDeployButton {
            display: none;
        }

        header [data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }

        .dashboard-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            background: rgba(15,23,42,0.76);
            border: 1px solid rgba(59,130,246,0.22);
            border-radius: 22px;
            padding: 15px 19px;
            margin-bottom: 12px;
            box-shadow: 0 18px 42px rgba(0,0,0,0.23);
            backdrop-filter: blur(16px);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .logo-box {
            width: 46px;
            height: 46px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #1d4ed8, #0891b2);
            color: white;
            font-size: 23px;
            box-shadow: 0 0 28px rgba(59,130,246,0.35);
        }

        .title-main {
            font-size: 25px;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .title-sub {
            margin-top: 4px;
            font-size: 13px;
            color: #94a3b8;
        }

        .header-badges {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .pill {
            border-radius: 999px;
            padding: 8px 13px;
            font-size: 13px;
            font-weight: 700;
            border: 1px solid rgba(148,163,184,0.22);
            background: rgba(15,23,42,0.74);
            color: #cbd5e1;
            white-space: nowrap;
        }

        .pill-green {
            border-color: rgba(34,197,94,0.35);
            background: rgba(34,197,94,0.12);
            color: #86efac;
        }

        .pill-blue {
            border-color: rgba(59,130,246,0.35);
            background: rgba(59,130,246,0.13);
            color: #93c5fd;
        }

        .section-label {
            margin: 4px 0 7px 0;
            font-size: 12px;
            font-weight: 800;
            color: #94a3b8;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .kpi-card {
            height: 112px;
            background:
                radial-gradient(circle at 85% 15%, rgba(59,130,246,0.18), transparent 36%),
                linear-gradient(145deg, rgba(17,24,39,0.96), rgba(15,23,42,0.86));
            border: 1px solid rgba(59,130,246,0.20);
            border-radius: 18px;
            padding: 14px 15px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.20);
        }

        .kpi-row-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .kpi-icon {
            width: 36px;
            height: 36px;
            border-radius: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(59,130,246,0.16);
            border: 1px solid rgba(59,130,246,0.25);
            font-size: 19px;
        }

        .kpi-label {
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            font-weight: 800;
        }

        .kpi-value {
            color: #f8fafc;
            font-size: 27px;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.05;
        }

        .kpi-sub {
            margin-top: 5px;
            font-size: 11.5px;
            color: #94a3b8;
            line-height: 1.32;
        }

        .analysis-card {
            background: rgba(15,23,42,0.74);
            border: 1px solid rgba(59,130,246,0.18);
            border-radius: 18px;
            padding: 13px 14px;
            margin-top: 6px;
            margin-bottom: 6px;
        }

        .analysis-title {
            color: #f8fafc;
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 4px;
        }

        .analysis-desc {
            color: #94a3b8;
            font-size: 13px;
            line-height: 1.42;
        }

        div[data-testid="metric-container"] {
            background:
                radial-gradient(circle at 90% 15%, rgba(59,130,246,0.16), transparent 40%),
                rgba(15,23,42,0.72);
            border: 1px solid rgba(59,130,246,0.18);
            border-radius: 16px;
            padding: 10px 13px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.16);
        }

        div[data-testid="metric-container"] label {
            color: #94a3b8 !important;
            font-weight: 700 !important;
            font-size: 12px !important;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 800 !important;
            font-size: 23px !important;
        }

        .stRadio > div {
            display: flex;
            gap: 9px;
            flex-wrap: wrap;
        }

        .stRadio label {
            background: rgba(15,23,42,0.70);
            border: 1px solid rgba(59,130,246,0.20);
            border-radius: 14px;
            padding: 8px 12px;
        }

        .stRadio label:hover {
            border-color: rgba(56,189,248,0.55);
        }

        div[data-testid="stVideo"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(59,130,246,0.24);
            box-shadow: 0 22px 50px rgba(0,0,0,0.28);
        }

        div[data-baseweb="select"] > div {
            background: rgba(15,23,42,0.95) !important;
            color: #e5e7eb !important;
            border-color: rgba(59,130,246,0.26) !important;
            border-radius: 13px !important;
        }

        .stSelectbox label,
        .stRadio label {
            color: #cbd5e1 !important;
            font-weight: 700 !important;
        }

        .stPlotlyChart {
            background: rgba(15,23,42,0.44);
            border-radius: 18px;
            border: 1px solid rgba(59,130,246,0.10);
        }

        .small-note {
            color: #94a3b8;
            font-size: 12px;
            line-height: 1.4;
            margin-top: 5px;
        }

        .method-note {
            background: rgba(245,158,11,0.10);
            border: 1px solid rgba(245,158,11,0.22);
            color: #fde68a;
            border-radius: 14px;
            padding: 10px 12px;
            font-size: 12.5px;
            line-height: 1.42;
            min-height: 74px;
        }

        .zone-heading {
            color: #f8fafc;
            font-weight: 800;
            font-size: 19px;
            margin-bottom: 2px;
        }

        .zone-caption {
            color: #94a3b8;
            font-size: 12.5px;
            margin-bottom: 8px;
        }

        .risk-text {
            font-weight: 800;
            color: #fdba74;
            text-align: right;
            font-size: 16px;
        }

        .quick-insight-title {
            color: #f8fafc;
            font-weight: 800;
            margin-top: 6px;
            margin-bottom: 5px;
        }

        @media (max-width: 1100px) {
            .dashboard-header {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        </style>
        """
    )


def header() -> None:
    html(
        """
        <div class="dashboard-header">
            <div class="header-left">
                <div class="logo-box">👥</div>
                <div>
                    <div class="title-main">Crowd Monitoring Dashboard</div>
                    <div class="title-sub">Intelligent Crowd Monitoring and Behavioral Analysis System</div>
                </div>
            </div>
            <div class="header-badges">
                <div class="pill pill-green">● Data Loaded</div>
                <div class="pill pill-blue">🧪 5-minute Shinjuku Experiment</div>
                <div class="pill">FIDTM · Zones · Density · Risk</div>
            </div>
        </div>
        """
    )


def section_label(text: str) -> None:
    html(f'<div class="section-label">{text}</div>')


def kpi_card(label: str, value: str, sub: str, icon: str) -> None:
    html(
        f"""
        <div class="kpi-card">
            <div class="kpi-row-top">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """
    )


def compact_text(text: str, max_sentences: int = 2) -> str:
    cleaned = " ".join(str(text).split())

    parts = []
    current = ""

    for char in cleaned:
        current += char
        if char in [".", "!", "?"]:
            parts.append(current.strip())
            current = ""
            if len(parts) >= max_sentences:
                break

    return " ".join(parts) if parts else cleaned


def render_kpis(kpis: dict) -> None:
    section_label("System KPIs")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        kpi_card("Avg Crowd", f"{kpis['avg_count']:.1f}", f"median {kpis['median_count']:.1f}", "👥")

    with c2:
        kpi_card(
            "Peak Count",
            f"{kpis['max_count']}",
            f"at {kpis['peak_time_label']} · frame {kpis['peak_frame_id']}",
            "📈",
        )

    with c3:
        kpi_card("Duration", kpis["duration_label"], f"{kpis['duration_sec']:.2f} seconds", "⏱️")

    with c4:
        kpi_card("Frames", f"{kpis['processed_frames']:,}", f"{kpis['estimated_fps']:.2f} video FPS", "🎞️")

    with c5:
        kpi_card("Alerts", f"{kpis['anomaly_count']}", "refined events", "⚠️")

    with c6:
        kpi_card("Avg FPS", f"{kpis['avg_pipeline_fps']:.2f}", "full offline pipeline", "⚡")


def get_video_path_for_dashboard(video_path: Path) -> Path:
    converted = PROJECT_ROOT / "results" / "videos_dashboard" / video_path.name
    return converted if converted.exists() else video_path


def render_video(data: dict) -> str:
    section_label("Video Stream")

    videos = data["videos"]
    midframe = data["midframe"]

    if not videos:
        st.warning("No rendered videos found. Showing midframe preview instead.")
        if midframe.exists():
            st.image(str(midframe), use_container_width=True)
        return "No video"

    video_names = list(videos.keys())
    default_idx = video_names.index("Zone Density + Risk") if "Zone Density + Risk" in video_names else 0

    selected_video = st.radio(
        "Video result type",
        video_names,
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_path = get_video_path_for_dashboard(videos[selected_video])

    try:
        st.video(str(selected_path))
    except Exception:
        st.warning("Video could not play in the browser. This is probably a codec issue.")
        if midframe.exists():
            st.image(str(midframe), use_container_width=True)

    st.caption(
        f"Showing {selected_video}. If blank, we will convert MP4 files to browser-friendly H.264 in the next step."
    )

    return selected_video


def render_zone_panel(data: dict) -> str:
    section_label("Zone Analysis")

    zone_df = data["zone_df"]
    tables = data["tables"]
    zones = data["available_zones"]

    zone_options = ["All Zones Summary"] + zones

    selected = st.selectbox(
        "Select zone",
        zone_options,
        index=0,
    )

    with st.container(border=True):
        if selected == "All Zones Summary":
            k = data["global_kpis"]
            summary = compact_text(data["global_summary_text"], max_sentences=2)

            h1, h2 = st.columns([3, 1])
            with h1:
                st.markdown('<div class="zone-heading">All Zones Summary</div>', unsafe_allow_html=True)
                st.markdown('<div class="zone-caption">Global overview across all manually annotated zones</div>', unsafe_allow_html=True)
            with h2:
                st.markdown('<div class="risk-text">HIGH</div>', unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Global Average", f"{k['avg_count']:.1f}", "persons / frame")
            with m2:
                st.metric("Peak Count", f"{k['max_count']}", f"at {k['peak_time_label']}")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("Main Hotspot", k["hotspot_zone"], k["hotspot_short"])
            with m4:
                st.metric("Hotspot Risk", f"{k['hotspot_high_critical_pct']:.1f}%", "HIGH/CRITICAL")

            st.markdown('<div class="quick-insight-title">Quick Insight</div>', unsafe_allow_html=True)
            st.info(summary)

            return selected

        zone_kpis = compute_zone_kpis(zone_df, selected, tables)
        insight = compact_text(build_zone_insight_text(zone_kpis), max_sentences=2)

        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown(
                f'<div class="zone-heading">{zone_kpis["zone_name"]} <code>{zone_kpis["zone_short_id"]}</code></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="zone-caption">Selected zone behavioral summary</div>', unsafe_allow_html=True)
        with h2:
            st.markdown(f'<div class="risk-text">{zone_kpis["current_risk"]}</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Average Count", f"{zone_kpis['avg_count']:.1f}", "persons / frame")
        with m2:
            st.metric("Peak Count", f"{zone_kpis['max_count']}", f"at {zone_kpis['peak_time_label']}")

        m3, m4 = st.columns(2)
        with m3:
            st.metric("Mean Density", f"{zone_kpis['mean_density']:.8f}", "pixel-based")
        with m4:
            st.metric("High/Critical", f"{zone_kpis['high_critical_pct']:.1f}%", "risk frames")

        m5, m6 = st.columns(2)
        with m5:
            st.metric("Current Count", f"{zone_kpis['current_count']}", "last frame")
        with m6:
            st.metric("Spike Events", f"{zone_kpis['spike_event_count']}", "refined rule")

        trends = zone_kpis["trend_summary"]

        st.markdown('<div class="quick-insight-title">Quick Insight</div>', unsafe_allow_html=True)
        st.info(insight)

        st.caption(
            f"Trend: {trends['ACCUMULATING_percent']:.1f}% accumulating · "
            f"{trends['DISPERSING_percent']:.1f}% dispersing · "
            f"{trends['STABLE_percent']:.1f}% stable."
        )

        return selected


def table(name: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return tables.get(name, pd.DataFrame())


def analysis_intro(title: str, text: str) -> None:
    html(
        f"""
        <div class="analysis-card">
            <div class="analysis-title">{title}</div>
            <div class="analysis-desc">{text}</div>
        </div>
        """
    )


def render_temporal_analysis(data: dict) -> None:
    frame_df = data["frame_df"]

    peak = frame_df.sort_values("total_count", ascending=False).iloc[0]
    avg = frame_df["total_count"].mean()

    analysis_intro(
        "Temporal Analysis",
        f"The timeline shows how the total crowd evolves across the 5-minute video. Average count is {avg:.1f}, and peak count is {int(peak['total_count'])} at {peak['timestamp_sec']:.2f} seconds.",
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.plotly_chart(charts.global_count_timeline(frame_df), use_container_width=True)

    with c2:
        st.plotly_chart(charts.count_change_proxy(frame_df), use_container_width=True)


def render_spatial_analysis(data: dict, selected_zone: str) -> None:
    tables = data["tables"]
    zone_df = data["zone_df"]

    zone_summary = table("zone_summary", tables)
    risk_percentage = table("risk_percentage", tables)
    density_trend = table("density_trend", tables)
    risk_duration = table("risk_duration", tables)

    if not zone_summary.empty:
        highest = zone_summary.sort_values("mean_count", ascending=False).iloc[0]
        desc = (
            f"Spatial analysis compares zones using count, density, and risk. "
            f"The highest average count zone is {highest['zone_name']} with mean count {highest['mean_count']:.1f}."
        )
    else:
        desc = "Spatial analysis compares zones using count, density, and risk."

    analysis_intro("Spatial / Zone Analysis", desc)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        metric = st.selectbox(
            "Compare zones by",
            ["mean_count", "max_count", "mean_density", "max_density"],
            index=0,
            help="This controls only the left chart below.",
        )

        st.plotly_chart(
            charts.zone_comparison_bar(zone_summary, metric=metric),
            use_container_width=True,
        )

    with c2:
        st.plotly_chart(
            charts.risk_percentage_stacked(risk_percentage),
            use_container_width=True,
        )

    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.plotly_chart(
            charts.density_trend_stacked(density_trend),
            use_container_width=True,
        )

    with c4:
        if selected_zone != "All Zones Summary":
            st.plotly_chart(
                charts.zone_count_timeline(zone_df, selected_zone),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                charts.high_critical_duration_bar(risk_duration),
                use_container_width=True,
            )


def render_anomaly_analysis(data: dict) -> None:
    tables = data["tables"]

    spike_summary = table("spike_summary", tables)
    multi_zone_timeline = table("multi_zone_timeline", tables)
    spike_events = table("spike_events", tables)
    multi_alerts = table("multi_zone_alerts", tables)

    spike_count = len(spike_events)
    alert_count = len(multi_alerts)

    top_zone_text = "No dominant anomaly zone was found."

    if not spike_summary.empty and "refined_spike_events" in spike_summary.columns:
        top_zone = spike_summary.sort_values("refined_spike_events", ascending=False).iloc[0]
        top_zone_text = (
            f"The zone with the most refined spike events is {top_zone['zone_name']} "
            f"with {int(top_zone['refined_spike_events'])} events."
        )

    analysis_intro(
        "Anomaly Detection",
        f"The refined anomaly layer detected {spike_count} sudden spike events and {alert_count} multi-zone alert events. {top_zone_text}",
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.plotly_chart(
            charts.refined_spike_events_bar(spike_summary),
            use_container_width=True,
        )

    with c2:
        st.plotly_chart(
            charts.multi_zone_alert_timeline(multi_zone_timeline),
            use_container_width=True,
        )

    if not multi_alerts.empty:
        st.markdown("#### Top Multi-Zone Alert Events")

        top_events = multi_alerts.sort_values("event_duration_sec", ascending=False).head(3)
        cols = st.columns(3)

        for i, (_, row) in enumerate(top_events.iterrows()):
            with cols[i]:
                st.metric(
                    "Alert Duration",
                    f"{row['event_duration_sec']:.1f}s",
                    f"Start {row['event_start_sec']:.1f}s · Peak {row['peak_total_count']:.0f}",
                )


def render_statistical_analysis(data: dict) -> None:
    tables = data["tables"]

    entropy = table("entropy_over_time", tables)
    corr = table("zone_count_correlation", tables)
    risk_transition = table("risk_transition_matrix", tables)
    lead_lag = table("lead_lag", tables)

    if not entropy.empty:
        mean_entropy = entropy["normalized_entropy"].mean()
        desc = (
            f"Statistical analysis describes how the crowd is distributed across zones. "
            f"Mean normalized entropy is {mean_entropy:.4f}; higher values mean the crowd is more evenly spread."
        )
    else:
        desc = "Statistical analysis describes how the crowd is distributed and how zones relate to each other."

    analysis_intro("Statistical Insights", desc)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.plotly_chart(charts.entropy_timeline(entropy), use_container_width=True)

    with c2:
        st.plotly_chart(
            charts.correlation_heatmap(corr, title="Zone Count Correlation Matrix"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.plotly_chart(charts.risk_transition_heatmap(risk_transition), use_container_width=True)

    with c4:
        st.markdown("#### Top Lead-Lag Relationships")

        if not lead_lag.empty:
            top = lead_lag.head(5)

            for _, row in top.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['zone_a']} ↔ {row['zone_b']}**")
                    st.caption(
                        f"Correlation: {row['best_correlation']:.3f} · "
                        f"Lag: {row['best_lag_seconds']:.2f}s"
                    )
                    st.write(row["interpretation"])
        else:
            st.info("Lead-lag table is not available.")


def render_analysis_area(data: dict, selected_zone: str) -> None:
    section_label("Analysis Explorer")

    analysis_type = st.selectbox(
        "Choose analysis type",
        [
            "1. Temporal analysis",
            "2. Spatial / zone analysis",
            "3. Anomaly detection",
            "4. Statistical insights",
        ],
        index=0,
        help="Select one analysis group. The charts below update automatically.",
    )

    if analysis_type.startswith("1."):
        render_temporal_analysis(data)
    elif analysis_type.startswith("2."):
        render_spatial_analysis(data, selected_zone)
    elif analysis_type.startswith("3."):
        render_anomaly_analysis(data)
    elif analysis_type.startswith("4."):
        render_statistical_analysis(data)


def methodology_footer() -> None:
    section_label("Methodology Notes")

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        html(
            """
            <div class="method-note">
                <b>Density:</b> pixel-based relative density from manual polygon area, not real-world persons/m².
            </div>
            """
        )

    with c2:
        html(
            """
            <div class="method-note">
                <b>Risk:</b> rule-based prototype thresholds from zone count, not certified safety limits.
            </div>
            """
        )

    with c3:
        html(
            """
            <div class="method-note">
                <b>Motion:</b> count-change is only a proxy. Real optical-flow congestion comes next.
            </div>
            """
        )


def main() -> None:
    inject_css()

    try:
        data = cached_dashboard_data()
    except Exception as exc:
        st.error(f"Could not load dashboard data: {exc}")
        st.stop()

    header()
    render_kpis(data["global_kpis"])

    left, right = st.columns([2.05, 1.0], gap="large")

    with left:
        render_video(data)

    with right:
        selected_zone = render_zone_panel(data)

    render_analysis_area(data, selected_zone)
    methodology_footer()

    st.caption(
        "Crowd Monitoring Dashboard · FIDTM localization · Zone density/risk · Refined temporal/spatial/statistical analysis"
    )


if __name__ == "__main__":
    main()