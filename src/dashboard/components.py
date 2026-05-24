"""
Reusable Streamlit dashboard components.

This module contains:
- global CSS
- KPI cards
- section headers
- risk badges
- selected-zone summary cards
- global summary cards
- insight boxes

The goal is to keep app.py clean and make the dashboard look professional.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


RISK_BADGE_CLASS = {
    "LOW": "risk-low",
    "MEDIUM": "risk-medium",
    "HIGH": "risk-high",
    "CRITICAL": "risk-critical",
}


def inject_global_css() -> None:
    """
    Inject global CSS for the Streamlit dashboard.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #070b14;
            --bg-panel: #0f172a;
            --bg-panel-soft: #111c31;
            --bg-card: #111827;
            --bg-card-hover: #162238;
            --border-main: rgba(59, 130, 246, 0.20);
            --border-soft: rgba(148, 163, 184, 0.16);
            --text-main: #e5e7eb;
            --text-muted: #94a3b8;
            --text-soft: #cbd5e1;
            --blue: #3b82f6;
            --cyan: #38bdf8;
            --green: #22c55e;
            --yellow: #f59e0b;
            --orange: #f97316;
            --red: #ef4444;
            --purple: #a78bfa;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(59,130,246,0.18), transparent 32%),
                radial-gradient(circle at 80% 0%, rgba(14,165,233,0.10), transparent 30%),
                linear-gradient(135deg, #050914 0%, #08111f 45%, #070b14 100%);
            color: var(--text-main);
        }

        .block-container {
            max-width: 1680px;
            padding-top: 1.1rem;
            padding-left: 1.6rem;
            padding-right: 1.6rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stHeader"] {
            background: rgba(7, 11, 20, 0.75);
            backdrop-filter: blur(10px);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07111f 0%, #050914 100%);
            border-right: 1px solid rgba(59,130,246,0.18);
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.85rem;
        }

        .main-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(59,130,246,0.18);
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.20);
            backdrop-filter: blur(14px);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .header-icon {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(37,99,235,0.8), rgba(14,165,233,0.55));
            color: white;
            font-size: 22px;
            box-shadow: 0 0 22px rgba(59,130,246,0.35);
        }

        .header-title {
            font-size: 25px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 2px;
            letter-spacing: -0.02em;
        }

        .header-subtitle {
            font-size: 13px;
            color: var(--text-muted);
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border-radius: 999px;
            padding: 8px 13px;
            font-size: 13px;
            font-weight: 700;
            border: 1px solid rgba(34,197,94,0.35);
            background: rgba(34,197,94,0.12);
            color: #86efac;
        }

        .experiment-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border-radius: 999px;
            padding: 8px 13px;
            font-size: 13px;
            font-weight: 700;
            border: 1px solid rgba(59,130,246,0.35);
            background: rgba(59,130,246,0.11);
            color: #93c5fd;
        }

        .time-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border-radius: 999px;
            padding: 8px 13px;
            font-size: 12px;
            border: 1px solid rgba(148,163,184,0.22);
            background: rgba(15,23,42,0.85);
            color: #cbd5e1;
        }

        .section-title {
            margin: 9px 0 9px 0;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #93a4bd;
        }

        .panel {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(59,130,246,0.18);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.22);
            backdrop-filter: blur(12px);
        }

        .panel-tight {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(59,130,246,0.18);
            border-radius: 18px;
            padding: 14px;
            box-shadow: 0 16px 40px rgba(0,0,0,0.18);
            backdrop-filter: blur(12px);
        }

        .kpi-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(145deg, rgba(17,24,39,0.98), rgba(15,23,42,0.92));
            border: 1px solid rgba(59,130,246,0.19);
            border-radius: 18px;
            padding: 17px 18px;
            min-height: 132px;
            box-shadow: 0 14px 34px rgba(0,0,0,0.18);
        }

        .kpi-card::before {
            content: "";
            position: absolute;
            right: -45px;
            top: -45px;
            width: 130px;
            height: 130px;
            background: radial-gradient(circle, rgba(59,130,246,0.22), transparent 68%);
        }

        .kpi-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .kpi-icon {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            background: rgba(59,130,246,0.14);
            border: 1px solid rgba(59,130,246,0.25);
        }

        .kpi-label {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .kpi-value {
            font-size: 29px;
            line-height: 1.05;
            color: #f8fafc;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .kpi-sub {
            margin-top: 7px;
            font-size: 12px;
            color: var(--text-muted);
            line-height: 1.35;
        }

        .zone-card {
            background:
                linear-gradient(145deg, rgba(15,23,42,0.98), rgba(17,24,39,0.92));
            border: 1px solid rgba(59,130,246,0.22);
            border-radius: 18px;
            padding: 18px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.015);
        }

        .zone-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .zone-name {
            font-size: 20px;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }

        .zone-short {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 8px;
            padding: 3px 8px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 800;
            background: rgba(59,130,246,0.18);
            color: #93c5fd;
            border: 1px solid rgba(59,130,246,0.25);
        }

        .risk-badge {
            border-radius: 999px;
            padding: 7px 11px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.04em;
            border: 1px solid;
        }

        .risk-low {
            color: #86efac;
            background: rgba(34,197,94,0.12);
            border-color: rgba(34,197,94,0.34);
        }

        .risk-medium {
            color: #fde68a;
            background: rgba(245,158,11,0.12);
            border-color: rgba(245,158,11,0.34);
        }

        .risk-high {
            color: #fdba74;
            background: rgba(249,115,22,0.14);
            border-color: rgba(249,115,22,0.38);
        }

        .risk-critical {
            color: #fca5a5;
            background: rgba(239,68,68,0.16);
            border-color: rgba(239,68,68,0.42);
        }

        .zone-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }

        .zone-metric {
            border: 1px solid rgba(148,163,184,0.14);
            background: rgba(15,23,42,0.60);
            border-radius: 14px;
            padding: 13px;
        }

        .zone-metric-label {
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .zone-metric-value {
            color: #f8fafc;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .zone-metric-sub {
            margin-top: 4px;
            color: #94a3b8;
            font-size: 11px;
        }

        .insight-box {
            background: rgba(8, 15, 28, 0.70);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 15px;
            padding: 14px 15px;
            color: #cbd5e1;
            font-size: 13px;
            line-height: 1.55;
        }

        .insight-title {
            color: #f8fafc;
            font-size: 14px;
            font-weight: 800;
            margin-bottom: 7px;
        }

        .video-hint {
            color: #94a3b8;
            font-size: 12px;
            margin-top: 8px;
            line-height: 1.45;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 12px;
            line-height: 1.45;
        }

        .warning-note {
            background: rgba(245,158,11,0.10);
            border: 1px solid rgba(245,158,11,0.25);
            color: #fde68a;
            border-radius: 14px;
            padding: 12px 14px;
            font-size: 13px;
            line-height: 1.45;
        }

        .success-note {
            background: rgba(34,197,94,0.10);
            border: 1px solid rgba(34,197,94,0.25);
            color: #bbf7d0;
            border-radius: 14px;
            padding: 12px 14px;
            font-size: 13px;
            line-height: 1.45;
        }

        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(59,130,246,0.25) !important;
            background: rgba(15,23,42,0.90) !important;
            color: #dbeafe !important;
            font-weight: 700 !important;
            transition: 0.18s ease-in-out !important;
        }

        .stButton > button:hover {
            border-color: rgba(56,189,248,0.70) !important;
            background: rgba(37,99,235,0.30) !important;
            color: white !important;
            transform: translateY(-1px);
        }

        .stRadio > div {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        div[data-testid="stVideo"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(59,130,246,0.22);
            box-shadow: 0 18px 45px rgba(0,0,0,0.24);
        }

        .stSelectbox label,
        .stRadio label,
        .stSlider label {
            color: #cbd5e1 !important;
            font-weight: 700 !important;
        }

        div[data-baseweb="select"] > div {
            background: rgba(15,23,42,0.92) !important;
            border-color: rgba(59,130,246,0.28) !important;
            color: #e5e7eb !important;
            border-radius: 12px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(15,23,42,0.80);
            border: 1px solid rgba(59,130,246,0.16);
            border-radius: 12px;
            padding: 9px 15px;
            color: #cbd5e1;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(37,99,235,0.30);
            color: #ffffff;
            border-color: rgba(56,189,248,0.55);
        }

        .dataframe {
            border-radius: 14px !important;
            overflow: hidden !important;
        }

        hr {
            border-color: rgba(148,163,184,0.14);
        }

        @media (max-width: 1100px) {
            .main-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }

            .zone-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """
    Render the dashboard top header.
    """
    st.markdown(
        """
        <div class="main-header">
            <div class="header-left">
                <div class="header-icon">👥</div>
                <div>
                    <div class="header-title">Crowd Monitoring Dashboard</div>
                    <div class="header-subtitle">Intelligent Crowd Monitoring and Behavioral Analysis System</div>
                </div>
            </div>
            <div class="header-right">
                <div class="status-pill">● Data Loaded</div>
                <div class="experiment-pill">🧪 5-minute Shinjuku Experiment</div>
                <div class="time-pill">🕒 308 sec · 8 zones · FIDTM outputs</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    """
    Render a small uppercase section title.
    """
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    sub: str,
    icon: str = "📊",
) -> None:
    """
    Render one KPI card.
    """
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level: str) -> str:
    """
    Return HTML for a risk badge.
    """
    risk = str(risk_level).upper()
    badge_class = RISK_BADGE_CLASS.get(risk, "risk-medium")

    return f'<span class="risk-badge {badge_class}">{risk}</span>'


def zone_kpi_card(zone_kpis: dict[str, Any], insight_text: str) -> None:
    """
    Render the selected-zone KPI card.
    """
    if not zone_kpis:
        st.markdown(
            """
            <div class="zone-card">
                <div class="zone-name">No zone selected</div>
                <div class="small-muted">Select a zone from the dropdown.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    risk_html = risk_badge(zone_kpis["current_risk"])

    high_seconds = zone_kpis.get("high_critical_seconds")
    high_seconds_text = "Not available" if high_seconds is None else f"{high_seconds:.2f} sec"

    accumulating = zone_kpis["trend_summary"]["ACCUMULATING_percent"]
    dispersing = zone_kpis["trend_summary"]["DISPERSING_percent"]
    stable = zone_kpis["trend_summary"]["STABLE_percent"]

    st.markdown(
        f"""
        <div class="zone-card">
            <div class="zone-card-header">
                <div>
                    <span class="zone-name">{zone_kpis['zone_name']}</span>
                    <span class="zone-short">{zone_kpis['zone_short_id']}</span>
                </div>
                <div>{risk_html}</div>
            </div>

            <div class="zone-grid">
                <div class="zone-metric">
                    <div class="zone-metric-label">Average Count</div>
                    <div class="zone-metric-value">{zone_kpis['avg_count']:.1f}</div>
                    <div class="zone-metric-sub">persons / frame</div>
                </div>

                <div class="zone-metric">
                    <div class="zone-metric-label">Peak Count</div>
                    <div class="zone-metric-value">{zone_kpis['max_count']}</div>
                    <div class="zone-metric-sub">at {zone_kpis['peak_time_label']}</div>
                </div>

                <div class="zone-metric">
                    <div class="zone-metric-label">Mean Pixel Density</div>
                    <div class="zone-metric-value" style="font-size:18px;">{zone_kpis['mean_density']:.8f}</div>
                    <div class="zone-metric-sub">relative image-space density</div>
                </div>

                <div class="zone-metric">
                    <div class="zone-metric-label">HIGH / CRITICAL</div>
                    <div class="zone-metric-value">{zone_kpis['high_critical_pct']:.2f}%</div>
                    <div class="zone-metric-sub">{high_seconds_text}</div>
                </div>

                <div class="zone-metric">
                    <div class="zone-metric-label">Current Count</div>
                    <div class="zone-metric-value">{zone_kpis['current_count']}</div>
                    <div class="zone-metric-sub">last processed frame</div>
                </div>

                <div class="zone-metric">
                    <div class="zone-metric-label">Refined Spike Events</div>
                    <div class="zone-metric-value">{zone_kpis['spike_event_count']}</div>
                    <div class="zone-metric-sub">selective anomaly rule</div>
                </div>
            </div>

            <div class="insight-box">
                <div class="insight-title">Zone Insight</div>
                {insight_text}
                <br><br>
                <b>Trend split:</b> {accumulating:.2f}% accumulating · {dispersing:.2f}% dispersing · {stable:.2f}% stable
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def global_summary_box(summary_text: str) -> None:
    """
    Render global summary when no zone-specific interpretation is needed.
    """
    st.markdown(
        f"""
        <div class="zone-card">
            <div class="zone-card-header">
                <div>
                    <span class="zone-name">Global Summary</span>
                    <span class="zone-short">ALL</span>
                </div>
                <div>{risk_badge("HIGH")}</div>
            </div>
            <div class="insight-box">
                <div class="insight-title">Experiment Insight</div>
                {summary_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note_box(text: str, kind: str = "success") -> None:
    """
    Render a styled note box.
    """
    css_class = "success-note" if kind == "success" else "warning-note"

    st.markdown(
        f"""
        <div class="{css_class}">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_video_warning() -> None:
    """
    Note about possible video codec issues.
    """
    note_box(
        "If a video does not play in the browser, the file is probably using OpenCV MP4 codec. "
        "The dashboard data and charts still work. Later we can convert only the selected videos to H.264.",
        kind="warning",
    )