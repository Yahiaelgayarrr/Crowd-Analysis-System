"""
Dashboard Plotly charts.

This module contains all interactive chart functions used by the Streamlit
dashboard. The goal is to keep app.py clean and make all charts reusable.

Charts included:
- global crowd timeline
- count change proxy
- zone count timeline
- zone density timeline
- zone comparison bars
- risk percentage stacked bar
- density trend stacked bar
- correlation heatmap
- entropy timeline
- refined spike events
- multi-zone alerts
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOT_TEMPLATE = "plotly_dark"

RISK_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#f59e0b",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}

TREND_COLORS = {
    "ACCUMULATING": "#ef4444",
    "DISPERSING": "#38bdf8",
    "STABLE": "#22c55e",
}


def apply_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    """
    Apply shared dark dashboard styling to Plotly figures.
    """
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,15,28,0.75)",
        font=dict(
            family="Inter, Segoe UI, Arial",
            color="#dbeafe",
            size=12,
        ),
        title=dict(
            font=dict(size=15, color="#f8fafc"),
            x=0.02,
            xanchor="left",
        ),
        margin=dict(l=30, r=25, t=55, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#334155",
            font_size=12,
            font_family="Inter, Segoe UI, Arial",
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.15)",
        zerolinecolor="rgba(148,163,184,0.25)",
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.15)",
        zerolinecolor="rgba(148,163,184,0.25)",
    )

    return fig


def global_count_timeline(frame_df: pd.DataFrame) -> go.Figure:
    """
    Global crowd count over time with rolling mean.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frame_df["timestamp_sec"],
            y=frame_df["total_count"],
            mode="lines",
            name="Total Count",
            line=dict(width=2.2, color="#38bdf8"),
            hovertemplate="Time: %{x:.2f}s<br>Total Count: %{y}<extra></extra>",
        )
    )

    if "rolling_mean_count_5s" in frame_df.columns:
        fig.add_trace(
            go.Scatter(
                x=frame_df["timestamp_sec"],
                y=frame_df["rolling_mean_count_5s"],
                mode="lines",
                name="5s Rolling Mean",
                line=dict(width=2, color="#a78bfa", dash="dot"),
                hovertemplate="Time: %{x:.2f}s<br>5s Mean: %{y:.2f}<extra></extra>",
            )
        )

    peak = frame_df.sort_values("total_count", ascending=False).iloc[0]

    fig.add_trace(
        go.Scatter(
            x=[peak["timestamp_sec"]],
            y=[peak["total_count"]],
            mode="markers+text",
            name="Peak",
            marker=dict(size=10, color="#ef4444", symbol="diamond"),
            text=[f"Peak {int(peak['total_count'])}"],
            textposition="top center",
            hovertemplate=(
                "Peak Time: %{x:.2f}s<br>"
                "Peak Count: %{y}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Global Crowd Count Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Total Count",
    )

    return apply_layout(fig, height=370)


def count_change_proxy(frame_df: pd.DataFrame) -> go.Figure:
    """
    Frame-to-frame count change proxy.
    This is not optical flow.
    """
    df = frame_df.copy()

    if "total_count_change" not in df.columns:
        df["total_count_change"] = df["total_count"].diff().fillna(0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp_sec"],
            y=df["total_count_change"],
            mode="lines",
            name="Count Change",
            line=dict(width=1.8, color="#f59e0b"),
            hovertemplate="Time: %{x:.2f}s<br>Count Change: %{y}<extra></extra>",
        )
    )

    if "rolling_mean_count_change_5s" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_sec"],
                y=df["rolling_mean_count_change_5s"],
                mode="lines",
                name="5s Rolling Change",
                line=dict(width=2.1, color="#38bdf8"),
                hovertemplate="Time: %{x:.2f}s<br>5s Change: %{y:.2f}<extra></extra>",
            )
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(226,232,240,0.45)",
    )

    fig.update_layout(
        title="Temporal Count-Change Proxy",
        xaxis_title="Time (seconds)",
        yaxis_title="Frame Count Change",
    )

    return apply_layout(fig, height=340)


def inference_fps_timeline(frame_df: pd.DataFrame) -> go.Figure:
    """
    Full pipeline FPS over time.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frame_df["timestamp_sec"],
            y=frame_df["inference_fps"],
            mode="lines",
            name="Pipeline FPS",
            line=dict(width=2, color="#22c55e"),
            hovertemplate="Time: %{x:.2f}s<br>FPS: %{y:.2f}<extra></extra>",
        )
    )

    mean_fps = frame_df["inference_fps"].mean()

    fig.add_hline(
        y=mean_fps,
        line_dash="dot",
        line_color="#a78bfa",
        annotation_text=f"Mean {mean_fps:.2f} FPS",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Full Pipeline FPS Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="FPS",
    )

    return apply_layout(fig, height=330)


def count_distribution(frame_df: pd.DataFrame) -> go.Figure:
    """
    Histogram of global crowd counts.
    """
    fig = px.histogram(
        frame_df,
        x="total_count",
        nbins=35,
        title="Global Count Distribution",
        labels={"total_count": "Total Count"},
        template=PLOT_TEMPLATE,
        color_discrete_sequence=["#38bdf8"],
    )

    fig.update_traces(
        marker_line_width=0,
        opacity=0.85,
        hovertemplate="Count Range: %{x}<br>Frames: %{y}<extra></extra>",
    )

    fig.update_layout(
        xaxis_title="Total Count",
        yaxis_title="Number of Frames",
    )

    return apply_layout(fig, height=330)


def zone_count_timeline(zone_df: pd.DataFrame, zone_name: str) -> go.Figure:
    """
    Selected zone count timeline.
    """
    df = zone_df[zone_df["zone_name"] == zone_name].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp_sec"],
            y=df["zone_count"],
            mode="lines",
            name="Zone Count",
            line=dict(width=2.2, color="#38bdf8"),
            hovertemplate="Time: %{x:.2f}s<br>Zone Count: %{y}<extra></extra>",
        )
    )

    peak = df.sort_values("zone_count", ascending=False).iloc[0]

    fig.add_trace(
        go.Scatter(
            x=[peak["timestamp_sec"]],
            y=[peak["zone_count"]],
            mode="markers",
            name="Peak",
            marker=dict(size=10, color="#ef4444", symbol="diamond"),
            hovertemplate="Peak Time: %{x:.2f}s<br>Peak Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Selected Zone Count Timeline — {zone_name}",
        xaxis_title="Time (seconds)",
        yaxis_title="Zone Count",
    )

    return apply_layout(fig, height=310)


def zone_density_timeline(zone_df: pd.DataFrame, zone_name: str) -> go.Figure:
    """
    Selected zone density timeline.
    """
    df = zone_df[zone_df["zone_name"] == zone_name].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp_sec"],
            y=df["zone_density_pixel"],
            mode="lines",
            name="Pixel-Based Density",
            line=dict(width=2.2, color="#a78bfa"),
            hovertemplate="Time: %{x:.2f}s<br>Density: %{y:.8f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Selected Zone Pixel-Based Density — {zone_name}",
        xaxis_title="Time (seconds)",
        yaxis_title="Density",
    )

    return apply_layout(fig, height=310)


def zone_comparison_bar(zone_summary: pd.DataFrame, metric: str = "mean_count") -> go.Figure:
    """
    Horizontal zone comparison bar.
    """
    if zone_summary.empty:
        return empty_figure("Zone comparison unavailable")

    metric_label = {
        "mean_count": "Mean Count",
        "max_count": "Peak Count",
        "mean_density": "Mean Density",
        "max_density": "Peak Density",
    }.get(metric, metric)

    df = zone_summary.sort_values(metric, ascending=True).copy()

    fig = px.bar(
        df,
        x=metric,
        y="zone_name",
        orientation="h",
        title=f"Zone Comparison — {metric_label}",
        labels={
            metric: metric_label,
            "zone_name": "Zone",
        },
        template=PLOT_TEMPLATE,
        color=metric,
        color_continuous_scale=["#0f172a", "#1d4ed8", "#38bdf8"],
    )

    fig.update_traces(
        hovertemplate="Zone: %{y}<br>" + metric_label + ": %{x}<extra></extra>",
    )

    fig.update_layout(coloraxis_showscale=False)

    return apply_layout(fig, height=370)


def risk_percentage_stacked(risk_percentage: pd.DataFrame) -> go.Figure:
    """
    Risk distribution stacked bar per zone.
    """
    if risk_percentage.empty:
        return empty_figure("Risk percentage unavailable")

    fig = go.Figure()

    for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        col = f"{risk}_percent"

        if col in risk_percentage.columns:
            fig.add_trace(
                go.Bar(
                    x=risk_percentage["zone_name"],
                    y=risk_percentage[col],
                    name=risk,
                    marker_color=RISK_COLORS[risk],
                    hovertemplate=(
                        "Zone: %{x}<br>"
                        f"{risk}: " + "%{y:.2f}%<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="Risk Level Distribution by Zone",
        xaxis_title="Zone",
        yaxis_title="Percentage of Frames (%)",
        barmode="stack",
        xaxis_tickangle=-30,
    )

    return apply_layout(fig, height=370)


def high_critical_duration_bar(risk_duration: pd.DataFrame) -> go.Figure:
    """
    HIGH/CRITICAL duration by zone.
    """
    if risk_duration.empty:
        return empty_figure("Risk duration unavailable")

    df = risk_duration.sort_values("high_or_critical_seconds", ascending=True).copy()

    fig = px.bar(
        df,
        x="high_or_critical_seconds",
        y="zone_name",
        orientation="h",
        title="HIGH/CRITICAL Risk Duration",
        labels={
            "high_or_critical_seconds": "Duration (seconds)",
            "zone_name": "Zone",
        },
        template=PLOT_TEMPLATE,
        color="high_or_critical_percent",
        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
    )

    fig.update_traces(
        hovertemplate=(
            "Zone: %{y}<br>"
            "Duration: %{x:.2f}s<br>"
            "Risk %: %{marker.color:.2f}%<extra></extra>"
        )
    )

    fig.update_layout(coloraxis_showscale=False)

    return apply_layout(fig, height=370)


def density_trend_stacked(density_trend: pd.DataFrame) -> go.Figure:
    """
    Accumulating/dispersing/stable percentage by zone.
    """
    if density_trend.empty:
        return empty_figure("Density trend unavailable")

    fig = go.Figure()

    for trend in ["ACCUMULATING", "DISPERSING", "STABLE"]:
        col = f"{trend}_percent"

        if col in density_trend.columns:
            fig.add_trace(
                go.Bar(
                    x=density_trend["zone_name"],
                    y=density_trend[col],
                    name=trend,
                    marker_color=TREND_COLORS[trend],
                    hovertemplate=(
                        "Zone: %{x}<br>"
                        f"{trend}: " + "%{y:.2f}%<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="Temporal Density Trend by Zone",
        xaxis_title="Zone",
        yaxis_title="Percentage of Frames (%)",
        barmode="stack",
        xaxis_tickangle=-30,
    )

    return apply_layout(fig, height=370)


def correlation_heatmap(corr_df: pd.DataFrame, title: str = "Zone Count Correlation") -> go.Figure:
    """
    Correlation heatmap for zones.
    """
    if corr_df.empty:
        return empty_figure("Correlation unavailable")

    df = corr_df.copy()

    first_col = df.columns[0]

    if first_col.lower().startswith("unnamed") or first_col == "zone_name":
        df = df.set_index(first_col)

    fig = px.imshow(
        df,
        text_auto=".2f",
        aspect="auto",
        title=title,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template=PLOT_TEMPLATE,
    )

    fig.update_traces(
        hovertemplate="Zone X: %{x}<br>Zone Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title="Corr"),
    )

    return apply_layout(fig, height=420)


def entropy_timeline(entropy_df: pd.DataFrame) -> go.Figure:
    """
    Crowd distribution entropy over time.
    """
    if entropy_df.empty:
        return empty_figure("Entropy unavailable")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=entropy_df["timestamp_sec"],
            y=entropy_df["normalized_entropy"],
            mode="lines",
            name="Normalized Entropy",
            line=dict(width=2.2, color="#22c55e"),
            hovertemplate="Time: %{x:.2f}s<br>Entropy: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Crowd Distribution Entropy Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized Entropy",
    )

    return apply_layout(fig, height=330)


def refined_spike_events_bar(spike_summary: pd.DataFrame) -> go.Figure:
    """
    Refined sudden spike event count by zone.
    """
    if spike_summary.empty or "refined_spike_events" not in spike_summary.columns:
        return empty_figure("Refined spike events unavailable")

    df = spike_summary.sort_values("refined_spike_events", ascending=True).copy()

    fig = px.bar(
        df,
        x="refined_spike_events",
        y="zone_name",
        orientation="h",
        title="Refined Sudden Spike Events by Zone",
        labels={
            "refined_spike_events": "Spike Events",
            "zone_name": "Zone",
        },
        template=PLOT_TEMPLATE,
        color="refined_spike_events",
        color_continuous_scale=["#0f172a", "#f59e0b", "#ef4444"],
    )

    fig.update_traces(
        hovertemplate="Zone: %{y}<br>Spike Events: %{x}<extra></extra>",
    )

    fig.update_layout(coloraxis_showscale=False)

    return apply_layout(fig, height=350)


def multi_zone_alert_timeline(multi_zone_timeline: pd.DataFrame) -> go.Figure:
    """
    Multi-zone high/critical count over time.
    """
    if multi_zone_timeline.empty:
        return empty_figure("Multi-zone alert timeline unavailable")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=multi_zone_timeline["timestamp_sec"],
            y=multi_zone_timeline["high_or_critical_zone_count"],
            mode="lines",
            name="HIGH/CRITICAL Zones",
            line=dict(width=2.2, color="#ef4444"),
            fill="tozeroy",
            hovertemplate=(
                "Time: %{x:.2f}s<br>"
                "High/Critical Zones: %{y}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Multi-Zone Risk Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Number of HIGH/CRITICAL Zones",
    )

    return apply_layout(fig, height=330)


def risk_transition_heatmap(matrix_df: pd.DataFrame) -> go.Figure:
    """
    Risk transition matrix heatmap.
    """
    if matrix_df.empty:
        return empty_figure("Risk transition matrix unavailable")

    df = matrix_df.copy()

    first_col = df.columns[0]

    if first_col.lower().startswith("unnamed") or first_col in ["from_risk", "risk_level"]:
        df = df.set_index(first_col)

    fig = px.imshow(
        df,
        text_auto=True,
        aspect="auto",
        title="Risk Transition Matrix",
        color_continuous_scale=["#0f172a", "#1d4ed8", "#38bdf8"],
        template=PLOT_TEMPLATE,
    )

    fig.update_traces(
        hovertemplate="From: %{y}<br>To: %{x}<br>Transitions: %{z}<extra></extra>"
    )

    return apply_layout(fig, height=360)


def selected_zone_risk_pie(zone_df: pd.DataFrame, zone_name: str) -> go.Figure:
    """
    Pie/donut chart for selected-zone risk distribution.
    """
    df = zone_df[zone_df["zone_name"] == zone_name].copy()

    if df.empty:
        return empty_figure("Selected zone risk unavailable")

    counts = (
        df["risk_level_text"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        .fillna(0)
        .reset_index()
    )

    counts.columns = ["risk_level", "frames"]

    fig = px.pie(
        counts,
        names="risk_level",
        values="frames",
        title=f"Risk Distribution — {zone_name}",
        hole=0.55,
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        template=PLOT_TEMPLATE,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="Risk: %{label}<br>Frames: %{value}<br>Share: %{percent}<extra></extra>",
    )

    return apply_layout(fig, height=310)


def empty_figure(title: str) -> go.Figure:
    """
    Empty placeholder figure.
    """
    fig = go.Figure()

    fig.add_annotation(
        text=title,
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#94a3b8"),
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return apply_layout(fig, height=320)