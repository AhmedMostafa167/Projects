"""Reusable Plotly figure builders.

Charts return `plotly.graph_objects.Figure` objects so pages can compose
multiple charts and apply layout tweaks (e.g. `update_layout(height=...)`)
without each chart hard-coding those decisions.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data import AGE_BIN_LABELS, DAY_NAMES

# Common color palette: muted enough to keep secondary charts readable,
# bright enough that the primary trace pops.
PRIMARY = "#3b82f6"   # blue-500
SECONDARY = "#ef4444" # red-500
MUTED = "#94a3b8"     # slate-400


def monthly_trend(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby("month").size().reset_index(name="crimes")
    fig = px.line(monthly, x="month", y="crimes", title="Crimes per month")
    fig.update_traces(line_color=PRIMARY, line_width=2.5)
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Crimes",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def hour_distribution(df: pd.DataFrame) -> go.Figure:
    hourly = df.groupby("hour").size().reset_index(name="crimes")
    fig = px.bar(hourly, x="hour", y="crimes", title="Crimes by hour of day")
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=2),
        xaxis_title="Hour (0-23)",
        yaxis_title="Crimes",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def dow_distribution(df: pd.DataFrame) -> go.Figure:
    by_dow = df.groupby("dow_name", observed=True).size().reset_index(name="crimes")
    by_dow["dow_name"] = pd.Categorical(by_dow["dow_name"], categories=DAY_NAMES, ordered=True)
    by_dow = by_dow.sort_values("dow_name")
    fig = px.bar(by_dow, x="dow_name", y="crimes", title="Crimes by day of week")
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Crimes",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def hour_by_area_heatmap(df: pd.DataFrame, top_n_areas: int = 15) -> go.Figure:
    top_areas = df["area"].value_counts().head(top_n_areas).index
    sub = df[df["area"].isin(top_areas)]
    pivot = sub.pivot_table(
        index="area", columns="hour", values="DR_NO", aggfunc="count", fill_value=0
    ).reindex(top_areas)
    fig = px.imshow(
        pivot.values,
        labels=dict(x="Hour", y="Area", color="Crimes"),
        x=pivot.columns,
        y=pivot.index,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"Crime intensity by hour and area (top {top_n_areas})",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def area_bar(df: pd.DataFrame) -> go.Figure:
    by_area = df["area"].value_counts().reset_index()
    by_area.columns = ["area", "crimes"]
    fig = px.bar(
        by_area.sort_values("crimes"),
        x="crimes",
        y="area",
        orientation="h",
        title="Crimes by area",
    )
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(
        height=max(400, 25 * len(by_area)),
        yaxis_title=None,
        xaxis_title="Crimes",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def top_crime_types(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df["crime_type"].value_counts().head(n).reset_index()
    top.columns = ["crime_type", "count"]
    fig = px.bar(
        top.sort_values("count"),
        x="count",
        y="crime_type",
        orientation="h",
        title=f"Top {n} crime types",
    )
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(
        yaxis_title=None,
        xaxis_title="Crimes",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def crime_type_by_area_heatmap(df: pd.DataFrame, top_n_types: int = 12, top_n_areas: int = 15) -> go.Figure:
    top_types = df["crime_type"].value_counts().head(top_n_types).index
    top_areas = df["area"].value_counts().head(top_n_areas).index
    sub = df[df["crime_type"].isin(top_types) & df["area"].isin(top_areas)]
    pivot = sub.pivot_table(
        index="crime_type", columns="area", values="DR_NO", aggfunc="count", fill_value=0
    )
    # Reorder by overall frequency for both axes.
    pivot = pivot.reindex(index=top_types, columns=top_areas)
    fig = px.imshow(
        pivot.values,
        labels=dict(x="Area", y="Crime type", color="Count"),
        x=pivot.columns,
        y=pivot.index,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"Top {top_n_types} crime types × top {top_n_areas} areas",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def age_distribution(df: pd.DataFrame) -> go.Figure:
    by_age = df["age_group"].value_counts().reindex(AGE_BIN_LABELS).reset_index()
    by_age.columns = ["age_group", "count"]
    fig = px.bar(by_age, x="age_group", y="count", title="Victim age distribution")
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Reported victims",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def sex_distribution(df: pd.DataFrame) -> go.Figure:
    by_sex = df["sex"].value_counts().reset_index()
    by_sex.columns = ["sex", "count"]
    sex_label = {"M": "Male", "F": "Female", "X": "Unknown / Other"}
    by_sex["sex"] = by_sex["sex"].map(sex_label).fillna("Unknown")
    fig = px.pie(by_sex, values="count", names="sex", title="Victim sex")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def descent_distribution(df: pd.DataFrame, n: int = 8) -> go.Figure:
    by = df["descent_label"].value_counts().head(n).reset_index()
    by.columns = ["descent", "count"]
    fig = px.bar(by.sort_values("count"), x="count", y="descent", orientation="h",
                 title=f"Top {n} victim descents (as reported)")
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(
        yaxis_title=None,
        xaxis_title="Reported victims",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def night_vs_day(df: pd.DataFrame) -> go.Figure:
    by = df.groupby(["area", "is_night"]).size().reset_index(name="crimes")
    by["period"] = by["is_night"].map({True: "Night (22-04)", False: "Day"})
    pivoted = by.pivot(index="area", columns="period", values="crimes").fillna(0)
    pivoted["total"] = pivoted.sum(axis=1)
    pivoted = pivoted.sort_values("total", ascending=True).drop(columns="total")
    fig = go.Figure()
    for col, color in zip(pivoted.columns, [MUTED, PRIMARY], strict=False):
        fig.add_trace(go.Bar(y=pivoted.index, x=pivoted[col], name=col, orientation="h", marker_color=color))
    fig.update_layout(
        barmode="stack",
        title="Night vs day crimes by area",
        height=max(400, 25 * len(pivoted)),
        xaxis_title="Crimes",
        yaxis_title=None,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
