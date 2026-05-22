"""Sidebar filter widgets.

Each page calls `sidebar_filters(df)` to render the same controls and
get back a filtered DataFrame. Keeping this in one place means the
controls are consistent across pages and easy to extend.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import apply_filters


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    min_date = df["date_occ"].min().date()
    max_date = df["date_occ"].max().date()
    date_range = st.sidebar.date_input(
        "Occurrence date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    # Streamlit returns a tuple of date objects when both are picked, or one
    # date while the user is still selecting the second.
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_range_ts = (
            pd.Timestamp(date_range[0]),
            pd.Timestamp(date_range[1]) + pd.Timedelta(days=1),
        )
    else:
        date_range_ts = None

    areas = st.sidebar.multiselect(
        "Areas",
        options=sorted(df["area"].dropna().unique()),
        default=None,
        placeholder="All areas",
    )

    top_types = df["crime_type"].value_counts().head(40).index.tolist()
    crime_types = st.sidebar.multiselect(
        "Crime types (top 40 by frequency shown)",
        options=top_types,
        default=None,
        placeholder="All types",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Source: LAPD via LA Open Data. "
        "Built with Streamlit + Plotly."
    )

    return apply_filters(df, date_range=date_range_ts, areas=areas, crime_types=crime_types)
