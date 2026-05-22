"""LA Crime Dashboard — Streamlit entry point (Overview page).

Run:
    streamlit run app.py

Additional pages live in `pages/` — Streamlit picks them up automatically
and renders them in the left sidebar.
"""

from pathlib import Path

import streamlit as st

from src.data import load_and_prepare
from src.filters import sidebar_filters
from src.viz import (
    hour_by_area_heatmap,
    monthly_trend,
    top_crime_types,
)

DATA_PATH = Path(__file__).parent / "crimes.csv"


@st.cache_data(show_spinner="Loading 185k crime records...")
def get_data():
    return load_and_prepare(DATA_PATH)


st.set_page_config(
    page_title="LA Crime Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

df_all = get_data()
df = sidebar_filters(df_all)

st.title("LA Crime Dashboard")
st.markdown(
    "Interactive view of the LAPD crime records dataset. "
    "Use the sidebar to filter by date, area, or crime type — every chart updates."
)

# --- KPI cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total crimes", f"{len(df):,}")
col2.metric("Unique areas", df["area"].nunique())
col3.metric(
    "Date range",
    f"{df['date_occ'].min().date()} → {df['date_occ'].max().date()}"
    if len(df)
    else "—",
)
col4.metric(
    "Top crime type",
    df["crime_type"].value_counts().index[0] if len(df) else "—",
)

st.divider()

# --- Monthly trend ---
st.plotly_chart(monthly_trend(df), use_container_width=True)

# --- Two-column section ---
left, right = st.columns([1.2, 1])
with left:
    st.plotly_chart(top_crime_types(df, n=12), use_container_width=True)
with right:
    st.plotly_chart(hour_by_area_heatmap(df, top_n_areas=10), use_container_width=True)

# --- Notes ---
with st.expander("Notes on this data"):
    st.markdown(
        "- **Date filter** uses occurrence date (`DATE OCC`), not report date.\n"
        "- The data has rows where a crime occurred years before being reported; "
        "those still appear under the date they actually happened.\n"
        "- Crime-type filter shows the top 40 most-frequent types in the sidebar; "
        "use the dataset directly if you need to filter to a long-tail type."
    )
