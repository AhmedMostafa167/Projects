"""Geography page — where crime happens, broken down by area."""

from pathlib import Path

import streamlit as st

from src.data import load_and_prepare
from src.filters import sidebar_filters
from src.viz import area_bar, crime_type_by_area_heatmap, night_vs_day

DATA_PATH = Path(__file__).parent.parent / "crimes.csv"


@st.cache_data(show_spinner="Loading data...")
def get_data():
    return load_and_prepare(DATA_PATH)


st.set_page_config(page_title="LA Crime — Geography", page_icon="📍", layout="wide")

df = sidebar_filters(get_data())

st.title("Geography")
st.markdown("Where crime is reported. Use the sidebar to filter by date or crime type.")

st.plotly_chart(area_bar(df), use_container_width=True)
st.plotly_chart(crime_type_by_area_heatmap(df), use_container_width=True)
st.plotly_chart(night_vs_day(df), use_container_width=True)

st.info(
    "**Why no map?** The raw data has street addresses (`LOCATION`) but no "
    "lat/lon coordinates. A real choropleth would require geocoding the "
    "~190k addresses — out of scope here, but called out as next-step work "
    "in the project README."
)
