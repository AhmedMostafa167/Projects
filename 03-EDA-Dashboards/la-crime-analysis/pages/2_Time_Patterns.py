"""Time-patterns page — when crime happens (hour, day of week, monthly trend)."""

from pathlib import Path

import streamlit as st

from src.data import NIGHT_HOURS, load_and_prepare
from src.filters import sidebar_filters
from src.viz import dow_distribution, hour_distribution, monthly_trend

DATA_PATH = Path(__file__).parent.parent / "crimes.csv"


@st.cache_data(show_spinner="Loading data...")
def get_data():
    return load_and_prepare(DATA_PATH)


st.set_page_config(page_title="LA Crime — Time", page_icon="⏰", layout="wide")

df = sidebar_filters(get_data())

st.title("Time patterns")
st.markdown("When crime happens — hour-of-day, day-of-week, monthly drift.")

# Hour + DoW side by side
left, right = st.columns(2)
with left:
    st.plotly_chart(hour_distribution(df), use_container_width=True)
with right:
    st.plotly_chart(dow_distribution(df), use_container_width=True)

# Full-width monthly trend
st.plotly_chart(monthly_trend(df), use_container_width=True)

# A small "did you know" stat
if len(df):
    night = df["is_night"].sum()
    night_share = night / len(df) * 100
    night_hours_str = ", ".join(f"{h:02d}:00" for h in sorted(NIGHT_HOURS))
    st.metric(
        f"Crimes between {night_hours_str}",
        f"{night:,}",
        f"{night_share:.1f}% of selection",
    )
