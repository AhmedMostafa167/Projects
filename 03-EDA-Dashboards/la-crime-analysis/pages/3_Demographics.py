"""Demographics page — reported victim age, sex, descent.

Important framing: this is *reported victim* data, not predicted offender
data. The sidebar callout below makes that explicit.
"""

from pathlib import Path

import streamlit as st

from src.data import load_and_prepare
from src.filters import sidebar_filters
from src.viz import age_distribution, descent_distribution, sex_distribution

DATA_PATH = Path(__file__).parent.parent / "crimes.csv"


@st.cache_data(show_spinner="Loading data...")
def get_data():
    return load_and_prepare(DATA_PATH)


st.set_page_config(page_title="LA Crime — Demographics", page_icon="👥", layout="wide")

df = sidebar_filters(get_data())

st.title("Reported victim demographics")
st.warning(
    "This page visualizes who is **reported as a victim**, not who commits crime. "
    "Demographic data also has well-known reporting biases — these charts should "
    "be read as 'who is reported to LAPD' rather than 'who is harmed'."
)

left, right = st.columns(2)
with left:
    st.plotly_chart(age_distribution(df), use_container_width=True)
with right:
    st.plotly_chart(sex_distribution(df), use_container_width=True)

st.plotly_chart(descent_distribution(df), use_container_width=True)
