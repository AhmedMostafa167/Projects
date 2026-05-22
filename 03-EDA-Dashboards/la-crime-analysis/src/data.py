"""Data loading and feature engineering.

The CSV is 27 MB / ~185k rows. Loaded once per Streamlit session via
`st.cache_data` and held in memory thereafter — re-parsing on every
interaction would make filtering feel sluggish.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AGE_BIN_LABELS = ["0-17", "18-25", "26-34", "35-44", "45-54", "55-64", "65+"]
AGE_BIN_EDGES = [-1, 17, 25, 34, 44, 54, 64, 200]

# Hours considered "night" for the night-crime breakdown.
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


DESCENT_MAP = {
    "A": "Other Asian", "B": "Black", "C": "Chinese", "D": "Cambodian",
    "F": "Filipino", "G": "Guamanian", "H": "Hispanic/Latino",
    "I": "Native American", "J": "Japanese", "K": "Korean", "L": "Laotian",
    "O": "Other", "P": "Pacific Islander", "S": "Samoan", "U": "Hawaiian",
    "V": "Vietnamese", "W": "White", "X": "Unknown", "Z": "Asian Indian",
}


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Read the CSV with appropriate dtypes for the date and time columns."""
    return pd.read_csv(
        csv_path,
        parse_dates=["Date Rptd", "DATE OCC"],
        dtype={"TIME OCC": str},
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns the dashboard plots on. Pure function -> easy to test."""
    out = df.copy()

    # Pad to four chars in case of 3-digit times like '900' (= 09:00).
    times = out["TIME OCC"].astype(str).str.zfill(4)
    out["hour"] = times.str[:2].astype(int).clip(0, 23)
    out["is_night"] = out["hour"].isin(NIGHT_HOURS)

    out["year"] = out["DATE OCC"].dt.year
    out["month"] = out["DATE OCC"].dt.to_period("M").dt.to_timestamp()
    out["dow"] = out["DATE OCC"].dt.dayofweek  # 0 = Monday
    out["dow_name"] = pd.Categorical(
        out["dow"].map(dict(enumerate(DAY_NAMES))),
        categories=DAY_NAMES,
        ordered=True,
    )

    out["age_group"] = pd.cut(
        out["Vict Age"], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS, ordered=True
    )

    out["descent_label"] = out["Vict Descent"].map(DESCENT_MAP).fillna("Unknown")

    # Rename for downstream readability
    out = out.rename(
        columns={
            "AREA NAME": "area",
            "Crm Cd Desc": "crime_type",
            "Vict Sex": "sex",
            "Weapon Desc": "weapon",
            "Status Desc": "status",
            "DATE OCC": "date_occ",
            "Date Rptd": "date_rptd",
        }
    )
    return out


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Combined entry point for the Streamlit app."""
    return add_features(load_raw(csv_path))


def apply_filters(
    df: pd.DataFrame,
    *,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    areas: list[str] | None = None,
    crime_types: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the sidebar filters. Each filter is a no-op if its arg is None/empty."""
    out = df
    if date_range:
        start, end = date_range
        out = out[(out["date_occ"] >= start) & (out["date_occ"] <= end)]
    if areas:
        out = out[out["area"].isin(areas)]
    if crime_types:
        out = out[out["crime_type"].isin(crime_types)]
    return out
