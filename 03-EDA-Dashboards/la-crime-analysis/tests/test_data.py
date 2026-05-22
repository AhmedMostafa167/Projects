"""Feature engineering and filtering are the pieces that need tests —
the visualizations are visual review territory."""

import pandas as pd

from src.data import add_features, apply_filters


def _toy_raw():
    return pd.DataFrame(
        {
            "DR_NO": [1, 2, 3, 4, 5],
            "Date Rptd": pd.to_datetime(
                ["2022-01-10", "2022-06-20", "2023-03-15", "2023-11-01", "2024-02-02"]
            ),
            "DATE OCC": pd.to_datetime(
                ["2022-01-05", "2022-06-15", "2023-03-10", "2023-10-25", "2024-02-01"]
            ),
            "TIME OCC": ["0830", "1430", "2300", "0200", "0930"],
            "AREA NAME": ["Central", "Hollywood", "Central", "Pacific", "Hollywood"],
            "Crm Cd Desc": ["THEFT", "BURGLARY", "VANDALISM", "THEFT", "BURGLARY"],
            "Vict Age": [20, 45, 70, 30, 16],
            "Vict Sex": ["M", "F", "M", "X", "F"],
            "Vict Descent": ["W", "H", "B", "A", "W"],
            "Weapon Desc": [None, None, "KNIFE", None, None],
            "Status Desc": ["IC", "IC", "AA", "IC", "IC"],
            "LOCATION": ["1st", "2nd", "3rd", "4th", "5th"],
        }
    )


def test_add_features_creates_expected_columns():
    df = add_features(_toy_raw())
    assert {"hour", "is_night", "year", "month", "dow", "dow_name", "age_group",
            "descent_label", "area", "crime_type"}.issubset(df.columns)


def test_hour_derived_from_time_occ():
    df = add_features(_toy_raw())
    assert df["hour"].tolist() == [8, 14, 23, 2, 9]


def test_is_night_flag():
    df = add_features(_toy_raw())
    # Hours 23 and 02 are in NIGHT_HOURS = {22,23,0,1,2,3}.
    assert df["is_night"].tolist() == [False, False, True, True, False]


def test_age_group_binning():
    df = add_features(_toy_raw())
    # ages 20, 45, 70, 30, 16
    assert df["age_group"].astype(str).tolist() == ["18-25", "45-54", "65+", "26-34", "0-17"]


def test_filter_by_date_range_inclusive():
    df = add_features(_toy_raw())
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-12-31")
    out = apply_filters(df, date_range=(start, end))
    assert len(out) == 2  # the two 2023 rows


def test_filter_by_area_and_type():
    df = add_features(_toy_raw())
    out = apply_filters(df, areas=["Hollywood"], crime_types=["BURGLARY"])
    assert len(out) == 2
    assert (out["area"] == "Hollywood").all()
    assert (out["crime_type"] == "BURGLARY").all()


def test_descent_label_mapping():
    df = add_features(_toy_raw())
    assert df["descent_label"].tolist() == ["White", "Hispanic/Latino", "Black", "Other Asian", "White"]
