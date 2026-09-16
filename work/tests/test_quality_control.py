"""QC detects dates, physical inconsistency and coverage failures."""

import pandas as pd
import xarray as xr
from soybean_yield_forecasting.era5.quality_control import validate_county_daily, validate_era5_dataset


def test_wrong_calendar_same_number_of_days():
    dataset = xr.Dataset(coords={"time": pd.date_range("1950-01-02", periods=365)})
    assert validate_era5_dataset(dataset, year=1950)[0] == "FAIL"


def test_missing_dates_and_incomplete_daily_coverage():
    frame = pd.DataFrame(
        {
            "county_fips": ["17001", "17003", "17001", "17003"],
            "date": pd.to_datetime(["1950-01-01", "1950-01-01", "1950-01-03", "1950-01-03"]),
        }
    )
    assert validate_county_daily(frame, 2)[0] == "FAIL"
    frame.loc[2, "date"] = pd.Timestamp("1950-01-02")
    assert validate_county_daily(frame, 2)[0] == "FAIL"


def test_mean_above_maximum_is_rejected():
    dataset = xr.Dataset(
        {"t2m_min": ("time", [280.0]), "t2m_mean": ("time", [300.0]), "t2m_max": ("time", [290.0])},
        coords={"time": pd.date_range("1950-01-01", periods=1)},
    )
    assert validate_era5_dataset(dataset, is_preflight=True)[0] == "FAIL"


def test_annual_479_county_cardinality():
    dates = pd.date_range("1952-01-01", "1952-12-31")
    frame = pd.MultiIndex.from_product(
        [[f"{index:05d}" for index in range(479)], dates], names=["county_fips", "date"]
    ).to_frame(index=False)
    assert len(frame) == 175314
    assert validate_county_daily(frame, 479, year=1952)[0] == "PASS"


def test_missing_or_nonfinite_values_rejected():
    frame = pd.DataFrame(
        {
            "county_fips": ["17001"],
            "date": pd.to_datetime(["1950-01-01"]),
            "total_precipitation": [float("inf")],
        }
    )
    assert validate_county_daily(frame, 1)[0] == "FAIL"
    frame["total_precipitation"] = float("nan")
    assert validate_county_daily(frame, 1)[0] == "FAIL"
