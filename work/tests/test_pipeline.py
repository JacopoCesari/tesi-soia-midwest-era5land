"""Original seven regression tests, retained through the refactor."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from soybean_yield_forecasting.era5.conversions import apply_evaporation_swap, apply_unit_conversions
from soybean_yield_forecasting.era5.manifest import ManifestManager
from soybean_yield_forecasting.era5.quality_control import validate_county_daily, validate_era5_dataset
from soybean_yield_forecasting.era5.variables import ACCUMULATED_VARIABLES


def test_leap_year_calendar_validation() -> None:
    """Test leap year calendar validation."""
    dates_common = pd.date_range("1950-01-01", "1950-12-31", freq="D")
    assert len(dates_common) == 365
    ds_common = xr.Dataset(coords={"valid_time": dates_common})
    status, notes = validate_era5_dataset(ds_common, year=1950, is_preflight=False)
    assert status == "PASS"
    dates_leap = pd.date_range("1952-01-01", "1952-12-31", freq="D")
    assert len(dates_leap) == 366
    ds_leap = xr.Dataset(coords={"valid_time": dates_leap})
    status_leap, notes_leap = validate_era5_dataset(ds_leap, year=1952, is_preflight=False)
    assert status_leap == "PASS"
    ds_incomplete = xr.Dataset(coords={"valid_time": dates_common})
    status_fail, notes_fail = validate_era5_dataset(ds_incomplete, year=1952, is_preflight=False)
    assert status_fail == "FAIL"
    assert "365 vs 366 expected" in notes_fail


def test_evaporation_swap() -> None:
    """Test evaporation swap."""
    dummy_data = np.ones((2, 2))
    ds = xr.Dataset(
        data_vars={
            "evabs": (["lat", "lon"], dummy_data * 10),
            "evaow": (["lat", "lon"], dummy_data * 20),
            "evavt": (["lat", "lon"], dummy_data * 30),
        }
    )
    ds_swapped = apply_evaporation_swap(ds)
    assert "evabs" not in ds_swapped.data_vars
    assert "evaow" not in ds_swapped.data_vars
    assert "evavt" not in ds_swapped.data_vars
    assert "evaporation_from_vegetation_transpiration" in ds_swapped.data_vars
    assert "evaporation_from_bare_soil" in ds_swapped.data_vars
    assert "evaporation_from_open_water_surfaces_excluding_oceans" in ds_swapped.data_vars
    assert ds_swapped["evaporation_from_vegetation_transpiration"].values[0, 0] == 10
    assert ds_swapped["evaporation_from_bare_soil"].values[0, 0] == 20
    assert ds_swapped["evaporation_from_open_water_surfaces_excluding_oceans"].values[0, 0] == 30


def test_county_daily_cardinality_and_uniqueness() -> None:
    """Test county daily cardinality and uniqueness."""
    counties = [f"{i:05d}" for i in range(479)]
    dates = pd.date_range("1950-06-01", periods=6, freq="D")
    records = []
    for d in dates:
        for c in counties:
            records.append({"date": d, "county_fips": c, "t2m_mean": 20.0, "tp": 5.0})
    df_valid = pd.DataFrame(records)
    status, msg = validate_county_daily(df_valid, expected_counties=479, expected_days=6)
    assert status == "PASS"
    df_missing_county = df_valid[df_valid["county_fips"] != "00000"].copy()
    status_fail, msg_fail = validate_county_daily(df_missing_county, expected_counties=479, expected_days=6)
    assert status_fail == "FAIL"
    df_dup = df_valid.copy()
    df_dup.iloc[1] = df_dup.iloc[0]
    status_dup, msg_dup = validate_county_daily(df_dup, expected_counties=479, expected_days=6)
    assert status_dup == "FAIL"
    assert "duplicate" in msg_dup


def test_manifest_checksum_validation() -> None:
    """Test manifest checksum validation."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / "manifest.csv"
        manifest = ManifestManager(manifest_path)
        test_file = tmp_dir / "sample.nc"
        test_file.write_bytes(b"A" * 2000)
        manifest.record(1950, "A", "daily_mean", test_file, "PASS")
        assert manifest.is_done(test_file, verify_checksum=False) is True
        assert manifest.is_done(test_file, verify_checksum=True) is True
        test_file.write_bytes(b"B" * 2000)
        assert manifest.is_done(test_file, verify_checksum=False) is True
        assert manifest.is_done(test_file, verify_checksum=True) is False


def test_unit_conversions() -> None:
    """Test unit conversions."""
    df_raw = pd.DataFrame(
        {"t2m_mean": [293.15], "surface_pressure": [101325.0], "tp": [0.025], "ssrd": [15000000.0]}
    )
    df_conv = apply_unit_conversions(df_raw)
    assert pytest.approx(df_conv["t2m_mean"].iloc[0], rel=0.0001) == 20.0
    assert pytest.approx(df_conv["surface_pressure"].iloc[0], rel=0.0001) == 1013.25
    assert pytest.approx(df_conv["tp"].iloc[0], rel=0.0001) == 25.0
    assert pytest.approx(df_conv["ssrd"].iloc[0], rel=0.0001) == 15.0


def test_channel_b_request_field_count() -> None:
    """Test channel b request field count."""
    num_vars = len(ACCUMULATED_VARIABLES)
    assert num_vars == 17
    fields_main_common = 365 * num_vars
    fields_main_leap = 366 * num_vars
    assert fields_main_common == 6205
    assert fields_main_leap == 6222
    fields_bound = 1 * num_vars
    assert fields_bound == 17
    assert fields_main_common + fields_bound == 6222
    assert fields_main_leap + fields_bound == 6239
    assert fields_main_leap + fields_bound < 12000


def test_spatial_weights_integrity(project_root) -> None:
    """Test spatial weights integrity."""
    weights_path = project_root / "../work/data/interim/spatial_weights/spatial_weights_479_counties.parquet"
    assert weights_path.exists()
    df_w = pd.read_parquet(weights_path)
    assert len(df_w) == 11953
    assert df_w["county_fips"].nunique() == 479
    sums = df_w.groupby("county_fips")["weight"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-05)
