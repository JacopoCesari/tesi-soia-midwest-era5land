"""Comprehensive tests for ARCO Zarr acquisition and county aggregation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from soybean_yield_forecasting.era5.arco import ARCOClient
from soybean_yield_forecasting.era5.pipeline import ARCOPipeline, run_year, run_year_arco
from soybean_yield_forecasting.era5.spatial_weights import load_spatial_weights


@pytest.fixture
def sample_weights_file(tmp_path: Path) -> Path:
    """Create a minimal valid spatial weights CSV fixture for 3 test counties."""
    weights_path = tmp_path / "spatial_weights.csv"
    data = [
        # county 17001
        {"county_fips": "17001", "latitude": 40.0, "longitude": -90.0, "weight": 0.6},
        {"county_fips": "17001", "latitude": 40.1, "longitude": -90.0, "weight": 0.4},
        # county 17003
        {"county_fips": "17003", "latitude": 40.0, "longitude": -89.0, "weight": 0.5},
        {"county_fips": "17003", "latitude": 40.1, "longitude": -89.0, "weight": 0.5},
        # county 17005
        {"county_fips": "17005", "latitude": 40.0, "longitude": -88.0, "weight": 1.0},
    ]
    pd.DataFrame(data).to_csv(weights_path, index=False)
    return weights_path


@pytest.fixture
def mock_hourly_slice() -> xr.Dataset:
    """Generate a synthetic 7-day hourly grid of ARCO variables for Midwest coordinates."""
    times = pd.date_range("1950-06-01 00:00:00", "1950-06-07 23:00:00", freq="1h")
    lats = [40.0, 40.1]
    lons = [-90.0, -89.0, -88.0]

    np.random.seed(42)
    shape = (len(times), len(lats), len(lons))

    # Temperatures in Kelvin (e.g. 293.15 K = 20 C)
    t2m = 293.15 + np.random.uniform(-5.0, 5.0, size=shape)
    d2m = t2m - np.random.uniform(2.0, 6.0, size=shape)
    # Wind in m/s
    u10 = np.random.uniform(-3.0, 3.0, size=shape)
    v10 = np.random.uniform(-3.0, 3.0, size=shape)
    # Surface pressure in Pa
    sp = 101325.0 + np.random.uniform(-500.0, 500.0, size=shape)
    # Precipitation in meters (e.g. 0.001 m/hr = 1 mm/hr)
    tp = np.random.exponential(scale=0.0002, size=shape)
    # Solar radiation in J/m^2 (hourly de-accumulated)
    ssrd = np.maximum(0.0, np.random.uniform(0.0, 2_000_000.0, size=shape))
    strd = np.random.uniform(1_000_000.0, 1_500_000.0, size=shape)
    # Soil water in m^3/m^3 (0 to 1)
    swvl1 = np.full(shape, 0.28)
    swvl2 = np.full(shape, 0.30)
    swvl3 = np.full(shape, 0.32)
    swvl4 = np.full(shape, 0.35)
    # Soil temp in Kelvin
    stl1 = np.full(shape, 292.0)
    stl2 = np.full(shape, 291.0)
    stl3 = np.full(shape, 290.0)
    stl4 = np.full(shape, 289.0)
    # Snow
    snowc = np.zeros(shape)
    sde = np.zeros(shape)
    # Skin temp
    skt = t2m + 1.0

    ds = xr.Dataset(
        data_vars={
            "t2m": (["valid_time", "latitude", "longitude"], t2m),
            "d2m": (["valid_time", "latitude", "longitude"], d2m),
            "u10": (["valid_time", "latitude", "longitude"], u10),
            "v10": (["valid_time", "latitude", "longitude"], v10),
            "sp": (["valid_time", "latitude", "longitude"], sp),
            "tp": (["valid_time", "latitude", "longitude"], tp),
            "ssrd": (["valid_time", "latitude", "longitude"], ssrd),
            "strd": (["valid_time", "latitude", "longitude"], strd),
            "swvl1": (["valid_time", "latitude", "longitude"], swvl1),
            "swvl2": (["valid_time", "latitude", "longitude"], swvl2),
            "swvl3": (["valid_time", "latitude", "longitude"], swvl3),
            "swvl4": (["valid_time", "latitude", "longitude"], swvl4),
            "stl1": (["valid_time", "latitude", "longitude"], stl1),
            "stl2": (["valid_time", "latitude", "longitude"], stl2),
            "stl3": (["valid_time", "latitude", "longitude"], stl3),
            "stl4": (["valid_time", "latitude", "longitude"], stl4),
            "snowc": (["valid_time", "latitude", "longitude"], snowc),
            "sde": (["valid_time", "latitude", "longitude"], sde),
            "skt": (["valid_time", "latitude", "longitude"], skt),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


def test_arco_pipeline_offline_execution(
    tmp_path: Path, sample_weights_file: Path, mock_hourly_slice: xr.Dataset
) -> None:
    """Test full offline ARCO acquisition, aggregation, unit conversion and derived features."""
    out_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"

    # Mock ARCOClient
    mock_client = MagicMock(spec=ARCOClient)
    mock_client.fetch_slice.return_value = mock_hourly_slice

    pipeline = ARCOPipeline(
        weights_path=sample_weights_file,
        output_directory=out_dir,
        expected_counties=3,
        arco_client=mock_client,
        cache_dir=cache_dir,
    )

    result_parquet = pipeline.run_year(
        year=1950,
        preflight=True,
        test_days=["01", "02", "03", "04", "05", "06", "07"],
        compute_derived=True,
    )

    assert result_parquet.exists()
    df = pd.read_parquet(result_parquet)

    # 3 counties x 7 days = 21 rows
    assert len(df) == 21
    assert df["county_fips"].nunique() == 3
    assert df["date"].nunique() == 7

    # Check canonical weather columns exist
    expected_core = [
        "air_temperature_mean",
        "air_temperature_minimum",
        "air_temperature_maximum",
        "dewpoint_temperature_mean",
        "eastward_wind_mean",
        "northward_wind_mean",
        "surface_pressure",
        "total_precipitation",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "volumetric_soil_water_layer_1",
        "soil_temperature_level_1",
        "skin_temperature_maximum",
    ]
    for col in expected_core:
        assert col in df.columns, f"Missing core variable {col}"

    # Check unit conversions:
    # Temperature should be in Celsius (~15 to 25 C), not Kelvin (>280)
    assert -20.0 < df["air_temperature_mean"].mean() < 40.0
    assert -20.0 < df["air_temperature_minimum"].mean() < 40.0
    assert -20.0 < df["air_temperature_maximum"].mean() < 40.0
    # Pressure should be in hPa (~1013), not Pa (>100000)
    assert 900.0 < df["surface_pressure"].mean() < 1100.0
    # Precipitation should be in mm
    assert df["total_precipitation"].min() >= 0.0
    # Radiation should be in MJ/m^2 (~5 to 40), not J/m^2 (>1e6)
    assert 0.0 <= df["surface_solar_radiation_downwards"].mean() < 50.0

    # Check derived features
    expected_derived = [
        "wind_speed",
        "relative_humidity",
        "vapor_pressure_deficit",
        "growing_degree_days",
        "heat_day_30",
        "heat_day_35",
        "et0_fao56",
        "p_minus_et0",
    ]
    for col in expected_derived:
        assert col in df.columns, f"Missing derived feature {col}"

    # Check physical constraints on derived features
    assert (df["wind_speed"] >= 0.0).all()
    assert (df["relative_humidity"].between(0.0, 105.0)).all()
    assert (df["vapor_pressure_deficit"] >= 0.0).all()
    assert (df["growing_degree_days"] >= 0.0).all()
    assert (df["heat_day_30"].isin([0, 1])).all()
    assert (df["et0_fao56"] >= 0.0).all()

    # Check QC and metadata sidecars
    qc_file = result_parquet.with_suffix(".qc.json")
    meta_file = result_parquet.with_suffix(".meta.json")
    assert qc_file.exists()
    assert meta_file.exists()

    qc_json = json.loads(qc_file.read_text(encoding="utf-8"))
    assert qc_json["status"] == "PASS"

    meta_json = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta_json["engine"] == "arco"
    assert meta_json["rows"] == 21


def test_arco_pipeline_idempotence(
    tmp_path: Path, sample_weights_file: Path, mock_hourly_slice: xr.Dataset
) -> None:
    """Test that a completed, validated year returns immediately on rerun without re-fetching."""
    out_dir = tmp_path / "output"
    mock_client = MagicMock(spec=ARCOClient)
    mock_client.fetch_slice.return_value = mock_hourly_slice

    pipeline = ARCOPipeline(
        weights_path=sample_weights_file,
        output_directory=out_dir,
        expected_counties=3,
        arco_client=mock_client,
    )

    first_path = pipeline.run_year(
        year=1950,
        preflight=True,
        test_days=["01", "02", "03", "04", "05", "06", "07"],
    )
    fetch_count = mock_client.fetch_slice.call_count
    assert fetch_count > 0

    # Second run should detect existing validated file and skip fetching
    second_path = pipeline.run_year(
        year=1950,
        preflight=True,
        test_days=["01", "02", "03", "04", "05", "06", "07"],
    )
    assert first_path == second_path
    assert mock_client.fetch_slice.call_count == fetch_count


def test_run_year_engine_routing(
    tmp_path: Path, sample_weights_file: Path, mock_hourly_slice: xr.Dataset
) -> None:
    """Test that top-level run_year defaults to ARCO engine and routes properly."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"

    mock_client = MagicMock(spec=ARCOClient)
    mock_client.fetch_slice.return_value = mock_hourly_slice

    out_file = run_year(
        year=1950,
        raw_directory=raw_dir,
        output_directory=out_dir,
        weights=sample_weights_file,
        expected_counties=3,
        arco_client=mock_client,
        preflight=True,
        test_days=["01", "02", "03", "04", "05", "06", "07"],
    )
    assert out_file.exists()
    assert out_file.name == "county_daily_1950.parquet"


@pytest.mark.network
def test_arco_live_inventory_probe() -> None:
    """Live network test: probe official ECMWF ARCO stores (requires network)."""
    client = ARCOClient()
    inv = client.discover_inventory()
    assert inv["base_url"] == "https://arco.datastores.ecmwf.int"
    assert "sfc-2m-temperature" in inv["stores"]
    meta = inv["stores"]["sfc-2m-temperature"]
    if meta["available"]:
        assert "t2m" in meta["variables"]
        assert "d2m" in meta["variables"]
