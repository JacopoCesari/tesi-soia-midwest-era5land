"""Tests for local transformations: aggregations, unit conversions, and derived features."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from soybean_yield_forecasting.era5.transformations import (
    aggregate_hourly_dataset_to_daily,
    apply_unit_conversions_arco,
    compute_all_derived_features,
    compute_climatic_water_balance,
    compute_fao56_et0,
    compute_growing_degree_days,
    compute_heat_stress_days,
    compute_relative_humidity,
    compute_rolling_features,
    compute_temporal_aggregations,
    compute_vapor_pressure_deficit,
    compute_wind_speed,
    joules_to_megajoules,
    kelvin_to_celsius,
    meters_to_mm,
    pa_to_hpa,
)


def test_unit_conversions():
    """Verify physical unit conversions and magnitude thresholds."""
    # Temperature K -> C
    assert abs(kelvin_to_celsius(300.15) - 27.0) < 1e-6
    assert abs(kelvin_to_celsius(273.15) - 0.0) < 1e-6

    # Pressure Pa -> hPa
    assert abs(pa_to_hpa(101325.0) - 1013.25) < 1e-6

    # Water m -> mm
    assert abs(meters_to_mm(0.025) - 25.0) < 1e-6

    # Radiation J/m² -> MJ/m²
    assert abs(joules_to_megajoules(15_000_000.0) - 15.0) < 1e-6

    # DataFrame conversion
    df = pd.DataFrame(
        {
            "air_temperature_mean": [290.0, 300.0],
            "surface_pressure": [101325.0, 100000.0],
            "total_precipitation": [0.010, 0.020],
            "surface_solar_radiation_downwards": [20_000_000.0, 15_000_000.0],
        }
    )
    converted = apply_unit_conversions_arco(df)
    assert 16.0 < converted["air_temperature_mean"].iloc[0] < 17.0
    assert 1013.0 < converted["surface_pressure"].iloc[0] < 1014.0
    assert converted["total_precipitation"].iloc[0] == 10.0
    assert converted["surface_solar_radiation_downwards"].iloc[0] == 20.0


def test_wind_speed():
    """Verify wind speed calculation sqrt(u10^2 + v10^2)."""
    assert abs(compute_wind_speed(3.0, 4.0) - 5.0) < 1e-6
    u = pd.Series([0.0, 6.0])
    v = pd.Series([0.0, 8.0])
    res = compute_wind_speed(u, v)
    assert res.iloc[0] == 0.0
    assert res.iloc[1] == 10.0


def test_relative_humidity_and_vpd():
    """Verify RH and VPD from temperature and dewpoint."""
    # When T = Tdew, RH = 100%, VPD = 0
    t = pd.Series([20.0, 25.0])
    td = pd.Series([20.0, 15.0])
    rh = compute_relative_humidity(t, td)
    vpd = compute_vapor_pressure_deficit(t, td)

    assert abs(rh.iloc[0] - 100.0) < 1e-3
    assert abs(vpd.iloc[0] - 0.0) < 1e-3

    # When T > Tdew, RH < 100%, VPD > 0
    assert rh.iloc[1] < 60.0
    assert vpd.iloc[1] > 1.0


def test_growing_degree_days():
    """Verify GDD base 10°C cutoff 30°C logic."""
    # Normal warm summer day: Tmax=28, Tmin=16 -> (28+16)/2 - 10 = 12
    assert abs(compute_growing_degree_days(28.0, 16.0) - 12.0) < 1e-6

    # Extreme hot day capped at 30: Tmax=36 (capped to 30), Tmin=20 -> (30+20)/2 - 10 = 15
    assert abs(compute_growing_degree_days(36.0, 20.0) - 15.0) < 1e-6

    # Cold day below base: Tmax=8, Tmin=2 -> adjusted to 10 and 10 -> (10+10)/2 - 10 = 0
    assert abs(compute_growing_degree_days(8.0, 2.0) - 0.0) < 1e-6


def test_heat_stress_days():
    """Verify heat stress indicator days (>= 30°C and >= 35°C)."""
    tmax = pd.Series([28.0, 31.0, 36.0])
    d30 = compute_heat_stress_days(tmax, 30.0)
    d35 = compute_heat_stress_days(tmax, 35.0)
    assert list(d30) == [0, 1, 1]
    assert list(d35) == [0, 0, 1]


def test_fao56_et0_and_climatic_water_balance():
    """Verify FAO-56 Penman-Monteith ET0 calculation and P - ET0."""
    tmean = pd.Series([22.0])
    tmax = pd.Series([28.0])
    tmin = pd.Series([16.0])
    ssrd = pd.Series([22.0])  # MJ/m²/day
    wind = pd.Series([2.5])  # m/s
    tdew = pd.Series([15.0])
    sp = pd.Series([1010.0])

    et0 = compute_fao56_et0(tmean, tmax, tmin, ssrd, wind, tdew, sp, latitude_deg=41.0, day_of_year=180)
    # Typical mid-latitude summer ET0 is in the 3.0 to 7.0 mm/day range
    assert 3.0 <= et0.iloc[0] <= 7.0

    p = pd.Series([10.0])
    bal = compute_climatic_water_balance(p, et0)
    assert abs(bal.iloc[0] - (10.0 - et0.iloc[0])) < 1e-6


def test_hourly_to_daily_gridded_aggregation():
    """Verify gridded hourly dataset aggregation into daily mean, min, max, sum."""
    times = pd.date_range("1950-06-01", periods=48, freq="1h")
    lats = [40.0, 41.0]
    lons = [-90.0, -89.0]

    # Hourly pattern: 0..47
    data_t2m = np.arange(48, dtype=np.float32).reshape(48, 1, 1).repeat(2, axis=1).repeat(2, axis=2)
    data_tp = np.ones((48, 2, 2), dtype=np.float32) * 0.001  # 1 mm per hour

    ds = xr.Dataset(
        data_vars={
            "t2m": (["time", "latitude", "longitude"], data_t2m),
            "tp": (["time", "latitude", "longitude"], data_tp),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    daily = aggregate_hourly_dataset_to_daily(
        ds,
        mean_vars=["t2m"],
        min_vars=["t2m"],
        max_vars=["t2m"],
        sum_vars=["tp"],
    )

    assert len(daily.time) == 2
    # Day 1 mean of 0..23 = 11.5
    assert abs(float(daily.t2m.isel(time=0, latitude=0, longitude=0)) - 11.5) < 1e-5
    # Day 1 min = 0.0, max = 23.0
    assert abs(float(daily.t2m_min.isel(time=0, latitude=0, longitude=0)) - 0.0) < 1e-5
    assert abs(float(daily.t2m_max.isel(time=0, latitude=0, longitude=0)) - 23.0) < 1e-5
    # Day 1 sum of tp = 24 * 0.001 = 0.024 m
    assert abs(float(daily.tp.isel(time=0, latitude=0, longitude=0)) - 0.024) < 1e-5


def test_rolling_and_temporal_aggregations():
    """Verify rolling windows and multi-day temporal block aggregations."""
    dates = pd.date_range("1950-06-01", periods=30, freq="1D")
    df = pd.DataFrame(
        {
            "county_fips": ["17001"] * 30,
            "date": dates,
            "air_temperature_mean": np.linspace(15.0, 25.0, 30),
            "total_precipitation": [2.0] * 30,
        }
    )
    # Rolling
    rolling_df = compute_rolling_features(df["air_temperature_mean"], windows=[7, 14])
    assert "air_temperature_mean_rolling_7d_mean" in rolling_df.columns
    assert "air_temperature_mean_rolling_14d_sum" in rolling_df.columns

    # Temporal aggregation (5-day blocks)
    block_df = compute_temporal_aggregations(df, step_days=5)
    assert len(block_df) == 6
    # 5 days * 2.0 mm = 10.0 mm
    assert abs(block_df["total_precipitation"].iloc[0] - 10.0) < 1e-5


def test_compute_all_derived_features():
    """Verify full derived features pipeline appends all expected columns."""
    dates = pd.date_range("1950-06-01", periods=10, freq="1D")
    df = pd.DataFrame(
        {
            "county_fips": ["17001"] * 10,
            "date": dates,
            "air_temperature_mean": [20.0] * 10,
            "air_temperature_minimum": [14.0] * 10,
            "air_temperature_maximum": [28.0] * 10,
            "dewpoint_temperature_mean": [12.0] * 10,
            "eastward_wind_mean": [3.0] * 10,
            "northward_wind_mean": [4.0] * 10,
            "surface_pressure": [1013.25] * 10,
            "total_precipitation": [5.0] * 10,
            "surface_solar_radiation_downwards": [20.0] * 10,
        }
    )

    enriched = compute_all_derived_features(df)
    expected_new = [
        "wind_speed",
        "relative_humidity",
        "vapor_pressure_deficit",
        "growing_degree_days",
        "heat_day_30",
        "heat_day_35",
        "et0_fao56",
        "p_minus_et0",
    ]
    for col in expected_new:
        assert col in enriched.columns
        assert not enriched[col].isna().any()
