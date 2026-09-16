"""Physical and agronomic transformations, aggregations, and derived meteorological features.

This module implements:
- Daily aggregations (mean, min, max, cumulative sum).
- Physical unit conversions (Kelvin to Celsius, Pa to hPa, m to mm, J/m² to MJ/m²).
- Exact derived meteorological features (wind speed, RH, VPD, GDD, heat days, rolling windows).
- Proxy agronomic indices (FAO-56 Penman-Monteith reference evapotranspiration ET0 and P_minus_ET0).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr


# -----------------------------------------------------------------------------
# Unit Conversions
# -----------------------------------------------------------------------------


def kelvin_to_celsius(values: np.ndarray | float | pd.Series) -> np.ndarray | float | pd.Series:
    """Convert temperature from Kelvin to degrees Celsius."""
    return values - 273.15


def pa_to_hpa(values: np.ndarray | float | pd.Series) -> np.ndarray | float | pd.Series:
    """Convert pressure from Pascals to hectopascals (hPa / mbar)."""
    return values / 100.0


def meters_to_mm(values: np.ndarray | float | pd.Series) -> np.ndarray | float | pd.Series:
    """Convert water depths from meters to millimeters (kg/m²)."""
    return values * 1000.0


def joules_to_megajoules(values: np.ndarray | float | pd.Series) -> np.ndarray | float | pd.Series:
    """Convert energy / radiant flux from Joules/m² to Megajoules/m²."""
    return values / 1_000_000.0


def apply_unit_conversions_arco(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply unit conversions to standard ARCO canonical columns."""
    df = frame.copy()
    temp_cols = [
        c
        for c in df.columns
        if any(marker in c for marker in ["temperature", "t2m", "skt", "dewpoint", "d2m", "stl"])
    ]
    for col in temp_cols:
        if col in df.columns and df[col].dropna().mean() > 100.0:
            df[col] = kelvin_to_celsius(df[col])

    if "surface_pressure" in df.columns and df["surface_pressure"].dropna().mean() > 50_000.0:
        df["surface_pressure"] = pa_to_hpa(df["surface_pressure"])

    water_cols = [c for c in df.columns if c in ["total_precipitation", "tp", "precip"]]
    for col in water_cols:
        # If mean is small (< 1.0 m per day, typically < 0.1 m), convert m to mm
        if col in df.columns and df[col].dropna().abs().max() < 5.0:
            df[col] = meters_to_mm(df[col])

    energy_cols = [c for c in df.columns if any(m in c for m in ["radiation", "ssrd", "strd"])]
    for col in energy_cols:
        if col in df.columns and df[col].dropna().abs().mean() > 10_000.0:
            df[col] = joules_to_megajoules(df[col])

    return df


# -----------------------------------------------------------------------------
# Gridded Daily Aggregation from Hourly ARCO Data
# -----------------------------------------------------------------------------


def aggregate_hourly_dataset_to_daily(
    dataset: xr.Dataset,
    mean_vars: Sequence[str] = (),
    min_vars: Sequence[str] = (),
    max_vars: Sequence[str] = (),
    sum_vars: Sequence[str] = (),
) -> xr.Dataset:
    """Aggregate an hourly gridded xarray Dataset into a daily Dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Hourly dataset with coordinate 'time'
    mean_vars : Sequence[str]
        Variables to aggregate using daily mean.
    min_vars : Sequence[str]
        Variables to aggregate using daily minimum.
    max_vars : Sequence[str]
        Variables to aggregate using daily maximum.
    sum_vars : Sequence[str]
        Variables to aggregate using daily sum (for de-accumulated precipitation & radiation).
    """
    time_coord = "valid_time" if "valid_time" in dataset.coords else "time"

    parts: list[xr.Dataset] = []
    if mean_vars:
        available = [v for v in mean_vars if v in dataset.data_vars]
        if available:
            parts.append(dataset[available].resample({time_coord: "1D"}).mean(dim=time_coord))

    if min_vars:
        available = [v for v in min_vars if v in dataset.data_vars]
        if available:
            min_ds = dataset[available].resample({time_coord: "1D"}).min(dim=time_coord)
            rename_map = {v: f"{v}_min" for v in available if not v.endswith("_min")}
            parts.append(min_ds.rename(rename_map))

    if max_vars:
        available = [v for v in max_vars if v in dataset.data_vars]
        if available:
            max_ds = dataset[available].resample({time_coord: "1D"}).max(dim=time_coord)
            rename_map = {v: f"{v}_max" for v in available if not v.endswith("_max")}
            parts.append(max_ds.rename(rename_map))

    if sum_vars:
        available = [v for v in sum_vars if v in dataset.data_vars]
        if available:
            parts.append(dataset[available].resample({time_coord: "1D"}).sum(dim=time_coord))

    if not parts:
        return dataset

    merged = xr.merge(parts, compat="override")
    # Normalize time coordinate to midnight UTC date
    daily_dates = pd.to_datetime(merged[time_coord].values).normalize()
    merged = merged.assign_coords({time_coord: daily_dates})
    if time_coord != "time":
        merged = merged.rename({time_coord: "time"})
    return merged


# -----------------------------------------------------------------------------
# Exact Derived Features (DERIVED_EXACT)
# -----------------------------------------------------------------------------


def compute_wind_speed(
    u10: np.ndarray | pd.Series | float, v10: np.ndarray | pd.Series | float
) -> np.ndarray | pd.Series | float:
    """Compute 10m wind speed magnitude from eastward (u10) and northward (v10) components.

    Formula:
        wind_speed = sqrt(u10^2 + v10^2)
    Classification: DERIVED_EXACT.
    """
    return np.sqrt(np.square(u10) + np.square(v10))


def compute_saturation_vapor_pressure(temperature_c: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    """Compute saturation vapor pressure e_s (kPa) using the Tetens formula (FAO-56 standard).

    Formula:
        e_s(T) = 0.61078 * exp((17.27 * T) / (T + 237.3))
    """
    return 0.61078 * np.exp((17.27 * temperature_c) / (temperature_c + 237.3))


def compute_relative_humidity(
    temperature_c: np.ndarray | pd.Series, dewpoint_c: np.ndarray | pd.Series
) -> np.ndarray | pd.Series:
    """Compute relative humidity RH (%) from air temperature and dewpoint temperature.

    Formula:
        RH = 100 * (e_a / e_s)
    clipped to [0.0, 100.0].
    Classification: DERIVED_EXACT.
    """
    es = compute_saturation_vapor_pressure(temperature_c)
    ea = compute_saturation_vapor_pressure(dewpoint_c)
    rh = 100.0 * (ea / np.maximum(es, 1e-6))
    if isinstance(rh, pd.Series):
        return rh.clip(lower=0.0, upper=100.0)
    return np.clip(rh, 0.0, 100.0)


def compute_vapor_pressure_deficit(
    temperature_c: np.ndarray | pd.Series, dewpoint_c: np.ndarray | pd.Series
) -> np.ndarray | pd.Series:
    """Compute Vapor Pressure Deficit VPD (kPa) from air temperature and dewpoint.

    Formula:
        VPD = max(0, e_s - e_a)
    Classification: DERIVED_EXACT.
    """
    es = compute_saturation_vapor_pressure(temperature_c)
    ea = compute_saturation_vapor_pressure(dewpoint_c)
    vpd = es - ea
    if isinstance(vpd, pd.Series):
        return vpd.clip(lower=0.0)
    return np.maximum(0.0, vpd)


def compute_growing_degree_days(
    tmax_c: np.ndarray | pd.Series | float,
    tmin_c: np.ndarray | pd.Series | float,
    base_c: float = 10.0,
    cutoff_c: float = 30.0,
) -> np.ndarray | pd.Series | float:
    """Compute Growing Degree Days (GDD) with standard agronomic base and cutoff.

    Formula:
        T_adj_max = min(max(tmax, base), cutoff)
        T_adj_min = min(max(tmin, base), cutoff)
        GDD = max(0.0, (T_adj_max + T_adj_min) / 2.0 - base)
    Classification: DERIVED_EXACT.
    """
    if isinstance(tmax_c, pd.Series):
        adj_max = tmax_c.clip(lower=base_c, upper=cutoff_c)
        adj_min = tmin_c.clip(lower=base_c, upper=cutoff_c)
        gdd = (adj_max + adj_min) / 2.0 - base_c
        return gdd.clip(lower=0.0)
    adj_max = np.clip(tmax_c, base_c, cutoff_c)
    adj_min = np.clip(tmin_c, base_c, cutoff_c)
    gdd = (adj_max + adj_min) / 2.0 - base_c
    return np.maximum(0.0, gdd)


def compute_heat_stress_days(
    tmax_c: np.ndarray | pd.Series, threshold_c: float = 30.0
) -> np.ndarray | pd.Series:
    """Indicator of days exceeding temperature threshold (e.g. 30°C or 35°C).

    Classification: DERIVED_EXACT.
    """
    if isinstance(tmax_c, pd.Series):
        return (tmax_c >= threshold_c).astype(int)
    return (np.asarray(tmax_c) >= threshold_c).astype(int)


def compute_rolling_features(
    series: pd.Series, windows: Sequence[int] = (7, 14, 30, 60), operations: Sequence[str] = ("mean", "sum")
) -> pd.DataFrame:
    """Compute multi-day rolling statistics over a daily time series.

    Classification: DERIVED_EXACT.
    """
    result = pd.DataFrame(index=series.index)
    col_name = series.name or "value"
    for w in windows:
        for op in operations:
            if op == "mean":
                result[f"{col_name}_rolling_{w}d_mean"] = series.rolling(window=w, min_periods=1).mean()
            elif op == "sum":
                result[f"{col_name}_rolling_{w}d_sum"] = series.rolling(window=w, min_periods=1).sum()
    return result


def compute_temporal_aggregations(
    frame: pd.DataFrame, step_days: int = 5, agg_funcs: dict[str, str] | None = None
) -> pd.DataFrame:
    """Aggregate a daily county time series into multi-day blocks (e.g. 5, 10, or 30 days).

    Classification: DERIVED_EXACT.
    """
    df = frame.copy()
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["county_fips", "date"] if "county_fips" in df.columns else "date").reset_index(drop=True)

    group_keys = ["county_fips"] if "county_fips" in df.columns else []

    def _block_group(group: pd.DataFrame) -> pd.DataFrame:
        period_idx = np.arange(len(group)) // step_days
        default_aggs = {
            col: "sum" if any(k in col for k in ["precip", "tp", "ssrd", "strd", "gdd"]) else "mean"
            for col in group.select_dtypes(include=[np.number]).columns
        }
        if agg_funcs:
            default_aggs.update(agg_funcs)
        aggregated = group.groupby(period_idx).agg(default_aggs)
        start_dates = group.groupby(period_idx)["date"].first()
        aggregated.insert(0, "period_start", start_dates)
        if "county_fips" in group.columns:
            aggregated.insert(0, "county_fips", group["county_fips"].iloc[0])
        return aggregated

    if group_keys:
        return df.groupby(group_keys, group_keys=False).apply(_block_group, include_groups=False).reset_index(drop=True)
    return _block_group(df).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Proxy Agronomic Evapotranspiration (DERIVED_PROXY)
# -----------------------------------------------------------------------------


def compute_fao56_et0(
    tmean_c: np.ndarray | pd.Series,
    tmax_c: np.ndarray | pd.Series,
    tmin_c: np.ndarray | pd.Series,
    surface_solar_radiation_mj: np.ndarray | pd.Series,
    wind_speed_10m: np.ndarray | pd.Series,
    dewpoint_c: np.ndarray | pd.Series,
    surface_pressure_hpa: np.ndarray | pd.Series | float = 1013.25,
    latitude_deg: float | np.ndarray = 40.0,
    day_of_year: int | np.ndarray = 180,
) -> np.ndarray | pd.Series:
    """Compute FAO-56 Penman-Monteith reference crop evapotranspiration (ET0) in mm/day.

    Methodology:
        Standardized FAO-56 equation for short grass reference crop:
            ET0 = (0.408 * Delta * (Rn - G) + gamma * (900 / (Tmean + 273)) * u2 * (es - ea))
                  / (Delta + gamma * (1 + 0.34 * u2))
        where:
        - Tmean: daily mean temperature (°C)
        - Rn: net radiation at the crop surface (MJ/m²/day)
        - G: soil heat flux density (MJ/m²/day), G approx 0 for daily time steps
        - u2: wind speed at 2m height (m/s), adjusted from 10m via FAO-56 logarithmic profile:
              u2 = u10 * 4.87 / ln(67.8 * 10 - 5.42) approx u10 * 0.748
        - es: saturation vapor pressure (kPa) = (es(Tmax) + es(Tmin)) / 2
        - ea: actual vapor pressure (kPa) = es(Tdew)
        - Delta: slope of vapor pressure curve (kPa/°C)
        - gamma: psychrometric constant (kPa/°C) = 0.665e-3 * P (kPa)

    Attribution:
        Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).
        Crop evapotranspiration-Guidelines for computing crop water requirements-FAO Irrigation and drainage paper 56.

    Classification: DERIVED_PROXY.
    NOTE: ET0_FAO56 is an agronomic index for reference grass and must NEVER be labeled
    as ECMWF IFS 'potential_evaporation'.
    """
    # 1. Atmospheric pressure P in kPa
    p_kpa = surface_pressure_hpa / 10.0

    # 2. Psychrometric constant gamma (kPa / °C)
    gamma = 0.000665 * p_kpa

    # 3. Vapor pressures
    es_tmax = compute_saturation_vapor_pressure(tmax_c)
    es_tmin = compute_saturation_vapor_pressure(tmin_c)
    es = (es_tmax + es_tmin) / 2.0
    ea = compute_saturation_vapor_pressure(dewpoint_c)

    # 4. Slope of saturation vapor pressure curve Delta (kPa / °C)
    delta = 4098.0 * (0.61078 * np.exp((17.27 * tmean_c) / (tmean_c + 237.3))) / np.square(tmean_c + 237.3)

    # 5. Wind speed adjusted to 2m (FAO-56 eq. 47)
    u2 = np.maximum(0.1, wind_speed_10m * 0.748)

    # 6. Solar & Net Radiation (MJ/m²/day)
    rs = np.maximum(0.0, surface_solar_radiation_mj)
    # Net solar radiation Rns with albedo 0.23 (FAO-56)
    rns = 0.77 * rs

    # Extraterrestrial radiation Ra approximation for net longwave scaling
    phi = np.radians(latitude_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * math.pi * day_of_year / 365.0)
    delta_solar = 0.409 * np.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)
    # sunset hour angle ws
    tan_prod = -np.tan(phi) * np.tan(delta_solar)
    ws = np.arccos(np.clip(tan_prod, -1.0, 1.0))
    gsc = 0.0820  # MJ / m² / min solar constant
    ra = (24.0 * 60.0 / math.pi) * gsc * dr * (
        ws * np.sin(phi) * np.sin(delta_solar) + np.cos(phi) * np.cos(delta_solar) * np.sin(ws)
    )
    rso = np.maximum(0.5, 0.75 * ra)
    rs_rso = np.clip(rs / rso, 0.3, 1.0)

    # Net longwave radiation Rnl (FAO-56 eq. 39)
    sigma = 4.903e-9  # Stefan-Boltzmann constant MJ / K^4 / m² / day
    tmax_k = tmax_c + 273.15
    tmin_k = tmin_c + 273.15
    rnl = (
        sigma
        * ((np.power(tmax_k, 4) + np.power(tmin_k, 4)) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.maximum(0.0, ea)))
        * (1.35 * rs_rso - 0.35)
    )
    rn = rns - rnl

    # 7. Penman-Monteith ET0 calculation
    numerator = 0.408 * delta * rn + gamma * (900.0 / (tmean_c + 273.15)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    et0 = numerator / denominator

    if isinstance(et0, pd.Series):
        return et0.clip(lower=0.0)
    return np.maximum(0.0, et0)


def compute_climatic_water_balance(
    total_precipitation_mm: np.ndarray | pd.Series, et0_mm: np.ndarray | pd.Series
) -> np.ndarray | pd.Series:
    """Compute Climatic Water Balance (P - ET0) in mm/day.

    Classification: DERIVED_PROXY.
    """
    return total_precipitation_mm - et0_mm


def compute_all_derived_features(county_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full set of derived meteorological and agronomic features.

    Appends:
    - wind_speed [DERIVED_EXACT]
    - relative_humidity [DERIVED_EXACT]
    - vapor_pressure_deficit [DERIVED_EXACT]
    - growing_degree_days [DERIVED_EXACT]
    - heat_day_30, heat_day_35 [DERIVED_EXACT]
    - et0_fao56 [DERIVED_PROXY]
    - p_minus_et0 [DERIVED_PROXY]
    """
    df = county_daily_df.copy()

    # Ensure required base columns exist
    tmean = df["air_temperature_mean"] if "air_temperature_mean" in df.columns else df.get("t2m_mean")
    tmin = df["air_temperature_minimum"] if "air_temperature_minimum" in df.columns else df.get("t2m_min", tmean)
    tmax = df["air_temperature_maximum"] if "air_temperature_maximum" in df.columns else df.get("t2m_max", tmean)
    tdew = df["dewpoint_temperature_mean"] if "dewpoint_temperature_mean" in df.columns else df.get("d2m", tmean)
    u10 = df["eastward_wind_mean"] if "eastward_wind_mean" in df.columns else df.get("u10", pd.Series(0.0, index=df.index))
    v10 = df["northward_wind_mean"] if "northward_wind_mean" in df.columns else df.get("v10", pd.Series(0.0, index=df.index))
    sp = df["surface_pressure"] if "surface_pressure" in df.columns else df.get("sp", pd.Series(1013.25, index=df.index))
    tp = df["total_precipitation"] if "total_precipitation" in df.columns else df.get("tp", pd.Series(0.0, index=df.index))
    ssrd = (
        df["surface_solar_radiation_downwards"]
        if "surface_solar_radiation_downwards" in df.columns
        else df.get("ssrd", pd.Series(15.0, index=df.index))
    )

    if tmean is not None:
        # Wind speed
        df["wind_speed"] = compute_wind_speed(u10, v10)

        # Humidity & VPD
        df["relative_humidity"] = compute_relative_humidity(tmean, tdew)
        df["vapor_pressure_deficit"] = compute_vapor_pressure_deficit(tmean, tdew)

        # GDD & Heat days
        df["growing_degree_days"] = compute_growing_degree_days(tmax, tmin, base_c=10.0, cutoff_c=30.0)
        df["heat_day_30"] = compute_heat_stress_days(tmax, 30.0)
        df["heat_day_35"] = compute_heat_stress_days(tmax, 35.0)

        # Day of year from date if available
        doy = 180
        if "date" in df.columns:
            doy = pd.to_datetime(df["date"]).dt.dayofyear.values

        # ET0 FAO-56 & P - ET0
        df["et0_fao56"] = compute_fao56_et0(
            tmean_c=tmean,
            tmax_c=tmax,
            tmin_c=tmin,
            surface_solar_radiation_mj=ssrd,
            wind_speed_10m=df["wind_speed"],
            dewpoint_c=tdew,
            surface_pressure_hpa=sp,
            latitude_deg=40.0,
            day_of_year=doy,
        )
        df["p_minus_et0"] = compute_climatic_water_balance(tp, df["et0_fao56"])

    return df
