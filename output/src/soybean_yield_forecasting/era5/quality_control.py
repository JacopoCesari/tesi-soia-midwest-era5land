"""Calendar, physical consistency, and county-day validation without network access."""

from __future__ import annotations

import calendar
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .variables import NORTH, SOUTH


def validate_era5_dataset(
    dataset: xr.Dataset,
    year: int | None = None,
    is_preflight: bool = False,
    *,
    expected_dates: Iterable | None = None,
) -> tuple[str, str]:
    """Check exact daily dates and the physical checks implemented by the pipeline.

    Channel B callers must pass the midnight boundary dates explicitly, or shift
    and crop the data first. A raw annual response includes an extra January 1.
    """
    if "latitude" in dataset.coords and "longitude" in dataset.coords:
        latitudes, longitudes = dataset.latitude.values, dataset.longitude.values
        if not len(latitudes) or not len(longitudes):
            return "FAIL", "Empty spatial grid"
        if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
            return "FAIL", "Non-finite grid coordinates"
        if np.any(np.diff(latitudes) >= 0) or np.any(np.diff(longitudes) <= 0):
            return "FAIL", "Latitude must decrease and longitude must increase"
        if latitudes.min() < SOUTH - 0.2 or latitudes.max() > NORTH + 0.2:
            return "FAIL", "Latitude outside the study domain"
    time_name = "valid_time" if "valid_time" in dataset.coords else "time"
    if time_name not in dataset.coords:
        return "FAIL", "Missing time coordinate"
    dates = pd.DatetimeIndex(dataset[time_name].values)
    if dates.empty or dates.hasnans or dates.has_duplicates:
        return "FAIL", "Empty, missing or duplicate timestamps"
    if not dates.is_monotonic_increasing or not dates.equals(dates.normalize()):
        return "FAIL", "Timestamps must increase at UTC midnight"
    if expected_dates is None and year is not None and not is_preflight:
        count = 366 if calendar.isleap(year) else 365
        if len(dates) != count:
            return "FAIL", f"Day count: {len(dates)} vs {count} expected"
        expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    if expected_dates is not None and not dates.equals(pd.DatetimeIndex(expected_dates)):
        return "FAIL", "Dates do not match the expected calendar"
    if len(dates) > 1 and not (np.diff(dates.values) == np.timedelta64(1, "D")).all():
        return "FAIL", "Unexpected missing dates"
    for name in ["tp", "total_precipitation", "ssrd", "surface_solar_radiation_downwards"]:
        if name in dataset and float(dataset[name].min()) < -1e-4:
            return "FAIL", f"Negative precipitation or solar radiation: {name}"
    for layer in range(1, 5):
        for name in [f"swvl{layer}", f"volumetric_soil_water_layer_{layer}"]:
            if name in dataset:
                if float(dataset[name].min()) < -0.05 or float(dataset[name].max()) > 1.05:
                    return "FAIL", f"Soil water outside the tolerated range: {name}"
    for minimum, mean, maximum in [
        ("t2m_min", "t2m_mean", "t2m_max"),
        ("air_temperature_minimum", "air_temperature_mean", "air_temperature_maximum"),
    ]:
        if minimum in dataset and maximum in dataset:
            if bool((dataset[minimum] > dataset[maximum] + 1e-4).any()):
                return "FAIL", "Thermal inconsistency: minimum > maximum"
            if mean in dataset and (
                bool((dataset[minimum] > dataset[mean] + 1e-4).any())
                or bool((dataset[mean] > dataset[maximum] + 1e-4).any())
            ):
                return "FAIL", "Thermal inconsistency: minimum <= mean <= maximum violated"
    return "PASS", f"Calendar and implemented physical checks passed ({len(dates)} days)"


def validate_county_daily(
    frame: pd.DataFrame,
    expected_counties: int = 479,
    expected_days: int | None = None,
    *,
    year: int | None = None,
    expected_variables: Iterable[str] | None = None,
    county_fips: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Validate complete county-day coverage and available physical checks."""
    if not {"date", "county_fips"}.issubset(frame):
        return "FAIL", "Missing county_fips or date"
    if frame.isna().any().any():
        return "FAIL", "Missing values in county-daily data"
    if frame.duplicated(["county_fips", "date"]).any():
        return "FAIL", "Found duplicate county-date keys"
    if frame.county_fips.nunique() != expected_counties:
        return "FAIL", "Unexpected county count"
    if county_fips is not None and set(frame.county_fips.astype(str)) != set(county_fips):
        return "FAIL", "County identifiers do not match the spatial weights"
    if not frame.groupby("date").county_fips.nunique().eq(expected_counties).all():
        return "FAIL", "Incomplete county coverage on one or more dates"
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(frame.date.unique(), errors="raise")).sort_values()
    except (TypeError, ValueError):
        return "FAIL", "Invalid dates"
    if dates.empty or not dates.equals(dates.normalize()):
        return "FAIL", "Expected nonempty daily dates at midnight"
    if expected_days is not None and len(dates) != expected_days:
        return "FAIL", f"Unexpected day count: {len(dates)} vs {expected_days}"
    calendar_dates = (
        pd.date_range(f"{year}-01-01", f"{year}-12-31") if year else pd.date_range(dates.min(), dates.max())
    )
    if not dates.equals(calendar_dates):
        return "FAIL", "Unexpected missing dates or incorrect year"
    if expected_variables is not None and set(frame) != {"county_fips", "date", *expected_variables}:
        return "FAIL", "Unexpected weather variable schema"
    numeric = frame.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        return "FAIL", "Non-finite weather values"
    # Reuse the physical checks; only the synthetic time axis is needed here.
    dataset = xr.Dataset(
        {name: ("record", values.to_numpy()) for name, values in numeric.items()}, coords={"time": dates}
    )
    status, note = validate_era5_dataset(dataset, is_preflight=True)
    if status != "PASS":
        return status, note
    return "PASS", f"Validated {len(frame)} rows, {expected_counties} counties and {len(dates)} days"


def write_quality_report(path: Path, checks: list[dict]) -> dict:
    """Write a machine-readable report with individual checks and overall status."""
    report = {
        "status": "PASS" if checks and all(check["status"] == "PASS" for check in checks) else "FAIL",
        "checks": checks,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report
