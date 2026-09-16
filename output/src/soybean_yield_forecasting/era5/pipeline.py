"""Explicit orchestration of downloads, raw validation and county aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .conversions import apply_evaporation_swap
from .county_aggregation import CountyAggregator
from .download import CDSDownloader
from .file_io import open_dataset_safe
from .manifest import ManifestManager
from .quality_control import validate_county_daily, validate_era5_dataset
from .schema import WEATHER_VARIABLES, canonicalize_weather
from .spatial_weights import download_area, load_spatial_weights
from .variables import DAILY_MAXIMUM_VARIABLES, DAILY_MEAN_VARIABLES, DAILY_MINIMUM_VARIABLES


def year_files(raw_directory: Path, year: int) -> tuple[dict[str, Path], Path]:
    """Locate the three daily-statistic files and midnight accumulation file."""
    statistics = {
        label: raw_directory / "raw_daily_stats" / f"era5_land_{year}_{statistic}.nc"
        for label, statistic in [("mean", "daily_mean"), ("min", "daily_minimum"), ("max", "daily_maximum")]
    }
    accumulated = raw_directory / "raw_accumulated_boundaries" / f"era5_land_{year}_accumulated_00utc.nc"
    return statistics, accumulated


def validate_raw_file(path: Path, year: int, subgroup: str, *, preflight: bool = False) -> tuple[str, str]:
    """Check each request's full calendar and variable inventory before recording PASS."""
    dataset = open_dataset_safe(path)
    if preflight:
        start, end = ("06-02", "06-08") if subgroup == "accumulated" else ("06-01", "06-07")
        dates = pd.date_range(f"{year}-{start}", f"{year}-{end}")
    elif subgroup == "accumulated":
        dates = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01")
    else:
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    status, note = validate_era5_dataset(dataset, expected_dates=dates)
    if status != "PASS":
        return status, note
    if subgroup == "accumulated":
        dataset = apply_evaporation_swap(dataset)
        expected = set(WEATHER_VARIABLES[20:])
    else:
        rename = {
            "t2m": {"daily_mean": "t2m_mean", "daily_minimum": "t2m_min", "daily_maximum": "t2m_max"}[
                subgroup
            ]
        }
        if subgroup == "daily_maximum":
            rename["skt"] = "skt_max"
        dataset = dataset.rename({old: new for old, new in rename.items() if old in dataset})
        if subgroup == "daily_mean":
            expected = set(WEATHER_VARIABLES[:20]) - {
                "air_temperature_minimum",
                "air_temperature_maximum",
                "skin_temperature_maximum",
            }
        elif subgroup == "daily_minimum":
            expected = {"air_temperature_minimum"}
        else:
            expected = {"air_temperature_maximum", "skin_temperature_maximum"}
    actual = set(canonicalize_weather(pd.DataFrame(columns=list(dataset.data_vars))).columns)
    if actual != expected:
        return (
            "FAIL",
            f"Unexpected {subgroup} variables; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
        )
    return "PASS", note


def run_year(
    year: int,
    raw_directory: Path,
    output_directory: Path,
    weights: Path,
    *,
    aggregate_only: bool = False,
    verify_checksum: bool = False,
    preflight: bool = False,
    expected_counties: int = 479,
) -> Path:
    """Run one explicitly selected year, recording raw checks and refusing invalid output."""
    manifest = ManifestManager(raw_directory / "manifest.csv")
    statistics, accumulated = year_files(raw_directory, year)
    spatial_weights = load_spatial_weights(weights, expected_counties)
    output = output_directory / f"county_daily_{year}.parquet"
    # A completed, validated year needs no raw NetCDF decompression/reaggregation.
    # Check every source record (and SHA-256 when requested) before taking this path.
    if output.exists() and all(
        manifest.is_done(path, verify_checksum=verify_checksum)
        for path in [*statistics.values(), accumulated]
    ):
        status, note = validate_county_daily(
            pd.read_parquet(output),
            expected_counties,
            expected_days=7 if preflight else None,
            year=None if preflight else year,
            expected_variables=WEATHER_VARIABLES,
            county_fips=spatial_weights.county_fips.unique(),
        )
        if status != "PASS":
            raise ValueError(f"Existing output is invalid: {note}")
        return output
    groups = [
        ("mean", "daily_mean", DAILY_MEAN_VARIABLES),
        ("min", "daily_minimum", DAILY_MINIMUM_VARIABLES),
        ("max", "daily_maximum", DAILY_MAXIMUM_VARIABLES),
    ]
    downloader = (
        None
        if aggregate_only
        else CDSDownloader(raw_directory, manifest, verify_checksum, area=download_area(spatial_weights))
    )
    for label, statistic, variables in groups:
        if downloader:
            statistics[label] = downloader.download_daily_stat_group(
                year,
                statistic,
                variables,
                test_days=[f"{day:02d}" for day in range(1, 8)] if preflight else None,
            )
        status, note = validate_raw_file(statistics[label], year, statistic, preflight=preflight)
        if not aggregate_only:
            manifest.record(year, "A", statistic, statistics[label], status, note)
        if status != "PASS":
            raise ValueError(note)
    if downloader:
        accumulated = downloader.download_accumulated_group(
            year, test_days=[f"{day:02d}" for day in range(2, 9)] if preflight else None
        )
    status, note = validate_raw_file(accumulated, year, "accumulated", preflight=preflight)
    if not aggregate_only:
        manifest.record(year, "B", "accumulated", accumulated, status, note)
    if status != "PASS":
        raise ValueError(note)
    if verify_checksum:
        for path in [*statistics.values(), accumulated]:
            if not manifest.is_done(path, verify_checksum=True):
                raise ValueError(f"Missing or invalid checksum record: {path}")
    if output.exists():
        frame = pd.read_parquet(output)
        status, note = validate_county_daily(
            frame,
            expected_counties,
            expected_days=7 if preflight else None,
            year=None if preflight else year,
            expected_variables=WEATHER_VARIABLES,
            county_fips=spatial_weights.county_fips.unique(),
        )
        if status != "PASS":
            raise ValueError(f"Existing output is invalid: {note}")
        return output
    return CountyAggregator(weights, output_directory, expected_counties).process_year(
        year, statistics, accumulated, is_preflight=preflight
    )
