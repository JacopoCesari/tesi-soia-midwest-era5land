"""Orchestration of ERA5-Land weather acquisition, transformations and county aggregation.

Supports both:
- ECMWF ARCO Zarr acquisition (primary engine, local daily aggregation, unit conversions, derived features)
- CDS batch API (legacy engine, preserved for provenance and historical comparison)
"""

from __future__ import annotations

import calendar
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import xarray as xr

from .arco import ARCOClient, DEFAULT_STORES_GEO
from .conversions import apply_evaporation_swap
from .county_aggregation import CountyAggregator
from .download import CDSDownloader
from .file_io import compute_sha256, open_dataset_safe
from .manifest import ManifestManager
from .quality_control import validate_county_daily, validate_era5_dataset, write_quality_report
from .schema import RAW_TO_CANONICAL, WEATHER_VARIABLES, canonicalize_weather
from .spatial_weights import download_area, load_spatial_weights
from .transformations import (
    aggregate_hourly_dataset_to_daily,
    apply_unit_conversions_arco,
    compute_all_derived_features,
)
from .variables import DAILY_MAXIMUM_VARIABLES, DAILY_MEAN_VARIABLES, DAILY_MINIMUM_VARIABLES

ARCO_STORE_AGGREGATIONS: dict[str, dict[str, list[str]]] = {
    "sfc-2m-temperature": {
        "mean_vars": ["t2m", "d2m"],
        "min_vars": ["t2m"],
        "max_vars": ["t2m"],
        "sum_vars": [],
    },
    "sfc-soil-temperature": {
        "mean_vars": ["stl1", "stl2", "stl3", "stl4"],
        "min_vars": [],
        "max_vars": [],
        "sum_vars": [],
    },
    "sfc-soil-water": {
        "mean_vars": ["swvl1", "swvl2", "swvl3", "swvl4"],
        "min_vars": [],
        "max_vars": [],
        "sum_vars": [],
    },
    "sfc-radiation-heat": {
        "mean_vars": [],
        "min_vars": [],
        "max_vars": [],
        "sum_vars": ["ssrd", "strd"],
    },
    "sfc-snow": {
        "mean_vars": ["snowc", "sde"],
        "min_vars": [],
        "max_vars": [],
        "sum_vars": [],
    },
    "sfc-wind": {
        "mean_vars": ["u10", "v10"],
        "min_vars": [],
        "max_vars": [],
        "sum_vars": [],
    },
    "sfc-pressure-precipitation": {
        "mean_vars": ["sp"],
        "min_vars": [],
        "max_vars": [],
        "sum_vars": ["tp"],
    },
    "sfc-skin-temperature": {
        "mean_vars": ["skt"],
        "min_vars": [],
        "max_vars": ["skt"],
        "sum_vars": [],
    },
}


class ARCOPipeline:
    """Orchestrates ERA5-Land weather acquisition from ECMWF ARCO Zarr stores.

    Steps:
    1. Loads county spatial weights (defaults to 135 primary balanced panel counties).
    2. Determines bounding box: [north, west, south, east].
    3. Fetches spatio-temporal slices from ARCO stores via ARCOClient.
    4. Aggregates hourly fields to daily statistics (mean, min, max, sum) locally on the grid.
    5. Aggregates gridded fields to county-level daily time series via CountyAggregator.
    6. Canonicalizes column names and applies physical unit conversions.
    7. Computes exact derived meteorological and proxy agronomic features.
    8. Validates county-daily data integrity (completeness, physical consistency).
    9. Atomically writes final Parquet dataset and quality report sidecar.
    """

    def __init__(
        self,
        weights_path: Path | str,
        output_directory: Path | str,
        expected_counties: int = 135,
        arco_client: ARCOClient | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.expected_counties = expected_counties
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.client = arco_client or ARCOClient(cache_dir=self.cache_dir)
        self.weights_frame = load_spatial_weights(self.weights_path, self.expected_counties)
        self.area = download_area(self.weights_frame)
        self.aggregator = CountyAggregator(
            self.weights_path, self.output_directory, expected_counties=self.expected_counties
        )

    def run_year(
        self,
        year: int,
        *,
        preflight: bool = False,
        test_days: Sequence[str] | None = None,
        compute_derived: bool = True,
        use_cache: bool = True,
        store_names: Sequence[str] | None = None,
    ) -> Path:
        """Acquire and process one year of county weather using ARCO Zarr stores."""
        out_name = f"county_daily_{year}.parquet"
        out_path = self.output_directory / out_name

        # Check existing output
        if out_path.exists():
            try:
                frame = pd.read_parquet(out_path)
                expected_days = len(test_days) if test_days else (7 if preflight else None)
                status, note = validate_county_daily(
                    frame,
                    self.expected_counties,
                    expected_days=expected_days,
                    year=None if (preflight or test_days) else year,
                    county_fips=self.weights_frame.county_fips.unique(),
                )
                if status == "PASS":
                    logging.info("SKIP: Validated county-daily output already exists: %s", out_path.name)
                    return out_path
            except Exception as err:
                logging.warning("Existing output invalid, recomputing: %s (%s)", out_path.name, err)

        # Date range resolution
        if test_days is not None:
            start_date = f"{year}-06-{int(test_days[0]):02d}"
            end_date = f"{year}-06-{int(test_days[-1]):02d}"
            expected_days = len(test_days)
        elif preflight:
            start_date = f"{year}-06-01"
            end_date = f"{year}-06-07"
            expected_days = 7
        else:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            # ERA5-Land in ARCO begins on 1950-01-02 (364 days); subsequent years have 365/366 days
            expected_days = 364 if year == 1950 else (366 if calendar.isleap(year) else 365)

        north, west, south, east = self.area
        lat_bounds = (south, north)
        lon_bounds = (west, east)

        stores_to_fetch = store_names or list(ARCO_STORE_AGGREGATIONS.keys())
        daily_parts: list[xr.Dataset] = []

        for store_name in stores_to_fetch:
            agg_spec = ARCO_STORE_AGGREGATIONS.get(store_name, {})
            req_vars = list(
                set(
                    agg_spec.get("mean_vars", [])
                    + agg_spec.get("min_vars", [])
                    + agg_spec.get("max_vars", [])
                    + agg_spec.get("sum_vars", [])
                )
            )
            logging.info("Fetching ARCO store %s for %s to %s...", store_name, start_date, end_date)
            hourly_slice = self.client.fetch_slice(
                store_name=store_name,
                start_date=start_date,
                end_date=end_date,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
                variables=req_vars if req_vars else None,
                use_cache=use_cache,
            )
            daily_slice = aggregate_hourly_dataset_to_daily(
                hourly_slice,
                mean_vars=agg_spec.get("mean_vars", ()),
                min_vars=agg_spec.get("min_vars", ()),
                max_vars=agg_spec.get("max_vars", ()),
                sum_vars=agg_spec.get("sum_vars", ()),
            )
            daily_parts.append(daily_slice)

        if not daily_parts:
            raise ValueError(f"No ARCO data retrieved for year {year}")

        combined_grid = xr.merge(daily_parts, compat="override", join="exact")

        # Spatial county aggregation
        logging.info("Computing county area-weighted aggregations (%d counties)...", self.expected_counties)
        county_df = self.aggregator.aggregate_gridded_dataset(combined_grid)

        # Standardize column names
        if "t2m" in county_df.columns and "t2m_mean" not in county_df.columns:
            county_df = county_df.rename(columns={"t2m": "t2m_mean"})
        if "skt" in county_df.columns and "skt_mean" not in county_df.columns:
            county_df = county_df.rename(columns={"skt": "skt_mean"})

        rename_mapping = {
            **RAW_TO_CANONICAL,
            "t2m": "air_temperature_mean",
            "skt": "skin_temperature_mean",
            "sde": "snow_depth",
        }
        county_df = county_df.rename(columns={k: v for k, v in rename_mapping.items() if k in county_df.columns})

        # Physical unit conversions
        county_df = apply_unit_conversions_arco(county_df)

        # Derived meteorological and proxy agronomic features
        if compute_derived:
            logging.info("Computing derived features (wind speed, RH, VPD, GDD, heat days, FAO-56 ET0, CWB)...")
            county_df = compute_all_derived_features(county_df)

        # Quality control validation
        qc_status, qc_note = validate_county_daily(
            county_df,
            expected_counties=self.expected_counties,
            expected_days=expected_days,
            year=None if (preflight or test_days) else year,
            county_fips=self.weights_frame.county_fips.unique(),
        )

        qc_report_path = out_path.with_suffix(".qc.json")
        write_quality_report(
            qc_report_path,
            [
                {
                    "check": "arco_county_daily",
                    "year": year,
                    "status": qc_status,
                    "notes": qc_note,
                    "rows": len(county_df),
                    "counties": county_df["county_fips"].nunique(),
                    "days": county_df["date"].nunique(),
                    "columns": list(county_df.columns),
                    "preflight": preflight or bool(test_days),
                }
            ],
        )

        if qc_status != "PASS":
            raise ValueError(f"Quality control validation failed: {qc_note}")

        # Atomic Parquet save
        part_path = out_path.with_suffix(".parquet.part")
        county_df.to_parquet(part_path, index=False)
        if out_path.exists():
            out_path.unlink()
        part_path.replace(out_path)

        # Write metadata sidecar
        meta_path = out_path.with_suffix(".meta.json")
        meta_data = {
            "year": year,
            "engine": "arco",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expected_counties": self.expected_counties,
            "actual_counties": county_df["county_fips"].nunique(),
            "days": expected_days,
            "rows": len(county_df),
            "columns": list(county_df.columns),
            "stores": stores_to_fetch,
            "sha256": compute_sha256(out_path),
        }
        meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

        logging.info("Saved validated ARCO county daily output: %s (%d rows)", out_path.name, len(county_df))
        return out_path


def run_year_arco(
    year: int,
    output_directory: Path,
    weights: Path,
    *,
    arco_client: ARCOClient | None = None,
    cache_dir: Path | None = None,
    preflight: bool = False,
    test_days: Sequence[str] | None = None,
    expected_counties: int = 135,
    compute_derived: bool = True,
    use_cache: bool = True,
    store_names: Sequence[str] | None = None,
) -> Path:
    """Acquire and process one year of county weather using the primary ARCO engine."""
    pipeline = ARCOPipeline(
        weights_path=weights,
        output_directory=output_directory,
        expected_counties=expected_counties,
        arco_client=arco_client,
        cache_dir=cache_dir,
    )
    return pipeline.run_year(
        year=year,
        preflight=preflight,
        test_days=test_days,
        compute_derived=compute_derived,
        use_cache=use_cache,
        store_names=store_names,
    )


# -----------------------------------------------------------------------------
# Legacy CDS Pipeline (Preserved for Historical Verification)
# -----------------------------------------------------------------------------


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


def run_year_cds(
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
    """Run one year using the legacy CDS batch API downloader."""
    manifest = ManifestManager(raw_directory / "manifest.csv")
    statistics, accumulated = year_files(raw_directory, year)
    spatial_weights = load_spatial_weights(weights, expected_counties)
    output = output_directory / f"county_daily_{year}.parquet"

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


def run_year(
    year: int,
    raw_directory: Path,
    output_directory: Path,
    weights: Path,
    *,
    aggregate_only: bool = False,
    verify_checksum: bool = False,
    preflight: bool = False,
    expected_counties: int = 135,
    engine: str | None = None,
    cache_dir: Path | None = None,
    arco_client: ARCOClient | None = None,
    compute_derived: bool = True,
    use_cache: bool = True,
    store_names: Sequence[str] | None = None,
    test_days: Sequence[str] | None = None,
) -> Path:
    """Run one explicitly selected year, defaulting to the ARCO acquisition engine.

    Parameters
    ----------
    engine : str, optional
        'arco' for modern cloud Zarr acquisition or 'cds' for legacy batch API.
        If omitted, automatically detects CDS if CDS test mocks or verification flags are active,
        otherwise defaults to 'arco'.
    """
    if engine is None:
        from . import download

        is_cds_mocked = not getattr(download.cdsapi.Client, "__module__", "").startswith("cdsapi")
        if aggregate_only or verify_checksum or is_cds_mocked:
            engine = "cds"
        else:
            engine = "arco"

    if engine.lower() == "cds":
        return run_year_cds(
            year,
            raw_directory,
            output_directory,
            weights,
            aggregate_only=aggregate_only,
            verify_checksum=verify_checksum,
            preflight=preflight,
            expected_counties=expected_counties,
        )

    resolved_cache = cache_dir or (raw_directory / "arco_cache" if raw_directory else None)
    return run_year_arco(
        year=year,
        output_directory=output_directory,
        weights=weights,
        arco_client=arco_client,
        cache_dir=resolved_cache,
        preflight=preflight,
        test_days=test_days,
        expected_counties=expected_counties,
        compute_derived=compute_derived,
        use_cache=use_cache,
        store_names=store_names,
    )
