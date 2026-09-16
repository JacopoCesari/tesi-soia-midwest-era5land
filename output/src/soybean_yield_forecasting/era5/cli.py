"""Command-line entry points for data preparation and offline validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import xarray as xr

from ..configuration import load_configuration
from .arco import ARCOClient
from .file_io import open_dataset_safe
from .manifest import ManifestManager
from .pipeline import ARCOPipeline, run_year, run_year_arco, run_year_cds, year_files
from .quality_control import validate_county_daily, validate_era5_dataset, write_quality_report
from .schema import WEATHER_VARIABLES, canonicalize_weather
from .spatial_weights import compute_spatial_weights, load_spatial_weights


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data.yaml"), help="Data YAML configuration"
    )
    return parser


def inventory_main(argv: Sequence[str] | None = None) -> None:
    """Discover and summarize available ECMWF ARCO Zarr stores and variables."""
    parser = _parser("Discover and inspect ECMWF ARCO ERA5-Land Zarr stores")
    parser.add_argument("--save", type=Path, help="Custom path to save JSON inventory")
    arguments = parser.parse_args(argv)
    configuration = load_configuration(arguments.config)
    paths = configuration.get("paths", {})
    save_path = arguments.save or paths.get("arco_inventory")
    client = ARCOClient()
    inv = client.discover_inventory(save_path=save_path)
    print(f"\nDiscovered {len(inv['stores'])} candidate stores:")
    for store_name, meta in inv["stores"].items():
        avail = "AVAILABLE" if meta["available"] else f"UNAVAILABLE (status {meta['status_code']})"
        var_count = len(meta["variables"])
        print(f"  - {store_name} ({meta['bucket']}): {avail} [{var_count} variables]")
        if meta["available"]:
            print(f"    Variables: {', '.join(meta['variables'][:10])}{'...' if var_count > 10 else ''}")
    print(f"\nTotal variables identified: {len(inv['all_variables'])}")
    if save_path:
        print(f"Inventory saved to: {save_path}")


def download_main(argv: Sequence[str] | None = None, *, aggregate_only: bool = False) -> None:
    """Download or aggregate explicitly chosen years; never start a run on empty arguments."""
    parser = _parser("Download and aggregate ERA5-Land county weather")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--test-year", type=int, help="One complete pilot year")
    selection.add_argument(
        "--primary-period",
        action="store_true",
        help="Weather years supporting the primary target (1950-2025)",
    )
    selection.add_argument("--start-year", type=int, help="First explicit year; requires --end-year")
    selection.add_argument(
        "--pilot",
        action="store_true",
        help="Run minimal 7-day preflight pilot (June 1-7, 1950)",
    )
    selection.add_argument(
        "--inventory",
        action="store_true",
        help="Query and print available ARCO Zarr stores and variables",
    )
    parser.add_argument("--end-year", type=int, help="Last explicit year")
    parser.add_argument(
        "--engine",
        choices=["arco", "cds"],
        default="arco",
        help="Acquisition engine: 'arco' (default, cloud Zarr) or 'cds' (legacy batch API)",
    )
    parser.add_argument(
        "--legacy-cds",
        action="store_true",
        help="Shortcut to use legacy CDS batch API engine instead of ARCO",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        default=aggregate_only,
        help="Read downloaded files without creating a remote client",
    )
    parser.add_argument(
        "--verify-checksum", action="store_true", help="Require SHA-256 matches against the manifest"
    )
    parser.add_argument("--raw-dir", type=Path, help="Override raw input/output directory")
    parser.add_argument("--output-dir", type=Path, help="Override county output directory")
    parser.add_argument("--cache-dir", type=Path, help="Override ARCO slice cache directory")
    parser.add_argument("--no-derived", action="store_true", help="Skip derived feature calculations")
    parser.add_argument(
        "--stores",
        nargs="+",
        help="Specific ARCO store names to acquire (defaults to all 8 stores)",
    )
    arguments = parser.parse_args(argv)

    if arguments.inventory:
        inventory_main(argv)
        return

    configuration = load_configuration(arguments.config)
    paths = configuration["paths"]
    is_preflight = False
    test_days = None

    if arguments.pilot:
        years = [1950]
        test_days = [f"{day:02d}" for day in range(1, 8)]
        is_preflight = True
    elif arguments.start_year is not None:
        if arguments.end_year is None or arguments.end_year < arguments.start_year:
            parser.error("--start-year requires --end-year >= --start-year")
        years = range(arguments.start_year, arguments.end_year + 1)
    elif arguments.end_year is not None:
        parser.error("--end-year requires --start-year")
    elif arguments.test_year is not None:
        years = [arguments.test_year]
    else:
        years = range(
            configuration["weather_period"]["start_year"], configuration["weather_period"]["end_year"] + 1
        )

    engine = "cds" if arguments.legacy_cds else arguments.engine
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for year in years:
        raw = arguments.raw_dir or (paths["preflight"] if is_preflight else paths["raw"])
        output = arguments.output_dir or (paths["preflight_output"] if is_preflight else paths["output"])
        try:
            result = run_year(
                year,
                raw,
                output,
                paths["weights"],
                aggregate_only=arguments.aggregate_only,
                verify_checksum=arguments.verify_checksum,
                expected_counties=configuration["expected_counties"],
                engine=engine,
                cache_dir=arguments.cache_dir or paths.get("arco_cache"),
                compute_derived=not arguments.no_derived,
                preflight=is_preflight,
                test_days=test_days,
                store_names=arguments.stores,
            )
            print(result)
        except (ValueError, OSError) as error:
            write_quality_report(
                paths["quality_reports"] / f"pipeline_{year}.json",
                [{"check": "pipeline", "year": year, "status": "FAIL", "notes": str(error)}],
            )
            raise


def weights_main(argv: Sequence[str] | None = None) -> None:
    """Recompute weights into a new directory, preserving the supplied artifacts."""
    parser = _parser("Compute ERA5-Land county area weights")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh output directory")
    arguments = parser.parse_args(argv)
    configuration = load_configuration(arguments.config)
    external = configuration["paths"]["external"]
    outputs = compute_spatial_weights(
        configuration["paths"]["counties"],
        external / "census_counties/cb_2020_us_county_500k.shp",
        arguments.output_dir,
        expected_counties=configuration["expected_counties"],
    )
    for output in outputs:
        print(output)


def _supplied_preflight_checks(configuration: dict, verify_checksum: bool) -> list[dict]:
    paths = configuration["paths"]
    checks = []
    weights = load_spatial_weights(paths["supplied_weights"], configuration["supplied_counties"])
    checks.append(
        {
            "check": "spatial_weights",
            "status": "PASS",
            "rows": len(weights),
            "counties": weights.county_fips.nunique(),
            "maximum_sum_error": float((weights.groupby("county_fips").weight.sum() - 1).abs().max()),
        }
    )
    manifest = ManifestManager(
        paths["historical_manifest"],
        path_map=paths["migration_map"],
        repository_root=paths["repository_root"],
        read_only=True,
    )
    for year in configuration["preflight_years"]:
        statistics, accumulated = year_files(paths["supplied_preflight"], year)
        for path in [*statistics.values(), accumulated]:
            dataset = open_dataset_safe(path)
            status, note = validate_era5_dataset(
                dataset, expected_dates=pd.date_range(f"{year}-06-01", periods=7)
            )
            checks.append(
                {
                    "check": "supplied_netcdf",
                    "file": str(path),
                    "status": status,
                    "notes": note,
                    "variables": list(dataset.data_vars),
                }
            )
            if verify_checksum:
                checks.append(
                    {
                        "check": "historical_checksum",
                        "file": str(path),
                        "status": "PASS" if manifest.is_done(path, True) else "FAIL",
                    }
                )
        mean = open_dataset_safe(statistics["mean"]).rename({"t2m": "t2m_mean"})
        minimum = open_dataset_safe(statistics["min"]).rename({"t2m": "t2m_min"})
        maximum = open_dataset_safe(statistics["max"]).rename({"t2m": "t2m_max"})
        status, note = validate_era5_dataset(
            xr.merge([mean, minimum, maximum], join="exact", compat="no_conflicts"), is_preflight=True
        )
        checks.append(
            {"check": "supplied_thermal_consistency", "year": year, "status": status, "notes": note}
        )
        supplied = paths["output"].parent / "supplied_preflight" / f"county_daily_{year}.parquet"
        frame = canonicalize_weather(pd.read_parquet(supplied))
        status, note = validate_county_daily(
            frame, configuration["supplied_counties"], 6, county_fips=weights.county_fips.unique()
        )
        checks.append(
            {
                "check": "supplied_county_daily",
                "year": year,
                "status": status,
                "notes": note,
                "scope": "Six aligned days, 21 weather fields; not a full-year or 37-field validation",
            }
        )
    return checks


def preflight_main(argv: Sequence[str] | None = None) -> None:
    """Default to offline validation; --download explicitly requests new seven-day files."""
    parser = _parser("Check supplied preflight artifacts or download a new reduced preflight")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download 7 days and all available fields; requires prior authorization",
    )
    parser.add_argument(
        "--engine",
        choices=["arco", "cds"],
        default="arco",
        help="Engine for preflight download: 'arco' (default) or 'cds' (legacy batch API)",
    )
    parser.add_argument(
        "--legacy-cds",
        action="store_true",
        help="Shortcut to use legacy CDS API for preflight download",
    )
    parser.add_argument("--verify-checksum", action="store_true")
    parser.add_argument("--report", type=Path, help="JSON QC report destination")
    arguments = parser.parse_args(argv)
    configuration = load_configuration(arguments.config)
    paths = configuration["paths"]
    checks = []
    try:
        if arguments.download:
            engine = "cds" if arguments.legacy_cds else arguments.engine
            for year in configuration["preflight_years"]:
                output = run_year(
                    year,
                    paths["preflight"],
                    paths["preflight_output"],
                    paths["weights"],
                    verify_checksum=arguments.verify_checksum,
                    preflight=True,
                    expected_counties=configuration["expected_counties"],
                    engine=engine,
                )
                checks.append({"check": "new_preflight", "year": year, "status": "PASS", "file": str(output)})
        else:
            checks = _supplied_preflight_checks(configuration, arguments.verify_checksum)
    except (ValueError, OSError) as error:
        checks.append({"check": "preflight", "status": "FAIL", "notes": str(error)})
    report_path = arguments.report or paths["quality_reports"] / "preflight.json"
    report = write_quality_report(report_path, checks)
    print(f"{report['status']}: {report_path}")
    if report["status"] != "PASS":
        raise SystemExit(1)


def validate_main(argv: Sequence[str] | None = None) -> None:
    """Validate annual county Parquet data and optionally its four raw-file checksums."""
    parser = _parser("Validate county-day output and emit a JSON quality report")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--file", type=Path, help="County Parquet path")
    parser.add_argument("--raw-dir", type=Path, help="Directory containing the operational manifest")
    parser.add_argument("--verify-checksum", action="store_true")
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)
    configuration = load_configuration(arguments.config)
    paths = configuration["paths"]
    checks = []
    try:
        weights = load_spatial_weights(paths["weights"], configuration["expected_counties"])
        checks.append({"check": "spatial_weights", "status": "PASS"})
        path = arguments.file or paths["output"] / f"county_daily_{arguments.year}.parquet"
        status, note = validate_county_daily(
            pd.read_parquet(path),
            configuration["expected_counties"],
            year=arguments.year,
            expected_variables=WEATHER_VARIABLES,
            county_fips=weights.county_fips.unique(),
        )
        checks.append({"check": "annual_county_daily", "status": status, "notes": note, "file": str(path)})
        if arguments.verify_checksum:
            raw = arguments.raw_dir or paths["raw"]
            manifest = ManifestManager(raw / "manifest.csv", read_only=True)
            statistics, accumulated = year_files(raw, arguments.year)
            for path in [*statistics.values(), accumulated]:
                checks.append(
                    {
                        "check": "raw_checksum",
                        "file": str(path),
                        "status": "PASS" if manifest.is_done(path, True) else "FAIL",
                    }
                )
    except (ValueError, OSError) as error:
        checks.append({"check": "validation", "status": "FAIL", "notes": str(error)})
    report_path = arguments.report or paths["quality_reports"] / f"county_daily_{arguments.year}.json"
    report = write_quality_report(report_path, checks)
    print(f"{report['status']}: {report_path}")
    if report["status"] != "PASS":
        raise SystemExit(1)
