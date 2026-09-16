"""Area-weighted daily aggregation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .conversions import apply_evaporation_swap, apply_unit_conversions
from .file_io import open_dataset_safe
from .quality_control import validate_county_daily, validate_era5_dataset, write_quality_report
from .schema import WEATHER_VARIABLES, canonicalize_weather
from .spatial_weights import load_spatial_weights


class CountyAggregator:
    """Compute area-weighted daily weather for the supplied counties."""

    def __init__(self, weights_parquet: Path, output_directory: Path, expected_counties: int = 479) -> None:
        self.weights_frame = load_spatial_weights(weights_parquet, expected_counties)
        self.expected_counties = expected_counties
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.weights_frame["latitude"] = np.round(self.weights_frame["latitude"].values, 2)
        self.weights_frame["longitude"] = np.round(self.weights_frame["longitude"].values, 2)

    def process_year(
        self,
        year: int,
        stat_files: dict[str, Path],
        accum_file: Path,
        *,
        is_preflight: bool = False,
        allow_supplied_sample: bool = False,
    ) -> Path:
        """Aggregate both channels with the existing area weights and convert units."""
        out_parquet = self.output_directory / f"county_daily_{year}.parquet"
        if out_parquet.exists():
            raise FileExistsError(f"Output already exists: {out_parquet}")
        logging.info("Aggregating county-daily channels A and B for year %d...", year)
        mean_dataset = open_dataset_safe(stat_files["mean"])
        minimum_dataset = open_dataset_safe(stat_files["min"])
        maximum_dataset = open_dataset_safe(stat_files["max"])
        if "t2m" in mean_dataset.data_vars:
            mean_dataset = mean_dataset.rename({"t2m": "t2m_mean"})
        if "t2m" in minimum_dataset.data_vars:
            minimum_dataset = minimum_dataset.rename({"t2m": "t2m_min"})
        if "t2m" in maximum_dataset.data_vars:
            maximum_dataset = maximum_dataset.rename({"t2m": "t2m_max"})
        if "skt" in maximum_dataset.data_vars:
            maximum_dataset = maximum_dataset.rename({"skt": "skt_max"})
        # A mismatch must not be silently relabelled by xarray's override join.
        for dataset in [mean_dataset, minimum_dataset, maximum_dataset]:
            status, note = validate_era5_dataset(dataset, year, is_preflight)
            if status != "PASS":
                raise ValueError(note)
        channel_a = xr.merge(
            [mean_dataset, minimum_dataset, maximum_dataset], compat="no_conflicts", join="exact"
        )
        time_coordinate_a = "valid_time" if "valid_time" in channel_a.coords else "time"
        channel_a = channel_a.assign_coords(
            {time_coordinate_a: pd.to_datetime(channel_a[time_coordinate_a].values).normalize()}
        ).rename({time_coordinate_a: "time"})
        channel_a = channel_a.assign_coords(
            {
                "latitude": np.round(channel_a["latitude"].values, 2),
                "longitude": np.round(channel_a["longitude"].values, 2),
            }
        )
        channel_b = open_dataset_safe(accum_file)
        status, note = validate_era5_dataset(channel_b, is_preflight=True)
        if status != "PASS":
            raise ValueError(note)
        channel_b = apply_evaporation_swap(channel_b)
        time_coordinate_b = "valid_time" if "valid_time" in channel_b.coords else "time"
        shifted_dates = pd.to_datetime(channel_b[time_coordinate_b].values).normalize() - pd.Timedelta(days=1)
        channel_b = channel_b.assign_coords({time_coordinate_b: shifted_dates}).rename(
            {time_coordinate_b: "time"}
        )
        channel_b = channel_b.assign_coords(
            {
                "latitude": np.round(channel_b["latitude"].values, 2),
                "longitude": np.round(channel_b["longitude"].values, 2),
            }
        )
        common_dates = np.intersect1d(channel_a.time.values, channel_b.time.values)
        if len(common_dates) == 0:
            raise ValueError(f"No common dates between channels A and B for year {year}")
        if not allow_supplied_sample and not np.array_equal(common_dates, channel_a.time.values):
            raise ValueError("Channel B does not cover every channel A date")
        aligned_channel_a = channel_a.sel(time=common_dates)
        aligned_channel_b = channel_b.sel(time=common_dates)
        combined_dataset = xr.merge([aligned_channel_a, aligned_channel_b], compat="override", join="inner")
        status, note = validate_era5_dataset(combined_dataset, year, is_preflight)
        if status != "PASS":
            raise ValueError(note)
        logging.info("Computing weighted county means for %d dates...", len(common_dates))
        year_frame = self.aggregate_gridded_dataset(combined_dataset)
        year_frame = apply_unit_conversions(year_frame)
        year_frame = canonicalize_weather(year_frame)
        qc_status, quality_message = validate_county_daily(
            year_frame,
            expected_counties=self.expected_counties,
            expected_days=len(common_dates),
            year=None if is_preflight else year,
            expected_variables=None if allow_supplied_sample else WEATHER_VARIABLES,
            county_fips=self.weights_frame.county_fips.unique(),
        )
        write_quality_report(
            out_parquet.with_suffix(".qc.json"),
            [
                {
                    "check": "county_daily",
                    "year": year,
                    "status": qc_status,
                    "notes": quality_message,
                    "rows": len(year_frame),
                    "weather_variables": len(year_frame.columns) - 2,
                    "supplied_sample": allow_supplied_sample,
                }
            ],
        )
        if qc_status != "PASS":
            raise ValueError(quality_message)
        else:
            logging.info("QC County-Daily: %s", quality_message)

        part_parquet = out_parquet.with_suffix(".parquet.part")
        year_frame.to_parquet(part_parquet, index=False)
        if out_parquet.exists():
            out_parquet.unlink()
        part_parquet.replace(out_parquet)
        logging.info(
            "Saved county-daily year %d: %s (%.2f MB, %d rows)",
            year,
            out_parquet.name,
            out_parquet.stat().st_size / 1000000.0,
            len(year_frame),
        )
        return out_parquet

    def aggregate_gridded_dataset(self, dataset: xr.Dataset) -> pd.DataFrame:
        """Compute area-weighted county means from any gridded daily dataset."""
        time_coord = "valid_time" if "valid_time" in dataset.coords else "time"
        ds = dataset.copy()
        if time_coord in ds.coords:
            normalized_dates = pd.to_datetime(ds[time_coord].values).normalize()
            ds = ds.assign_coords({time_coord: normalized_dates})
            if time_coord != "time":
                ds = ds.rename({time_coord: "time"})

        ds = ds.assign_coords(
            {
                "latitude": np.round(ds["latitude"].values, 2),
                "longitude": np.round(ds["longitude"].values, 2),
            }
        )

        county_frames = []
        for fips, group in self.weights_frame.groupby("county_fips"):
            county_latitudes = xr.DataArray(group["latitude"].values, dims="points")
            county_longitudes = xr.DataArray(group["longitude"].values, dims="points")
            point_weights = xr.DataArray(group["weight"].values, dims="points")
            points = ds.sel(latitude=county_latitudes, longitude=county_longitudes)
            county_aggregate = (
                (points * point_weights).sum(dim="points", skipna=False).to_dataframe().reset_index()
            )
            county_aggregate["county_fips"] = fips
            county_frames.append(county_aggregate)

        year_frame = pd.concat(county_frames).reset_index(drop=True)
        if "time" in year_frame.columns:
            year_frame.rename(columns={"time": "date"}, inplace=True)

        columns_to_drop = [
            column
            for column in ["points", "latitude", "longitude", "number", "expver"]
            if column in year_frame.columns
        ]
        if columns_to_drop:
            year_frame.drop(columns=columns_to_drop, inplace=True)

        return year_frame
