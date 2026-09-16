"""CDS requests and response handling."""

from __future__ import annotations

import calendar
import logging
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Any, Sequence

warnings.warn(
    "CDSDownloader is deprecated and preserved for historical verification and legacy comparison only. "
    "ECMWF ARCO Zarr (ARCOClient and ARCOPipeline) is the primary acquisition engine.",
    DeprecationWarning,
    stacklevel=2,
)

import cdsapi
import pandas as pd
import requests
import xarray as xr

from .file_io import get_safe_path, open_dataset_safe
from .manifest import ManifestManager
from .quality_control import validate_era5_dataset
from .variables import ACCUMULATED_VARIABLES, CDS_AREA, DATASET_DAILY_STATS, DATASET_HOURLY


class CDSDownloader:
    """Retrieve ERA5-Land files with resumable manifest checks."""

    def __init__(
        self,
        base_directory: Path,
        manifest: ManifestManager,
        verify_checksum: bool = False,
        *,
        area: Sequence[float] = CDS_AREA,
        accumulated_variables: Sequence[str] = ACCUMULATED_VARIABLES,
    ) -> None:
        self.base_directory = base_directory
        self.manifest = manifest
        self.verify_checksum = verify_checksum
        self._client = None
        self.area = list(area)
        self.accumulated_variables = list(accumulated_variables)
        self.raw_statistics_directory = base_directory / "raw_daily_stats"
        self.raw_accumulations_directory = base_directory / "raw_accumulated_boundaries"
        self.raw_statistics_directory.mkdir(parents=True, exist_ok=True)
        self.raw_accumulations_directory.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> Any:
        """Create the CDS client only when an actual retrieval is requested."""
        if self._client is None:
            self._client = cdsapi.Client()
        return self._client

    def download_daily_stat_group(
        self, year: int, statistic: str, variables: list[str], test_days: list[str] | None = None
    ) -> Path:
        """Download one daily-statistic group; test_days selects a reduced June sample."""
        target_name = f"era5_land_{year}_{statistic}.nc"
        target_path = self.raw_statistics_directory / target_name
        if self.manifest.is_done(target_path, verify_checksum=self.verify_checksum):
            logging.info("SKIP (validated in manifest): %s", target_name)
            return target_path
        if target_path.exists():
            raise FileExistsError(f"Unverified file already exists; choose a fresh directory: {target_path}")
        if not variables:
            raise ValueError("At least one variable is required")
        if test_days is None:
            chunks = [
                self._download_daily_chunk(
                    year, month, statistic, variables,
                    list(range(1, calendar.monthrange(year, month)[1] + 1)),
                )
                for month in range(1, 13)
            ]
            self._merge_daily_chunks(chunks, target_path)
            return target_path
        request = {
            "variable": variables,
            "year": str(year),
            "month": ["06"],
            "day": test_days,
            "daily_statistic": statistic,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": self.area,
        }
        part_path = target_path.with_suffix(".nc.part")
        if part_path.exists():
            part_path.unlink()
        logging.info(
            "DOWNLOAD CHANNEL A: year %d, statistic %s (%d variables)", year, statistic, len(variables)
        )
        self.client.retrieve(DATASET_DAILY_STATS, request).download(str(part_path))
        if zipfile.is_zipfile(part_path):
            extraction_directory = self.raw_statistics_directory / f"_tmp_{year}_{statistic}"
            extraction_directory.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(part_path, "r") as archive:
                archive.extractall(extraction_directory)
            if part_path.exists():
                part_path.unlink()
            netcdf_files = list(extraction_directory.glob("*.nc"))
            if len(netcdf_files) == 1:
                if target_path.exists():
                    target_path.unlink()
                netcdf_files[0].replace(target_path)
            elif len(netcdf_files) > 1:
                logging.info("Merging %d NetCDF variables for %s...", len(netcdf_files), statistic)
                opened = []
                for file_handle in netcdf_files:
                    dataset = xr.open_dataset(get_safe_path(file_handle))
                    dataset.load()
                    opened.append(dataset)
                merged = xr.merge(opened, compat="override", join="override")
                for dataset in opened:
                    dataset.close()
                if target_path.exists():
                    target_path.unlink()
                merged.to_netcdf(get_safe_path(target_path))
                merged.close()
            shutil.rmtree(extraction_directory, ignore_errors=True)
        else:
            if target_path.exists():
                target_path.unlink()
            part_path.replace(target_path)
        logging.info("Completed: %s (%.1f MB)", target_path.name, target_path.stat().st_size / 1000000.0)
        return target_path

    def _download_daily_chunk(
        self, year: int, month: int, statistic: str, variables: list[str], days: list[int]
    ) -> Path:
        """Cache validated monthly blocks; bisect days only on an explicit CDS cost limit.

        Annual daily-statistics requests were rejected by live CDS on 2026-09-15.
        Authentication, licence, network and other errors must not trigger splitting.
        """
        tag = f"{statistic}_{month:02d}_{days[0]:02d}_{days[-1]:02d}"
        target = self.raw_statistics_directory / "chunks" / str(year) / f"{tag}.nc"
        if self.manifest.is_done(target, verify_checksum=self.verify_checksum):
            logging.info("SKIP validated daily block: %d %s", year, tag)
            return target
        if target.exists():
            raise FileExistsError(f"Unverified daily block already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".nc.part")
        if part.exists():
            part.unlink()
        request = {
            "variable": variables, "year": str(year), "month": [f"{month:02d}"],
            "day": [f"{day:02d}" for day in days], "daily_statistic": statistic,
            "time_zone": "utc+00:00", "frequency": "1_hourly", "area": self.area,
        }
        logging.info("DOWNLOAD CHANNEL A block: %d %s (%d variables)", year, tag, len(variables))
        try:
            self.client.retrieve(DATASET_DAILY_STATS, request).download(str(part))
        except requests.HTTPError as error:
            if not (
                error.response is not None and error.response.status_code == 403
                and "cost limits exceeded" in str(error).lower() and len(days) > 1
            ):
                raise
            midpoint = len(days) // 2
            logging.info("CDS cost limit: splitting %d %s into two day blocks", year, tag)
            children = [self._download_daily_chunk(year, month, statistic, variables, section)
                        for section in (days[:midpoint], days[midpoint:])]
            self._merge_daily_chunks(children, target)
        else:
            self._finalize_download(part, target, year, tag)
        dataset = open_dataset_safe(target)
        status, note = validate_era5_dataset(
            dataset, expected_dates=pd.to_datetime([f"{year}-{month:02d}-{day:02d}" for day in days])
        )
        if len(dataset.data_vars) != len(variables):
            status, note = "FAIL", "Unexpected variable count in daily block"
        dataset.close()
        self.manifest.record(year, "A", tag, target, status, note)
        if status != "PASS":
            raise ValueError(note)
        return target

    @staticmethod
    def _merge_daily_chunks(chunks: list[Path], target: Path) -> None:
        """Concatenate exact grids/variable inventories, preserving raw values and files."""
        opened = [open_dataset_safe(path) for path in chunks]
        try:
            if any(set(ds.data_vars) != set(opened[0].data_vars) for ds in opened):
                raise ValueError("Variable inventory differs between daily blocks")
            time_name = "valid_time" if "valid_time" in opened[0].dims else "time"
            combined = xr.concat(opened, dim=time_name, join="exact", compat="equals")
            status, note = validate_era5_dataset(combined)
            if status != "PASS":
                raise ValueError(note)
            part = target.with_suffix(".nc.part")
            combined.to_netcdf(get_safe_path(part))
            combined.close()
            part.replace(target)
        finally:
            for dataset in opened:
                dataset.close()

    def download_accumulated_group(self, year: int, test_days: list[str] | None = None) -> Path:
        """Download midnight accumulations using a target-year request and next January 1."""
        target_name = f"era5_land_{year}_accumulated_00utc.nc"
        target_path = self.raw_accumulations_directory / target_name
        if self.manifest.is_done(target_path, verify_checksum=self.verify_checksum):
            logging.info("SKIP (validated in manifest): %s", target_name)
            return target_path
        if target_path.exists():
            raise FileExistsError(f"Unverified file already exists; choose a fresh directory: {target_path}")
        if not self.accumulated_variables:
            raise ValueError("At least one accumulated variable is required")
        if len(self.accumulated_variables) * (366 if calendar.isleap(year) else 365) >= 12000:
            raise ValueError("Accumulated request reaches the 12000-field limit")
        part_path = target_path.with_suffix(".nc.part")
        if part_path.exists():
            part_path.unlink()
        if test_days is not None:
            request = {
                "variable": self.accumulated_variables,
                "year": str(year),
                "month": "06",
                "day": test_days,
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": self.area,
            }
            logging.info(
                "DOWNLOAD CHANNEL B (preflight): year %d (%d variables)",
                year,
                len(self.accumulated_variables),
            )
            self.client.retrieve(DATASET_HOURLY, request).download(str(part_path))
            self._finalize_download(part_path, target_path, year, "accum")
        else:
            main_part = self.raw_accumulations_directory / f"_part_main_{year}.nc"
            boundary_part = self.raw_accumulations_directory / f"_part_bound_{year}.nc"
            main_request = {
                "variable": self.accumulated_variables,
                "year": str(year),
                "month": [f"{month:02d}" for month in range(1, 13)],
                "day": [f"{day:02d}" for day in range(1, 32)],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": self.area,
            }
            boundary_request = {
                "variable": self.accumulated_variables,
                "year": str(year + 1),
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": self.area,
            }
            logging.info("DOWNLOAD CHANNEL B (part 1/2): year %d (365/366 days)", year)
            self.client.retrieve(DATASET_HOURLY, main_request).download(str(main_part))
            main_dataset = open_dataset_safe(main_part)
            logging.info("DOWNLOAD CHANNEL B (part 2/2): boundary 01-01-%d", year + 1)
            self.client.retrieve(DATASET_HOURLY, boundary_request).download(str(boundary_part))
            boundary_dataset = open_dataset_safe(boundary_part)
            time_dimension = "valid_time" if "valid_time" in main_dataset.dims else "time"
            full_dataset = xr.concat([main_dataset, boundary_dataset], dim=time_dimension)
            if target_path.exists():
                target_path.unlink()
            full_dataset.to_netcdf(get_safe_path(target_path))
            main_dataset.close()
            boundary_dataset.close()
            full_dataset.close()
            if main_part.exists():
                main_part.unlink()
            if boundary_part.exists():
                boundary_part.unlink()
        logging.info("Completed: %s (%.1f MB)", target_path.name, target_path.stat().st_size / 1000000.0)
        return target_path

    def _finalize_download(self, part_path: Path, target_path: Path, year: int, tag: str) -> None:
        """Unpack CDS ZIP responses and merge their variable files."""
        if zipfile.is_zipfile(part_path):
            extraction_directory = self.raw_accumulations_directory / f"_tmp_{year}_{tag}"
            extraction_directory.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(part_path, "r") as archive:
                archive.extractall(extraction_directory)
            if part_path.exists():
                part_path.unlink()
            netcdf_files = list(extraction_directory.glob("*.nc"))
            if len(netcdf_files) == 1:
                if target_path.exists():
                    target_path.unlink()
                netcdf_files[0].replace(target_path)
            elif len(netcdf_files) > 1:
                logging.info("Merging %d NetCDF variable files for %s...", len(netcdf_files), tag)
                opened = []
                for file_handle in netcdf_files:
                    dataset = xr.open_dataset(get_safe_path(file_handle))
                    dataset.load()
                    opened.append(dataset)
                merged = xr.merge(opened, compat="override", join="override")
                for dataset in opened:
                    dataset.close()
                if target_path.exists():
                    target_path.unlink()
                merged.to_netcdf(get_safe_path(target_path))
                merged.close()
            shutil.rmtree(extraction_directory, ignore_errors=True)
        else:
            if target_path.exists():
                target_path.unlink()
            part_path.replace(target_path)
