"""ARCO (Analysis-Ready Cloud-Optimized) Zarr client and acquisition engine for ERA5-Land.

This module implements:
- Programmatic discovery of available stores and variable metadata via .zmetadata probing.
- Lazy, chunk-aware access via xarray and zarr over HTTP/S3 endpoints.
- Authentication handling via environment variables (CDSAPI_KEY, ARCO_API_KEY) and ~/.cdsapirc.
- Spatial and temporal subsetting respecting store chunking.
- Resumable checkpointing, local caching, and atomic writes.
- Provenance logging of store sources, variables, time ranges, and spatial bounding boxes.
- Dry-run and minimal pilot execution modes.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .file_io import compute_sha256


DEFAULT_ARCO_BASE_URL = "https://arco.datastores.ecmwf.int"

DEFAULT_STORES_GEO = {
    "sfc-2m-temperature": "cadl-arco-geo-007",
    "sfc-soil-temperature": "cadl-arco-geo-006",
    "sfc-soil-water": "cadl-arco-geo-005",
    "sfc-radiation-heat": "cadl-arco-geo-010",
    "sfc-snow": "cadl-arco-geo-030",
    "sfc-wind": "cadl-arco-geo-008",
    "sfc-pressure-precipitation": "cadl-arco-geo-009",
    "sfc-skin-temperature": "cadl-arco-geo-043",
}

DEFAULT_STORES_TIME = {
    "sfc-2m-temperature": "cadl-arco-time-007",
    "sfc-soil-temperature": "cadl-arco-time-006",
    "sfc-soil-water": "cadl-arco-time-005",
    "sfc-radiation-heat": "cadl-arco-time-010",
    "sfc-snow": "cadl-arco-time-030",
    "sfc-wind": "cadl-arco-time-008",
    "sfc-pressure-precipitation": "cadl-arco-time-009",
    "sfc-skin-temperature": "cadl-arco-time-043",
}


@dataclass
class ARCOStoreMetadata:
    """Metadata of an ARCO Zarr store discovered programmatically."""

    store_name: str
    bucket: str
    store_type: str
    url: str
    status_code: int
    available: bool
    variables: list[str]
    dimensions: dict[str, int]
    chunks: dict[str, list[int]]
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_arco_credentials() -> str | None:
    """Retrieve CDS / ARCO API key from environment variables or ~/.cdsapirc."""
    key = os.environ.get("CDSAPI_KEY") or os.environ.get("ARCO_API_KEY") or os.environ.get("ECMWF_API_KEY")
    if key:
        return key.strip()

    rc_path = Path.home() / ".cdsapirc"
    if rc_path.exists():
        try:
            for line in rc_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("key:"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return None


def _safe_remove(path: Path, max_attempts: int = 5, base_delay: float = 0.2) -> None:
    """Safely remove a file or directory tree, retrying if temporarily locked on Windows."""
    if not path.exists():
        return

    gc.collect()
    for attempt in range(1, max_attempts + 1):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return
        except (PermissionError, OSError):
            gc.collect()
            time.sleep(base_delay * attempt)

    # If direct removal failed, rename to a temporary trash name to free the destination path
    try:
        trash = path.parent / f".trash_{path.name}_{int(time.time())}"
        path.rename(trash)
        if trash.is_dir():
            shutil.rmtree(trash, ignore_errors=True)
        else:
            trash.unlink(missing_ok=True)
    except Exception:
        pass


def _safe_replace_directory(src: Path, dst: Path, max_attempts: int = 15, base_delay: float = 0.5) -> None:
    """Safely replace dst directory with src directory on Windows.

    Windows-specific environments (especially OneDrive synced folders and active
    antivirus/indexer services) can transiently hold read locks on newly written
    chunk files. This function invokes garbage collection to release any internal
    Python handles and performs retries with exponential backoff. If direct rename
    is persistently denied, it falls back to copying the directory contents.
    """
    gc.collect()

    if dst.exists():
        _safe_remove(dst)

    for attempt in range(1, max_attempts + 1):
        try:
            gc.collect()
            src.rename(dst)
            return
        except (PermissionError, OSError) as err:
            if attempt == max_attempts:
                logging.warning(
                    "Direct rename of %s to %s failed after %d attempts (%s). Attempting fallback copy...",
                    src.name,
                    dst.name,
                    max_attempts,
                    err,
                )
                try:
                    if not dst.exists():
                        shutil.copytree(src, dst)
                    _safe_remove(src)
                    return
                except Exception as fallback_err:
                    raise PermissionError(
                        f"Failed to replace directory {dst.name} with {src.name} after {max_attempts} retries: {fallback_err}"
                    ) from err
            gc.collect()
            time.sleep(min(3.0, base_delay * (1.3 ** attempt)))


class ARCOClient:
    """Client for retrieving and subsetting ECMWF ARCO ERA5-Land Zarr stores."""

    def __init__(
        self,
        base_url: str = DEFAULT_ARCO_BASE_URL,
        store_type: str = "geoChunked",
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        cache_dir: Path | str | None = None,
        stores_map: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.store_type = store_type
        self.api_key = api_key or get_arco_credentials()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if stores_map is not None:
            self.stores_map = stores_map
        elif self.store_type == "timeChunked":
            self.stores_map = DEFAULT_STORES_TIME
        else:
            self.stores_map = DEFAULT_STORES_GEO

    def get_auth_headers(self) -> dict[str, str]:
        """Generate HTTP headers for authenticated requests."""
        headers = {
            "Accept": "application/json, */*",
            "User-Agent": "SoybeanYieldForecasting-ARCO/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_store_url(self, store_name: str) -> str:
        """Construct the full URL to a specific Zarr store."""
        bucket = self.stores_map.get(store_name, f"cadl-arco-{self.store_type[:3]}-001")
        filename = f"{self.store_type}.zarr"
        return f"{self.base_url}/{bucket}/arco/reanalysis_era5_land/{store_name}/{filename}"

    def probe_store_metadata(self, store_name: str) -> ARCOStoreMetadata:
        """Query the remote store's .zmetadata to verify availability and structure."""
        url = self.build_store_url(store_name)
        metadata_url = f"{url}/.zmetadata"
        headers = self.get_auth_headers()

        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(metadata_url, headers=headers, timeout=self.timeout)
                if response.status_code in (200, 401, 403, 404):
                    break
            except (requests.ConnectionError, requests.Timeout) as err:
                if attempt == self.max_retries:
                    logging.warning("Failed probing %s after %d attempts: %s", store_name, attempt, err)
                    break
                sleep_time = self.backoff_factor**attempt
                logging.info("Retry %d/%d for %s after %.1fs", attempt, self.max_retries, store_name, sleep_time)
                time.sleep(sleep_time)

        if response is None or response.status_code != 200:
            status = response.status_code if response is not None else 500
            return ARCOStoreMetadata(
                store_name=store_name,
                bucket=self.stores_map.get(store_name, ""),
                store_type=self.store_type,
                url=url,
                status_code=status,
                available=False,
                variables=[],
                dimensions={},
                chunks={},
                attributes={},
            )

        data = response.json()
        meta = data.get("metadata", {})
        root_attrs = meta.get(".zattrs", {})

        variables = []
        dimensions: dict[str, int] = {}
        chunks: dict[str, list[int]] = {}

        coord_keys = {"time", "valid_time", "latitude", "longitude"}
        for key, val in meta.items():
            if key.endswith("/.zarray"):
                var_name = key.split("/")[0]
                chunks[var_name] = val.get("chunks", [])
                if var_name in coord_keys:
                    shape = val.get("shape", [])
                    if shape:
                        dimensions[var_name] = shape[0]
                else:
                    variables.append(var_name)

        return ARCOStoreMetadata(
            store_name=store_name,
            bucket=self.stores_map.get(store_name, ""),
            store_type=self.store_type,
            url=url,
            status_code=200,
            available=True,
            variables=sorted(set(variables)),
            dimensions=dimensions,
            chunks=chunks,
            attributes=root_attrs,
        )

    def discover_inventory(self, save_path: Path | None = None) -> dict[str, Any]:
        """Programmatically query all candidate stores to build the live inventory."""
        logging.info("Discovering ARCO stores inventory from %s (%s)...", self.base_url, self.store_type)
        inventory: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": self.base_url,
            "store_type": self.store_type,
            "has_credentials": bool(self.api_key),
            "stores": {},
            "all_variables": {},
        }

        for store_name in self.stores_map:
            meta = self.probe_store_metadata(store_name)
            inventory["stores"][store_name] = meta.to_dict()
            if meta.available:
                for var in meta.variables:
                    inventory["all_variables"][var] = {
                        "store": store_name,
                        "url": meta.url,
                        "chunks": meta.chunks.get(var, []),
                    }

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
            logging.info("Inventory cached to: %s", save_path)

        return inventory

    def open_store_lazy(self, store_name: str) -> xr.Dataset:
        """Open an ARCO Zarr store lazily with authentication."""
        url = self.build_store_url(store_name)
        storage_options = {"headers": self.get_auth_headers()}
        logging.info("Opening ARCO Zarr lazily: %s", url)
        return xr.open_zarr(url, consolidated=True, storage_options=storage_options, chunks=None)

    def fetch_slice(
        self,
        store_name: str,
        start_date: str,
        end_date: str,
        lat_bounds: tuple[float, float],
        lon_bounds: tuple[float, float],
        variables: Sequence[str] | None = None,
        use_cache: bool = True,
    ) -> xr.Dataset:
        """Fetch a specific spatio-temporal slice from an ARCO store with caching.

        Parameters
        ----------
        store_name : str
            Identifier of the store (e.g. 'sfc-2m-temperature')
        start_date : str
            Start date string (e.g. '1950-06-01')
        end_date : str
            End date string (e.g. '1950-06-07' or '1950-06-07T23:00:00')
        lat_bounds : tuple[float, float]
            (min_latitude, max_latitude) in degrees North
        lon_bounds : tuple[float, float]
            (min_longitude, max_longitude) in degrees East
        variables : Sequence[str], optional
            Variables to extract from this store
        use_cache : bool
            Whether to read and write locally cached slices
        """
        # Formulate cache key
        min_lat, max_lat = round(min(lat_bounds), 2), round(max(lat_bounds), 2)
        min_lon, max_lon = round(min(lon_bounds), 2), round(max(lon_bounds), 2)
        tag = f"{store_name}_{start_date[:10]}_{end_date[:10]}_lat{min_lat}_{max_lat}_lon{min_lon}_{max_lon}"
        variable_key = ",".join(sorted(variables or ()))
        cache_digest = sha256(f"{tag}|{variable_key}".encode()).hexdigest()[:16]
        cache_name = f"{store_name}_{start_date[:10]}_{end_date[:10]}_{cache_digest}.zarr"
        cache_path = self.cache_dir / cache_name if self.cache_dir else None

        if use_cache and cache_path and cache_path.exists():
            logging.info("Loading cached ARCO slice: %s", cache_path.name)
            try:
                cached = xr.open_zarr(cache_path, consolidated=True).load()
                if not cached.data_vars:
                    raise ValueError("cache contains no data variables")
                for variable in cached.data_vars:
                    if np.isnan(cached[variable].values).any():
                        raise ValueError(f"cached variable {variable} contains unexpected NaNs")
                return cached
            except Exception as err:
                logging.warning(
                    "Discarding incomplete or invalid ARCO cache %s: %s",
                    cache_path.name,
                    err,
                )
                _safe_remove(cache_path)

        ds = self.open_store_lazy(store_name)

        # Slice time
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        # Ensure end_date covers full day if given as YYYY-MM-DD
        if len(end_date) == 10:
            slice_end = f"{end_date}T23:59:59"
        else:
            slice_end = end_date
        sub_ds = ds.sel({time_coord: slice(start_date, slice_end)})

        # Slice latitude and longitude with striping for robust connection management
        max_lat_span = 1.0  # Max degrees per latitude strip (~10 points in latitude)
        if (max_lat - min_lat) > max_lat_span:
            strips = []
            cur_lat = min_lat
            while cur_lat < max_lat:
                next_lat = min(cur_lat + max_lat_span, max_lat)
                lat_slice = (
                    slice(cur_lat, next_lat + 0.05)
                    if next_lat >= max_lat
                    else slice(cur_lat, next_lat - 0.01)
                )
                lon_slice = slice(min_lon - 0.05, max_lon + 0.05)
                loaded_strip = None
                for strip_attempt in range(1, self.max_retries + 1):
                    try:
                        strip_ds = sub_ds.sel(latitude=lat_slice, longitude=lon_slice)
                        if variables:
                            available = [v for v in variables if v in strip_ds.data_vars]
                            strip_ds = strip_ds[available]
                        if strip_attempt == 1:
                            logging.info(
                                "Downloading ARCO strip lat [%.2f, %.2f] (shape %s)...",
                                cur_lat,
                                next_lat,
                                dict(strip_ds.sizes),
                            )
                        else:
                            logging.info(
                                "Attempt %d/%d downloading ARCO strip lat [%.2f, %.2f]...",
                                strip_attempt,
                                self.max_retries,
                                cur_lat,
                                next_lat,
                            )
                        loaded_strip = strip_ds.load()
                        has_nan = any(np.isnan(loaded_strip[v].values).any() for v in loaded_strip.data_vars)
                        if has_nan:
                            raise ValueError(f"Strip lat [{cur_lat:.2f}, {next_lat:.2f}] contains unexpected NaNs")
                        break
                    except Exception as err:
                        if strip_attempt == self.max_retries:
                            raise
                        sleep_time = max(3.0, self.backoff_factor**strip_attempt)
                        logging.warning(
                            "Retry %d/%d downloading strip [%.2f, %.2f] after %.1fs due to: %s",
                            strip_attempt,
                            self.max_retries,
                            cur_lat,
                            next_lat,
                            sleep_time,
                            err,
                        )
                        time.sleep(sleep_time)
                strips.append(loaded_strip)
                cur_lat = next_lat
            loaded = xr.concat(strips, dim="latitude")
            lat_idx = loaded.get_index("latitude")
            if lat_idx.duplicated().any():
                loaded = loaded.sel(latitude=~lat_idx.duplicated())
        else:
            lat_slice = slice(min_lat - 0.05, max_lat + 0.05)
            lon_slice = slice(min_lon - 0.05, max_lon + 0.05)
            loaded = None
            for single_attempt in range(1, self.max_retries + 1):
                try:
                    slice_ds = sub_ds.sel(latitude=lat_slice, longitude=lon_slice)
                    if variables:
                        available = [v for v in variables if v in slice_ds.data_vars]
                        slice_ds = slice_ds[available]
                    if single_attempt == 1:
                        logging.info("Downloading slice %s (shape %s)...", tag, dict(slice_ds.sizes))
                    else:
                        logging.info("Attempt %d/%d downloading slice %s...", single_attempt, self.max_retries, tag)
                    loaded = slice_ds.load()
                    has_nan = any(np.isnan(loaded[v].values).any() for v in loaded.data_vars)
                    if has_nan:
                        raise ValueError(f"Slice {tag} contains unexpected NaNs")
                    break
                except Exception as err:
                    if single_attempt == self.max_retries:
                        raise
                    sleep_time = max(3.0, self.backoff_factor**single_attempt)
                    logging.warning(
                        "Retry %d/%d downloading slice %s after %.1fs due to: %s",
                        single_attempt,
                        self.max_retries,
                        tag,
                        sleep_time,
                        err,
                    )
                    time.sleep(sleep_time)

        # Integrity verification before caching
        for v in loaded.data_vars:
            nan_count = int(np.isnan(loaded[v].values).sum())
            if nan_count > 0:
                raise ValueError(
                    f"Refusing to cache corrupted store {store_name}: variable {v} has {nan_count} NaNs"
                )

        if use_cache and cache_path:
            # Keep the temporary component short enough for Windows/Zarr atomic
            # metadata filenames while retaining the descriptive final cache key.
            temp_path = cache_path.parent / f".{cache_digest}.zarr.part"
            if temp_path.exists():
                _safe_remove(temp_path)
            loaded.to_zarr(temp_path, consolidated=True)
            _safe_replace_directory(temp_path, cache_path)
            logging.info("Cached ARCO slice: %s", cache_path.name)

        return loaded
