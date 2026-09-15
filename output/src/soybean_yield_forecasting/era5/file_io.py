"""Portable NetCDF loading and checksums."""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

import xarray as xr


def get_safe_path(path: Path) -> Path:
    """Return a Windows short path when available to support NetCDF4 Unicode paths."""
    if os.name != "nt":
        return path
    try:
        candidate_path = path.resolve()
        parent = candidate_path.parent
        buffer = ctypes.create_unicode_buffer(500)
        result = ctypes.windll.kernel32.GetShortPathNameW(str(parent), buffer, 500)
        if result > 0:
            return Path(buffer.value) / candidate_path.name
    except Exception:
        pass
    return path


def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 digest without loading the entire file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file_handle:
        while chunk := file_handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def open_dataset_safe(file_path: Path) -> xr.Dataset:
    """Load a NetCDF file or CDS ZIP into memory and release file handles."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if zipfile.is_zipfile(file_path):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with zipfile.ZipFile(file_path, "r") as archive:
                archive.extractall(temporary_directory)
            netcdf_files = sorted(Path(temporary_directory).glob("*.nc"))
            if not netcdf_files:
                raise ValueError(f"No NetCDF files in ZIP: {file_path}")
            opened = []
            for candidate_path in netcdf_files:
                dataset = xr.open_dataset(get_safe_path(candidate_path))
                dataset.load()
                opened.append(dataset)
            if len(opened) == 1:
                merged = opened[0]
            else:
                merged = xr.merge(opened, compat="override", join="override")
            for dataset in opened:
                dataset.close()
            return merged
    else:
        dataset = xr.open_dataset(get_safe_path(file_path))
        dataset.load()
        dataset.close()
        return dataset
