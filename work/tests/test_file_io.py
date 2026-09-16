"""NetCDF and zipped CDS responses load without keeping file handles open."""

import zipfile

import xarray as xr
from soybean_yield_forecasting.era5.file_io import open_dataset_safe


def test_zipped_variables_are_merged_and_handles_released(tmp_path):
    first = tmp_path / "temperature.nc"
    second = tmp_path / "precipitation.nc"
    xr.Dataset({"t2m": ("time", [290.0])}).to_netcdf(first, engine="scipy")
    xr.Dataset({"tp": ("time", [0.001])}).to_netcdf(second, engine="scipy")
    archive_path = tmp_path / "response.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(first, first.name)
        archive.write(second, second.name)
    dataset = open_dataset_safe(archive_path)
    assert set(dataset.data_vars) == {"t2m", "tp"}
    assert float(dataset.t2m[0]) == 290.0
    # Windows would reject this rename if a handle remained open.
    archive_path.rename(tmp_path / "closed_response.zip")
