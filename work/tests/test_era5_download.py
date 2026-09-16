"""Mocked complete-year and preflight pipeline checks with no CDS network access."""

import pandas as pd
import pytest
from soybean_yield_forecasting.era5.pipeline import run_year
from soybean_yield_forecasting.era5.schema import WEATHER_VARIABLES


@pytest.mark.parametrize("year,days", [(1950, 365), (1952, 366)])
def test_complete_year_requests_boundary_aggregation_and_resume(tmp_path, fake_cds, year, days, monkeypatch):
    weights = tmp_path / "weights.parquet"
    pd.DataFrame({"county_fips": ["17001"], "lat": [40.0], "lon": [-90.0], "weight": [1.0]}).to_parquet(
        weights
    )
    raw, output = tmp_path / "raw", tmp_path / "output"
    result = run_year(year, raw, output, weights, verify_checksum=True, expected_counties=1)
    frame = pd.read_parquet(result)
    assert frame.shape == (days, 39)
    assert set(frame.columns) == {"date", "county_fips", *WEATHER_VARIABLES}
    assert frame.date.min() == pd.Timestamp(year, 1, 1)
    assert frame.date.max() == pd.Timestamp(year, 12, 31)
    assert frame.total_precipitation.iloc[0] == pytest.approx(0.001)
    assert frame.total_precipitation.iloc[-1] == 0  # Comes from the next January 1 request.
    assert frame.air_temperature_mean.iloc[0] == pytest.approx(16.85)
    assert frame.surface_pressure.iloc[0] == pytest.approx(1013.25)
    assert len(fake_cds) == 38  # 12 months x 3 statistics + annual/boundary accumulations.
    assert all(request["area"] == [40.0, -90.0, 40.0, -90.0] for _, request in fake_cds)
    boundary = fake_cds[-1][1]
    assert boundary["year"] == str(year + 1)
    assert boundary["month"] == ["01"] and boundary["day"] == ["01"]
    assert len(fake_cds[-2][1]["variable"]) * days < 12000
    from soybean_yield_forecasting.era5 import pipeline

    def unexpected_raw_validation(*args, **kwargs):
        raise AssertionError("Completed years must not decompress/revalidate raw NetCDF")

    monkeypatch.setattr(pipeline, "validate_raw_file", unexpected_raw_validation)
    assert run_year(year, raw, output, weights, verify_checksum=True, expected_counties=1) == result
    assert len(fake_cds) == 38
    assert (
        run_year(year, raw, output, weights, aggregate_only=True, verify_checksum=True, expected_counties=1)
        == result
    )
    # A different county with the same row count must not be accepted by the shortcut.
    original_output = result.read_bytes()
    frame.assign(county_fips="17003").to_parquet(result, index=False)
    with pytest.raises(ValueError, match="Existing output is invalid"):
        run_year(year, raw, output, weights, verify_checksum=True, expected_counties=1)
    result.write_bytes(original_output)
    # Raw checksum failure must prevent the shortcut and never trigger overwriting.
    source = pipeline.year_files(raw, year)[0]["mean"]
    payload = source.read_bytes()
    source.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(FileExistsError, match="Unverified"):
        run_year(year, raw, output, weights, verify_checksum=True, expected_counties=1)
    assert len(fake_cds) == 38


def test_daily_month_chunks_resume_after_interruption(tmp_path, fake_cds, monkeypatch):
    from soybean_yield_forecasting.era5.download import CDSDownloader
    from soybean_yield_forecasting.era5.manifest import ManifestManager

    downloader = CDSDownloader(tmp_path, ManifestManager(tmp_path / "manifest.csv"), True)
    original = downloader.client.retrieve

    def interrupted(dataset, request):
        assert len(request["month"]) == 1
        if request["month"] == ["03"]:
            raise ConnectionError("simulated interruption after two completed months")
        return original(dataset, request)

    monkeypatch.setattr(downloader.client, "retrieve", interrupted)
    with pytest.raises(ConnectionError):
        downloader.download_daily_stat_group(1952, "daily_mean", ["2m_temperature"])
    assert len(fake_cds) == 2
    monkeypatch.setattr(downloader.client, "retrieve", original)
    result = downloader.download_daily_stat_group(1952, "daily_mean", ["2m_temperature"])
    from soybean_yield_forecasting.era5.file_io import open_dataset_safe
    assert len(fake_cds) == 12  # January/February must not be requested again.
    assert open_dataset_safe(result).sizes["valid_time"] == 366


def test_cost_limit_splits_days_but_auth_failure_does_not(tmp_path, fake_cds, monkeypatch):
    import requests
    from soybean_yield_forecasting.era5.download import CDSDownloader
    from soybean_yield_forecasting.era5.manifest import ManifestManager

    downloader = CDSDownloader(tmp_path, ManifestManager(tmp_path / "manifest.csv"), True)
    original = downloader.client.retrieve

    def limited(dataset, request):
        if len(request["day"]) > 10:
            response = requests.Response()
            response.status_code = 403
            raise requests.HTTPError("cost limits exceeded", response=response)
        return original(dataset, request)

    monkeypatch.setattr(downloader.client, "retrieve", limited)
    result = downloader.download_daily_stat_group(1950, "daily_mean", ["2m_temperature"])
    from soybean_yield_forecasting.era5.file_io import open_dataset_safe
    assert open_dataset_safe(result).sizes["valid_time"] == 365
    assert all(len(request["day"]) <= 10 for _, request in fake_cds)

    def forbidden(dataset, request):
        response = requests.Response()
        response.status_code = 403
        raise requests.HTTPError("licence not accepted", response=response)

    monkeypatch.setattr(downloader.client, "retrieve", forbidden)
    with pytest.raises(requests.HTTPError, match="licence"):
        downloader.download_daily_stat_group(1951, "daily_mean", ["2m_temperature"])


def test_new_preflight_has_seven_aligned_days_and_all_fields(tmp_path, fake_cds):
    weights = tmp_path / "weights.parquet"
    pd.DataFrame({"county_fips": ["17001"], "lat": [40.0], "lon": [-90.0], "weight": [1.0]}).to_parquet(
        weights
    )
    path = run_year(1950, tmp_path / "raw", tmp_path / "out", weights, preflight=True, expected_counties=1)
    frame = pd.read_parquet(path)
    assert len(frame) == 7 and len(frame.columns) == 39
    assert frame.date.max() == pd.Timestamp("1950-06-07")
    assert fake_cds[-1][1]["day"] == ["02", "03", "04", "05", "06", "07", "08"]


def test_request_limit_rejected_before_retrieval(tmp_path, fake_cds):
    from soybean_yield_forecasting.era5.download import CDSDownloader
    from soybean_yield_forecasting.era5.manifest import ManifestManager

    downloader = CDSDownloader(
        tmp_path,
        ManifestManager(tmp_path / "manifest.csv"),
        accumulated_variables=[f"variable_{index}" for index in range(40)],
    )
    with pytest.raises(ValueError, match="12000-field"):
        downloader.download_accumulated_group(1952)
    assert fake_cds == []


def test_existing_unverified_raw_file_is_preserved(tmp_path, fake_cds):
    from soybean_yield_forecasting.era5.download import CDSDownloader
    from soybean_yield_forecasting.era5.manifest import ManifestManager

    downloader = CDSDownloader(tmp_path, ManifestManager(tmp_path / "manifest.csv"))
    path = tmp_path / "raw_daily_stats/era5_land_1950_daily_mean.nc"
    path.write_bytes(b"original source content")
    with pytest.raises(FileExistsError):
        downloader.download_daily_stat_group(1950, "daily_mean", ["2m_temperature"])
    assert path.read_bytes() == b"original source content" and not fake_cds
