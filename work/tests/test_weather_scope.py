"""Weather optimizations retain every weighted cell and the first crop campaign."""

from soybean_yield_forecasting.configuration import load_configuration
from soybean_yield_forecasting.era5 import cli
from soybean_yield_forecasting.era5.spatial_weights import download_area, load_spatial_weights


def test_selected_area_preserves_all_weight_cells(project_root):
    config = load_configuration(project_root / "configs/data.yaml")
    weights = load_spatial_weights(config["paths"]["weights"], 135)
    north, west, south, east = download_area(weights)
    assert [north, west, south, east] == [45.8, -96.8, 36.0, -80.5]
    assert weights.latitude.between(south, north).all()
    assert weights.longitude.between(west, east).all()
    assert (round((north - south) * 10) + 1) * (round((east - west) * 10) + 1) == 16236


def test_primary_weather_includes_1950_without_1950_yield(project_root, monkeypatch):
    config_path = project_root / "configs/data.yaml"
    config = load_configuration(config_path)
    assert config["primary_period"] == {"start_year": 1951, "end_year": 2025}
    years = []

    def record_year(year, raw, output, weights, **options):
        assert options["expected_counties"] == 135
        assert weights == config["paths"]["weights"]
        years.append(year)
        return output / f"county_daily_{year}.parquet"

    monkeypatch.setattr(cli, "run_year", record_year)
    cli.download_main(["--config", str(config_path), "--primary-period", "--aggregate-only"])
    assert years == list(range(1950, 2026))
