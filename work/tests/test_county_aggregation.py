"""Numerical parity against preserved, real preflight outputs."""

import pandas as pd
import pytest
from soybean_yield_forecasting.era5.county_aggregation import CountyAggregator
from soybean_yield_forecasting.era5.pipeline import year_files
from soybean_yield_forecasting.era5.schema import canonicalize_weather


@pytest.mark.parametrize("year", [1950, 2025])
def test_supplied_preflight_values_preserved(project_root, tmp_path, year):
    weights = project_root / "../work/data/interim/spatial_weights/spatial_weights_479_counties.parquet"
    statistics, accumulated = year_files(project_root / "../work/data/raw/era5_land/supplied_preflight", year)
    output = CountyAggregator(weights, tmp_path).process_year(
        year, statistics, accumulated, is_preflight=True, allow_supplied_sample=True
    )
    actual = pd.read_parquet(output).sort_values(["county_fips", "date"]).reset_index(drop=True)
    expected = canonicalize_weather(
        pd.read_parquet(
            project_root
            / f"../work/data/interim/county_daily_weather/supplied_preflight/county_daily_{year}.parquet"
        )
    )
    expected = expected.sort_values(["county_fips", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_like=True, rtol=1e-10, atol=1e-10)
    assert len(actual) == 479 * 6


def test_partial_samples_cannot_pass_as_complete_year(project_root, tmp_path):
    weights = project_root / "../work/data/interim/spatial_weights/spatial_weights_479_counties.parquet"
    statistics, accumulated = year_files(project_root / "../work/data/raw/era5_land/supplied_preflight", 1950)
    with pytest.raises(ValueError, match="Day count"):
        CountyAggregator(weights, tmp_path).process_year(1950, statistics, accumulated)
