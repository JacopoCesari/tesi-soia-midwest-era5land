"""Validate supplied weight files and reject invalid normalizations."""

import pandas as pd
import pytest
from soybean_yield_forecasting.era5.spatial_weights import load_spatial_weights


def test_supplied_weight_formats_match(project_root):
    directory = project_root / "../work/data/interim/spatial_weights"
    parquet = load_spatial_weights(directory / "spatial_weights_479_counties.parquet")
    csv = load_spatial_weights(directory / "spatial_weights_479_counties.csv")
    pd.testing.assert_frame_equal(parquet, csv, check_exact=False, rtol=1e-10, atol=1e-14)
    assert len(parquet) == 11953


def test_invalid_weight_sum_rejected(tmp_path):
    path = tmp_path / "weights.parquet"
    pd.DataFrame({"county_fips": ["17001"], "lat": [40.0], "lon": [-90.0], "weight": [0.8]}).to_parquet(path)
    with pytest.raises(ValueError, match="sum to one"):
        load_spatial_weights(path, 1)
