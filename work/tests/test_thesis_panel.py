"""Independently reconcile the selected panel and weights with immutable sources."""

import hashlib
import json
import runpy
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from soybean_yield_forecasting.configuration import load_configuration
from soybean_yield_forecasting.data.nass import read_survey
from soybean_yield_forecasting.data.panel import select_complete_counties, validate_balanced_panel
from soybean_yield_forecasting.era5.spatial_weights import load_spatial_weights


def test_selected_target_is_exact_complete_source_subset(project_root):
    config = load_configuration(project_root / "configs/data.yaml")
    paths = config["paths"]
    source = read_survey(paths["external"] / "soybean_yield_479_counties_1950_2025.xlsx", "yield_bu_per_acre")
    # Independent set-intersection calculation, rather than the selection implementation.
    sets = [
        set(source.loc[(source.year == year) & np.isfinite(source.yield_bu_per_acre), "county_fips"])
        for year in range(1951, 2026)
    ]
    eligible = set.intersection(*sets)
    target = pd.read_csv(paths["target"], dtype={"county_fips": str})
    assert len(eligible) == config["expected_counties"] == 135
    assert len(target) == 10125 and set(target.county_fips) == eligible
    validate_balanced_panel(target, "yield_bu_per_acre", **config["primary_period"])
    expected = (
        source.loc[source.county_fips.isin(eligible) & (source.year >= 1951), target.columns]
        .sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(target, expected, check_exact=True)
    acreage = pd.read_csv(paths["acreage"], dtype={"county_fips": str})
    validate_balanced_panel(acreage, "acres_harvested")
    pd.testing.assert_frame_equal(target[["county_fips", "year"]], acreage[["county_fips", "year"]])
    original = load_spatial_weights(paths["supplied_weights"])
    selected = load_spatial_weights(paths["weights"], 135)
    expected_weights = (
        original.loc[original.county_fips.isin(eligible)]
        .sort_values(["county_fips", "latitude", "longitude"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(selected, expected_weights, rtol=1e-14, atol=1e-14)
    assert len(selected) == 3344 and set(selected.county_fips) == eligible
    provenance = json.loads(paths["provenance"].read_text())
    for key in ("target", "counties", "acreage", "weights"):
        assert (
            hashlib.sha256(paths[key].read_bytes()).hexdigest()
            == provenance["derived_sha256"][paths[key].name]
        )


def test_incomplete_nonfinite_and_duplicate_records():
    frame = pd.DataFrame(
        {
            "county_fips": ["17001"] * 3 + ["17003"] * 3 + ["17005"] * 2,
            "year": [2023, 2024, 2025] * 2 + [2023, 2025],
            "yield_bu_per_acre": [10.0, 11.0, 12.0, 10.0, np.inf, 12.0, 10.0, 12.0],
        }
    )
    selected = select_complete_counties(frame, "yield_bu_per_acre", start_year=2023, end_year=2025)
    assert set(selected.county_fips) == {"17001"}
    with pytest.raises(ValueError, match="Duplicate"):
        select_complete_counties(
            pd.concat([frame, frame.iloc[:1]]), "yield_bu_per_acre", start_year=2023, end_year=2025
        )


def test_delivery_excludes_internal_and_full_sources(project_root, tmp_path):
    exporter = runpy.run_path(str(project_root.parent / "work/export_delivery.py"))
    output = tmp_path / "delivery.zip"
    exporter["export_delivery"](project_root, output)
    with ZipFile(output) as archive:
        names = archive.namelist()
        assert "data/target/soybean_yield_1951_2025.csv" in names
        assert "data/auxiliary/spatial_weights.csv" in names
        assert "src/soybean_yield_forecasting/era5/pipeline.py" in names
        excluded = (
            "work/",
            "scratch/",
            "../work/data/external/",
            "../work/data/raw/",
            "../work/data/interim/",
            "references/papers/",
            "thesis/notes/",
            "reports/",
            ".git/",
            ".venv/",
        )
        assert not any(name.startswith(excluded) for name in names)
        assert not any(name.endswith((".xlsx", ".nc", ".pdf")) for name in names)
        assert archive.testzip() is None
    with pytest.raises(FileExistsError):
        exporter["export_delivery"](project_root, output)
