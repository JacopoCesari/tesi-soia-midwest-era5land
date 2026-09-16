"""Verify supplied NASS measurements and preserve optional missing pairs."""

import hashlib
import json

import pandas as pd
from soybean_yield_forecasting.data.nass import read_survey
from soybean_yield_forecasting.data.panel import validate_balanced_panel


def test_source_files_preserved_byte_for_byte(project_root):
    entries = json.loads(
        (project_root / "../work/data/external/file_migration.json").read_text(encoding="utf-8")
    )
    assert len(entries) == 40
    for entry in entries:
        path = project_root.parent / entry["new_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], str(path)


def test_primary_panel_continuity_and_optional_separation(project_root):
    external = project_root / "../work/data/external/usda_nass_and_county_boundaries"
    yield_file = external / "soybean_yield_479_counties_1950_2025.xlsx"
    acreage_file = external / "soybean_acres_479_counties_1950_2025.xlsx"
    yields = read_survey(yield_file, "yield_bu_per_acre", "historical")
    acreage = read_survey(acreage_file, "acres_harvested", "historical")
    validate_balanced_panel(
        yields, "yield_bu_per_acre", start_year=1950, end_year=2010, expected_counties=479
    )
    validate_balanced_panel(acreage, "acres_harvested", start_year=1950, end_year=2010, expected_counties=479)
    assert len(yields) == len(acreage) == 29219
    pd.testing.assert_frame_equal(yields[["county_fips", "year"]], acreage[["county_fips", "year"]])
    assert "acres_harvested" not in yields.columns
    assert "median_acres_1950_2025" not in yields.columns
    optional = read_survey(yield_file, "yield_bu_per_acre", "recent")
    assert len(optional) == 6268 and optional.year.between(2011, 2025).all()
