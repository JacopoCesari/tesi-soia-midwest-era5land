"""Build the author-selected complete 1951-2025 panel from immutable local sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from soybean_yield_forecasting.configuration import load_configuration
from soybean_yield_forecasting.data.nass import read_survey
from soybean_yield_forecasting.data.panel import select_complete_counties, validate_balanced_panel
from soybean_yield_forecasting.era5.spatial_weights import load_spatial_weights


def prepare(configuration: dict) -> dict:
    """Write new derived files or verify identical existing files; never replace data."""
    paths = configuration["paths"]
    period = configuration["primary_period"]
    count = configuration["expected_counties"]
    sources = {
        "yield": paths["external"] / "soybean_yield_479_counties_1950_2025.xlsx",
        "acreage": paths["external"] / "soybean_acres_479_counties_1950_2025.xlsx",
        "weights": paths["supplied_weights"],
    }
    source_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    yields = read_survey(sources["yield"], "yield_bu_per_acre")
    selected = select_complete_counties(yields, "yield_bu_per_acre", **period)
    validate_balanced_panel(selected, "yield_bu_per_acre", expected_counties=count, **period)
    counties = selected[["county_fips", "state", "county"]].drop_duplicates()
    if len(counties) != count:
        raise ValueError("County names must be stable within the selected panel")
    acreage = read_survey(sources["acreage"], "acres_harvested")
    acreage = (
        acreage.loc[
            acreage.county_fips.isin(counties.county_fips)
            & acreage.year.between(period["start_year"], period["end_year"])
        ]
        .sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )
    validate_balanced_panel(acreage, "acres_harvested", expected_counties=count, **period)
    pd.testing.assert_frame_equal(selected[["county_fips", "year"]], acreage[["county_fips", "year"]])
    weights = load_spatial_weights(sources["weights"], configuration["supplied_counties"])
    weights = weights.loc[weights.county_fips.isin(counties.county_fips)].sort_values(
        ["county_fips", "latitude", "longitude"]
    )
    if set(weights.county_fips) != set(counties.county_fips):
        raise ValueError("Spatial weights must cover exactly the selected counties")
    frames = {
        "target": selected[["county_fips", "year", "yield_bu_per_acre"]],
        "counties": counties,
        "acreage": acreage[["county_fips", "year", "acres_harvested"]],
        "weights": weights,
    }
    contents = {
        key: frame.to_csv(index=False, lineterminator="\n").encode("utf-8") for key, frame in frames.items()
    }
    report = {
        "decision_date": "2026-09-15",
        "selection": "Finite SURVEY yield in every year; no imputation; within the supplied 479 counties",
        "rationale": "Maximum usable crop-year coverage; 1950 yield excluded because October 1949 weather is unavailable",
        "period": period,
        "counties": count,
        "observations": len(selected),
        "source_yield_observations": len(yields),
        "excluded_source_observations": len(yields) - len(selected),
        "counties_by_state": counties.groupby("state").size().to_dict(),
        "weight_rows": len(weights),
        "source_sha256": {sources[key].name: value for key, value in source_hashes.items()},
        "derived_sha256": {
            paths[key].name: hashlib.sha256(value).hexdigest() for key, value in contents.items()
        },
        "acreage": "Complete for the same selected county-year keys; auxiliary only, never a predictor or weight",
    }
    contents["provenance"] = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Check every conflict and source hash before writing any derived file.
    for key, content in contents.items():
        if paths[key].exists() and paths[key].read_bytes() != content:
            raise FileExistsError(f"Different derived data already exists: {paths[key]}")
    for key, path in sources.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[key]:
            raise ValueError(f"Source changed during preparation: {path.name}")
    for key, content in contents.items():
        if not paths[key].exists():
            paths[key].parent.mkdir(parents=True, exist_ok=True)
            with paths[key].open("xb") as output:
                output.write(content)
    load_spatial_weights(paths["weights"], count)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    arguments = parser.parse_args()
    print(json.dumps(prepare(load_configuration(arguments.config)), indent=2))
