"""Read the supplied USDA NASS SURVEY workbooks without modifying them."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def read_survey(
    path: Path,
    measure: Literal["yield_bu_per_acre", "acres_harvested"],
    period: Literal["all", "historical", "recent"] = "all",
) -> pd.DataFrame:
    """Read the full source export; primary sample selection is handled separately.

    Missing optional observations remain absent or missing. Acreage is exposed
    only for the panel-construction audit, with no model or weighting interface.
    """
    if measure not in ("yield_bu_per_acre", "acres_harvested") or period not in (
        "all",
        "historical",
        "recent",
    ):
        raise ValueError("Unsupported measure or period")
    sheet = "balanced_1950_2010" if period == "historical" else "all_years_survey"
    frame = pd.read_excel(path, sheet_name=sheet, dtype={"county_fips": str})
    frame = frame[["county_fips", "year", "state", "county", measure]].copy()
    frame["county_fips"] = frame.county_fips.str.zfill(5)
    if period != "all":
        start, end = (1950, 2010) if period == "historical" else (2011, 2025)
        frame = frame[frame.year.between(start, end)]
    if frame.duplicated(["county_fips", "year"]).any():
        raise ValueError("Duplicate SURVEY county-year records")
    if not frame.county_fips.str.fullmatch(r"\d{5}").all():
        raise ValueError("Invalid county FIPS")
    return frame.reset_index(drop=True)
