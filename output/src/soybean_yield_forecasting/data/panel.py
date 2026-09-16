"""Verify the existing balanced sample without inventing a selection threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_balanced_panel(
    frame: pd.DataFrame,
    measure: str,
    *,
    start_year: int = 1951,
    end_year: int = 2025,
    expected_counties: int = 135,
) -> None:
    """Require the county-year Cartesian product and nonmissing supplied measurements."""
    if not {"county_fips", "year", measure}.issubset(frame):
        raise ValueError("Missing panel fields")
    if frame[["county_fips", "year", measure]].isna().any().any():
        raise ValueError("Missing panel identifiers or measurements")
    if not np.isfinite(frame[measure]).all():
        raise ValueError("Nonfinite panel measurements")
    if frame.duplicated(["county_fips", "year"]).any():
        raise ValueError("Duplicate county-year observations")
    if frame.county_fips.nunique() != expected_counties:
        raise ValueError("Unexpected county count")
    years = set(range(start_year, end_year + 1))
    if (
        set(frame.year) != years
        or not frame.groupby("county_fips").year.agg(set).map(lambda values: values == years).all()
    ):
        raise ValueError("Every county must cover the complete primary period")


def select_complete_counties(
    frame: pd.DataFrame, measure: str, *, start_year: int, end_year: int
) -> pd.DataFrame:
    """Keep only counties with a finite observation in every requested year.

    Author-selected temporal completeness rule; no acreage threshold or imputation.
    Duplicate keys are an error, never an invitation to average source records.
    """
    if start_year > end_year:
        raise ValueError("Invalid panel period")
    if frame.duplicated(["county_fips", "year"]).any():
        raise ValueError("Duplicate county-year observations")
    selected = frame.loc[frame.year.between(start_year, end_year)].copy()
    selected = selected.loc[np.isfinite(selected[measure])]
    counts = selected.groupby("county_fips").year.nunique()
    counties = counts.index[counts == end_year - start_year + 1]
    return (
        selected.loc[selected.county_fips.isin(counties)]
        .sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )
