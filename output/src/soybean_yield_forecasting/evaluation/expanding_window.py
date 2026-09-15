"""Year-grouped rolling-origin splits for complete county panels."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpandingWindowSplit:
    """Positional row indices for iloc and explicit crop-year metadata."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    train_years: tuple[int, ...]
    test_year: int


def expanding_window_splits(
    records: pd.DataFrame,
    initial_training_years: int | None = None,
    year_column: str = "year",
    county_column: str = "county_fips",
) -> Iterator[ExpandingWindowSplit]:
    """Keep complete years together and expand the training set after each test year.

    The author must explicitly choose the initial training length. Reject duplicate
    county-years, calendar gaps and changing county sets. This splitter does not
    establish whether past target labels had been published at a forecast origin.
    """
    if (
        isinstance(initial_training_years, bool)
        or not isinstance(initial_training_years, Integral)
        or initial_training_years < 1
    ):
        raise ValueError("initial_training_years must be a positive integer")
    if records.empty or not {year_column, county_column}.issubset(records):
        raise ValueError("Nonempty county-year records are required")
    if records[[year_column, county_column]].isna().any().any():
        raise ValueError("County and year definitions cannot be missing")
    if (
        not records[year_column]
        .map(lambda year: isinstance(year, Integral) and not isinstance(year, bool))
        .all()
    ):
        raise ValueError("Years must be integers")
    if records.duplicated([county_column, year_column]).any():
        raise ValueError("Duplicate county-year records")
    years = sorted(int(year) for year in records[year_column].unique())
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Calendar years must be consecutive")
    if len(years) <= initial_training_years:
        raise ValueError("At least one test year must follow the initial training window")
    county_sets = records.groupby(year_column)[county_column].agg(frozenset)
    if not county_sets.map(lambda counties: counties == county_sets.iloc[0]).all():
        raise ValueError("Each year must contain the same complete county set")
    for position in range(initial_training_years, len(years)):
        test_year = years[position]
        yield ExpandingWindowSplit(
            np.flatnonzero(records[year_column].to_numpy() < test_year),
            np.flatnonzero(records[year_column].to_numpy() == test_year),
            tuple(years[:position]),
            test_year,
        )
