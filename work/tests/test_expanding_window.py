"""Complete years stay indivisible in every expanding training window."""

import pandas as pd
import pytest
from soybean_yield_forecasting.configuration import load_configuration
from soybean_yield_forecasting.evaluation.expanding_window import expanding_window_splits


def records():
    return pd.MultiIndex.from_product(
        [range(1950, 2011), ["17001", "17003"]], names=["year", "county_fips"]
    ).to_frame(index=False)


def test_explicit_forty_year_window(project_root):
    frame = records().sample(frac=1, random_state=4)
    configuration = load_configuration(project_root / "configs/modeling.yaml")
    assert configuration["validation"]["initial_training_years"] is None
    with pytest.raises(ValueError, match="positive integer"):
        list(expanding_window_splits(frame, configuration["validation"]["initial_training_years"]))
    splits = list(expanding_window_splits(frame, 40))
    assert len(splits) == 21
    assert splits[0].train_years == tuple(range(1950, 1990))
    assert splits[0].test_year == 1990 and splits[-1].test_year == 2010
    for split in splits:
        train, test = frame.iloc[split.train_indices], frame.iloc[split.test_indices]
        assert train.year.max() == split.test_year - 1
        assert set(test.year) == {split.test_year}
        assert set(train.county_fips) == set(test.county_fips)
        assert set(train.year).isdisjoint(test.year)


def test_initial_training_size_is_configurable():
    assert next(expanding_window_splits(records(), 10)).test_year == 1960


@pytest.mark.parametrize(
    "kind", ["duplicate", "gap", "missing_county", "missing_year", "fractional_year", "too_short"]
)
def test_inconsistent_panels_rejected(kind):
    frame = records()
    if kind == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif kind == "gap":
        frame = frame[frame.year != 1960]
    elif kind == "missing_county":
        frame = frame.iloc[1:]
    elif kind == "missing_year":
        frame.loc[0, "year"] = None
    elif kind == "fractional_year":
        frame = frame.assign(year=frame.year + 0.5)
    else:
        frame = frame[frame.year < 1990]
    with pytest.raises(ValueError):
        list(expanding_window_splits(frame, 40))
