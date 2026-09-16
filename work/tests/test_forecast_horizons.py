"""Forecast origin calendar arithmetic and strict future-weather exclusion."""

from datetime import date

import pandas as pd
import pytest
from soybean_yield_forecasting.configuration import configured_forecast_horizons, load_configuration
from soybean_yield_forecasting.features.forecast_horizons import (
    build_forecast_horizons,
    filter_weather_at_origin,
    validate_weather_cutoff,
)


def test_twelve_horizons_cross_calendar_year():
    # Synthetic harvest reference for testing only, not an agronomic decision.
    horizons = build_forecast_horizons(2000, harvest_month=10, harvest_day=15)
    assert [horizon.months_before_harvest for horizon in horizons] == list(range(12, 0, -1))
    assert horizons[0].forecast_origin == date(1999, 10, 15)
    assert horizons[3].forecast_origin == date(2000, 1, 15)
    assert horizons[-1].forecast_origin == date(2000, 9, 15)
    assert all(horizon.target_year == 2000 for horizon in horizons)
    changed = build_forecast_horizons(2000, harvest_month=9, harvest_day=15)
    assert all(first.forecast_origin != second.forecast_origin for first, second in zip(horizons, changed))


def test_month_end_and_leap_day_are_independent():
    leap = build_forecast_horizons(2000, harvest_month=3, harvest_day=31)
    ordinary = build_forecast_horizons(2001, harvest_month=3, harvest_day=31)
    assert leap[-1].forecast_origin == date(2000, 2, 29)
    assert ordinary[-1].forecast_origin == date(2001, 2, 28)
    assert leap[-2].forecast_origin == date(2000, 1, 31)


@pytest.mark.parametrize("horizons", [[0], [13], [-1], [1.5], [True], [1, 1], []])
def test_invalid_horizons(horizons):
    with pytest.raises(ValueError):
        build_forecast_horizons(2000, harvest_month=10, harvest_day=15, horizons=horizons)


def test_reference_remains_explicit(project_root):
    configuration = load_configuration(project_root / "configs/features.yaml")
    with pytest.raises(ValueError, match="remain open"):
        configured_forecast_horizons(2000, configuration)
    with pytest.raises(ValueError):
        build_forecast_horizons(2000, harvest_month=2, harvest_day=30)


def test_cutoff_filter_returns_separate_labelled_data():
    horizon = build_forecast_horizons(2000, harvest_month=10, harvest_day=15)[0]
    weather = pd.DataFrame(
        {"date": ["1999-10-14", "1999-10-15", "1999-10-16"], "total_precipitation": [1, 2, 999]}
    )
    with pytest.raises(ValueError, match="after forecast_origin"):
        validate_weather_cutoff(weather, horizon)
    filtered = filter_weather_at_origin(weather, horizon)
    assert filtered.total_precipitation.tolist() == [1, 2]
    assert filtered.target_year.eq(2000).all()
    assert filtered.months_before_harvest.eq(12).all()
    assert len(weather) == 3 and "target_year" not in weather
    validate_weather_cutoff(filtered, horizon)


@pytest.mark.parametrize("value", [None, "invalid", "1999-10-15T12:00:00Z"])
def test_invalid_daily_weather_dates(value):
    horizon = build_forecast_horizons(2000, harvest_month=10, harvest_day=15)[0]
    with pytest.raises(ValueError):
        filter_weather_at_origin(pd.DataFrame({"date": [value]}), horizon)
