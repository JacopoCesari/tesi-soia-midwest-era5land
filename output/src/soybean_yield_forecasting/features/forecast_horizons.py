"""Twelve separate monthly forecast origins and strict observation cutoffs."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from numbers import Integral
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ForecastHorizon:
    """An origin for a target crop year; forecast_origin may be in the prior year."""

    target_year: int
    forecast_origin: date
    months_before_harvest: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.months_before_harvest, bool)
            or not isinstance(self.months_before_harvest, Integral)
            or not 1 <= self.months_before_harvest <= 12
        ):
            raise ValueError("months_before_harvest must be an integer from 1 through 12")
        if isinstance(self.target_year, bool) or not isinstance(self.target_year, Integral):
            raise ValueError("target_year must be an integer")
        if not isinstance(self.forecast_origin, date):
            raise ValueError("forecast_origin must be a date")
        if self.forecast_origin.year not in (self.target_year - 1, self.target_year):
            raise ValueError("forecast_origin must be in the target year or its previous year")


def build_forecast_horizons(
    target_year: int, *, harvest_month: int, harvest_day: int, horizons: Iterable[int] = range(12, 0, -1)
) -> tuple[ForecastHorizon, ...]:
    """Subtract calendar months from an explicitly supplied harvest month and day.

    Each origin is computed independently from the harvest reference. If that day
    does not exist in an origin month, clamp to that month's last day. The harvest
    reference itself must be a valid date; there is no default harvest date.
    """
    for value in (target_year, harvest_month, harvest_day):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("Year, month and day must be explicit integers")
    reference = date(target_year, harvest_month, harvest_day)
    requested = tuple(horizons)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Supply distinct monthly horizons")
    origins = []
    for months in requested:
        if isinstance(months, bool) or not isinstance(months, Integral) or not 1 <= months <= 12:
            raise ValueError("Horizons must be integers from 1 through 12")
        year, month_index = divmod(reference.year * 12 + reference.month - 1 - months, 12)
        month = month_index + 1
        day = min(reference.day, calendar.monthrange(year, month)[1])
        origins.append(ForecastHorizon(target_year, date(year, month, day), months))
    return tuple(origins)


def _weather_dates(weather: pd.DataFrame, date_column: str) -> pd.Series:
    dates = pd.to_datetime(weather[date_column], errors="raise", utc=True)
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Weather dates must be nonmissing daily UTC dates at midnight")
    return dates


def validate_weather_cutoff(
    weather: pd.DataFrame, horizon: ForecastHorizon, date_column: str = "date"
) -> None:
    """Reject any daily weather observation later than this horizon's inclusive cutoff."""
    dates = _weather_dates(weather, date_column)
    if dates.gt(pd.Timestamp(horizon.forecast_origin, tz="UTC")).any():
        raise ValueError("Weather after forecast_origin is invalid for this horizon")


def filter_weather_at_origin(
    weather: pd.DataFrame, horizon: ForecastHorizon, date_column: str = "date"
) -> pd.DataFrame:
    """Return a separate horizon-labelled copy using only dates through the origin.

    This is a daily date convention, not a claim about ERA5 publication latency.
    No feature lookback, future completion or imputation is performed.
    """
    dates = _weather_dates(weather, date_column)
    subset = weather.loc[dates.le(pd.Timestamp(horizon.forecast_origin, tz="UTC"))].copy()
    subset["target_year"] = horizon.target_year
    subset["forecast_origin"] = pd.Timestamp(horizon.forecast_origin)
    subset["months_before_harvest"] = horizon.months_before_harvest
    validate_weather_cutoff(subset, horizon, date_column)
    return subset
