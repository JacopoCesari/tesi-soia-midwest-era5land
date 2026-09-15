"""Load repository configuration with paths relative to the configuration directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .features.forecast_horizons import ForecastHorizon, build_forecast_horizons


def load_configuration(path: Path) -> dict[str, Any]:
    """Load a YAML mapping; resolve its path settings relative to the project root."""
    path = Path(path).resolve()
    configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if "paths" in configuration:
        configuration["paths"] = {
            key: (path.parent.parent / value).resolve() for key, value in configuration["paths"].items()
        }
    return configuration


def configured_forecast_horizons(
    target_year: int, configuration: dict[str, Any]
) -> tuple[ForecastHorizon, ...]:
    """Require the still-open harvest reference before constructing forecast dates."""
    reference = configuration["harvest_reference"]
    if reference.get("month") is None or reference.get("day") is None:
        raise ValueError("Harvest reference month and day remain open; supply verified values explicitly")
    return build_forecast_horizons(
        target_year,
        harvest_month=reference["month"],
        harvest_day=reference["day"],
        horizons=configuration["months_before_harvest"],
    )
