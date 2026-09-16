"""Documented English names at the boundary between CDS files and county data."""

from __future__ import annotations

import pandas as pd

RAW_TO_CANONICAL = {
    "t2m_mean": "air_temperature_mean",
    "t2m_min": "air_temperature_minimum",
    "t2m_max": "air_temperature_maximum",
    "skt_max": "skin_temperature_maximum",
    "d2m": "dewpoint_temperature_mean",
    "u10": "eastward_wind_mean",
    "v10": "northward_wind_mean",
    "sp": "surface_pressure",
    **{f"stl{layer}": f"soil_temperature_level_{layer}" for layer in range(1, 5)},
    **{f"swvl{layer}": f"volumetric_soil_water_layer_{layer}" for layer in range(1, 5)},
    "src": "skin_reservoir_content",
    "snowc": "snow_cover",
    "rsn": "snow_density",
    "sd": "snow_depth_water_equivalent",
    "tp": "total_precipitation",
    "e": "total_evaporation",
    "evatc": "evaporation_from_the_top_of_canopy",
    "pev": "potential_evaporation",
    "sro": "surface_runoff",
    "ssro": "sub_surface_runoff",
    "sf": "snowfall",
    "smlt": "snowmelt",
    "ssrd": "surface_solar_radiation_downwards",
    "strd": "surface_thermal_radiation_downwards",
    "ssr": "surface_net_solar_radiation",
    "str": "surface_net_thermal_radiation",
    "sshf": "surface_sensible_heat_flux",
    "slhf": "surface_latent_heat_flux",
}
CORRECTED_EVAPORATION_VARIABLES = (
    "evaporation_from_bare_soil",
    "evaporation_from_vegetation_transpiration",
    "evaporation_from_open_water_surfaces_excluding_oceans",
)
WEATHER_VARIABLES = tuple(RAW_TO_CANONICAL.values()) + CORRECTED_EVAPORATION_VARIABLES


def canonicalize_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename source fields without altering numerical values or original files."""
    renamed = frame.rename(columns=RAW_TO_CANONICAL).copy()
    if renamed.columns.duplicated().any():
        raise ValueError("Conflicting raw and canonical weather fields")
    return renamed
