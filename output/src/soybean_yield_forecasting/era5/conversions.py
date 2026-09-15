"""ERA5 physical identities and legacy unit conversions."""

from __future__ import annotations

import pandas as pd
import xarray as xr


def apply_evaporation_swap(dataset: xr.Dataset) -> xr.Dataset:
    """Restore the physical identities of the three swapped evaporation parameters."""
    transpiration_variable = (
        "evavt" if "evavt" in dataset.data_vars else "evatp" if "evatp" in dataset.data_vars else None
    )
    if "evabs" in dataset.data_vars and "evaow" in dataset.data_vars and (transpiration_variable is not None):
        transpiration = dataset["evabs"].copy()
        bare_soil_evaporation = dataset["evaow"].copy()
        open_water_evaporation = dataset[transpiration_variable].copy()
        dataset = dataset.drop_vars(["evabs", "evaow", transpiration_variable])
        dataset["evaporation_from_vegetation_transpiration"] = transpiration
        dataset["evaporation_from_bare_soil"] = bare_soil_evaporation
        dataset["evaporation_from_open_water_surfaces_excluding_oceans"] = open_water_evaporation
    return dataset


def apply_unit_conversions(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert raw CDS units using the existing magnitude guards; see the data dictionary."""
    frame = frame.copy()
    temperature_columns = [
        column
        for column in frame.columns
        if any(
            (
                temperature_marker in column
                for temperature_marker in ["t2m", "stl", "temperature", "skt", "d2m"]
            )
        )
    ]
    for column in temperature_columns:
        if frame[column].mean() > 100:
            frame[column] = frame[column] - 273.15
    if "surface_pressure" in frame.columns and frame["surface_pressure"].mean() > 50000:
        frame["surface_pressure"] = frame["surface_pressure"] / 100.0
    if "sp" in frame.columns and frame["sp"].mean() > 50000:
        frame["sp"] = frame["sp"] / 100.0
    water_columns = [
        "tp",
        "total_precipitation",
        "e",
        "total_evaporation",
        "pev",
        "potential_evaporation",
        "sro",
        "surface_runoff",
        "ssro",
        "sub_surface_runoff",
        "sf",
        "snowfall",
        "smlt",
        "snowmelt",
        "src",
        "skin_reservoir_content",
        "sd",
        "snow_depth_water_equivalent",
        "evatc",
        "evaporation_from_the_top_of_canopy",
        "evaporation_from_bare_soil",
        "evaporation_from_vegetation_transpiration",
        "evaporation_from_open_water_surfaces_excluding_oceans",
    ]
    for column in water_columns:
        if column in frame.columns and frame[column].abs().max() < 10.0:
            frame[column] = frame[column] * 1000.0
    energy_columns = [
        "ssrd",
        "surface_solar_radiation_downwards",
        "strd",
        "surface_thermal_radiation_downwards",
        "ssr",
        "surface_net_solar_radiation",
        "str",
        "surface_net_thermal_radiation",
        "sshf",
        "surface_sensible_heat_flux",
        "slhf",
        "surface_latent_heat_flux",
    ]
    for column in energy_columns:
        if column in frame.columns and frame[column].abs().mean() > 10000.0:
            frame[column] = frame[column] / 1000000.0
    return frame
