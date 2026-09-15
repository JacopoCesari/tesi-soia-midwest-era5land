"""CDS variable selections and regional domain."""

NORTH, WEST, SOUTH, EAST = (47.7, -97.0, 35.8, -80.4)
CDS_AREA = [NORTH, WEST, SOUTH, EAST]
DAY_MODE = "UTC"
DATASET_DAILY_STATS = "derived-era5-land-daily-statistics"
DATASET_HOURLY = "reanalysis-era5-land"
DAILY_MEAN_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "skin_reservoir_content",
    "snow_cover",
    "snow_density",
    "snow_depth_water_equivalent",
]
DAILY_MINIMUM_VARIABLES = ["2m_temperature"]
DAILY_MAXIMUM_VARIABLES = ["2m_temperature", "skin_temperature"]
ACCUMULATED_VARIABLES = [
    "total_precipitation",
    "total_evaporation",
    "evaporation_from_the_top_of_canopy",
    "evaporation_from_bare_soil",
    "evaporation_from_open_water_surfaces_excluding_oceans",
    "evaporation_from_vegetation_transpiration",
    "potential_evaporation",
    "surface_runoff",
    "sub_surface_runoff",
    "snowfall",
    "snowmelt",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
]
EVAPORATION_SWAP_MAP = {
    "evabs": "evaporation_from_vegetation_transpiration",
    "evaow": "evaporation_from_bare_soil",
    "evavt": "evaporation_from_open_water_surfaces_excluding_oceans",
    "evatp": "evaporation_from_open_water_surfaces_excluding_oceans",
}
