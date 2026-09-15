"""Shared local paths and deterministic CDS responses; no live credentials."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "output"


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def fake_cds(monkeypatch):
    from soybean_yield_forecasting.era5 import download

    requests = []
    aliases = {
        "2m_temperature": "t2m",
        "2m_dewpoint_temperature": "d2m",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "surface_pressure": "sp",
        "skin_temperature": "skt",
        "skin_reservoir_content": "src",
        "snow_cover": "snowc",
        "snow_density": "rsn",
        "snow_depth_water_equivalent": "sd",
        "total_precipitation": "tp",
        "total_evaporation": "e",
        "potential_evaporation": "pev",
        "evaporation_from_the_top_of_canopy": "evatc",
        "evaporation_from_bare_soil": "evabs",
        "evaporation_from_open_water_surfaces_excluding_oceans": "evaow",
        "evaporation_from_vegetation_transpiration": "evavt",
        "surface_runoff": "sro",
        "sub_surface_runoff": "ssro",
        "snowfall": "sf",
        "snowmelt": "smlt",
        "surface_solar_radiation_downwards": "ssrd",
        "surface_thermal_radiation_downwards": "strd",
        "surface_net_solar_radiation": "ssr",
        "surface_net_thermal_radiation": "str",
        "surface_sensible_heat_flux": "sshf",
        "surface_latent_heat_flux": "slhf",
        **{f"soil_temperature_level_{layer}": f"stl{layer}" for layer in range(1, 5)},
        **{f"volumetric_soil_water_layer_{layer}": f"swvl{layer}" for layer in range(1, 5)},
    }

    class Response:
        def __init__(self, request):
            self.request = request

        def download(self, path):
            request = self.request
            months = request["month"] if isinstance(request["month"], list) else [request["month"]]
            dates = pd.date_range(f"{request['year']}-01-01", f"{request['year']}-12-31")
            dates = dates[
                dates.month.isin([int(month) for month in months])
                & dates.day.isin([int(day) for day in request["day"]])
            ]
            values = {}
            for variable in request["variable"]:
                alias = aliases[variable]
                if "temperature" in variable:
                    value = {"daily_minimum": 280.0, "daily_mean": 290.0, "daily_maximum": 300.0}.get(
                        request.get("daily_statistic"), 290.0
                    )
                elif variable == "surface_pressure":
                    value = 101325.0
                elif "radiation" in variable or "heat_flux" in variable:
                    value = 15000000.0
                elif "volumetric" in variable:
                    value = 0.3
                else:
                    value = 0.001
                array = np.full((len(dates), 1, 1), value)
                if alias == "tp":
                    array[:, 0, 0] = np.arange(len(dates)) * 1e-6
                values[alias] = (("valid_time", "latitude", "longitude"), array)
            xr.Dataset(
                values, coords={"valid_time": dates, "latitude": [40.0], "longitude": [-90.0]}
            ).to_netcdf(path, engine="scipy")

    class Client:
        def retrieve(self, dataset, request):
            requests.append((dataset, request))
            return Response(request)

    monkeypatch.setattr(download.cdsapi, "Client", Client)
    return requests
