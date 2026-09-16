"""County-grid intersection weights in the EPSG:5070 equal-area projection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .variables import EAST, NORTH, SOUTH, WEST


def download_area(weights: pd.DataFrame) -> list[float]:
    """Smallest grid-aligned CDS rectangle containing every contributing cell center.

    Use validated weights. Removing other counties does not alter retained weights
    or grid resolution. CDS area ordering is north, west, south, east.
    """
    return [
        round(float(weights.latitude.max()), 1),
        round(float(weights.longitude.min()), 1),
        round(float(weights.latitude.min()), 1),
        round(float(weights.longitude.max()), 1),
    ]


def load_spatial_weights(path: Path, expected_counties: int = 479) -> pd.DataFrame:
    """Read existing or canonical weights and validate their coverage and normalization."""
    path = Path(path)
    frame = (
        pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"county_fips": str})
    )
    frame = frame.rename(columns={"lat": "latitude", "lon": "longitude"})
    required = ["county_fips", "latitude", "longitude", "weight"]
    if not set(required).issubset(frame) or frame[required].isna().any().any():
        raise ValueError("Spatial weights have missing fields or values")
    frame["county_fips"] = frame.county_fips.astype(str).str.zfill(5)
    if not frame.county_fips.str.fullmatch(r"\d{5}").all():
        raise ValueError("County FIPS must contain five digits")
    if frame.county_fips.nunique() != expected_counties:
        raise ValueError("Unexpected county coverage in spatial weights")
    if frame.duplicated(["county_fips", "latitude", "longitude"]).any():
        raise ValueError("Duplicate county-grid weight")
    if not np.isfinite(frame[["latitude", "longitude", "weight"]]).all().all() or (frame.weight <= 0).any():
        raise ValueError("Spatial weights and coordinates must be finite; weights must be positive")
    if (
        not frame.latitude.between(SOUTH - 1e-6, NORTH + 1e-6).all()
        or not frame.longitude.between(WEST - 1e-6, EAST + 1e-6).all()
    ):
        raise ValueError("Weight coordinates outside the regional grid")
    if not np.allclose(frame.groupby("county_fips").weight.sum(), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Spatial weights must sum to one for every county")
    return frame


def compute_spatial_weights(
    county_list: Path,
    boundaries: Path,
    output_directory: Path,
    area: tuple[float, float, float, float] = (NORTH, WEST, SOUTH, EAST),
    expected_counties: int = 479,
) -> tuple[Path, Path]:
    """Intersect 0.1-degree cells with counties, preserving the original algorithm.

    Refuse existing output files. Geographic libraries are imported only for this
    computation; ERA5 downloads and basic validation do not require geopandas.
    """
    import geopandas as gpd
    from shapely.geometry import box

    output_directory = Path(output_directory)
    output_parquet = output_directory / f"spatial_weights_{expected_counties}_counties.parquet"
    output_csv = output_directory / f"spatial_weights_{expected_counties}_counties.csv"
    if output_parquet.exists() or output_csv.exists():
        raise FileExistsError("Spatial weights already exist; choose a separate output directory")
    counties = (
        pd.read_csv(county_list, dtype={"county_fips": str})
        if county_list.suffix == ".csv"
        else pd.read_excel(county_list, sheet_name="county_list_479", dtype={"county_fips": str})
    )
    county_fips = set(counties.county_fips.str.zfill(5))
    geometries = gpd.read_file(boundaries)
    selected = geometries[geometries.GEOID.isin(county_fips)][
        ["GEOID", "NAME", "STATE_NAME", "geometry"]
    ].copy()
    if len(county_fips) != expected_counties or len(selected) != expected_counties:
        raise ValueError("County list and boundary coverage must match")
    selected = selected.to_crs(epsg=5070)
    north, west, south, east = area
    latitudes = np.round(np.arange(north, south - 0.05, -0.1), 1)
    longitudes = np.round(np.arange(west, east + 0.05, 0.1), 1)
    cells = [
        {
            "latitude": latitude,
            "longitude": longitude,
            "geometry": box(longitude - 0.05, latitude - 0.05, longitude + 0.05, latitude + 0.05),
        }
        for latitude in latitudes
        for longitude in longitudes
    ]
    grid = gpd.GeoDataFrame(cells, crs="EPSG:4326").to_crs(epsg=5070)
    intersections = gpd.overlay(selected, grid, how="intersection")
    intersections["intersection_area"] = intersections.geometry.area
    county_totals = intersections.groupby("GEOID").intersection_area.transform("sum")
    intersections["weight"] = intersections.intersection_area / county_totals
    weights = intersections[["GEOID", "latitude", "longitude", "weight"]].rename(
        columns={"GEOID": "county_fips"}
    )
    if not np.allclose(weights.groupby("county_fips").weight.sum(), 1.0, atol=1e-5):
        raise ValueError("Computed weight sums are invalid")
    output_directory.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(output_parquet, index=False)
    weights.to_csv(output_csv, index=False)
    return output_parquet, output_csv
