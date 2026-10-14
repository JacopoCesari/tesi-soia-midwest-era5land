"""
02_compute_spatial_weights.py

Computes exact planar intersection weights between the ERA5-Land regular grid
(0.1 deg x 0.1 deg) and irregular county boundary polygons using an equal-area
projection (EPSG:5070 Albers Conical Equal Area).
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
COUNTIES_CSV = ROOT / "data" / "auxiliary" / "counties.csv"
SHAPEFILE = ROOT / "data" / "raw" / "census_counties" / "cb_2020_us_county_500k.shp"
OUTPUT_WEIGHTS_CSV = ROOT / "data" / "auxiliary" / "spatial_weights.csv"

# Regional bounding box for the study domain
NORTH, WEST, SOUTH, EAST = 47.7, -97.0, 35.8, -80.4


def main():
    print("Loading county boundaries and panel definitions...")
    df_counties = pd.read_csv(COUNTIES_CSV, dtype={"county_fips": str})
    fips_set = set(df_counties["county_fips"].str.zfill(5))

    # Load Census polygons and filter to the 135 selected counties
    gdf_counties = gpd.read_file(SHAPEFILE)
    gdf_counties = gdf_counties[gdf_counties["GEOID"].isin(fips_set)].copy()
    
    # Reproject county geometries to equal-area projection
    gdf_counties = gdf_counties.to_crs(epsg=5070)

    print("Generating regular ERA5-Land grid cells...")
    latitudes = np.round(np.arange(NORTH, SOUTH - 0.05, -0.1), 1)
    longitudes = np.round(np.arange(WEST, EAST + 0.05, 0.1), 1)

    # Build 0.1-degree rectangular grid cells centered around each coordinate
    cells = []
    for lat in latitudes:
        for lon in longitudes:
            cells.append({
                "latitude": lat,
                "longitude": lon,
                "geometry": box(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05),
            })
    
    gdf_grid = gpd.GeoDataFrame(cells, crs="EPSG:4326").to_crs(epsg=5070)

    print("Computing polygon geometric intersections...")
    intersections = gpd.overlay(gdf_counties, gdf_grid, how="intersection")
    intersections["intersection_area"] = intersections.geometry.area

    # Normalize weights so that the sum of weights equals 1.0 for each county
    county_totals = intersections.groupby("GEOID")["intersection_area"].transform("sum")
    intersections["weight"] = intersections["intersection_area"] / county_totals

    weights = intersections[["GEOID", "latitude", "longitude", "weight"]].rename(
        columns={"GEOID": "county_fips"}
    )
    weights = weights.sort_values(["county_fips", "latitude", "longitude"]).reset_index(drop=True)

    OUTPUT_WEIGHTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(OUTPUT_WEIGHTS_CSV, index=False)
    
    print(f"Spatial weights computed successfully ({len(weights)} county-cell pairs saved).")


if __name__ == "__main__":
    main()
