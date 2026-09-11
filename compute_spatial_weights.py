import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


def main():
    parser = argparse.ArgumentParser(description="Calcolo dei pesi spaziali areali per le 479 contee su griglia ERA5-Land 0.1°.")
    parser.add_argument("--data-dir", type=Path, default=Path("01_Dati_Soia_e_Target"), help="Cartella dei dati agronomici e shapefile (default: 01_Dati_Soia_e_Target).")
    parser.add_argument("--out-dir", type=Path, default=Path("era5_land_daily"), help="Cartella di destinazione per i pesi spaziali (default: era5_land_daily).")
    args = parser.parse_args()

    t0 = time.time()
    data_dir = args.data_dir
    out_dir = args.out_dir
    shp_path = data_dir / "census_counties" / "cb_2020_us_county_500k.shp"
    excel_path = data_dir / "soybean_yield_479_counties_1950_2025.xlsx"

    print("Loading 479 counties from Excel...")
    df_counties = pd.read_excel(excel_path, sheet_name="county_list_479")
    df_counties["county_fips"] = df_counties["county_fips"].astype(str).str.zfill(5)
    fips_list = set(df_counties["county_fips"])

    print("Loading US Census counties shapefile...")
    gdf = gpd.read_file(shp_path)
    gdf_479 = gdf[gdf["GEOID"].isin(fips_list)][["GEOID", "NAME", "STATE_NAME", "geometry"]].copy()
    print(f"Matched {len(gdf_479)} counties. Projecting to EPSG:5070 (Albers Equal Area)...")
    gdf_479 = gdf_479.to_crs(epsg=5070)

    # Build grid cells for the regional bounding box
    north, west, south, east = 47.7, -97.0, 35.8, -80.4
    lats = np.round(np.arange(north, south - 0.05, -0.1), 1)
    lons = np.round(np.arange(west, east + 0.05, 0.1), 1)

    print(f"Building grid cells ({len(lats)} lats x {len(lons)} lons = {len(lats)*len(lons)} cells)...")
    cell_records = []
    step = 0.1
    half = step / 2.0
    for lat in lats:
        for lon in lons:
            cell_box = box(lon - half, lat - half, lon + half, lat + half)
            cell_records.append({"lat": lat, "lon": lon, "geometry": cell_box})

    gdf_cells = gpd.GeoDataFrame(cell_records, crs="EPSG:4326").to_crs(epsg=5070)
    print(f"Total grid cells: {len(gdf_cells)}. Performing spatial intersection with 479 counties...")

    # Spatial intersection using overlay
    intersections = gpd.overlay(gdf_479, gdf_cells, how="intersection")
    intersections["intersection_area"] = intersections.geometry.area

    # Normalize weights so sum of weights per county is exactly 1.0
    county_totals = intersections.groupby("GEOID")["intersection_area"].transform("sum")
    intersections["weight"] = intersections["intersection_area"] / county_totals

    weights_df = intersections[["GEOID", "lat", "lon", "weight"]].copy()
    weights_df.rename(columns={"GEOID": "county_fips"}, inplace=True)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "spatial_weights_479_counties.parquet"
    out_csv = out_dir / "spatial_weights_479_counties.csv"

    weights_df.to_parquet(out_parquet, index=False)
    weights_df.to_csv(out_csv, index=False)

    print(f"Completed in {time.time() - t0:.1f}s!")
    print(f"Saved {len(weights_df):,} cell-county intersections to {out_parquet.name} and {out_csv.name}.")
    print(f"Counties covered: {weights_df['county_fips'].nunique()} of 479.")
    
    # Check weight sum per county
    sums = weights_df.groupby("county_fips")["weight"].sum()
    print(f"Weight sum min: {sums.min():.6f}, max: {sums.max():.6f}")

if __name__ == "__main__":
    main()
