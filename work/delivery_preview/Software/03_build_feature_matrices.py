"""
03_build_feature_matrices.py

Transforms daily county agrometeorological series into 12 monthly forecast
horizon feature matrices (H=12 down to H=1).

Campaign definition:
November 1 of year (Y-1) through October 31 of year Y (12 elapsed campaign months).
For fitted horizons H=1..11, elapsed months are m = 13 - H; H=12 uses none.
- H=12: Baseline prior to campaign start (0 weather features, spatial coords only).
- H=11: End of December Y-1 (2 months of weather).
...
- H=1:  End of October Y (all 12 campaign months).

Indicators (17 per month):
- 8 flux sums: P, SSRD, ET0, P_minus_ET0, GDD, HD30, HD35, EDD30.
- 8 state means: T_mean, T_max, T_min, T_dew, SM1, SM2, SM3, VPD.
- 1 integrated state: SM_root = 0.07*SM1 + 0.21*SM2 + 0.72*SM3.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEATHER_DIR = ROOT / "data" / "weather" / "daily_counties"
WEIGHTS_CSV = ROOT / "data" / "auxiliary" / "spatial_weights.csv"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
OUT_DIR = ROOT / "output" / "data" / "processed" / "model_datasets"

# Mapping from daily parquet columns to thesis variable notation
FLUX_MAP = {
    "total_precipitation": "P",
    "surface_solar_radiation_downwards": "SSRD",
    "et0_fao56": "ET0",
    "p_minus_et0": "P_minus_ET0",
    "growing_degree_days": "GDD",
    "heat_day_30": "HD30",
    "heat_day_35": "HD35",
    "edd_30": "EDD30",
}

STATE_MAP = {
    "air_temperature_mean": "T_mean",
    "air_temperature_maximum": "T_max",
    "air_temperature_minimum": "T_min",
    "dewpoint_temperature_mean": "T_dew",
    "volumetric_soil_water_layer_1": "SM1",
    "volumetric_soil_water_layer_2": "SM2",
    "volumetric_soil_water_layer_3": "SM3",
    "vapor_pressure_deficit": "VPD",
}

ALL_INDICATORS = list(FLUX_MAP.values()) + list(STATE_MAP.values()) + ["SM_root"]


def compute_normalized_centroids(weights_df):
    """Compute area-weighted centroid (lat, lon) per county, normalized to [0, 1]."""
    weights_df = weights_df.copy()
    weights_df["w_lat"] = weights_df["latitude"] * weights_df["weight"]
    weights_df["w_lon"] = weights_df["longitude"] * weights_df["weight"]

    grouped = weights_df.groupby("county_fips").agg(
        sum_w=("weight", "sum"),
        sum_wlat=("w_lat", "sum"),
        sum_wlon=("w_lon", "sum")
    ).reset_index()

    grouped["lat"] = grouped["sum_wlat"] / grouped["sum_w"]
    grouped["lon"] = grouped["sum_wlon"] / grouped["sum_w"]

    grouped["lat_norm"] = (grouped["lat"] - grouped["lat"].min()) / (grouped["lat"].max() - grouped["lat"].min())
    grouped["lon_norm"] = (grouped["lon"] - grouped["lon"].min()) / (grouped["lon"].max() - grouped["lon"].min())
    
    return grouped[["county_fips", "lat_norm", "lon_norm"]]


def aggregate_month(df_year, cal_month):
    """Aggregate daily weather into monthly sums for flux variables and means for state variables."""
    sub = df_year[df_year["date"].dt.month == cal_month].copy()
    if sub.empty:
        return pd.DataFrame()

    if "edd_30" not in sub.columns and "air_temperature_maximum" in sub.columns:
        sub["edd_30"] = np.maximum(0.0, sub["air_temperature_maximum"] - 30.0)

    # Sum fluxes and average states
    agg_dict = {col: "sum" for col in FLUX_MAP.keys() if col in sub.columns}
    agg_dict.update({col: "mean" for col in STATE_MAP.keys() if col in sub.columns})

    monthly = sub.groupby("county_fips").agg(agg_dict).reset_index()
    monthly = monthly.rename(columns={**FLUX_MAP, **STATE_MAP})

    # Weighted root-zone soil moisture (0-100 cm depth)
    if {"SM1", "SM2", "SM3"}.issubset(monthly.columns):
        monthly["SM_root"] = 0.07 * monthly["SM1"] + 0.21 * monthly["SM2"] + 0.72 * monthly["SM3"]

    return monthly


def main():
    print("Initializing feature matrix builder...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_csv(TARGET_CSV)
    years = sorted(target_df["year"].unique())
    weights = pd.read_csv(WEIGHTS_CSV)
    weights["county_fips"] = weights["county_fips"].astype(str).str.zfill(5)
    
    centroids = compute_normalized_centroids(weights)

    # Cache loaded yearly parquets to avoid reloading them repeatedly
    weather_cache = {}
    def get_weather(y):
        if y not in weather_cache:
            p = WEATHER_DIR / f"county_daily_{y}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"])
                df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
                weather_cache[y] = df
            else:
                weather_cache[y] = None
        return weather_cache[y]

    # Pre-aggregate monthly weather: (year, month) -> monthly dataframe
    monthly_data = {}
    print("Aggregating daily weather series to campaign months...")
    for yr in range(min(years) - 1, max(years) + 1):
        df_yr = get_weather(yr)
        if df_yr is not None:
            for m in range(1, 13):
                monthly_data[(yr, m)] = aggregate_month(df_yr, m)

    # Build feature matrices for each horizon H from 12 down to 1
    for H in range(12, 0, -1):
        num_months = 0 if H == 12 else 13 - H
        print(f"Building feature matrix for Horizon H={H:02d} ({num_months} elapsed campaign months)...")

        rows = []
        for yr in years:
            base = centroids.copy()
            base["crop_year"] = yr

            # Campaign months: month 1 = Nov(Y-1), month 2 = Dec(Y-1), months 3..12 = Jan..Oct(Y)
            for cm in range(1, num_months + 1):
                cal_m = 11 if cm == 1 else (12 if cm == 2 else cm - 2)
                cal_y = yr - 1 if cm <= 2 else yr
                
                mdf = monthly_data.get((cal_y, cal_m), pd.DataFrame())
                if not mdf.empty:
                    mdf_cols = {col: f"{col}_m{cm:02d}" for col in ALL_INDICATORS if col in mdf.columns}
                    mdf_renamed = mdf[["county_fips"] + [c for c in ALL_INDICATORS if c in mdf.columns]].rename(columns=mdf_cols)
                    base = base.merge(mdf_renamed, on="county_fips", how="left")

            rows.append(base)

        df_H = pd.concat(rows, ignore_index=True)
        # Reorder columns: identifiers, coordinates, then monthly features
        feat_cols = [c for c in df_H.columns if c not in ("county_fips", "crop_year", "lat_norm", "lon_norm")]
        df_H = df_H[["county_fips", "crop_year", "lat_norm", "lon_norm"] + feat_cols]

        out_path = OUT_DIR / f"features_H{H:02d}.parquet"
        df_H.to_parquet(out_path, index=False)
        print(f"  Saved H={H:02d}: shape {df_H.shape} -> {out_path.name}")

    print("All 12 feature matrices constructed successfully.")


if __name__ == "__main__":
    main()
