"""Build 12 horizon-specific feature matrices from daily county weather parquets.

Campaign definition: November 1 of (year-1) through October 31 of year.
Harvest cutoff: October 31 (consolidated 2026-09-29).
Horizons H=12..1: H=12 → end of Nov Y-1 (0 weather features); H=1 → end of Oct Y.

Output: one Parquet per horizon at output/data/processed/model_datasets/features_H{H:02d}.parquet
Schema per row: county_fips, crop_year, lat_norm, lon_norm,
                [for H<12] {var}_{month} for each elapsed campaign month and each of 17 indicators.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (run from repo root or output/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # repo root: Tesi/
WEATHER_DIR = ROOT / "data" / "weather" / "daily_counties"
WEIGHTS_CSV = ROOT / "data" / "auxiliary" / "spatial_weights.csv"
TARGET_CSV  = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
OUT_DIR     = ROOT / "output" / "data" / "processed" / "model_datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Column mapping: parquet column → thesis indicator name
# Exactly K=16 distinct bioclimatic indicators (7 flux sums, 9 state means)
# ---------------------------------------------------------------------------
K_FLUX_MAP: dict[str, str] = {
    "total_precipitation":               "P",
    "surface_solar_radiation_downwards": "SSRD",
    "et0_fao56":                         "ET0",
    "p_minus_et0":                       "P_minus_ET0",
    "growing_degree_days":               "GDD",
    "heat_day_30":                       "HD30",
    "heat_day_35":                       "HD35",
}

K_STATE_MAP: dict[str, str] = {
    "air_temperature_mean":              "T_mean",
    "air_temperature_maximum":           "T_max",
    "air_temperature_minimum":           "T_min",
    "dewpoint_temperature_mean":         "T_dew",
    "volumetric_soil_water_layer_1":      "SM1",
    "volumetric_soil_water_layer_2":      "SM2",
    "volumetric_soil_water_layer_3":      "SM3",
    "vapor_pressure_deficit":            "VPD",
}

FLUX_INDICATORS = list(K_FLUX_MAP.values())        # 7 flux indicators
STATE_INDICATORS = list(K_STATE_MAP.values()) + ["SM_root"]  # 9 state indicators (SM_root derived)
ALL_INDICATOR_NAMES: list[str] = FLUX_INDICATORS + STATE_INDICATORS  # Exactly 16 indicators

HARVEST_MONTH = 10   # October
HARVEST_DAY   = 31   # consolidated 2026-09-29

# Campaign months: Nov(Y-1)=1, Dec(Y-1)=2, Jan(Y)=3, ..., Oct(Y)=12
def campaign_month_to_calendar(m: int, target_year: int) -> tuple[int, int]:
    """Return (calendar_month, calendar_year) for campaign month m (1=Nov Y-1)."""
    calendar_month = (10 + m) % 12 or 12
    calendar_year = target_year - 1 if m <= 2 else target_year
    return calendar_month, calendar_year


def load_weather_year(year: int) -> pd.DataFrame:
    path = WEATHER_DIR / f"county_daily_{year}.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_campaign_month(
    df_year: pd.DataFrame, cal_month: int
) -> pd.DataFrame:
    """Aggregate one calendar month from a daily parquet. Returns one row per county."""
    mask = df_year["date"].dt.month == cal_month
    sub = df_year.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for fips, grp in sub.groupby("county_fips"):
        row: dict = {"county_fips": str(fips)}
        # Flux: sum
        for raw_col, ind_name in K_FLUX_MAP.items():
            if raw_col in grp.columns:
                row[ind_name] = float(grp[raw_col].sum())
            else:
                row[ind_name] = float("nan")
        # State: mean
        for raw_col, ind_name in K_STATE_MAP.items():
            if raw_col in grp.columns:
                row[ind_name] = float(grp[raw_col].mean())
            else:
                row[ind_name] = float("nan")
        
        # SM_root: depth-weighted column (0-100 cm, Eq. 3.14: 0.07*SM1 + 0.21*SM2 + 0.72*SM3)
        row["SM_root"] = 0.07 * row["SM1"] + 0.21 * row["SM2"] + 0.72 * row["SM3"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_centroid_coords(weights: pd.DataFrame) -> pd.DataFrame:
    """Compute area-weighted centroid (lat, lon) per county from the grid weights."""
    weights = weights.copy()
    weights["county_fips"] = weights["county_fips"].astype(str)
    agg = (
        weights
        .assign(
            wlat=lambda d: d["latitude"] * d["weight"],
            wlon=lambda d: d["longitude"] * d["weight"],
        )
        .groupby("county_fips")
        .agg(
            sum_w=("weight", "sum"),
            sum_wlat=("wlat", "sum"),
            sum_wlon=("wlon", "sum"),
        )
        .reset_index()
    )
    agg["lat_centroid"] = agg["sum_wlat"] / agg["sum_w"]
    agg["lon_centroid"] = agg["sum_wlon"] / agg["sum_w"]
    return agg[["county_fips", "lat_centroid", "lon_centroid"]]


def normalize_coords(centroids: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize lat/lon to [0,1] using the full sample range."""
    centroids = centroids.copy()
    centroids["lat_norm"] = (
        (centroids["lat_centroid"] - centroids["lat_centroid"].min())
        / (centroids["lat_centroid"].max() - centroids["lat_centroid"].min())
    )
    centroids["lon_norm"] = (
        (centroids["lon_centroid"] - centroids["lon_centroid"].min())
        / (centroids["lon_centroid"].max() - centroids["lon_centroid"].min())
    )
    return centroids


def build_all_horizons(
    target_years: list[int],
    centroids_norm: pd.DataFrame,
    verbose: bool = True,
) -> dict[int, pd.DataFrame]:
    """Build feature DataFrames for all 12 horizons across all target years.

    Returns dict: horizon_H -> DataFrame with one row per (county_fips, crop_year).
    H=12: only spatial coords (P_met=0).
    H=11..1: spatial coords + accumulated monthly weather from campaign month 1..m.
    """
    # Cache: (year, campaign_month) -> monthly aggregated DataFrame
    monthly_cache: dict[tuple[int, int], pd.DataFrame] = {}

    def get_monthly(target_year: int, campaign_m: int) -> pd.DataFrame:
        key = (target_year, campaign_m)
        if key not in monthly_cache:
            cal_month, cal_year = campaign_month_to_calendar(campaign_m, target_year)
            try:
                df_year = load_weather_year(cal_year)
                agg = aggregate_campaign_month(df_year, cal_month)
                agg["crop_year"] = target_year
                agg["campaign_month"] = campaign_m
                monthly_cache[key] = agg
            except FileNotFoundError:
                if verbose:
                    print(f"  WARNING: missing parquet for year {cal_year}", flush=True)
                monthly_cache[key] = pd.DataFrame()
        return monthly_cache[key]

    horizon_frames: dict[int, list[pd.DataFrame]] = {H: [] for H in range(1, 13)}

    for crop_year in target_years:
        if verbose and crop_year % 10 == 1:
            print(f"  Processing crop year {crop_year}...", flush=True)

        # Spatial coords for this year (same for all horizons)
        base = centroids_norm[["county_fips", "lat_norm", "lon_norm"]].copy()
        base["crop_year"] = crop_year

        # H=12: zero weather features
        horizon_frames[12].append(base.copy())

        # H=11..1: accumulate monthly blocks
        for H in range(11, 0, -1):
            m = 13 - H  # number of elapsed campaign months
            # Collect campaign months 1..m
            monthly_dfs = []
            for cm in range(1, m + 1):
                mdf = get_monthly(crop_year, cm)
                if mdf.empty:
                    break
                monthly_dfs.append(mdf)

            if len(monthly_dfs) < m:
                # Incomplete weather data for this horizon/year — skip
                continue

            # Pivot: one column per (indicator, campaign_month)
            combined = base.copy()
            for cm, mdf in enumerate(monthly_dfs, start=1):
                mdf_pivot = mdf.set_index("county_fips")
                for ind_name in ALL_INDICATOR_NAMES:
                    col_label = f"{ind_name}_m{cm:02d}"
                    combined = combined.merge(
                        mdf_pivot[[ind_name]].rename(columns={ind_name: col_label}).reset_index(),
                        on="county_fips",
                        how="left",
                    )
            horizon_frames[H].append(combined)

    # Concatenate all years per horizon
    result = {}
    for H in range(1, 13):
        if horizon_frames[H]:
            df = pd.concat(horizon_frames[H], ignore_index=True)
            # Ensure consistent column order: county_fips, crop_year, lat_norm, lon_norm, features...
            feat_cols = [c for c in df.columns if c not in ("county_fips", "crop_year", "lat_norm", "lon_norm")]
            df = df[["county_fips", "crop_year", "lat_norm", "lon_norm"] + feat_cols]
            result[H] = df
    return result


def main() -> None:
    print("=== Build Feature Matrices ===", flush=True)

    # Load inputs
    weights   = pd.read_csv(WEIGHTS_CSV)
    target_df = pd.read_csv(TARGET_CSV)
    target_years = sorted(target_df["year"].unique().tolist())
    panel_fips = set(target_df["county_fips"].unique())

    print(f"Target years: {min(target_years)}–{max(target_years)} ({len(target_years)} years)", flush=True)
    print(f"Panel counties: {len(panel_fips)}", flush=True)

    # Build centroids restricted to the 135-county panel
    weights_panel = weights[weights["county_fips"].isin(panel_fips)]
    centroids = build_centroid_coords(weights_panel)
    centroids_norm = normalize_coords(centroids)
    print(f"Centroids computed: {len(centroids_norm)} counties", flush=True)

    # Check available indicator columns vs expected K=16
    sample = pd.read_parquet(WEATHER_DIR / "county_daily_1990.parquet")
    missing_flux = [c for c in K_FLUX_MAP.keys() if c not in sample.columns]
    missing_state = [c for c in K_STATE_MAP.keys() if c not in sample.columns]
    if missing_flux or missing_state:
        print(f"  WARNING: missing parquet columns — flux: {missing_flux}, state: {missing_state}", flush=True)
    print(f"Indicators used: {len(ALL_INDICATOR_NAMES)} ({ALL_INDICATOR_NAMES})", flush=True)

    # Build feature matrices for all campaign years
    # Weather needed: 1950 (for Nov–Dec Y-1=1950 of the 1951 campaign) through 2025
    weather_years_needed = list(range(min(target_years) - 1, max(target_years) + 1))
    print(f"Building horizons H=12..1 for {len(target_years)} crop years...", flush=True)
    horizon_dfs = build_all_horizons(target_years, centroids_norm, verbose=True)

    # Save one Parquet per horizon
    manifest = {}
    for H, df in sorted(horizon_dfs.items()):
        out_path = OUT_DIR / f"features_H{H:02d}.parquet"
        df.to_parquet(out_path, index=False)
        n_feat = len([c for c in df.columns if c not in ("county_fips", "crop_year", "lat_norm", "lon_norm")])
        manifest[H] = {"path": str(out_path), "rows": len(df), "n_features": n_feat, "shape": list(df.shape)}
        print(f"  H={H:2d} -> {len(df):6,} rows, {n_feat:3d} weather features saved to {out_path.name}", flush=True)

    # Save manifest
    manifest_path = OUT_DIR / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}", flush=True)
    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()
