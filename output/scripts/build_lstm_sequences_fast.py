"""
build_lstm_sequences_fast.py -- Fast Vectorized Sequence Builder for LSTM (30d, 10d, 5d).

Builds and caches full-season sequences (H=1) for all 75 target years (1951-2025).
Any horizon H in [1..11] is obtained via zero-cost slicing: X_H = X_H1[:, :S_H, :].
Features per time step: 17 bioclimatic indicators + lat_norm + lon_norm = 19 features.
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = ROOT / "data" / "weather" / "daily_counties"
TARGET_CSV  = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
FEAT_DIR    = ROOT / "output" / "data" / "processed" / "model_datasets"
SEQ_DIR     = ROOT / "output" / "data" / "processed" / "lstm_sequences"
SEQ_DIR.mkdir(parents=True, exist_ok=True)

# 17 Bioclimatic variables in fixed canonical order
FLUX_COLS = [
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "et0_fao56",
    "p_minus_et0",
    "growing_degree_days",
    "heat_day_30",
    "heat_day_35",
    "edd_30",
]

STATE_COLS = [
    "air_temperature_mean",
    "air_temperature_maximum",
    "air_temperature_minimum",
    "dewpoint_temperature_mean",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "vapor_pressure_deficit",
]

CAMPAIGN_MONTH_DAYS = {
    1: 30, 2: 31, 3: 31, 4: 28, 5: 31, 6: 30,
    7: 31, 8: 30, 9: 31, 10: 31, 11: 30, 12: 31,
}

def windows_per_month(W: int) -> dict[int, int]:
    return {cm: CAMPAIGN_MONTH_DAYS[cm] // W for cm in range(1, 13)}

def seq_len_at_H(H: int, W: int) -> int:
    if H == 12:
        return 0
    m = 13 - H
    wpm = windows_per_month(W)
    return sum(wpm[cm] for cm in range(1, m + 1))

def campaign_to_calendar(cm: int, target_year: int) -> tuple[int, int]:
    cal_month = (10 + cm) % 12 or 12
    cal_year  = target_year - 1 if cm <= 2 else target_year
    return cal_month, cal_year

def load_county_coordinates(panel_fips: list[str]) -> np.ndarray:
    feat_h1 = pd.read_parquet(FEAT_DIR / "features_H01.parquet")
    feat_h1["county_fips"] = feat_h1["county_fips"].astype(str)
    coords_df = feat_h1[["county_fips", "lat_norm", "lon_norm"]].drop_duplicates()
    coords_map = coords_df.set_index("county_fips")[["lat_norm", "lon_norm"]].to_dict("index")
    coords_arr = np.zeros((len(panel_fips), 2), dtype=np.float32)
    for i, f in enumerate(panel_fips):
        coords_arr[i, 0] = coords_map[f]["lat_norm"]
        coords_arr[i, 1] = coords_map[f]["lon_norm"]
    return coords_arr

def build_all_sequences(W: int) -> None:
    cache_file = SEQ_DIR / f"seq_{W}d_full_v3.npz"
    if cache_file.exists():
        print(f"[{W}d] Cache already exists at {cache_file.name}. Skipping.")
        return

    print(f"\n[{W}d] Building full-season sequences (W={W} days, 1951-2025)...", flush=True)
    t0 = time.time()

    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    panel_fips = sorted(yield_df["county_fips"].unique().tolist())
    target_years = sorted(yield_df["year"].unique().tolist())

    n_counties = len(panel_fips)
    n_years    = len(target_years)
    n_obs      = n_counties * n_years
    S_full     = seq_len_at_H(1, W)
    INPUT_SIZE = 19  # 17 bioclimatic + 2 spatial coordinates

    # 1. Coordinate array for counties [135, 2]
    coords_arr = load_county_coordinates(panel_fips)  # [n_counties, 2]

    # Pre-allocate output arrays
    X_seq = np.zeros((n_obs, S_full, INPUT_SIZE), dtype=np.float32)
    county_arr = np.empty(n_obs, dtype=object)
    year_arr   = np.zeros(n_obs, dtype=np.int32)

    # Pre-fill county and year arrays
    fips_to_idx = {f: i for i, f in enumerate(panel_fips)}
    for yi, yr in enumerate(target_years):
        for ci, fips in enumerate(panel_fips):
            idx = yi * n_counties + ci
            county_arr[idx] = fips
            year_arr[idx]   = yr
            # Pre-broadcast lat/lon coordinates across all time steps
            X_seq[idx, :, 17] = coords_arr[ci, 0]  # lat_norm
            X_seq[idx, :, 18] = coords_arr[ci, 1]  # lon_norm

    # 2. Iterate by calendar year to load each daily parquet ONCE
    # Target years 1951-2025 require daily weather from 1950 to 2025 (76 calendar years)
    all_cal_years = sorted(list(range(min(target_years) - 1, max(target_years) + 1)))
    daily_cache: dict[int, pd.DataFrame] = {}

    wpm = windows_per_month(W)

    for cal_y in all_cal_years:
        pq_path = WEATHER_DIR / f"county_daily_{cal_y}.parquet"
        if not pq_path.exists():
            continue
        df = pd.read_parquet(pq_path)
        df["date"] = pd.to_datetime(df["date"])
        df["county_fips"] = df["county_fips"].astype(str)
        if "edd_30" not in df.columns and "air_temperature_maximum" in df.columns:
            df["edd_30"] = np.maximum(0.0, df["air_temperature_maximum"] - 30.0)
        daily_cache[cal_y] = df

        # Evict old years
        for old_y in list(daily_cache):
            if old_y < cal_y - 1:
                del daily_cache[old_y]

        # For target years that use this calendar year
        for target_year in [cal_y, cal_y + 1]:
            if target_year not in target_years:
                continue
            yi = target_years.index(target_year)

            # Check which campaign months fall in this cal_y
            for cm in range(1, 13):
                m_cal, y_cal = campaign_to_calendar(cm, target_year)
                if y_cal != cal_y:
                    continue

                n_win = wpm[cm]
                if n_win == 0:
                    continue

                # Calculate offset of first window in this month
                win_start_offset = sum(wpm[c] for c in range(1, cm))

                # Filter month data and sort by (county_fips, date)
                m_mask = df["date"].dt.month == m_cal
                df_m = df[m_mask].sort_values(["county_fips", "date"])

                for fips, grp in df_m.groupby("county_fips"):
                    if fips not in fips_to_idx:
                        continue
                    ci = fips_to_idx[fips]
                    obs_i = yi * n_counties + ci

                    # Vectorized extraction of blocks
                    flux_vals  = grp[FLUX_COLS].values   # [days, 8]
                    state_vals = grp[STATE_COLS].values  # [days, 8]
                    sm1 = grp["volumetric_soil_water_layer_1"].values
                    sm2 = grp["volumetric_soil_water_layer_2"].values
                    sm3 = grp["volumetric_soil_water_layer_3"].values
                    sm_root = 0.07 * sm1 + 0.21 * sm2 + 0.72 * sm3  # [days]

                    for wi in range(n_win):
                        d_start = wi * W
                        d_end   = d_start + W
                        if d_end > len(flux_vals):
                            continue
                        curr_win = win_start_offset + wi

                        # 8 flux sums
                        X_seq[obs_i, curr_win, :8] = np.sum(flux_vals[d_start:d_end], axis=0)
                        # 8 state means
                        X_seq[obs_i, curr_win, 8:16] = np.mean(state_vals[d_start:d_end], axis=0)
                        # 1 root soil moisture mean
                        X_seq[obs_i, curr_win, 16] = np.mean(sm_root[d_start:d_end])

    np.savez_compressed(
        cache_file,
        X_seq=X_seq,
        counties=county_arr,
        years=year_arr,
    )
    print(f"[{W}d] Done in {time.time()-t0:.1f}s. Saved {cache_file.name}, shape={X_seq.shape}", flush=True)

if __name__ == "__main__":
    for w in [30, 10, 5]:
        build_all_sequences(w)
