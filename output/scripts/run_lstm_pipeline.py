"""
run_lstm_pipeline.py — Resumable LSTM grid search for soybean yield anomaly forecasting.

VARIANTE A: OLS detrend (train-only) → LSTM predicts anomaly residuals.
Architecture: LSTM with lat_norm + lon_norm concatenated at each time step (18 inputs).
Spatial encoding: continuous coordinates, no county embedding (consistent with tabular models).
Fixed from literature (Khaki & Wang 2019; Khaki, Wang & Archontoulis 2020):
  - Optimizer: Adam, lr = 3e-4 (identical in both papers)
  - Weight init: Xavier uniform (identical in both papers)
  - Batch size searched: {25, 64} (Khaki 2020 CNN-RNN used 25; Khaki 2019 DNN used 64)

Grid (36 configs): hidden_size × num_layers × dropout × batch_size
  hidden_size  ∈ {64, 128, 256}
  num_layers   ∈ {1, 2}
  dropout      ∈ {0.0, 0.2, 0.4}   NOTE: no-op when num_layers=1 (PyTorch behaviour)
  batch_size   ∈ {25, 64}

Weather aggregation frequencies: 30-day (monthly, reuses tabular Parquet), 10-day, 5-day.
  - 30-day: reshaped directly from existing features_H*.parquet (no re-aggregation).
  - 10-day, 5-day: aggregated from daily county Parquets using calendar-month window slicing.

Validation protocol: expanding-window on 1985–1995 (11 folds), matching tabular pipeline.
Priority horizon order: H=1, H=4, H=7 → then remaining H if time permits.
Horizons and frequencies are configurable via CLI arguments.

CHECKPOINT / RESUME: After each completed config (all 11 val folds), results are written
to work/reports/experiments/lstm/grid_checkpoint.json. On restart, completed configs are
detected by key "{freq}d_H{H:02d}_cfg{cfg_idx:03d}" and skipped automatically.

NOTE (future experiment): Replace the direct linear projection (hidden → 1) with an
intermediate dense head: Linear(hidden, 32) → ReLU → Linear(32, 1). This adds a non-linear
mixing stage between the LSTM hidden state and the scalar output. Architectural flag
`use_dense_head` left as a TODO marker in LSTMForecaster.

PREREQUISITES:
  pip install torch --index-url https://download.pytorch.org/whl/cpu

Usage:
  python output/scripts/run_lstm_pipeline.py                          # default: all freqs, H=1,4,7
  python output/scripts/run_lstm_pipeline.py --freq 30 --horizons 1 4 7
  python output/scripts/run_lstm_pipeline.py --freq 10 --horizons 1
  python output/scripts/run_lstm_pipeline.py --freq 5 10 30 --horizons 1 4 7 2 3 5 6 8 9 10 11
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Guard: PyTorch required
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: PyTorch not installed. Run:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
WEATHER_DIR = ROOT / "data" / "weather" / "daily_counties"
TARGET_CSV  = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
WEIGHTS_CSV = ROOT / "data" / "auxiliary" / "spatial_weights.csv"
FEAT_DIR    = ROOT / "output" / "data" / "processed" / "model_datasets"
SEQ_DIR     = ROOT / "output" / "data" / "processed" / "lstm_sequences"
REPORT_DIR  = ROOT / "work" / "reports" / "experiments" / "lstm"

SEQ_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Partition constants (matches tabular pipeline)
# ---------------------------------------------------------------------------
BASE_TRAIN_START = 1951
VAL_START        = 1985
VAL_END          = 1995
TEST_START       = 1996
TEST_END         = 2025

# ---------------------------------------------------------------------------
# Variable definitions (K=16, matches build_feature_matrices.py)
# ---------------------------------------------------------------------------
FLUX_RAW = {
    "total_precipitation":               "P",
    "surface_solar_radiation_downwards": "SSRD",
    "et0_fao56":                         "ET0",
    "p_minus_et0":                       "P_minus_ET0",
    "growing_degree_days":               "GDD",
    "heat_day_30":                       "HD30",
    "heat_day_35":                       "HD35",
}
STATE_RAW = {
    "air_temperature_mean":             "T_mean",
    "air_temperature_maximum":          "T_max",
    "air_temperature_minimum":          "T_min",
    "dewpoint_temperature_mean":        "T_dew",
    "volumetric_soil_water_layer_1":    "SM1",
    "volumetric_soil_water_layer_2":    "SM2",
    "volumetric_soil_water_layer_3":    "SM3",
    "vapor_pressure_deficit":           "VPD",
}
FLUX_NAMES  = list(FLUX_RAW.values())           # 7 — aggregated as SUM per window
STATE_NAMES = list(STATE_RAW.values()) + ["SM_root"]  # 9 — aggregated as MEAN per window
ALL_VARS    = FLUX_NAMES + STATE_NAMES          # 16 total, in fixed order

# Campaign month → (calendar_month, calendar_year_offset)
# campaign month 1 = Nov Y-1, 2 = Dec Y-1, 3 = Jan Y, ..., 12 = Oct Y
def campaign_to_calendar(cm: int, target_year: int) -> tuple[int, int]:
    cal_month = (10 + cm) % 12 or 12
    cal_year  = target_year - 1 if cm <= 2 else target_year
    return cal_month, cal_year

# Days in each campaign month (non-leap calendar; Feb=28 for consistent window count)
CAMPAIGN_MONTH_DAYS = {
    1: 30,   # Nov
    2: 31,   # Dec
    3: 31,   # Jan
    4: 28,   # Feb (non-leap, gives min windows across years)
    5: 31,   # Mar
    6: 30,   # Apr
    7: 31,   # May
    8: 30,   # Jun
    9: 31,   # Jul
    10: 31,  # Aug
    11: 30,  # Sep
    12: 31,  # Oct
}

def windows_per_month(W: int) -> dict[int, int]:
    """Number of complete W-day windows per campaign month (using non-leap Feb=28)."""
    return {cm: CAMPAIGN_MONTH_DAYS[cm] // W for cm in range(1, 13)}

def seq_len_at_H(H: int, W: int) -> int:
    """Total number of windows at horizon H for window size W."""
    if H == 12:
        return 0
    m = 13 - H  # elapsed campaign months
    wpm = windows_per_month(W)
    return sum(wpm[cm] for cm in range(1, m + 1))

# ---------------------------------------------------------------------------
# Coordinate loading
# ---------------------------------------------------------------------------
def load_centroids() -> pd.DataFrame:
    """Load and normalise county centroids (matches build_feature_matrices.py)."""
    weights = pd.read_csv(WEIGHTS_CSV)
    weights["county_fips"] = weights["county_fips"].astype(str)
    agg = (
        weights
        .assign(wlat=lambda d: d["latitude"] * d["weight"],
                wlon=lambda d: d["longitude"] * d["weight"])
        .groupby("county_fips")
        .agg(sum_w=("weight","sum"), sum_wlat=("wlat","sum"), sum_wlon=("wlon","sum"))
        .reset_index()
    )
    agg["lat"] = agg["sum_wlat"] / agg["sum_w"]
    agg["lon"] = agg["sum_wlon"] / agg["sum_w"]
    agg["lat_norm"] = (agg["lat"] - agg["lat"].min()) / (agg["lat"].max() - agg["lat"].min())
    agg["lon_norm"] = (agg["lon"] - agg["lon"].min()) / (agg["lon"].max() - agg["lon"].min())
    return agg[["county_fips", "lat_norm", "lon_norm"]]

# ---------------------------------------------------------------------------
# Sequence building — 30-day (reshape from tabular features)
# ---------------------------------------------------------------------------
def build_sequences_30d(H: int, panel_fips: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 30-day sequences for horizon H by reshaping existing tabular features.
    Returns:
        X_seq  : (n_obs, m, 16)  — weather features per campaign month
        X_stat : (n_obs, 2)      — [lat_norm, lon_norm]
        county_arr : (n_obs,)    — county_fips strings
        year_arr   : (n_obs,)    — crop_year ints
    """
    if H == 12:
        return None, None, None, None

    feat = pd.read_parquet(FEAT_DIR / f"features_H{H:02d}.parquet")
    feat["county_fips"] = feat["county_fips"].astype(str)
    feat = feat[feat["county_fips"].isin(set(panel_fips))]
    feat = feat.sort_values(["crop_year", "county_fips"]).reset_index(drop=True)

    m = 13 - H  # elapsed months
    # Reconstruct [n_obs, m, 16] from flat {var}_m{cm} columns in FIXED var order
    X_seq = np.zeros((len(feat), m, 16), dtype=np.float32)
    for cm in range(1, m + 1):
        for vi, vname in enumerate(ALL_VARS):
            col = f"{vname}_m{cm:02d}"
            X_seq[:, cm - 1, vi] = feat[col].values.astype(np.float32)

    X_stat   = feat[["lat_norm", "lon_norm"]].values.astype(np.float32)
    counties = feat["county_fips"].values
    years    = feat["crop_year"].values.astype(np.int32)
    return X_seq, X_stat, counties, years

# ---------------------------------------------------------------------------
# Sequence building — 5-day / 10-day (from daily parquets)
# ---------------------------------------------------------------------------
def _load_daily_year(year: int) -> pd.DataFrame | None:
    path = WEATHER_DIR / f"county_daily_{year}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["county_fips"] = df["county_fips"].astype(str)
    return df

def _aggregate_window(grp: pd.DataFrame) -> dict:
    """Aggregate a W-day block for one county: sum flux, mean state, compute SM_root."""
    row = {}
    for raw_col, name in FLUX_RAW.items():
        row[name] = float(grp[raw_col].sum()) if raw_col in grp.columns else np.nan
    for raw_col, name in STATE_RAW.items():
        row[name] = float(grp[raw_col].mean()) if raw_col in grp.columns else np.nan
    row["SM_root"] = 0.07 * row["SM1"] + 0.21 * row["SM2"] + 0.72 * row["SM3"]
    return row

def build_sequences_submonthly(
    W: int,
    H: int,
    panel_fips: list[str],
    centroids: pd.DataFrame,
    all_target_years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build W-day sequences for horizon H from daily county Parquets.
    Uses non-leap Feb (28 days) window count for consistency across years.

    Returns:
        X_seq  : (n_obs, seq_len, 16)
        X_stat : (n_obs, 2)
        county_arr : (n_obs,)
        year_arr   : (n_obs,)
    """
    if H == 12:
        return None, None, None, None

    wpm = windows_per_month(W)
    m   = 13 - H
    S   = seq_len_at_H(H, W)  # total windows

    fips_list  = sorted(panel_fips)
    fips_index = {f: i for i, f in enumerate(fips_list)}
    n_counties = len(fips_list)
    n_years    = len(all_target_years)
    n_obs      = n_counties * n_years

    X_seq  = np.full((n_obs, S, 16), np.nan, dtype=np.float32)
    X_stat = np.zeros((n_obs, 2), dtype=np.float32)
    county_arr = np.empty(n_obs, dtype=object)
    year_arr   = np.zeros(n_obs, dtype=np.int32)

    # Fill static coords
    coord_map = centroids.set_index("county_fips")[["lat_norm", "lon_norm"]].to_dict("index")

    # Cache daily parquets per calendar year to avoid re-loading
    daily_cache: dict[int, pd.DataFrame] = {}

    for yi, target_year in enumerate(all_target_years):
        # Load required calendar years (Y-1 for Nov/Dec, Y for Jan-Oct)
        for cal_year in (target_year - 1, target_year):
            if cal_year not in daily_cache:
                df = _load_daily_year(cal_year)
                daily_cache[cal_year] = df
                # Keep cache small: evict years no longer needed
                to_drop = [y for y in list(daily_cache) if y < target_year - 1]
                for y in to_drop:
                    del daily_cache[y]

        # Build per-county sequences
        window_offset = 0
        county_window_data: dict[str, list[dict]] = {f: [] for f in fips_list}

        for cm in range(1, m + 1):
            n_win = wpm[cm]
            if n_win == 0:
                continue
            cal_month, cal_year = campaign_to_calendar(cm, target_year)
            df_year = daily_cache.get(cal_year)
            if df_year is None:
                window_offset += n_win
                continue

            # Filter to this calendar month
            mask = df_year["date"].dt.month == cal_month
            df_month = df_year.loc[mask].copy()
            df_month = df_month.sort_values("date")

            for fips in fips_list:
                county_days = df_month[df_month["county_fips"] == fips].sort_values("date")
                for wi in range(n_win):
                    start = wi * W
                    end   = start + W
                    block = county_days.iloc[start:end]
                    if len(block) < W:
                        # Incomplete block — use NaN (signals missing data)
                        county_window_data[fips].append({v: np.nan for v in ALL_VARS})
                    else:
                        county_window_data[fips].append(_aggregate_window(block))

        # Write into arrays
        for fips in fips_list:
            ci   = fips_index[fips]
            obs_i = yi * n_counties + ci
            windows = county_window_data[fips]
            county_arr[obs_i] = fips
            year_arr[obs_i]   = target_year
            if fips in coord_map:
                X_stat[obs_i, 0] = coord_map[fips]["lat_norm"]
                X_stat[obs_i, 1] = coord_map[fips]["lon_norm"]
            for wi, w_dict in enumerate(windows[:S]):
                for vi, vname in enumerate(ALL_VARS):
                    X_seq[obs_i, wi, vi] = w_dict.get(vname, np.nan)

    return X_seq, X_stat, county_arr, year_arr

# ---------------------------------------------------------------------------
# Sequence cache: build once, save to disk
# ---------------------------------------------------------------------------
def get_or_build_sequences(
    freq: int,
    H: int,
    panel_fips: list[str],
    centroids: pd.DataFrame,
    all_target_years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None, None]:
    """Load cached sequences from disk, building them if needed."""
    if H == 12:
        return None, None, None, None

    cache_path = SEQ_DIR / f"seq_{freq}d_H{H:02d}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X_seq"].astype(np.float32),
            data["X_stat"].astype(np.float32),
            data["counties"],
            data["years"].astype(np.int32),
        )

    print(f"    Building sequences {freq}d H={H} ...", end=" ", flush=True)
    t0 = time.time()

    if freq == 30:
        X_seq, X_stat, counties, years = build_sequences_30d(H, panel_fips)
    else:
        X_seq, X_stat, counties, years = build_sequences_submonthly(
            freq, H, panel_fips, centroids, all_target_years
        )

    if X_seq is None:
        return None, None, None, None

    np.savez_compressed(
        cache_path,
        X_seq=X_seq, X_stat=X_stat, counties=counties, years=years,
    )
    print(f"done ({time.time()-t0:.1f}s), shape={X_seq.shape}", flush=True)
    return X_seq, X_stat, counties, years

# ---------------------------------------------------------------------------
# Detrending (OLS, train-only per county — matches tabular pipeline)
# ---------------------------------------------------------------------------
def fit_trends(yield_df: pd.DataFrame, train_years: list[int]) -> dict[str, tuple[float, float]]:
    train = yield_df[yield_df["year"].isin(train_years)]
    params = {}
    for fips, grp in train.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        t_bar  = t_vals.mean(); y_bar = y_vals.mean()
        beta   = np.sum((t_vals - t_bar) * (y_vals - y_bar)) / np.sum((t_vals - t_bar) ** 2)
        alpha  = y_bar - beta * t_bar
        params[str(fips)] = (alpha, beta)
    return params

def compute_anomaly(yield_df: pd.DataFrame, trend_params: dict, years: list[int]) -> pd.DataFrame:
    sub = yield_df[yield_df["year"].isin(years)].copy()
    sub["trend"]   = sub.apply(lambda r: trend_params[str(r["county_fips"])][0]
                                + trend_params[str(r["county_fips"])][1] * r["year"], axis=1)
    sub["anomaly"] = sub["yield_bu_per_acre"] - sub["trend"]
    return sub[["county_fips", "year", "anomaly"]].rename(columns={"year": "crop_year"})

# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    """
    LSTM regressor for soybean yield anomaly forecasting.

    Input per time step: 16 weather features + 2 spatial coords (lat_norm, lon_norm) = 18.
    Output: scalar anomaly prediction.

    NOTE (future experiment): Replace the direct linear head with a dense intermediate
    layer: Linear(hidden_size, 32) → ReLU → Dropout(p) → Linear(32, 1).
    This non-linear mixing stage may improve learning when hidden_size is large.
    Activate via `use_dense_head=True` flag (currently not implemented in grid).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_dense_head: bool = False,  # TODO: add to grid in future ablation
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        # Head: direct linear projection (current)
        self.head = nn.Linear(hidden_size, 1)
        self.use_dense_head = use_dense_head  # future hook
        self._xavier_init()

    def _xavier_init(self) -> None:
        """Xavier uniform init for all weight matrices (Khaki & Wang 2019, 2020)."""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        last    = out[:, -1, :]         # take last time step hidden state
        return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# Training & validation helpers
# ---------------------------------------------------------------------------
def train_epoch(
    model: LSTMForecaster,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(
    model: LSTMForecaster,
    X: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy()
    y_np = y.numpy()
    rmse  = float(np.sqrt(np.mean((preds - y_np) ** 2)))
    ss_res = np.sum((y_np - preds) ** 2)
    ss_tot = np.sum(y_np ** 2)  # anomaly: zero-mean denominator
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2


def fit_lstm(
    X_tr: np.ndarray,  # [n_train, seq_len, 18]
    y_tr: np.ndarray,  # [n_train]
    X_va: np.ndarray,  # [n_val, seq_len, 18]
    y_va: np.ndarray,  # [n_val]
    config: dict,
    max_epochs: int = 100,
    patience: int = 7,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Train LSTM with given config and return (val_rmse, val_r2).
    Uses Adam lr=3e-4 (Khaki & Wang 2019, 2020) with early stopping.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_size  = X_tr.shape[2]  # 18 = 16 weather + 2 coords
    hidden_size = config["hidden_size"]
    num_layers  = config["num_layers"]
    dropout     = config["dropout"]
    batch_size  = config["batch_size"]

    model     = LSTMForecaster(input_size, hidden_size, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # fixed from literature
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_rmse = float("inf")
    best_val_r2   = float("nan")
    patience_left = patience

    for epoch in range(max_epochs):
        train_epoch(model, loader, optimizer, criterion)
        val_rmse, val_r2 = evaluate(model, X_va_t, y_va_t)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_r2   = val_r2
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return best_val_rmse, best_val_r2


# ---------------------------------------------------------------------------
# Feature scaling (train-only, consistent with tabular pipeline)
# ---------------------------------------------------------------------------
def scale_sequences(
    X_tr: np.ndarray,
    X_va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    StandardScaler fit on training set, applied to val/test.
    Flattens [n, seq, feat] → [n*seq, feat] for fit, then restores shape.
    """
    n_tr, S, F = X_tr.shape
    flat_tr = X_tr.reshape(-1, F)
    # Mask NaNs for mean/std computation
    mean = np.nanmean(flat_tr, axis=0)
    std  = np.nanstd(flat_tr, axis=0)
    std[std == 0] = 1.0

    def scale(X: np.ndarray) -> np.ndarray:
        n, s, f = X.shape
        flat = (X.reshape(-1, f) - mean) / std
        flat = np.nan_to_num(flat, nan=0.0)  # replace any residual NaN with 0
        return flat.reshape(n, s, f).astype(np.float32)

    return scale(X_tr), scale(X_va)


def build_fold_arrays(
    X_seq: np.ndarray,   # [n_obs, S, 16]
    X_stat: np.ndarray,  # [n_obs, 2]
    counties: np.ndarray,
    years: np.ndarray,
    anomaly_df: pd.DataFrame,  # county_fips, crop_year, anomaly
    train_years: list[int],
    val_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Extract train/val arrays for one fold, scale features, return targets.
    Returns None if there is insufficient data.
    """
    # Index masks
    tr_mask = np.isin(years, train_years)
    va_mask  = years == val_year

    if tr_mask.sum() == 0 or va_mask.sum() == 0:
        return None

    # Build input: concatenate lat/lon to each time step
    S = X_seq.shape[1]
    stat_exp = X_stat[:, np.newaxis, :].repeat(S, axis=1)  # [n_obs, S, 2]
    X_full = np.concatenate([X_seq, stat_exp], axis=2)     # [n_obs, S, 18]

    X_tr_raw = X_full[tr_mask]
    X_va_raw = X_full[va_mask]
    X_tr_sc, X_va_sc = scale_sequences(X_tr_raw, X_va_raw)

    # Build targets: align anomaly_df to array order
    anom_map = anomaly_df.set_index(["county_fips", "crop_year"])["anomaly"].to_dict()

    def get_targets(mask: np.ndarray) -> np.ndarray:
        targets = []
        for i in np.where(mask)[0]:
            key = (str(counties[i]), int(years[i]))
            targets.append(anom_map.get(key, np.nan))
        return np.array(targets, dtype=np.float32)

    y_tr = get_targets(tr_mask)
    y_va = get_targets(va_mask)

    # Drop rows where target is NaN
    valid_tr = ~np.isnan(y_tr)
    valid_va = ~np.isnan(y_va)
    if valid_tr.sum() < 10 or valid_va.sum() < 1:
        return None

    return X_tr_sc[valid_tr], y_tr[valid_tr], X_va_sc[valid_va], y_va[valid_va]


# ---------------------------------------------------------------------------
# Hyperparameter grid (36 configs)
# ---------------------------------------------------------------------------
GRID: list[dict] = [
    {"hidden_size": h, "num_layers": l, "dropout": d, "batch_size": b}
    for h, l, d, b in product([64, 128, 256], [1, 2], [0.0, 0.2, 0.4], [25, 64])
]

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def save_checkpoint(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def checkpoint_key(freq: int, H: int, cfg_idx: int) -> str:
    return f"{freq}d_H{H:02d}_cfg{cfg_idx:03d}"

# ---------------------------------------------------------------------------
# Grid search for one (freq, H)
# ---------------------------------------------------------------------------
def grid_search_one(
    freq: int,
    H: int,
    X_seq: np.ndarray,
    X_stat: np.ndarray,
    counties: np.ndarray,
    years: np.ndarray,
    yield_df: pd.DataFrame,
    all_years: list[int],
    checkpoint: dict,
    checkpoint_path: Path,
) -> dict[int, dict]:
    """
    Run 36-config expanding-window grid search on validation partition (1985–1995).
    Skips configs already in checkpoint. Returns {cfg_idx: result_dict}.
    """
    val_years = [y for y in all_years if VAL_START <= y <= VAL_END]
    results   = {}

    for cfg_idx, config in enumerate(GRID):
        key = checkpoint_key(freq, H, cfg_idx)
        if key in checkpoint:
            # Already computed — restore from checkpoint
            results[cfg_idx] = checkpoint[key]
            continue

        t_cfg = time.time()
        fold_rmses = []
        fold_r2s   = []

        for val_year in val_years:
            train_years = [y for y in all_years if y < val_year]
            tp = fit_trends(yield_df, train_years)

            anom_train = compute_anomaly(yield_df, tp, train_years)
            anom_val   = compute_anomaly(yield_df, tp, [val_year])
            anom_all   = pd.concat([anom_train, anom_val], ignore_index=True)

            fold_data = build_fold_arrays(
                X_seq, X_stat, counties, years,
                anom_all, train_years, val_year,
            )
            if fold_data is None:
                continue

            X_tr, y_tr, X_va, y_va = fold_data
            val_rmse, val_r2 = fit_lstm(X_tr, y_tr, X_va, y_va, config)
            fold_rmses.append(val_rmse)
            fold_r2s.append(val_r2)

        pooled_rmse = float(np.mean(fold_rmses)) if fold_rmses else float("nan")
        pooled_r2   = float(np.mean(fold_r2s))   if fold_r2s   else float("nan")
        elapsed     = time.time() - t_cfg

        result = {
            "val_rmse": pooled_rmse,
            "val_r2":   pooled_r2,
            "n_folds":  len(fold_rmses),
            "config":   config,
            "elapsed_s": round(elapsed, 1),
            "timestamp": datetime.utcnow().isoformat(),
        }
        results[cfg_idx]    = result
        checkpoint[key]     = result
        save_checkpoint(checkpoint, checkpoint_path)

        print(
            f"      cfg {cfg_idx+1:02d}/{len(GRID)}"
            f"  h={config['hidden_size']} l={config['num_layers']}"
            f"  dr={config['dropout']} bs={config['batch_size']}"
            f"  → val_rmse={pooled_rmse:.4f}  ({elapsed:.0f}s)",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Best-params extraction and summary
# ---------------------------------------------------------------------------
def extract_best_params(checkpoint: dict) -> dict:
    """
    For each (freq, H) combination, select config with lowest val_rmse.
    Returns nested dict: best_params[f"{freq}d"][f"H{H:02d}"] = {config + metrics}.
    """
    best: dict = {}
    for key, result in checkpoint.items():
        parts = key.split("_")
        freq_str = parts[0]          # e.g. "30d"
        h_str    = parts[1]          # e.g. "H01"
        if freq_str not in best:
            best[freq_str] = {}
        if h_str not in best[freq_str] or result["val_rmse"] < best[freq_str][h_str]["val_rmse"]:
            best[freq_str][h_str] = result
    return best


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    freqs: list[int],
    horizons: list[int],
    max_epochs: int = 100,
    patience: int = 7,
) -> None:
    t0 = time.time()
    print("=== LSTM Grid Search Pipeline (Variante A — OLS anomaly) ===", flush=True)
    print(f"Frequencies: {freqs}-day | Horizons: {horizons}", flush=True)
    print(f"Grid: {len(GRID)} configs | Validation: {VAL_START}–{VAL_END} | Seed: 42", flush=True)
    print(f"Checkpoint: {REPORT_DIR / 'grid_checkpoint.json'}", flush=True)

    # Load panel
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    panel_fips  = sorted(yield_df["county_fips"].unique().tolist())
    all_years   = sorted(yield_df["year"].unique().tolist())
    centroids   = load_centroids()
    # Restrict centroids to panel counties
    centroids = centroids[centroids["county_fips"].isin(set(panel_fips))]
    print(f"Panel: {len(panel_fips)} counties, {min(all_years)}–{max(all_years)}", flush=True)

    checkpoint_path = REPORT_DIR / "grid_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    n_already_done = len(checkpoint)
    if n_already_done > 0:
        print(f"Resuming: {n_already_done} configs already completed in checkpoint.", flush=True)

    # Main loop: freq → H → grid
    for freq in freqs:
        print(f"\n{'='*60}", flush=True)
        print(f"FREQUENCY: {freq}-day windows", flush=True)
        for H in horizons:
            if H == 12:
                print(f"  H=12: naive baseline, skipping LSTM.", flush=True)
                continue
            print(f"\n  --- Horizon H={H} ({seq_len_at_H(H, freq)} windows) ---", flush=True)

            # Load / build sequences
            X_seq, X_stat, counties, years = get_or_build_sequences(
                freq, H, panel_fips, centroids, all_years
            )
            if X_seq is None:
                print(f"  SKIP H={H}: no data.", flush=True)
                continue

            # Run grid search with checkpoint
            print(f"  Grid search: {len(GRID)} configs × {VAL_END-VAL_START+1} val folds...", flush=True)
            grid_search_one(
                freq=freq, H=H,
                X_seq=X_seq, X_stat=X_stat,
                counties=counties, years=years,
                yield_df=yield_df, all_years=all_years,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
            )

    # Save best params
    best = extract_best_params(checkpoint)
    best_path = REPORT_DIR / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print("GRID SEARCH COMPLETE — Best configs per (freq, H):", flush=True)
    for freq_str, h_dict in sorted(best.items()):
        print(f"\n  {freq_str}:", flush=True)
        for h_str, rec in sorted(h_dict.items()):
            cfg = rec["config"]
            print(
                f"    {h_str}: RMSE={rec['val_rmse']:.4f}"
                f"  h={cfg['hidden_size']} l={cfg['num_layers']}"
                f"  dr={cfg['dropout']} bs={cfg['batch_size']}",
                flush=True,
            )

    elapsed_total = (time.time() - t0) / 60
    print(f"\nTotal elapsed: {elapsed_total:.1f} min", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Best params: {best_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resumable LSTM grid search")
    parser.add_argument(
        "--freq", nargs="+", type=int, default=[30, 10, 5],
        help="Aggregation frequencies in days (default: 30 10 5)",
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 4, 7],
        help="Forecast horizons H to evaluate (default: 1 4 7)",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience",   type=int, default=7)
    args = parser.parse_args()

    run_pipeline(
        freqs=args.freq,
        horizons=args.horizons,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
