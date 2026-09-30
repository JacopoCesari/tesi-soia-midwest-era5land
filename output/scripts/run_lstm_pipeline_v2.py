"""
run_lstm_pipeline_v2.py — Resumable LSTM grid search for soybean yield anomaly forecasting.

NOTE: v1 (run_lstm_pipeline.py) used lat/lon as static spatial features concatenated per
time step (input_size=18). Those 167 checkpoint entries are the 'lat/lon baseline' variant,
comparable to tabular models. This v2 is the 'county embedding' architecture with pre-training.
The two variants serve as ablations for the spatial encoding choice.

NOTE (future ablation): Variante B = no OLS detrend, temporal attention captures trend.
NOTE (future ablation): Dense head between LSTM state and output scalar.
NOTE (future ablation): Variante A+ pre-training can be disabled (--no-pretrain flag) to
measure the delta between Xavier init and warm-start init.

Architecture changes from v1:
- Input per time step: 16 weather features ONLY (no lat/lon)
- County embedding dim=16, fixed (based on 135 counties)
  - Concatenated to the LSTM hidden state AT THE OUTPUT
- num_layers extended to {1, 2, 3, 4}
- Grid: 72 configs (hidden_size × num_layers × dropout × batch_size)
- Two-phase training:
  1. PRE-TRAINING: train a universal LSTM on ALL panel data (1951-1979 train, 1980-1984 val).
  2. FINE-TUNING: initialize LSTM weights from pre-trained model. Use discriminative LR.

PREREQUISITES:
  pip install torch --index-url https://download.pytorch.org/whl/cpu

Usage:
  python output/scripts/run_lstm_pipeline_v2.py
  python output/scripts/run_lstm_pipeline_v2.py --freq 30 --horizons 1 4 7
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
FEAT_DIR    = ROOT / "output" / "data" / "processed" / "model_datasets"
SEQ_DIR     = ROOT / "output" / "data" / "processed" / "lstm_sequences"
PRETRAIN_DIR= SEQ_DIR  # Save pretrain weights alongside sequences
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
# Constants and configs
# ---------------------------------------------------------------------------
LR = 3e-4          # fixed from Khaki & Wang (2019, 2020)
LR_LSTM_FINETUNE = 3e-5   # 1/10 of LR for LSTM weights during fine-tuning
COUNTY_EMBEDDING_DIM = 16  # fixed
N_COUNTIES = 135

GRID = [
    {"hidden_size": h, "num_layers": l, "dropout": d, "batch_size": b}
    for h, l, d, b in product([64, 128, 256], [1, 2, 3, 4], [0.0, 0.2, 0.4], [25, 64])
]  # 72 configs

# Pre-training period (before validation and test partitions)
PRETRAIN_TRAIN_YEARS = list(range(1951, 1980))  # 1951-1979
PRETRAIN_VAL_YEARS = list(range(1980, 1985))    # 1980-1984

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
def campaign_to_calendar(cm: int, target_year: int) -> tuple[int, int]:
    cal_month = (10 + cm) % 12 or 12
    cal_year  = target_year - 1 if cm <= 2 else target_year
    return cal_month, cal_year

# Days in each campaign month (non-leap calendar; Feb=28 for consistent window count)
CAMPAIGN_MONTH_DAYS = {
    1: 30, 2: 31, 3: 31, 4: 28, 5: 31, 6: 30,
    7: 31, 8: 30, 9: 31, 10: 31, 11: 30, 12: 31,
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
# Sequence building — 30-day (reshape from tabular features)
# ---------------------------------------------------------------------------
def build_sequences_30d(H: int, panel_fips: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 30-day sequences for horizon H by reshaping existing tabular features.
    Returns:
        X_seq  : (n_obs, m, 16)  — weather features per campaign month
        county_arr : (n_obs,)    — county_fips strings
        year_arr   : (n_obs,)    — crop_year ints
    """
    if H == 12:
        return None, None, None

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

    counties = feat["county_fips"].values
    years    = feat["crop_year"].values.astype(np.int32)
    return X_seq, counties, years

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
    all_target_years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build W-day sequences for horizon H from daily county Parquets.
    Returns:
        X_seq  : (n_obs, seq_len, 16)
        county_arr : (n_obs,)
        year_arr   : (n_obs,)
    """
    if H == 12:
        return None, None, None

    wpm = windows_per_month(W)
    m   = 13 - H
    S   = seq_len_at_H(H, W)  # total windows

    fips_list  = sorted(panel_fips)
    fips_index = {f: i for i, f in enumerate(fips_list)}
    n_counties = len(fips_list)
    n_years    = len(all_target_years)
    n_obs      = n_counties * n_years

    X_seq  = np.full((n_obs, S, 16), np.nan, dtype=np.float32)
    county_arr = np.empty(n_obs, dtype=object)
    year_arr   = np.zeros(n_obs, dtype=np.int32)

    # Cache daily parquets per calendar year to avoid re-loading
    daily_cache: dict[int, pd.DataFrame] = {}

    for yi, target_year in enumerate(all_target_years):
        for cal_year in (target_year - 1, target_year):
            if cal_year not in daily_cache:
                df = _load_daily_year(cal_year)
                daily_cache[cal_year] = df
                to_drop = [y for y in list(daily_cache) if y < target_year - 1]
                for y in to_drop:
                    del daily_cache[y]

        # Build per-county sequences
        county_window_data: dict[str, list[dict]] = {f: [] for f in fips_list}

        for cm in range(1, m + 1):
            n_win = wpm[cm]
            if n_win == 0:
                continue
            cal_month, cal_year = campaign_to_calendar(cm, target_year)
            df_year = daily_cache.get(cal_year)
            if df_year is None:
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
            for wi, w_dict in enumerate(windows[:S]):
                for vi, vname in enumerate(ALL_VARS):
                    X_seq[obs_i, wi, vi] = w_dict.get(vname, np.nan)

    return X_seq, county_arr, year_arr

# ---------------------------------------------------------------------------
# Sequence cache: build once, save to disk
# ---------------------------------------------------------------------------
def get_or_build_sequences(
    freq: int,
    H: int,
    panel_fips: list[str],
    all_target_years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """Load cached sequences from disk, building them if needed."""
    if H == 12:
        return None, None, None

    cache_path = SEQ_DIR / f"seq_{freq}d_H{H:02d}_v2.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X_seq"].astype(np.float32),
            data["counties"],
            data["years"].astype(np.int32),
        )

    print(f"    Building sequences {freq}d H={H} ...", end=" ", flush=True)
    t0 = time.time()

    if freq == 30:
        X_seq, counties, years = build_sequences_30d(H, panel_fips)
    else:
        X_seq, counties, years = build_sequences_submonthly(
            freq, H, panel_fips, all_target_years
        )

    if X_seq is None:
        return None, None, None

    np.savez_compressed(
        cache_path,
        X_seq=X_seq, counties=counties, years=years,
    )
    print(f"done ({time.time()-t0:.1f}s), shape={X_seq.shape}", flush=True)
    return X_seq, counties, years

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
# Feature scaling
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
    mean = np.nanmean(flat_tr, axis=0)
    std  = np.nanstd(flat_tr, axis=0)
    std[std == 0] = 1.0

    def scale(X: np.ndarray) -> np.ndarray:
        n, s, f = X.shape
        flat = (X.reshape(-1, f) - mean) / std
        flat = np.nan_to_num(flat, nan=0.0)
        return flat.reshape(n, s, f).astype(np.float32)

    return scale(X_tr), scale(X_va)

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
def _xavier_init(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "weight" in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

class PretrainLSTM(nn.Module):
    """
    Used for the pre-training phase. No county embedding.
    input [batch, seq, 16] → LSTM(hidden_size, num_layers) → h_T [batch, hidden_size] → Linear(hidden_size, 1) → scalar
    """
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        _xavier_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMForecaster(nn.Module):
    """
    Used for fine-tuning. Has county embedding.
    input [batch, seq, 16] → LSTM(hidden_size, num_layers) → h_T [batch, hidden_size]
    county_idx [batch] → Embedding(135, 16) → e_c [batch, 16]
    concat([h_T, e_c]) → Linear(hidden_size+16, 1) → scalar
    """
    def __init__(self, hidden_size: int, num_layers: int, dropout: float, n_counties: int, emb_dim: int):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.county_emb = nn.Embedding(n_counties, emb_dim)
        
        # NOTE (future): Replace Linear head with: Linear(hidden+16, 32) → ReLU → Linear(32, 1)
        # This is the 'dense head' variant noted in decisions.md
        self.head = nn.Linear(hidden_size + emb_dim, 1)
        
        _xavier_init(self)

    def forward(self, x: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h_t = out[:, -1, :]
        e_c = self.county_emb(c_idx)
        merged = torch.cat([h_t, e_c], dim=-1)
        return self.head(merged).squeeze(-1)


# ---------------------------------------------------------------------------
# Training & validation helpers
# ---------------------------------------------------------------------------
def build_fold_arrays_v2(
    X_seq: np.ndarray,   # [n_obs, S, 16]
    counties: np.ndarray,
    years: np.ndarray,
    anomaly_df: pd.DataFrame,  # county_fips, crop_year, anomaly
    train_years: list[int],
    val_years: list[int],
    panel_fips: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Extract train/val arrays for one fold, scale features, return targets and county indices.
    """
    tr_mask = np.isin(years, train_years)
    va_mask = np.isin(years, val_years)

    if tr_mask.sum() == 0 or va_mask.sum() == 0:
        return None

    X_tr_raw = X_seq[tr_mask]
    X_va_raw = X_seq[va_mask]
    X_tr_sc, X_va_sc = scale_sequences(X_tr_raw, X_va_raw)

    anom_map = anomaly_df.set_index(["county_fips", "crop_year"])["anomaly"].to_dict()
    fips_to_idx = {f: i for i, f in enumerate(panel_fips)}

    def get_targets_and_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        targets = []
        indices = []
        for i in np.where(mask)[0]:
            fips = str(counties[i])
            key = (fips, int(years[i]))
            targets.append(anom_map.get(key, np.nan))
            indices.append(fips_to_idx.get(fips, 0))
        return np.array(targets, dtype=np.float32), np.array(indices, dtype=np.int64)

    y_tr, c_tr = get_targets_and_indices(tr_mask)
    y_va, c_va = get_targets_and_indices(va_mask)

    valid_tr = ~np.isnan(y_tr)
    valid_va = ~np.isnan(y_va)
    if valid_tr.sum() < 10 or valid_va.sum() < 1:
        return None

    return (
        X_tr_sc[valid_tr], y_tr[valid_tr], c_tr[valid_tr],
        X_va_sc[valid_va], y_va[valid_va], c_va[valid_va]
    )


def evaluate_pretrain(model: PretrainLSTM, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy()
    y_np = y.numpy()
    rmse = float(np.sqrt(np.mean((preds - y_np) ** 2)))
    return rmse

def evaluate_finetune(model: LSTMForecaster, X: torch.Tensor, c: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        preds = model(X, c).numpy()
    y_np = y.numpy()
    rmse = float(np.sqrt(np.mean((preds - y_np) ** 2)))
    ss_res = np.sum((y_np - preds) ** 2)
    ss_tot = np.sum(y_np ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2


def get_or_pretrain_weights(
    freq: int,
    H: int,
    hidden_size: int,
    num_layers: int,
    X_seq: np.ndarray,
    counties: np.ndarray,
    years: np.ndarray,
    yield_df: pd.DataFrame,
    panel_fips: list[str],
) -> dict:
    pretrain_path = PRETRAIN_DIR / f"pretrain_{freq}d_H{H:02d}_h{hidden_size}_l{num_layers}.pt"
    if pretrain_path.exists():
        return torch.load(pretrain_path, map_location="cpu", weights_only=True)

    print(f"      Running PRE-TRAINING for h={hidden_size} l={num_layers} H={H}...", flush=True)
    
    tp = fit_trends(yield_df, PRETRAIN_TRAIN_YEARS)
    anom_train = compute_anomaly(yield_df, tp, PRETRAIN_TRAIN_YEARS)
    anom_val   = compute_anomaly(yield_df, tp, PRETRAIN_VAL_YEARS)
    anom_all   = pd.concat([anom_train, anom_val], ignore_index=True)

    fold_data = build_fold_arrays_v2(
        X_seq, counties, years, anom_all,
        PRETRAIN_TRAIN_YEARS, PRETRAIN_VAL_YEARS, panel_fips
    )
    
    if fold_data is None:
        raise ValueError("Not enough pre-training data.")

    X_tr, y_tr, _, X_va, y_va, _ = fold_data
    
    model = PretrainLSTM(hidden_size, num_layers, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val_rmse = float("inf")
    patience = 15
    patience_left = patience
    best_state = None

    for epoch in range(200):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_rmse = evaluate_pretrain(model, X_va_t, y_va_t)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.lstm.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    print(f"      Pre-training finished. Best Val RMSE: {best_val_rmse:.4f}", flush=True)
    
    # Save the LSTM state dict (exclude head)
    torch.save(best_state, pretrain_path)
    return best_state


def fit_lstm_v2(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    c_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    c_va: np.ndarray,
    config: dict,
    pretrained_lstm_state: dict | None,
    n_counties: int,
    emb_dim: int,
    max_epochs: int = 100,
    patience: int = 7,
    seed: int = 42,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_size = config["hidden_size"]
    num_layers  = config["num_layers"]
    dropout     = config["dropout"]
    batch_size  = config["batch_size"]

    model = LSTMForecaster(hidden_size, num_layers, dropout, n_counties, emb_dim)
    
    if pretrained_lstm_state is not None:
        model.lstm.load_state_dict(pretrained_lstm_state, strict=False)

    optimizer = torch.optim.Adam([
        {"params": model.lstm.parameters(), "lr": LR_LSTM_FINETUNE},
        {"params": model.county_emb.parameters(), "lr": LR},
        {"params": model.head.parameters(), "lr": LR},
    ])
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    c_tr_t = torch.tensor(c_tr, dtype=torch.int64)
    
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    c_va_t = torch.tensor(c_va, dtype=torch.int64)

    dataset = TensorDataset(X_tr_t, c_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_rmse = float("inf")
    best_val_r2   = float("nan")
    patience_left = patience

    for epoch in range(max_epochs):
        model.train()
        for X_b, c_b, y_b in loader:
            optimizer.zero_grad()
            preds = model(X_b, c_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_rmse, val_r2 = evaluate_finetune(model, X_va_t, c_va_t, y_va_t)
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
    counties: np.ndarray,
    years: np.ndarray,
    yield_df: pd.DataFrame,
    all_years: list[int],
    panel_fips: list[str],
    checkpoint: dict,
    checkpoint_path: Path,
) -> dict[int, dict]:
    val_years = [y for y in all_years if VAL_START <= y <= VAL_END]
    results   = {}
    
    # Preload pre-trained weights to avoid re-training per config
    pretrained_cache = {}

    for cfg_idx, config in enumerate(GRID):
        key = checkpoint_key(freq, H, cfg_idx)
        if key in checkpoint:
            results[cfg_idx] = checkpoint[key]
            continue

        h = config["hidden_size"]
        l = config["num_layers"]
        
        # Load or run pre-training for this h, l combination
        pt_key = (h, l)
        if pt_key not in pretrained_cache:
            pretrained_cache[pt_key] = get_or_pretrain_weights(
                freq, H, h, l, X_seq, counties, years, yield_df, panel_fips
            )
        pretrained_lstm_state = pretrained_cache[pt_key]

        t_cfg = time.time()
        fold_rmses = []
        fold_r2s   = []

        for val_year in val_years:
            train_years = [y for y in all_years if y < val_year]
            tp = fit_trends(yield_df, train_years)

            anom_train = compute_anomaly(yield_df, tp, train_years)
            anom_val   = compute_anomaly(yield_df, tp, [val_year])
            anom_all   = pd.concat([anom_train, anom_val], ignore_index=True)

            fold_data = build_fold_arrays_v2(
                X_seq, counties, years, anom_all,
                train_years, [val_year], panel_fips
            )
            if fold_data is None:
                continue

            X_tr, y_tr, c_tr, X_va, y_va, c_va = fold_data
            
            val_rmse, val_r2 = fit_lstm_v2(
                X_tr, y_tr, c_tr, X_va, y_va, c_va,
                config, pretrained_lstm_state,
                n_counties=N_COUNTIES, emb_dim=COUNTY_EMBEDDING_DIM
            )
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
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    freqs: list[int],
    horizons: list[int],
) -> None:
    t0 = time.time()
    print("=== LSTM Grid Search Pipeline (v2 — County Embedding & Pre-training) ===", flush=True)
    print(f"Frequencies: {freqs}-day | Horizons: {horizons}", flush=True)
    print(f"Grid: {len(GRID)} configs | Validation: {VAL_START}–{VAL_END} | Seed: 42", flush=True)
    
    checkpoint_path = REPORT_DIR / "grid_checkpoint_v2.json"
    print(f"Checkpoint: {checkpoint_path}", flush=True)

    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    panel_fips  = sorted(yield_df["county_fips"].unique().tolist())
    all_years   = sorted(yield_df["year"].unique().tolist())
    
    print(f"Panel: {len(panel_fips)} counties, {min(all_years)}–{max(all_years)}", flush=True)

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

            X_seq, counties, years = get_or_build_sequences(
                freq, H, panel_fips, all_years
            )
            if X_seq is None:
                print(f"  SKIP H={H}: no data.", flush=True)
                continue

            print(f"  Grid search: {len(GRID)} configs × {VAL_END-VAL_START+1} val folds...", flush=True)
            grid_search_one(
                freq=freq, H=H,
                X_seq=X_seq, counties=counties, years=years,
                yield_df=yield_df, all_years=all_years, panel_fips=panel_fips,
                checkpoint=checkpoint, checkpoint_path=checkpoint_path,
            )

    print(f"\nPipeline finished in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM grid search v2.")
    parser.add_argument("--freq", type=int, nargs="+", default=[30, 10, 5],
                        help="Window frequency in days.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 7, 2, 3, 5, 6, 8, 9, 10, 11],
                        help="List of horizons (H).")
    args = parser.parse_args()
    
    run_pipeline(freqs=args.freq, horizons=args.horizons)
