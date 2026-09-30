"""
run_lstm_pipeline_v2.py — Resumable LSTM Pipeline with Temporal Attention, County Embedding, and Dense MLP Head.

Features:
- 17 Bioclimatic indicators per time step (including EDD30, exactly matching tabular models).
- Bahdanau Temporal Attention layer along campaign time steps:
    e_t = v^T tanh(W_a h_t + b_a), alpha_t = softmax(e_t), c = sum_t (alpha_t * h_t)
  Extracts native explainability weights alpha_t for Chapter 5/6 thesis figures.
- County Embedding (d=16) at output head for parsimonious spatial encoding.
- Dense MLP Head: Linear(hidden + 16 (+ 1), 32) -> ReLU -> Dropout -> Linear(32, 1)
  capturing non-linear climate x geography interactions.
- Dual specifications:
  1. Weather-Only (Tier 4 core benchmark)
  2. Integrated (Weather + train-only detrended prior-year yield anomaly epsilon_{t-1})
- Two-phase training:
  1. Universal pre-training on pooled panel (1951-1979 train, 1980-1984 val).
  2. Expanding-window fine-tuning with discriminative learning rates (ULMFiT style).
- Resumable checkpointing for both validation tuning and test evaluation.

Usage:
  python output/scripts/run_lstm_pipeline_v2.py --mode tune --freq 30 --horizons 1 2 3
  python output/scripts/run_lstm_pipeline_v2.py --mode test --freq 30 --horizons 1 2 3
  python output/scripts/run_lstm_pipeline_v2.py --mode all --freq 30 --grid standard
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
ROOT         = Path(__file__).resolve().parents[2]
WEATHER_DIR  = ROOT / "data" / "weather" / "daily_counties"
TARGET_CSV   = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
FEAT_DIR     = ROOT / "output" / "data" / "processed" / "model_datasets"
SEQ_DIR      = ROOT / "output" / "data" / "processed" / "lstm_sequences"
PRETRAIN_DIR = SEQ_DIR
REPORT_DIR   = ROOT / "work" / "reports" / "experiments" / "lstm"
OUTPUT_DIR   = ROOT / "output" / "data" / "processed"

SEQ_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Partition constants (strictly matching research protocol)
# ---------------------------------------------------------------------------
BASE_TRAIN_START = 1951
VAL_START        = 1985
VAL_END          = 1995
TEST_START       = 1996
TEST_END         = 2025

# ---------------------------------------------------------------------------
# Hyperparameter Constants & Grids
# ---------------------------------------------------------------------------
LR = 3e-4                   # Khaki & Wang (2019, 2020)
LR_LSTM_FINETUNE = 3e-5     # 1/10 LR for pre-trained weights during fine-tuning
COUNTY_EMBEDDING_DIM = 16   # Fixed spatial latent dimension for 135 counties
N_COUNTIES = 135

def build_pruned_grid() -> list[dict]:
    """
    Smart pruned grid (44 configs):
    Keeps all parameter levels (hidden: 64,128,256; layers: 1,2,3,4; dropout: 0.0,0.2,0.4; batch: 25,64)
    while excluding only pathologically bad or redundant combinations:
    1. l=1 with multiple dropouts (in nn.LSTM dropout only applies between recurrent layers).
    2. l>=3 with dropout=0.0 (deep networks without dropout overfit severely on small panel data).
    3. l=4 with hidden=256 (>1.8M params, mathematically untrainable on 5k observations).
    """
    configs = []
    for h in [64, 128, 256]:
        for l in [1, 2, 3, 4]:
            for d in [0.0, 0.2, 0.4]:
                for b in [25, 64]:
                    if l == 1 and d > 0.0:
                        continue
                    if l >= 3 and d == 0.0:
                        continue
                    if l == 4 and h == 256:
                        continue
                    configs.append({"hidden_size": h, "num_layers": l, "dropout": d, "batch_size": b})
    return configs

GRIDS = {
    # Smart pruned 44-configuration grid (no parameter dropped, only pathological combos removed)
    "pruned": build_pruned_grid(),
    # Full unconstrained 72-configuration grid
    "full": [
        {"hidden_size": h, "num_layers": l, "dropout": d, "batch_size": b}
        for h, l, d, b in product([64, 128, 256], [1, 2, 3, 4], [0.0, 0.2, 0.4], [25, 64])
    ],
    # Streamlined 18-configuration grid
    "standard": [
        {"hidden_size": h, "num_layers": l, "dropout": d, "batch_size": 64}
        for h, l, d in product([64, 128, 256], [1, 2, 3], [0.1, 0.3])
    ],
    # Rapid 8-configuration grid
    "compact": [
        {"hidden_size": h, "num_layers": l, "dropout": d, "batch_size": 64}
        for h, l, d in product([64, 128], [1, 2], [0.1, 0.3])
    ],
}

# Pre-training window: 1951-1979 train, 1980-1984 validation (strictly before 1985)
PRETRAIN_TRAIN_YEARS = list(range(1951, 1980))
PRETRAIN_VAL_YEARS   = list(range(1980, 1985))

# ---------------------------------------------------------------------------
# Bioclimatic Indicator Definitions (17 indicators, identical to tabular models)
# ---------------------------------------------------------------------------
FLUX_RAW = {
    "total_precipitation":               "P",
    "surface_solar_radiation_downwards": "SSRD",
    "et0_fao56":                         "ET0",
    "p_minus_et0":                       "P_minus_ET0",
    "growing_degree_days":               "GDD",
    "heat_day_30":                       "HD30",
    "heat_day_35":                       "HD35",
    "edd_30":                            "EDD30",
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

FLUX_NAMES  = list(FLUX_RAW.values())                  # 8 flux indicators (summed)
STATE_NAMES = list(STATE_RAW.values()) + ["SM_root"]   # 9 state indicators (averaged)
ALL_VARS    = FLUX_NAMES + STATE_NAMES                 # 17 total indicators in fixed order
INPUT_SIZE  = len(ALL_VARS)                            # 17

def campaign_to_calendar(cm: int, target_year: int) -> tuple[int, int]:
    """Map campaign month (1=Nov Y-1 ... 12=Oct Y) to calendar (month, year)."""
    cal_month = (10 + cm) % 12 or 12
    cal_year  = target_year - 1 if cm <= 2 else target_year
    return cal_month, cal_year

CAMPAIGN_MONTH_DAYS = {
    1: 30, 2: 31, 3: 31, 4: 28, 5: 31, 6: 30,
    7: 31, 8: 30, 9: 31, 10: 31, 11: 30, 12: 31,
}

def windows_per_month(W: int) -> dict[int, int]:
    """Number of complete W-day windows per campaign month (non-leap Feb=28)."""
    return {cm: CAMPAIGN_MONTH_DAYS[cm] // W for cm in range(1, 13)}

def seq_len_at_H(H: int, W: int) -> int:
    """Total number of elapsed windows at horizon H for window frequency W."""
    if H == 12:
        return 0
    m = 13 - H
    wpm = windows_per_month(W)
    return sum(wpm[cm] for cm in range(1, m + 1))


# ---------------------------------------------------------------------------
# Sequence Building (30-day from tabular parquet, sub-monthly from daily parquet)
# ---------------------------------------------------------------------------
def build_sequences_30d(H: int, panel_fips: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 30-day sequences for horizon H from existing tabular feature matrix."""
    if H == 12:
        return None, None, None

    feat = pd.read_parquet(FEAT_DIR / f"features_H{H:02d}.parquet")
    feat["county_fips"] = feat["county_fips"].astype(str)
    feat = feat[feat["county_fips"].isin(set(panel_fips))]
    feat = feat.sort_values(["crop_year", "county_fips"]).reset_index(drop=True)

    m = 13 - H
    X_seq = np.zeros((len(feat), m, INPUT_SIZE), dtype=np.float32)
    for cm in range(1, m + 1):
        for vi, vname in enumerate(ALL_VARS):
            col = f"{vname}_m{cm:02d}"
            X_seq[:, cm - 1, vi] = feat[col].values.astype(np.float32)

    counties = feat["county_fips"].values
    years    = feat["crop_year"].values.astype(np.int32)
    return X_seq, counties, years


def _load_daily_year(year: int) -> pd.DataFrame | None:
    path = WEATHER_DIR / f"county_daily_{year}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["county_fips"] = df["county_fips"].astype(str)
    # Calculate edd_30 on the fly if not stored in parquet
    if "edd_30" not in df.columns and "air_temperature_maximum" in df.columns:
        df["edd_30"] = np.maximum(0.0, df["air_temperature_maximum"] - 30.0)
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
    """Build W-day sequences for horizon H from daily county parquets."""
    if H == 12:
        return None, None, None

    wpm = windows_per_month(W)
    m   = 13 - H
    S   = seq_len_at_H(H, W)

    fips_list  = sorted(panel_fips)
    fips_index = {f: i for i, f in enumerate(fips_list)}
    n_counties = len(fips_list)
    n_years    = len(all_target_years)
    n_obs      = n_counties * n_years

    X_seq      = np.full((n_obs, S, INPUT_SIZE), np.nan, dtype=np.float32)
    county_arr = np.empty(n_obs, dtype=object)
    year_arr   = np.zeros(n_obs, dtype=np.int32)

    daily_cache: dict[int, pd.DataFrame] = {}

    for yi, target_year in enumerate(all_target_years):
        for cal_year in (target_year - 1, target_year):
            if cal_year not in daily_cache:
                df = _load_daily_year(cal_year)
                daily_cache[cal_year] = df
                to_drop = [y for y in list(daily_cache) if y < target_year - 1]
                for y in to_drop:
                    del daily_cache[y]

        county_window_data: dict[str, list[dict]] = {f: [] for f in fips_list}

        for cm in range(1, m + 1):
            n_win = wpm[cm]
            if n_win == 0:
                continue
            cal_month, cal_year = campaign_to_calendar(cm, target_year)
            df_year = daily_cache.get(cal_year)
            if df_year is None:
                continue

            mask = df_year["date"].dt.month == cal_month
            df_month = df_year.loc[mask].copy().sort_values("date")

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


def get_or_build_sequences(
    freq: int,
    H: int,
    panel_fips: list[str],
    all_target_years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """Load cached sequences from disk, building and caching if needed."""
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

    print(f"    Building sequences {freq}d H={H} (17 vars)...", end=" ", flush=True)
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
# Detrending (OLS, train-only per county)
# ---------------------------------------------------------------------------
def fit_trends(yield_df: pd.DataFrame, train_years: list[int]) -> dict[str, tuple[float, float]]:
    train = yield_df[yield_df["year"].isin(train_years)]
    params = {}
    for fips, grp in train.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        t_bar  = t_vals.mean()
        y_bar  = y_vals.mean()
        beta   = np.sum((t_vals - t_bar) * (y_vals - y_bar)) / np.sum((t_vals - t_bar) ** 2)
        alpha  = y_bar - beta * t_bar
        params[str(fips)] = (float(alpha), float(beta))
    return params


def compute_anomaly(yield_df: pd.DataFrame, trend_params: dict, years: list[int]) -> pd.DataFrame:
    sub = yield_df[yield_df["year"].isin(years)].copy()
    sub["trend"]   = sub.apply(lambda r: trend_params[str(r["county_fips"])][0]
                                + trend_params[str(r["county_fips"])][1] * r["year"], axis=1)
    sub["anomaly"] = sub["yield_bu_per_acre"] - sub["trend"]
    return sub[["county_fips", "year", "anomaly", "trend"]].rename(columns={"year": "crop_year"})


# ---------------------------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------------------------
def scale_sequences(
    X_tr: np.ndarray,
    X_va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """StandardScaler fit on training set, applied to val/test."""
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
# Neural Network Modules: Attention, PretrainLSTM, LSTMForecaster
# ---------------------------------------------------------------------------
def _xavier_init(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)


class TemporalAttention(nn.Module):
    """
    Bahdanau Temporal Attention across campaign time steps:
      e_t = v_a^T tanh(W_a h_t + b_a)
      alpha_t = softmax(e_t)
      context = sum_{t=1}^T alpha_t * h_t
    """
    def __init__(self, hidden_size: int, attn_size: int = 32):
        super().__init__()
        self.w_a = nn.Linear(hidden_size, attn_size, bias=True)
        self.v_a = nn.Linear(attn_size, 1, bias=False)
        _xavier_init(self)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            context: [batch_size, hidden_size]
            attn_weights: [batch_size, seq_len]
        """
        u = torch.tanh(self.w_a(hidden_states))          # [B, S, attn_size]
        scores = self.v_a(u).squeeze(-1)                 # [B, S]
        attn_weights = torch.softmax(scores, dim=-1)     # [B, S]
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1) # [B, H]
        return context, attn_weights


def get_activation(act_name: str) -> nn.Module:
    act = act_name.lower()
    if act == "gelu":
        return nn.GELU()
    elif act in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(0.1)
    elif act == "relu":
        return nn.ReLU()
    else:
        raise ValueError(f"Unknown activation: {act_name}")


class PretrainLSTM(nn.Module):
    """
    Universal LSTM pre-trained on panel data (1951-1979) without county embedding.
    """
    def __init__(self, hidden_size: int, num_layers: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_size, attn_size=32)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        _xavier_init(self)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        context, attn = self.attention(out)
        pred = self.head(context).squeeze(-1)
        return pred, attn


class LSTMForecaster(nn.Module):
    """
    Main LSTM Forecaster with Temporal Attention, County Embedding, and Dense MLP Head.
    Supports both Weather-Only and Integrated (with lagged yield anomaly).
    """
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_counties: int = N_COUNTIES,
        emb_dim: int = COUNTY_EMBEDDING_DIM,
        include_lag_yield: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_size, attn_size=32)
        self.county_emb = nn.Embedding(n_counties, emb_dim)
        self.include_lag_yield = include_lag_yield

        in_head = hidden_size + emb_dim + (1 if include_lag_yield else 0)
        self.head = nn.Sequential(
            nn.Linear(in_head, 32),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        _xavier_init(self)

    def forward(
        self,
        x: torch.Tensor,
        c_idx: torch.Tensor,
        lag_y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        context, attn = self.attention(out)
        e_c = self.county_emb(c_idx)

        if self.include_lag_yield and lag_y is not None:
            features = torch.cat([context, e_c, lag_y], dim=-1)
        else:
            features = torch.cat([context, e_c], dim=-1)

        pred = self.head(features).squeeze(-1)
        return pred, attn


# ---------------------------------------------------------------------------
# Training & Evaluation Helpers
# ---------------------------------------------------------------------------
def build_fold_arrays_v2(
    X_seq: np.ndarray,
    counties: np.ndarray,
    years: np.ndarray,
    yield_df: pd.DataFrame,
    trend_params: dict,
    train_years: list[int],
    val_years: list[int],
    panel_fips: list[str],
    include_lag_yield: bool = False,
) -> tuple | None:
    """Extract train/val arrays, scale features, and build targets + county indices + lag anomaly."""
    tr_mask = np.isin(years, train_years)
    va_mask = np.isin(years, val_years)

    if tr_mask.sum() == 0 or va_mask.sum() == 0:
        return None

    X_tr_sc, X_va_sc = scale_sequences(X_seq[tr_mask], X_seq[va_mask])

    # Compute current-year and prior-year anomalies strictly using training trend params
    all_rel_years = sorted(list(set(train_years + val_years + [min(train_years) - 1])))
    anom_df = compute_anomaly(yield_df, trend_params, all_rel_years)
    anom_map = anom_df.set_index(["county_fips", "crop_year"])["anomaly"].to_dict()
    fips_to_idx = {f: i for i, f in enumerate(panel_fips)}

    def extract_fold_vectors(mask: np.ndarray):
        targets = []
        indices = []
        lags    = []
        for i in np.where(mask)[0]:
            fips = str(counties[i])
            yr   = int(years[i])
            targets.append(anom_map.get((fips, yr), np.nan))
            indices.append(fips_to_idx.get(fips, 0))
            if include_lag_yield:
                # Prior year anomaly epsilon_{c, yr-1}
                lags.append(anom_map.get((fips, yr - 1), 0.0))
        return (
            np.array(targets, dtype=np.float32),
            np.array(indices, dtype=np.int64),
            np.array(lags, dtype=np.float32).reshape(-1, 1) if include_lag_yield else None,
        )

    y_tr, c_tr, lag_tr = extract_fold_vectors(tr_mask)
    y_va, c_va, lag_va = extract_fold_vectors(va_mask)

    valid_tr = ~np.isnan(y_tr)
    valid_va = ~np.isnan(y_va)
    if valid_tr.sum() < 10 or valid_va.sum() < 1:
        return None

    # Scale lag yield if included
    if include_lag_yield and lag_tr is not None and lag_va is not None:
        mean_lag = np.nanmean(lag_tr[valid_tr])
        std_lag  = np.nanstd(lag_tr[valid_tr])
        std_lag  = 1.0 if std_lag == 0 else std_lag
        lag_tr = (lag_tr - mean_lag) / std_lag
        lag_va = (lag_va - mean_lag) / std_lag

    return (
        X_tr_sc[valid_tr], y_tr[valid_tr], c_tr[valid_tr], lag_tr[valid_tr] if include_lag_yield else None,
        X_va_sc[valid_va], y_va[valid_va], c_va[valid_va], lag_va[valid_va] if include_lag_yield else None,
    )


def evaluate_pretrain(model: PretrainLSTM, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds, _ = model(X)
        preds_np = preds.numpy()
    y_np = y.numpy()
    return float(np.sqrt(np.mean((preds_np - y_np) ** 2)))


def evaluate_finetune(
    model: LSTMForecaster,
    X: torch.Tensor,
    c: torch.Tensor,
    y: torch.Tensor,
    lag: torch.Tensor | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        preds, attn = model(X, c, lag)
        preds_np = preds.numpy()
        attn_np  = attn.numpy()
    y_np = y.numpy()
    rmse = float(np.sqrt(np.mean((preds_np - y_np) ** 2)))
    ss_res = np.sum((y_np - preds_np) ** 2)
    ss_tot = np.sum(y_np ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2, preds_np, attn_np


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
    activation: str = "gelu",
) -> dict:
    """Pre-train universal LSTM on 1951-1979 and cache weights."""
    pretrain_path = PRETRAIN_DIR / f"pretrain_{freq}d_H{H:02d}_h{hidden_size}_l{num_layers}_{activation}_v2.pt"
    if pretrain_path.exists():
        return torch.load(pretrain_path, map_location="cpu", weights_only=True)

    print(f"      Pre-training LSTM+Attention h={hidden_size} l={num_layers} H={H} ({activation})...", flush=True)
    tp = fit_trends(yield_df, PRETRAIN_TRAIN_YEARS)

    fold_data = build_fold_arrays_v2(
        X_seq, counties, years, yield_df, tp,
        PRETRAIN_TRAIN_YEARS, PRETRAIN_VAL_YEARS, panel_fips,
        include_lag_yield=False,
    )
    if fold_data is None:
        raise ValueError("Not enough pre-training data.")

    X_tr, y_tr, _, _, X_va, y_va, _, _ = fold_data

    model = PretrainLSTM(hidden_size, num_layers, dropout=0.2, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
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
        for X_b, y_b in loader:
            optimizer.zero_grad()
            preds, _ = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_rmse = evaluate_pretrain(model, X_va_t, y_va_t)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                "lstm": model.lstm.state_dict(),
                "attention": model.attention.state_dict(),
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    print(f"      Pre-training finished. Best Val RMSE: {best_val_rmse:.4f}", flush=True)
    torch.save(best_state, pretrain_path)
    return best_state


def fit_lstm_v2(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    c_tr: np.ndarray,
    lag_tr: np.ndarray | None,
    X_va: np.ndarray,
    y_va: np.ndarray,
    c_va: np.ndarray,
    lag_va: np.ndarray | None,
    config: dict,
    pretrained_state: dict | None,
    include_lag_yield: bool = False,
    activation: str = "gelu",
    max_epochs: int = 100,
    patience: int = 7,
    seed: int = 42,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Train/Fine-tune LSTMForecaster for one fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_size = config["hidden_size"]
    num_layers  = config["num_layers"]
    dropout     = config["dropout"]
    batch_size  = config["batch_size"]

    model = LSTMForecaster(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_counties=N_COUNTIES,
        emb_dim=COUNTY_EMBEDDING_DIM,
        include_lag_yield=include_lag_yield,
        activation=activation,
    )

    if pretrained_state is not None:
        model.lstm.load_state_dict(pretrained_state["lstm"], strict=False)
        model.attention.load_state_dict(pretrained_state["attention"], strict=False)

    optimizer = torch.optim.Adam([
        {"params": model.lstm.parameters(), "lr": LR_LSTM_FINETUNE},
        {"params": model.attention.parameters(), "lr": LR_LSTM_FINETUNE},
        {"params": model.county_emb.parameters(), "lr": LR},
        {"params": model.head.parameters(), "lr": LR},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    c_tr_t = torch.tensor(c_tr, dtype=torch.int64)
    lag_tr_t = torch.tensor(lag_tr, dtype=torch.float32) if include_lag_yield else None

    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    c_va_t = torch.tensor(c_va, dtype=torch.int64)
    lag_va_t = torch.tensor(lag_va, dtype=torch.float32) if include_lag_yield else None

    if include_lag_yield:
        dataset = TensorDataset(X_tr_t, c_tr_t, y_tr_t, lag_tr_t)
    else:
        dataset = TensorDataset(X_tr_t, c_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_rmse = float("inf")
    best_val_r2   = float("nan")
    best_preds    = None
    best_attn     = None
    patience_left = patience

    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            if include_lag_yield:
                X_b, c_b, y_b, lag_b = batch
                preds, _ = model(X_b, c_b, lag_b)
            else:
                X_b, c_b, y_b = batch
                preds, _ = model(X_b, c_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_rmse, val_r2, val_preds, val_attn = evaluate_finetune(model, X_va_t, c_va_t, y_va_t, lag_va_t)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_r2   = val_r2
            best_preds    = val_preds
            best_attn     = val_attn
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return best_val_rmse, best_val_r2, best_preds, best_attn


# ---------------------------------------------------------------------------
# Checkpoint Helpers
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
# Validation Tuning (Expanding Window 1985-1995)
# ---------------------------------------------------------------------------
def run_validation_tuning(
    freqs: list[int],
    horizons: list[int],
    grid_name: str = "pruned",
    activation: str = "gelu",
) -> dict:
    """Run expanding validation search across 1985-1995 to select optimal hyperparameters."""
    checkpoint_path = REPORT_DIR / "grid_checkpoint_v2.json"
    checkpoint = load_checkpoint(checkpoint_path)
    grid = GRIDS[grid_name]

    print(f"\n{'='*70}", flush=True)
    print(f"STAGE 1: VALIDATION TUNING (1985–1995 Expanding Window)", flush=True)
    print(f"Grid: {grid_name.upper()} ({len(grid)} configs) | Activation: {activation.upper()} | Checkpoint: {checkpoint_path.name}", flush=True)
    print(f"{'='*70}", flush=True)

    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    panel_fips = sorted(yield_df["county_fips"].unique().tolist())
    all_years  = sorted(yield_df["year"].unique().tolist())
    val_years  = [y for y in all_years if VAL_START <= y <= VAL_END]

    for freq in freqs:
        print(f"\n>> FREQUENCY: {freq}-day sequences", flush=True)
        for H in horizons:
            if H == 12:
                continue
            print(f"\n  --- Horizon H={H:02d} ({seq_len_at_H(H, freq)} steps) ---", flush=True)
            X_seq, counties, years = get_or_build_sequences(freq, H, panel_fips, all_years)
            if X_seq is None:
                continue

            pretrained_cache = {}

            for cfg_idx, config in enumerate(grid):
                key = checkpoint_key(freq, H, cfg_idx)
                if key in checkpoint:
                    continue

                h = config["hidden_size"]
                l = config["num_layers"]
                pt_key = (h, l)
                if pt_key not in pretrained_cache:
                    pretrained_cache[pt_key] = get_or_pretrain_weights(
                        freq, H, h, l, X_seq, counties, years, yield_df, panel_fips, activation=activation
                    )
                pt_state = pretrained_cache[pt_key]

                t_cfg = time.time()
                fold_rmses = []
                fold_r2s   = []

                for val_year in val_years:
                    train_years = [y for y in all_years if y < val_year]
                    tp = fit_trends(yield_df, train_years)
                    fold_data = build_fold_arrays_v2(
                        X_seq, counties, years, yield_df, tp,
                        train_years, [val_year], panel_fips,
                        include_lag_yield=False,
                    )
                    if fold_data is None:
                        continue

                    X_tr, y_tr, c_tr, _, X_va, y_va, c_va, _ = fold_data
                    v_rmse, v_r2, _, _ = fit_lstm_v2(
                        X_tr, y_tr, c_tr, None, X_va, y_va, c_va, None,
                        config, pt_state, include_lag_yield=False, activation=activation,
                    )
                    fold_rmses.append(v_rmse)
                    fold_r2s.append(v_r2)

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
                checkpoint[key] = result
                save_checkpoint(checkpoint, checkpoint_path)

                print(
                    f"      cfg {cfg_idx+1:02d}/{len(grid)}"
                    f"  h={config['hidden_size']} l={config['num_layers']}"
                    f"  dr={config['dropout']} bs={config['batch_size']}"
                    f"  → val_rmse={pooled_rmse:.4f}  ({elapsed:.0f}s)",
                    flush=True,
                )

    return checkpoint


# ---------------------------------------------------------------------------
# Out-of-Sample Test Evaluation (1996-2025 Expanding Window)
# ---------------------------------------------------------------------------
def run_test_forecast(
    freq: int,
    horizons: list[int],
    variants: list[str],
    activation: str = "gelu",
) -> None:
    """
    Run 30 expanding folds (1996-2025) on test partition using optimal validation config.
    Evaluates Weather-Only and/or Integrated, saving predictions, metrics, and attention weights.
    """
    chk_path = REPORT_DIR / "grid_checkpoint_v2.json"
    if not chk_path.exists():
        print(f"ERROR: {chk_path} not found. Run validation tuning first (--mode tune).")
        return

    checkpoint = load_checkpoint(chk_path)
    test_chk_path = REPORT_DIR / "test_checkpoint_v2.json"
    test_checkpoint = load_checkpoint(test_chk_path)

    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    panel_fips = sorted(yield_df["county_fips"].unique().tolist())
    all_years  = sorted(yield_df["year"].unique().tolist())
    test_years = [y for y in all_years if TEST_START <= y <= TEST_END]

    print(f"\n{'='*70}", flush=True)
    print(f"STAGE 2: OUT-OF-SAMPLE TEST FORECAST (1996–2025)", flush=True)
    print(f"Frequency: {freq}-day | Horizons: {horizons} | Variants: {variants} | Activation: {activation.upper()}", flush=True)
    print(f"{'='*70}", flush=True)

    all_preds_records = []
    all_attn_records  = []
    all_metrics_records = []

    for H in horizons:
        if H == 12:
            continue
        print(f"\n>> Evaluating Horizon H={H:02d} ({freq}-day windows)...", flush=True)

        # Find best validation config for (freq, H)
        prefix = f"{freq}d_H{H:02d}_cfg"
        matching_cfgs = [v for k, v in checkpoint.items() if k.startswith(prefix) and not np.isnan(v["val_rmse"])]
        if not matching_cfgs:
            print(f"  No validation results for {freq}d H={H}. Falling back to default config.")
            best_config = {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "batch_size": 64}
        else:
            best_match = min(matching_cfgs, key=lambda x: x["val_rmse"])
            best_config = best_match["config"]
            print(f"  Best Val Config: h={best_config['hidden_size']} l={best_config['num_layers']} "
                  f"dr={best_config['dropout']} bs={best_config['batch_size']} (RMSE={best_match['val_rmse']:.4f})")

        X_seq, counties, years = get_or_build_sequences(freq, H, panel_fips, all_years)
        if X_seq is None:
            continue

        h = best_config["hidden_size"]
        l = best_config["num_layers"]
        pt_state = get_or_pretrain_weights(freq, H, h, l, X_seq, counties, years, yield_df, panel_fips, activation=activation)

        for variant in variants:
            include_lag = (variant == "integrated")
            model_tag = f"LSTM_{variant}"

            y_trues_all = []
            y_preds_all = []

            for test_y in test_years:
                test_key = f"{variant}_{freq}d_H{H:02d}_{test_y}"
                train_years = [y for y in all_years if y < test_y]
                tp = fit_trends(yield_df, train_years)

                fold_data = build_fold_arrays_v2(
                    X_seq, counties, years, yield_df, tp,
                    train_years, [test_y], panel_fips,
                    include_lag_yield=include_lag,
                )
                if fold_data is None:
                    continue

                X_tr, y_tr, c_tr, lag_tr, X_te, y_te, c_te, lag_te = fold_data

                if test_key in test_checkpoint:
                    res = test_checkpoint[test_key]
                    y_pred_te = np.array(res["preds"])
                    attn_te   = np.array(res["attn"]) if "attn" in res else None
                else:
                    _, _, y_pred_te, attn_te = fit_lstm_v2(
                        X_tr, y_tr, c_tr, lag_tr, X_te, y_te, c_te, lag_te,
                        best_config, pt_state, include_lag_yield=include_lag, activation=activation,
                    )
                    test_checkpoint[test_key] = {
                        "preds": [float(p) for p in y_pred_te],
                        "attn":  attn_te.tolist() if attn_te is not None else None,
                    }
                    save_checkpoint(test_checkpoint, test_chk_path)

                # Fetch trends for actual yield calculation
                sub_trend = compute_anomaly(yield_df, tp, [test_y])
                trend_map = sub_trend.set_index("county_fips")["trend"].to_dict()

                # Record per-county predictions and attention weights
                fips_idx_to_fips = {i: f for i, f in enumerate(panel_fips)}
                for i in range(len(y_te)):
                    c_fips = fips_idx_to_fips.get(int(c_te[i]), "unknown")
                    tr_val = trend_map.get(c_fips, np.nan)
                    all_preds_records.append({
                        "model": model_tag,
                        "freq": freq,
                        "H": H,
                        "year": test_y,
                        "county_fips": c_fips,
                        "y_true_anomaly": float(y_te[i]),
                        "y_pred_anomaly": float(y_pred_te[i]),
                        "trend": float(tr_val),
                        "y_true_yield": float(y_te[i] + tr_val),
                        "y_pred_yield": float(y_pred_te[i] + tr_val),
                    })
                    if attn_te is not None:
                        all_attn_records.append({
                            "model": model_tag,
                            "freq": freq,
                            "H": H,
                            "year": test_y,
                            "county_fips": c_fips,
                            "attn_weights": [round(float(w), 5) for w in attn_te[i]],
                        })

                y_trues_all.extend(y_te)
                y_preds_all.extend(y_pred_te)

            # Compute pooled out-of-sample metrics
            y_t = np.array(y_trues_all)
            y_p = np.array(y_preds_all)
            rmse_val = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
            mae_val  = float(np.mean(np.abs(y_t - y_p)))
            ss_res   = np.sum((y_t - y_p) ** 2)
            ss_tot   = np.sum(y_t ** 2)
            r2_val   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            all_metrics_records.append({
                "model": model_tag,
                "freq": freq,
                "H": H,
                "R2_OOS": round(r2_val, 4),
                "RMSE": round(rmse_val, 4),
                "MAE": round(mae_val, 4),
                "n_obs": len(y_t),
            })
            print(f"    [{model_tag:18s}] R2_OOS: {r2_val:.4f} | RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f}", flush=True)

    # Save output artifacts
    if all_preds_records:
        df_p = pd.DataFrame(all_preds_records)
        df_p.to_parquet(OUTPUT_DIR / f"predictions_lstm_{freq}d.parquet", index=False)
        print(f"\nSaved predictions to output/data/processed/predictions_lstm_{freq}d.parquet", flush=True)

    if all_attn_records:
        df_a = pd.DataFrame(all_attn_records)
        df_a.to_parquet(OUTPUT_DIR / f"attention_weights_lstm_{freq}d.parquet", index=False)
        print(f"Saved attention weights to output/data/processed/attention_weights_lstm_{freq}d.parquet", flush=True)

    if all_metrics_records:
        df_m = pd.DataFrame(all_metrics_records)
        df_m.to_csv(OUTPUT_DIR / f"evaluation_summary_lstm_{freq}d.csv", index=False)
        print(f"Saved evaluation metrics to output/data/processed/evaluation_summary_lstm_{freq}d.csv", flush=True)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM pipeline v2 with Attention and County Embedding.")
    parser.add_argument("--mode", choices=["tune", "test", "all"], default="tune",
                        help="Execution mode: 'tune' (validation search), 'test' (test evaluation), 'all' (both).")
    parser.add_argument("--freq", type=int, nargs="+", default=[30],
                        help="Window frequency in days (30, 10, or 5).")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 10],
                        help="List of forecast horizons H (default: 1 3 6 10).")
    parser.add_argument("--grid", choices=["pruned", "full", "standard", "compact"], default="pruned",
                        help="Hyperparameter grid size (pruned=44, full=72, standard=18, compact=8).")
    parser.add_argument("--activation", choices=["gelu", "relu", "leaky_relu"], default="gelu",
                        help="MLP head activation function (gelu, relu, leaky_relu). Default: gelu.")
    parser.add_argument("--variants", nargs="+", choices=["weather", "integrated"], default=["weather", "integrated"],
                        help="Model variants to evaluate in test mode ('weather', 'integrated').")
    args = parser.parse_args()

    if args.mode in ("tune", "all"):
        run_validation_tuning(freqs=args.freq, horizons=args.horizons, grid_name=args.grid, activation=args.activation)

    if args.mode in ("test", "all"):
        for f in args.freq:
            run_test_forecast(freq=f, horizons=args.horizons, variants=args.variants, activation=args.activation)
