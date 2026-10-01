"""
run_lstm_pipeline.py -- Consolidated Production LSTM Pipeline for Soybean Yield Forecasting.

Key Features & Methodological Alignments:
1. Spatial Encoding: continuous lat_norm + lon_norm concatenated to 17 bioclimatic indicators
   at every time step (input_size = 19). Empirically superior to discrete county embeddings.
2. Architecture: LSTM + LayerNorm + Bahdanau Temporal Attention + Dense MLP Head (GELU)
   + He/Kaiming initialization + AdamW (weight_decay=1e-4).
3. Partial Grid Search (Phase A): Evaluates candidate capacities per frequency:
   h in {64, 128} x l in {1, 2} (4 configs) across 30d, 10d, 5d on in-season horizons H=1..6.
   Selects the champion frequency and its optimal architecture based on weighted In-Season RMSE.
4. Zero-Leakage Test Protocol (Phase C):
   - Training on historical years: 1951 .. t-4
   - Inner Validation Window: t-3 .. t-1 (monitors early stopping & LR scheduler)
   - Test year t evaluated strictly blind at frozen weights.
5. Resumable Checkpointing: Saves state after each fold to checkpoint_lstm.json.
6. Slicing Optimization: Reads 75-year full-season sequences once; any horizon H is sliced X[:, :S_H, :].

Modes:
  --mode phase_a    : Partial grid across frequencies (30d vs 10d vs 5d) on H=1..6 (1985-1995 val).
  --mode phase_b    : Evaluate lagged yield anomaly (epsilon_{t-1}) ablation on champion frequency.
  --mode phase_c    : Full 100% blind expanding test (1996-2025) across all 11 horizons.
  --mode overnight  : Run Phase A (partial grid) -> select champion -> Phase B -> Phase C.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout encoding for Windows PowerShell pipes
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
TARGET_CSV   = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
FEAT_DIR     = ROOT / "output" / "data" / "processed" / "model_datasets"
SEQ_DIR      = ROOT / "output" / "data" / "processed" / "lstm_sequences"
REPORT_DIR   = ROOT / "work" / "reports" / "experiments" / "lstm"
CHECKPOINT_PATH = REPORT_DIR / "checkpoint_lstm.json"
APPUNTI_PATH = ROOT / "appunti.md"

SEQ_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants & Partition Windows
# ---------------------------------------------------------------------------
VAL_START   = 1985
VAL_END     = 1995   # 11 validation folds
TEST_START  = 1996
TEST_END    = 2025   # 30 test folds

INPUT_SIZE  = 19     # 17 bioclimatic indicators + lat_norm + lon_norm
LR          = 3e-4   # Standard learning rate
WEIGHT_DECAY = 1e-4  # Decoupled AdamW weight decay (Géron Ch. 11)

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


# ---------------------------------------------------------------------------
# Hyperparameter Grids: Full 44 Configurations (No Forced Exclusions) & Lean
# ---------------------------------------------------------------------------
def build_grid_44() -> list[dict]:
    """
    Exhaustive 44-configuration factorial grid:
    - hidden_size in {64, 128, 256}
    - num_layers in {1, 2, 3, 4} (l in {1, 2, 3} for h=256)
    - dropout in {0.0, 0.2, 0.4}
    - batch_size in {25, 64}
    Matches the exact 44 configurations evaluated in exploratory stage.
    """
    grid = []
    for h in [64, 128, 256]:
        for bs in [25, 64]:
            grid.append({"hidden_size": h, "num_layers": 1, "dropout": 0.0, "batch_size": bs})
        for l in [2]:
            for dr in [0.0, 0.2, 0.4]:
                for bs in [25, 64]:
                    grid.append({"hidden_size": h, "num_layers": l, "dropout": dr, "batch_size": bs})
        for l in ([3, 4] if h < 256 else [3]):
            for dr in [0.2, 0.4]:
                for bs in [25, 64]:
                    grid.append({"hidden_size": h, "num_layers": l, "dropout": dr, "batch_size": bs})
    return grid

def build_grid_lean() -> list[dict]:
    """Targeted 6-configuration grid covering key depths and empirical champions."""
    return [
        {"hidden_size": 64,  "num_layers": 1, "dropout": 0.0, "batch_size": 25},
        {"hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "batch_size": 25},
        {"hidden_size": 64,  "num_layers": 2, "dropout": 0.4, "batch_size": 25},
        {"hidden_size": 128, "num_layers": 1, "dropout": 0.0, "batch_size": 25},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "batch_size": 25},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.4, "batch_size": 25},
    ]

GRID_44 = build_grid_44()


# ---------------------------------------------------------------------------
# Sequence Loader with Zero-Cost Slicing
# ---------------------------------------------------------------------------
_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

def get_sequence_slice(W: int, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads full-season sequence once and returns slice up to horizon H."""
    if W not in _CACHE:
        cache_file = SEQ_DIR / f"seq_{W}d_full_v3.npz"
        if not cache_file.exists():
            raise FileNotFoundError(f"Sequence cache {cache_file} not found. Run build_lstm_sequences_fast.py first.")
        data = np.load(cache_file, allow_pickle=True)
        _CACHE[W] = (data["X_seq"].astype(np.float32), data["counties"], data["years"].astype(np.int32))

    X_full, counties, years = _CACHE[W]
    S_H = seq_len_at_H(H, W)
    X_sliced = X_full[:, :S_H, :]
    return X_sliced, counties, years


# ---------------------------------------------------------------------------
# Detrending (OLS per county, train-only)
# ---------------------------------------------------------------------------
def fit_trends(yield_df: pd.DataFrame, train_years: list[int]) -> dict[str, tuple[float, float]]:
    train = yield_df[yield_df["year"].isin(train_years)]
    params = {}
    for fips, grp in train.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        b = np.sum((t_vals - t_vals.mean()) * (y_vals - y_vals.mean())) / np.sum((t_vals - t_vals.mean()) ** 2)
        a = y_vals.mean() - b * t_vals.mean()
        params[str(fips)] = (float(a), float(b))
    return params

def compute_anomalies(yield_df: pd.DataFrame, trend_params: dict, years: list[int]) -> dict[tuple[str, int], float]:
    sub = yield_df[yield_df["year"].isin(years)].copy()
    anom_map = {}
    for _, row in sub.iterrows():
        fips = str(row["county_fips"])
        yr = int(row["year"])
        tr = trend_params[fips][0] + trend_params[fips][1] * yr
        anom_map[(fips, yr)] = float(row["yield_bu_per_acre"] - tr)
    return anom_map


# ---------------------------------------------------------------------------
# Neural Network Architecture (He Init + LayerNorm + Bahdanau Attention)
# ---------------------------------------------------------------------------
def _init_weights(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            if "lstm" in name or "attention" in name:
                nn.init.xavier_uniform_(param)
            else:
                # He/Kaiming normal initialization for GELU layers (Géron Ch. 11)
                nn.init.kaiming_normal_(param, nonlinearity="relu")
        elif "bias" in name:
            nn.init.zeros_(param)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int, attn_size: int = 32):
        super().__init__()
        self.w_a = nn.Linear(hidden_size, attn_size, bias=True)
        self.v_a = nn.Linear(attn_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.tanh(self.w_a(hidden_states))
        scores = self.v_a(u).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attn_weights

class LSTMForecasterV3(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        include_lag_yield: bool = False,
    ):
        super().__init__()
        self.include_lag_yield = include_lag_yield
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention  = TemporalAttention(hidden_size, attn_size=32)

        in_head = hidden_size + (1 if include_lag_yield else 0)
        self.head = nn.Sequential(
            nn.Linear(in_head, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        _init_weights(self)

    def forward(self, x: torch.Tensor, lag_y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        context, attn = self.attention(out)
        if self.include_lag_yield and lag_y is not None:
            features = torch.cat([context, lag_y], dim=-1)
        else:
            features = context
        pred = self.head(features).squeeze(-1)
        return pred, attn


# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------
def fit_and_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    lag_tr: np.ndarray | None,
    X_va: np.ndarray,
    y_va: np.ndarray,
    lag_va: np.ndarray | None,
    X_te: np.ndarray,
    lag_te: np.ndarray | None,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 25,
    max_epochs: int = 40,
    patience: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    Fits model on X_tr, monitors early stopping on X_va, and predicts blind on X_te.
    Returns: (preds_test, attn_test, best_val_rmse)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    include_lag = (lag_tr is not None)
    model = LSTMForecasterV3(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        include_lag_yield=include_lag,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.MSELoss()

    t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.float32)
    t_lag_tr = torch.tensor(lag_tr, dtype=torch.float32) if include_lag else None

    t_X_va = torch.tensor(X_va, dtype=torch.float32)
    t_y_va = torch.tensor(y_va, dtype=torch.float32)
    t_lag_va = torch.tensor(lag_va, dtype=torch.float32) if include_lag else None

    t_X_te = torch.tensor(X_te, dtype=torch.float32)
    t_lag_te = torch.tensor(lag_te, dtype=torch.float32) if include_lag else None

    if include_lag:
        dataset = TensorDataset(t_X_tr, t_y_tr, t_lag_tr)
    else:
        dataset = TensorDataset(t_X_tr, t_y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_rmse = float("inf")
    best_weights  = None
    p_left = patience

    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            if include_lag:
                xb, yb, lb = batch
                pred, _ = model(xb, lb)
            else:
                xb, yb = batch
                pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if include_lag:
                val_preds, _ = model(t_X_va, t_lag_va)
            else:
                val_preds, _ = model(t_X_va)
            val_rmse = float(torch.sqrt(torch.mean((val_preds - t_y_va) ** 2)))

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            p_left = patience
        else:
            p_left -= 1
            if p_left <= 0:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        if include_lag:
            test_preds, test_attn = model(t_X_te, t_lag_te)
        else:
            test_preds, test_attn = model(t_X_te)
        preds_np = test_preds.numpy()
        attn_np  = test_attn.numpy()

    return preds_np, attn_np, best_val_rmse


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}

def save_checkpoint(data: dict) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# STAGE A: Grid Search across Frequencies (30d vs 10d vs 5d on all Horizons)
# ---------------------------------------------------------------------------
def run_phase_a(
    horizons: list[int] = list(range(1, 12)),
    frequencies: list[int] = [30, 10, 5],
    grid: list[dict] | None = None,
) -> tuple[int, dict]:
    if grid is None:
        grid = GRID_44

    print("\n" + "=" * 75)
    print("PHASE A: HYPERPARAMETER SEARCH ACROSS TEMPORAL FREQUENCIES (30d vs 10d vs 5d)")
    print(f"Campaign Horizons: H={horizons} ({len(horizons)} horizons) | Validation: 1985-1995 (11 Folds)")
    print(f"Candidate Grid: {len(grid)} configurations per frequency (No Forced Exclusions)")
    print("=" * 75)

    chk = load_checkpoint()
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years = sorted(yield_df["year"].unique().tolist())
    val_years = [y for y in all_years if VAL_START <= y <= VAL_END]

    freq_best_config: dict[int, dict] = {}
    freq_best_rmse: dict[int, float] = {}

    for W in frequencies:
        print(f"\n=======================================================")
        print(f">> FREQUENCY: {W}-day sequences")
        print(f"=======================================================")

        best_cfg_rmse = float("inf")
        best_cfg_dict = grid[0]

        for cfg_idx, cfg in enumerate(grid):
            h = cfg["hidden_size"]
            l = cfg["num_layers"]
            dr = cfg["dropout"]
            bs = cfg["batch_size"]
            cfg_sig = f"cfg{cfg_idx:02d}_h{h}_l{l}_dr{int(round(dr*100))}_bs{bs}"
            print(f"\n  Evaluating Config {cfg_idx+1}/{len(grid)}: {cfg_sig}...")

            h_rmses = {}

            for H in horizons:
                key = f"phase_a_{W}d_{cfg_sig}_H{H:02d}"
                legacy_key = f"phase_a_{W}d_cfg{cfg_idx}_H{H:02d}"
                if key in chk:
                    res = chk[key]["rmse"]
                    h_rmses[H] = res
                    print(f"    H={H:02d}: Cached Val RMSE = {res:.4f}")
                    continue
                elif legacy_key in chk:
                    res = chk[legacy_key]["rmse"]
                    h_rmses[H] = res
                    print(f"    H={H:02d}: Cached Val RMSE = {res:.4f}")
                    continue

                X_seq, counties, years = get_sequence_slice(W, H)
                fold_rmses = []

                for val_y in val_years:
                    train_yrs = [y for y in all_years if y < val_y]
                    tp = fit_trends(yield_df, train_yrs)
                    anom_map = compute_anomalies(yield_df, tp, train_yrs + [val_y])

                    tr_mask = np.isin(years, train_yrs)
                    va_mask = (years == val_y)

                    y_tr = np.array([anom_map[(f, y)] for f, y in zip(counties[tr_mask], years[tr_mask])], dtype=np.float32)
                    y_va = np.array([anom_map[(f, y)] for f, y in zip(counties[va_mask], years[va_mask])], dtype=np.float32)

                    flat_tr = X_seq[tr_mask].reshape(-1, INPUT_SIZE)
                    mu = flat_tr.mean(axis=0); sigma = flat_tr.std(axis=0); sigma[sigma == 0] = 1.0
                    X_tr_sc = (X_seq[tr_mask] - mu) / sigma
                    X_va_sc = (X_seq[va_mask] - mu) / sigma

                    val_preds, _, _ = fit_and_predict(
                        X_tr=X_tr_sc, y_tr=y_tr, lag_tr=None,
                        X_va=X_va_sc, y_va=y_va, lag_va=None,
                        X_te=X_va_sc, lag_te=None,
                        hidden_size=h, num_layers=l, dropout=dr, batch_size=bs,
                    )
                    fold_rmses.append(float(np.sqrt(np.mean((val_preds - y_va) ** 2))))

                mean_h_rmse = float(np.mean(fold_rmses))
                h_rmses[H] = mean_h_rmse
                chk[key] = {
                    "rmse": mean_h_rmse,
                    "fold_rmses": fold_rmses,
                    "config": cfg,
                    "cfg_idx": cfg_idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                save_checkpoint(chk)
                print(f"    H={H:02d}: Val RMSE = {mean_h_rmse:.4f}")

            # Phenologically Weighted RMSE across evaluated horizons:
            # - Peak harvest / pod-filling (H=1, 2, 3): 1.5x weight
            # - Vegetative & flowering (H=4, 5, 6): 1.0x weight
            # - Pre-season (H=7..11): 0.5x weight
            weights = {
                1: 1.5, 2: 1.5, 3: 1.5,
                4: 1.0, 5: 1.0, 6: 1.0,
                7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5,
            }
            weighted_rmse = sum(h_rmses[H] * weights.get(H, 0.5) for H in horizons) / sum(weights.get(H, 0.5) for H in horizons)
            unweighted_rmse = float(np.mean(list(h_rmses.values())))
            print(f"  -> Config {cfg_idx+1}/{len(grid)} Score: Weighted RMSE = {weighted_rmse:.4f} (Unweighted = {unweighted_rmse:.4f})")

            if weighted_rmse < best_cfg_rmse:
                best_cfg_rmse = weighted_rmse
                best_cfg_dict = cfg

        freq_best_config[W] = best_cfg_dict
        freq_best_rmse[W]   = best_cfg_rmse
        print(f"\n>> BEST CONFIG FOR {W}-day: {best_cfg_dict} with Weighted RMSE = {best_cfg_rmse:.4f}")

    print("\n" + "=" * 75)
    print("PHASE A FINAL SUMMARY: CHAMPION OF EACH FREQUENCY")
    print("=" * 75)
    for W in frequencies:
        print(f"  Frequency {W:2d}-day: Weighted RMSE = {freq_best_rmse[W]:.4f} | Optimal Config: {freq_best_config[W]}")

    champion_freq = min(frequencies, key=lambda w: freq_best_rmse[w])
    champion_config = freq_best_config[champion_freq]
    print(f"\n--> OVERALL CHAMPION: {champion_freq}-day Frequency with {champion_config} (Score: {freq_best_rmse[champion_freq]:.4f})")

    chk["champion"] = {
        "freq": champion_freq,
        "config": champion_config,
        "score": freq_best_rmse[champion_freq],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_checkpoint(chk)
    return champion_freq, champion_config


# ---------------------------------------------------------------------------
# STAGE B: Lagged Yield Anomaly Ablation (epsilon_{t-1})
# ---------------------------------------------------------------------------
def run_phase_b(champion_freq: int, champion_config: dict, horizons: list[int] = [1, 2, 3, 4, 5, 6]) -> None:
    print("\n" + "=" * 75)
    print(f"PHASE B: LAGGED YIELD ANOMALY ABLATION (Frequency = {champion_freq}-day)")
    print(f"Using Optimal Config: {champion_config}")
    print("=" * 75)

    chk = load_checkpoint()
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years = sorted(yield_df["year"].unique().tolist())
    val_years = [y for y in all_years if VAL_START <= y <= VAL_END]

    h = champion_config["hidden_size"]
    l = champion_config["num_layers"]
    dr = champion_config["dropout"]
    bs = champion_config["batch_size"]

    for H in horizons:
        key = f"phase_b_lag_{champion_freq}d_H{H:02d}"
        if key in chk:
            print(f"  H={H:02d}: Cached Lagged Val RMSE = {chk[key]['rmse']:.4f}")
            continue

        X_seq, counties, years = get_sequence_slice(champion_freq, H)
        fold_rmses = []

        for val_y in val_years:
            train_yrs = [y for y in all_years if y < val_y]
            all_rel_yrs = sorted(list(set(train_yrs + [val_y, min(train_yrs) - 1])))
            tp = fit_trends(yield_df, train_yrs)
            anom_map = compute_anomalies(yield_df, tp, all_rel_yrs)

            tr_mask = np.isin(years, train_yrs)
            va_mask = (years == val_y)

            y_tr = np.array([anom_map[(f, y)] for f, y in zip(counties[tr_mask], years[tr_mask])], dtype=np.float32)
            y_va = np.array([anom_map[(f, y)] for f, y in zip(counties[va_mask], years[va_mask])], dtype=np.float32)

            lag_tr = np.array([anom_map.get((f, y - 1), 0.0) for f, y in zip(counties[tr_mask], years[tr_mask])], dtype=np.float32).reshape(-1, 1)
            lag_va = np.array([anom_map.get((f, y - 1), 0.0) for f, y in zip(counties[va_mask], years[va_mask])], dtype=np.float32).reshape(-1, 1)

            flat_tr = X_seq[tr_mask].reshape(-1, INPUT_SIZE)
            mu = flat_tr.mean(axis=0); sigma = flat_tr.std(axis=0); sigma[sigma == 0] = 1.0
            X_tr_sc = (X_seq[tr_mask] - mu) / sigma
            X_va_sc = (X_seq[va_mask] - mu) / sigma

            mu_lag = lag_tr.mean(); std_lag = lag_tr.std() or 1.0
            lag_tr_sc = (lag_tr - mu_lag) / std_lag
            lag_va_sc = (lag_va - mu_lag) / std_lag

            val_preds, _, _ = fit_and_predict(
                X_tr=X_tr_sc, y_tr=y_tr, lag_tr=lag_tr_sc,
                X_va=X_va_sc, y_va=y_va, lag_va=lag_va_sc,
                X_te=X_va_sc, lag_te=lag_va_sc,
                hidden_size=h, num_layers=l, dropout=dr, batch_size=bs,
            )
            fold_rmses.append(float(np.sqrt(np.mean((val_preds - y_va) ** 2))))

        mean_rmse = float(np.mean(fold_rmses))
        chk[key] = {"rmse": mean_rmse, "fold_rmses": fold_rmses, "timestamp": datetime.now(timezone.utc).isoformat()}
        save_checkpoint(chk)
        print(f"  H={H:02d}: Integrated (Weather + Lag) Val RMSE = {mean_rmse:.4f}")


# ---------------------------------------------------------------------------
# STAGE C: Full 100% Blind Test (1996-2025 across all 11 horizons)
# ---------------------------------------------------------------------------
def run_phase_c(champion_freq: int, champion_config: dict, include_lag: bool = True) -> None:
    print("\n" + "=" * 75)
    print("PHASE C: FULL 100% BLIND OUT-OF-SAMPLE TEST FORECAST (1996-2025)")
    print(f"Frequency = {champion_freq}-day | Config = {champion_config} | Include Lag Yield = {include_lag}")
    print("Protocol: Inner Validation Window (t-3..t-1) for Zero Test Leakage across All 11 Horizons")
    print("=" * 75)

    chk = load_checkpoint()
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years = sorted(yield_df["year"].unique().tolist())
    test_years = [y for y in all_years if TEST_START <= y <= TEST_END]

    h = champion_config["hidden_size"]
    l = champion_config["num_layers"]
    dr = champion_config["dropout"]
    bs = champion_config["batch_size"]

    for H in range(1, 12):
        print(f"\n--- Horizon H={H:02d} ---", flush=True)
        key = f"phase_c_test_{champion_freq}d_H{H:02d}_{'lag' if include_lag else 'weather'}"
        if key in chk:
            print(f"  Cached Test RMSE = {chk[key]['rmse_pooled']:.4f}, R2 = {chk[key]['r2_pooled']:.4f}")
            continue

        X_seq, counties, years = get_sequence_slice(champion_freq, H)
        y_true_all = []
        y_pred_all = []
        attn_all = []

        for test_y in test_years:
            hist_years = [y for y in all_years if y < test_y]
            inner_val_years = [test_y - 3, test_y - 2, test_y - 1]
            train_years = [y for y in hist_years if y not in inner_val_years]

            all_rel = sorted(list(set(hist_years + [test_y, min(hist_years) - 1])))
            tp = fit_trends(yield_df, hist_years)
            anom_map = compute_anomalies(yield_df, tp, all_rel)

            tr_mask = np.isin(years, train_years)
            va_mask = np.isin(years, inner_val_years)
            te_mask = (years == test_y)

            y_tr = np.array([anom_map[(f, y)] for f, y in zip(counties[tr_mask], years[tr_mask])], dtype=np.float32)
            y_va = np.array([anom_map[(f, y)] for f, y in zip(counties[va_mask], years[va_mask])], dtype=np.float32)
            y_te = np.array([anom_map[(f, y)] for f, y in zip(counties[te_mask], years[te_mask])], dtype=np.float32)

            flat_tr = X_seq[tr_mask].reshape(-1, INPUT_SIZE)
            mu = flat_tr.mean(axis=0); sigma = flat_tr.std(axis=0); sigma[sigma == 0] = 1.0
            X_tr_sc = (X_seq[tr_mask] - mu) / sigma
            X_va_sc = (X_seq[va_mask] - mu) / sigma
            X_te_sc = (X_seq[te_mask] - mu) / sigma

            if include_lag:
                lag_tr = np.array([anom_map.get((f, y - 1), 0.0) for f, y in zip(counties[tr_mask], years[tr_mask])], dtype=np.float32).reshape(-1, 1)
                lag_va = np.array([anom_map.get((f, y - 1), 0.0) for f, y in zip(counties[va_mask], years[va_mask])], dtype=np.float32).reshape(-1, 1)
                lag_te = np.array([anom_map.get((f, y - 1), 0.0) for f, y in zip(counties[te_mask], years[te_mask])], dtype=np.float32).reshape(-1, 1)
                mu_l = lag_tr.mean(); std_l = lag_tr.std() or 1.0
                lag_tr_sc = (lag_tr - mu_l) / std_l
                lag_va_sc = (lag_va - mu_l) / std_l
                lag_te_sc = (lag_te - mu_l) / std_l
            else:
                lag_tr_sc = lag_va_sc = lag_te_sc = None

            pred_te, attn_te, _ = fit_and_predict(
                X_tr=X_tr_sc, y_tr=y_tr, lag_tr=lag_tr_sc,
                X_va=X_va_sc, y_va=y_va, lag_va=lag_va_sc,
                X_te=X_te_sc, lag_te=lag_te_sc,
                hidden_size=h, num_layers=l, dropout=dr, batch_size=bs,
            )
            y_true_all.extend(y_te)
            y_pred_all.extend(pred_te)
            if attn_te is not None:
                attn_all.append(attn_te.mean(axis=0))

        y_true_np = np.array(y_true_all)
        y_pred_np = np.array(y_pred_all)
        rmse_p = float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)))
        mae_p  = float(np.mean(np.abs(y_true_np - y_pred_np)))
        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
        ss_tot = np.sum(y_true_np ** 2)
        r2_p   = float(1.0 - ss_res / ss_tot)
        hit_rate = float(np.mean(np.sign(y_true_np) == np.sign(y_pred_np)))

        mean_attn_vector = [float(v) for v in np.mean(attn_all, axis=0)] if len(attn_all) > 0 else []
        print(f"  H={H:02d} Final Test: RMSE={rmse_p:.4f} | R2_OOS={r2_p:.4f} | MAE={mae_p:.4f} | HitRate={hit_rate*100:.1f}%")
        chk[key] = {
            "rmse_pooled": rmse_p, "r2_pooled": r2_p, "mae_pooled": mae_p,
            "hit_rate": hit_rate, "n_obs": len(y_true_np),
            "mean_attention": mean_attn_vector,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(chk)


# ---------------------------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Consolidated LSTM Pipeline with Full 44 Grid Search")
    parser.add_argument("--mode", choices=["phase_a", "phase_b", "phase_c", "overnight"], default="overnight")
    parser.add_argument("--grid", choices=["44", "lean"], default="44", help="Grid type: 44 for exhaustive 44 configs, lean for 6 configs")
    parser.add_argument("--freqs", nargs="+", type=int, default=[30, 10, 5], help="Aggregation frequencies in days (default: 30 10 5)")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(range(1, 12)), help="Horizons to evaluate (default: 1..11)")
    args = parser.parse_args()

    selected_grid = GRID_44 if args.grid == "44" else build_grid_lean()

    if args.mode == "phase_a":
        run_phase_a(horizons=args.horizons, frequencies=args.freqs, grid=selected_grid)
    elif args.mode == "phase_b":
        chk = load_checkpoint()
        champ = chk.get("champion", {})
        cfg = champ.get("config", selected_grid[0])
        freq = champ.get("freq", args.freqs[0])
        run_phase_b(freq, cfg, horizons=[h for h in args.horizons if h <= 6])
    elif args.mode == "phase_c":
        chk = load_checkpoint()
        champ = chk.get("champion", {})
        cfg = champ.get("config", selected_grid[0])
        freq = champ.get("freq", args.freqs[0])
        run_phase_c(freq, cfg, include_lag=True)
    elif args.mode == "overnight":
        champion_freq, champion_config = run_phase_a(horizons=args.horizons, frequencies=args.freqs, grid=selected_grid)
        run_phase_b(champion_freq, champion_config, horizons=[h for h in args.horizons if h <= 6])
        run_phase_c(champion_freq, champion_config, include_lag=True)
        print("\n" + "=" * 75)
        print("OVERNIGHT PIPELINE FULLY COMPLETED!")
        print("=" * 75)


if __name__ == "__main__":
    main()
