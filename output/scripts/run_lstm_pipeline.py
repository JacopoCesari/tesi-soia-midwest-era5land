"""
run_lstm_pipeline.py -- Production LSTM Pipeline for Soybean Yield Forecasting.

Methodological Architecture:
1. Spatial Encoding: Continuous normalized coordinates (lat_norm, lon_norm) concatenated
   with 17 bioclimatic indicators at every temporal window (input_size = 19).
2. Backbone: Recurrent LSTM + nn.LayerNorm + Bahdanau Temporal Attention.
3. Multi-Head & Loss Support:
   - "gelu": MLP Head with GELU activation (canonical baseline)
   - "linear": Direct linear projection (non-squashing, preserves raw latent scale)
   - "prelu": MLP Head with Parametric ReLU (non-saturating negative gradient)
   - "asym_huber": Asymmetric Huber Loss (penalizes underestimation of severe droughts)
4. Hierarchical Two-Stage Tuning Protocol:
   - Phase A: Macro-screening of temporal frequencies (30d, 10d, 5d) and backbone capacity (GRID_44).
   - Micro-Grid: Local box search ("one above, one below") across the 4 output heads at locked frequency and depth.
   - Phase B: Prior-year yield anomaly ablation (epsilon_{t-1}).
   - Phase C: 100% blind out-of-sample evaluation (1996-2025) with Inner Validation Split (t-3..t-1).

Modes:
  --mode full_pipeline : Execute Phase A -> Microgrid -> Phase B -> Phase C (resumable from checkpoint).
  --mode overnight     : Backward-compatible alias for full_pipeline.
  --mode phase_a       : Run macro grid search (30d, 10d, 5d) on 1985-1995 validation.
  --mode microgrid     : Run local box search for the 4 heads on champion frequency.
  --mode phase_b       : Run lagged yield ablation.
  --mode phase_c       : Run blind out-of-sample test evaluation (1996-2025).
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
LR          = 3e-4   # Standard learning rate (Khaki et al. 2020)
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
# Hyperparameter Grids: Full 44 Configurations & Lean
# ---------------------------------------------------------------------------
def build_grid_44() -> list[dict]:
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
# Loss Functions & Neural Architecture
# ---------------------------------------------------------------------------
class AsymmetricHuberLoss(nn.Module):
    """
    Asymmetric Huber Loss: Quadratic near zero, linear on large residuals,
    with asymmetric weighting alpha > 1.0 penalizing underestimation of drought collapses.
    """
    def __init__(self, delta: float = 1.0, alpha: float = 1.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = target - pred  # err < 0 means pred > target (underestimating a drought drop)
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * (err ** 2),
            self.delta * (abs_err - 0.5 * self.delta)
        )
        weights = torch.where(err < -self.delta, self.alpha, 1.0)
        return torch.mean(weights * huber)


def _init_weights(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            if "lstm" in name or "attention" in name:
                nn.init.xavier_uniform_(param)
            else:
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


class LSTMForecaster(nn.Module):
    """
    Production LSTM Model with selectable output heads:
    - 'gelu': MLP Head with GELU activation
    - 'linear': Direct linear projection (non-squashing)
    - 'prelu': MLP Head with Parametric ReLU (non-saturating negative slope)
    """
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        head_type: str = "gelu",
        include_lag_yield: bool = False,
    ):
        super().__init__()
        self.include_lag_yield = include_lag_yield
        self.head_type = head_type
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
        if head_type == "linear":
            self.head = nn.Linear(in_head, 1)
        elif head_type == "prelu":
            self.head = nn.Sequential(
                nn.Linear(in_head, 32),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
        else:  # default 'gelu'
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

# Backward-compatible alias
LSTMForecasterV3 = LSTMForecaster


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
    head_type: str = "gelu",
    loss_type: str = "mse",
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    Fits model on X_tr, monitors early stopping on X_va, and predicts blind on X_te.
    Returns: (preds_test, attn_test, best_val_rmse)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    include_lag = (lag_tr is not None)
    model = LSTMForecaster(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        head_type=head_type,
        include_lag_yield=include_lag,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    if loss_type == "asym_huber":
        criterion = AsymmetricHuberLoss(delta=1.0, alpha=1.5)
    else:
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
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_checkpoint(data: dict) -> None:
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# STAGE A: Multi-Frequency Macro Grid Search (GRID_44)
# ---------------------------------------------------------------------------
def run_phase_a(
    horizons: list[int] = list(range(1, 12)),
    frequencies: list[int] = [30, 10, 5],
    grid: list[dict] = GRID_44,
) -> tuple[int, dict]:
    print("=" * 75)
    print("PHASE A: MULTI-FREQUENCY GRID SEARCH (1985-1995 VALIDATION EXPANDING WINDOW)")
    print(f"Configurations per frequency: {len(grid)} | Frequencies: {frequencies} | Horizons: {horizons}")
    print("=" * 75)

    chk = load_checkpoint()
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years = sorted(yield_df["year"].unique().tolist())
    val_years = [y for y in all_years if VAL_START <= y <= VAL_END]

    freq_best_config_per_H: dict[int, dict[int, dict]] = {}
    freq_best_rmse_per_H: dict[int, dict[int, float]] = {}

    for W in frequencies:
        print(f"\n>>> PROCESSING {W}-DAY FREQUENCY (GRID OF {len(grid)} CONFIGURATIONS) <<<")
        for cfg_idx, cfg in enumerate(grid):
            h = cfg["hidden_size"]
            l = cfg["num_layers"]
            dr = cfg["dropout"]
            bs = cfg["batch_size"]
            sig = f"cfg{cfg_idx:02d}_h{h}_l{l}_dr{int(round(dr*100))}_bs{bs}"

            for H in horizons:
                key = f"phase_a_{W}d_{sig}_H{H:02d}"
                legacy_key = f"phase_a_{W}d_cfg{cfg_idx}_H{H:02d}"

                if key in chk:
                    continue
                if legacy_key in chk:
                    chk[key] = chk[legacy_key]
                    save_checkpoint(chk)
                    continue

                X_seq, counties, years = get_sequence_slice(W, H)
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

                    flat_tr = X_seq[tr_mask].reshape(-1, INPUT_SIZE)
                    mu = flat_tr.mean(axis=0); sigma = flat_tr.std(axis=0); sigma[sigma == 0] = 1.0
                    X_tr_sc = (X_seq[tr_mask] - mu) / sigma
                    X_va_sc = (X_seq[va_mask] - mu) / sigma

                    _, _, v_rmse = fit_and_predict(
                        X_tr=X_tr_sc, y_tr=y_tr, lag_tr=None,
                        X_va=X_va_sc, y_va=y_va, lag_va=None,
                        X_te=X_va_sc, lag_te=None,
                        hidden_size=h, num_layers=l, dropout=dr, batch_size=bs,
                        head_type="gelu", loss_type="mse",
                    )
                    fold_rmses.append(v_rmse)

                mean_val_rmse = float(np.mean(fold_rmses))
                chk[key] = {
                    "rmse": mean_val_rmse, "fold_rmses": fold_rmses,
                    "config": cfg, "cfg_idx": cfg_idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                save_checkpoint(chk)
                print(f"  [{W}d] {sig} | H={H:02d} -> Val RMSE: {mean_val_rmse:.4f}")

        # Per-horizon optimal configurations
        horizon_best_cfg: dict[int, dict] = {}
        horizon_best_rmse: dict[int, float] = {}
        for H in horizons:
            best_H_rmse = float("inf")
            best_H_cfg = grid[0]
            for c_idx, c_cfg in enumerate(grid):
                ch = c_cfg["hidden_size"]
                cl = c_cfg["num_layers"]
                cdr = c_cfg["dropout"]
                cbs = c_cfg["batch_size"]
                sig = f"cfg{c_idx:02d}_h{ch}_l{cl}_dr{int(round(cdr*100))}_bs{cbs}"
                k = f"phase_a_{W}d_{sig}_H{H:02d}"
                leg_k = f"phase_a_{W}d_cfg{c_idx}_H{H:02d}"
                if k in chk:
                    r_val = chk[k]["rmse"]
                elif leg_k in chk:
                    r_val = chk[leg_k]["rmse"]
                else:
                    continue
                if r_val < best_H_rmse:
                    best_H_rmse = r_val
                    best_H_cfg = c_cfg
            horizon_best_cfg[H] = best_H_cfg
            horizon_best_rmse[H] = best_H_rmse
            print(f"  Horizon H={H:02d}: Optimal Config = {best_H_cfg} | Val RMSE = {best_H_rmse:.4f}")

        freq_best_config_per_H[W] = horizon_best_cfg
        freq_best_rmse_per_H[W]   = horizon_best_rmse
        print(f"\n>> COMPLETED {W}-day: Mean Per-Horizon Val RMSE = {np.mean(list(horizon_best_rmse.values())):.4f}")

    print("\n" + "=" * 75)
    print("PHASE A FINAL SUMMARY: CHAMPION OF EACH FREQUENCY")
    print("=" * 75)
    for W in frequencies:
        mean_val = float(np.mean(list(freq_best_rmse_per_H[W].values())))
        print(f"  Frequency {W:2d}-day: Mean Per-Horizon Val RMSE = {mean_val:.4f}")

    champion_freq = min(frequencies, key=lambda w: np.mean(list(freq_best_rmse_per_H[w].values())))
    champion_configs_per_H = freq_best_config_per_H[champion_freq]
    print(f"\n--> OVERALL CHAMPION FREQUENCY: {champion_freq}-day Frequency")
    for H in sorted(champion_configs_per_H.keys()):
        print(f"    H={H:02d}: {champion_configs_per_H[H]} (Val RMSE = {freq_best_rmse_per_H[champion_freq][H]:.4f})")

    chk["champion"] = {
        "freq": champion_freq,
        "config": champion_configs_per_H[1],
        "score": float(np.mean(list(freq_best_rmse_per_H[champion_freq].values()))),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    chk["optimal_params_per_horizon"] = {
        str(w): {str(h): c for h, c in horizon_map.items()}
        for w, horizon_map in freq_best_config_per_H.items()
    }
    save_checkpoint(chk)

    out_optimal_json = ROOT / "output" / "data" / "processed" / "hyperparameters_optimal_lstm.json"
    with open(out_optimal_json, "w") as f:
        json.dump(
            {str(H): champion_configs_per_H[H] for H in sorted(champion_configs_per_H.keys())},
            f, indent=2
        )
    print(f">> Saved per-horizon optimal parameters to {out_optimal_json}")

    return champion_freq, champion_configs_per_H


# ---------------------------------------------------------------------------
# MICRO-GRID: Local Box Tuning for the 4 Output Heads ("One Above, One Below")
# ---------------------------------------------------------------------------
def run_phase_microgrid(
    champion_freq: int,
    champion_configs_per_H: dict,
    horizons: list[int] = [1, 2, 3, 4, 5, 6],
) -> dict:
    print("\n" + "=" * 75)
    print(f"PHASE MICROGRID: LOCAL BOX SEARCH FOR 4 OUTPUT HEADS ({champion_freq}-day)")
    print("Variants: 1. GELU (Baseline) | 2. Linear | 3. PReLU | 4. Asymmetric Huber")
    print("=" * 75)

    chk = load_checkpoint()
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years = sorted(yield_df["year"].unique().tolist())
    val_years = [y for y in all_years if VAL_START <= y <= VAL_END]

    variants = [
        {"name": "gelu",       "head": "gelu",   "loss": "mse"},
        {"name": "linear",     "head": "linear", "loss": "mse"},
        {"name": "prelu",      "head": "prelu",  "loss": "mse"},
        {"name": "asym_huber", "head": "prelu",  "loss": "asym_huber"},
    ]

    base_cfg = champion_configs_per_H.get(1, champion_configs_per_H.get("1", list(champion_configs_per_H.values())[0]))
    h_star = base_cfg["hidden_size"]
    l_star = 2  # Locked to depth 2 as empirically confirmed
    dr_star = base_cfg["dropout"]
    bs_star = base_cfg.get("batch_size", 25)

    h_candidates = sorted(list(set([max(64, h_star // 2), h_star, min(256, h_star * 2)])))
    dr_candidates = sorted(list(set([max(0.0, round(dr_star - 0.1, 2)), dr_star, min(0.4, round(dr_star + 0.1, 2))])))

    variant_champions = {}

    for var in variants:
        v_name = var["name"]
        head_t = var["head"]
        loss_t = var["loss"]
        print(f"\n>> Tuning Variant: {v_name.upper()} (head={head_t}, loss={loss_t})")

        best_v_rmse = float("inf")
        best_v_cfg = None

        for h in h_candidates:
            for dr in dr_candidates:
                cfg_rmse_list = []
                for H in horizons:
                    sig = f"h{h}_l{l_star}_dr{int(round(dr*100))}_bs{bs_star}"
                    key = f"microgrid_{champion_freq}d_{v_name}_{sig}_H{H:02d}"

                    if key in chk:
                        cfg_rmse_list.append(chk[key]["rmse"])
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

                        flat_tr = X_seq[tr_mask].reshape(-1, INPUT_SIZE)
                        mu = flat_tr.mean(axis=0); sigma = flat_tr.std(axis=0); sigma[sigma == 0] = 1.0
                        X_tr_sc = (X_seq[tr_mask] - mu) / sigma
                        X_va_sc = (X_seq[va_mask] - mu) / sigma

                        _, _, v_rmse = fit_and_predict(
                            X_tr=X_tr_sc, y_tr=y_tr, lag_tr=None,
                            X_va=X_va_sc, y_va=y_va, lag_va=None,
                            X_te=X_va_sc, lag_te=None,
                            hidden_size=h, num_layers=l_star, dropout=dr, batch_size=bs_star,
                            head_type=head_t, loss_type=loss_t,
                        )
                        fold_rmses.append(v_rmse)

                    h_rmse = float(np.mean(fold_rmses))
                    chk[key] = {
                        "rmse": h_rmse, "fold_rmses": fold_rmses,
                        "variant": v_name, "head": head_t, "loss": loss_t,
                        "config": {"hidden_size": h, "num_layers": l_star, "dropout": dr, "batch_size": bs_star},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    save_checkpoint(chk)
                    cfg_rmse_list.append(h_rmse)

                mean_cfg_rmse = float(np.mean(cfg_rmse_list))
                print(f"   Config {sig}: Mean In-Season Val RMSE = {mean_cfg_rmse:.4f}")
                if mean_cfg_rmse < best_v_rmse:
                    best_v_rmse = mean_cfg_rmse
                    best_v_cfg = {
                        "hidden_size": h, "num_layers": l_star, "dropout": dr,
                        "batch_size": bs_star, "head": head_t, "loss": loss_t
                    }

        variant_champions[v_name] = {
            "config": best_v_cfg,
            "val_rmse": best_v_rmse,
        }
        print(f"   => Champion for {v_name.upper()}: {best_v_cfg} (Val RMSE = {best_v_rmse:.4f})")

    chk["variant_champions"] = variant_champions
    save_checkpoint(chk)

    out_heads_json = ROOT / "output" / "data" / "processed" / "hyperparameters_optimal_lstm_heads.json"
    with open(out_heads_json, "w") as f:
        json.dump(variant_champions, f, indent=2)
    print(f"\n>> Saved optimal head configurations to {out_heads_json}")

    return variant_champions


# ---------------------------------------------------------------------------
# STAGE B: Lagged Yield Anomaly Ablation (epsilon_{t-1})
# ---------------------------------------------------------------------------
def run_phase_b(champion_freq: int, champion_configs: dict, horizons: list[int] = [1, 2, 3, 4, 5, 6]) -> None:
    print("\n" + "=" * 75)
    print(f"PHASE B: LAGGED YIELD ANOMALY ABLATION (Frequency = {champion_freq}-day)")
    print("Protocol: Per-Horizon Optimal Tuning")
    print("=" * 75)

    chk = load_checkpoint()
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years = sorted(yield_df["year"].unique().tolist())
    val_years = [y for y in all_years if VAL_START <= y <= VAL_END]

    for H in horizons:
        cfg = champion_configs.get(H, champion_configs.get(str(H), champion_configs)) if isinstance(champion_configs, dict) else champion_configs
        h = cfg["hidden_size"]
        l = cfg["num_layers"]
        dr = cfg["dropout"]
        bs = cfg["batch_size"]
        print(f"  H={H:02d} using optimal config: h={h}, l={l}, dr={dr}, bs={bs}")
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

            _, _, v_rmse = fit_and_predict(
                X_tr=X_tr_sc, y_tr=y_tr, lag_tr=lag_tr_sc,
                X_va=X_va_sc, y_va=y_va, lag_va=lag_va_sc,
                X_te=X_va_sc, lag_te=lag_va_sc,
                hidden_size=h, num_layers=l, dropout=dr, batch_size=bs,
            )
            fold_rmses.append(v_rmse)

        mean_v = float(np.mean(fold_rmses))
        chk[key] = {
            "rmse": mean_v, "fold_rmses": fold_rmses,
            "config": cfg, "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(chk)
        print(f"  H={H:02d} Complete -> Lagged Val RMSE: {mean_v:.4f}")


# ---------------------------------------------------------------------------
# STAGE C: Full 100% Blind Test (1996-2025 across all 11 horizons)
# ---------------------------------------------------------------------------
def run_phase_c(champion_freq: int, champion_configs: dict, include_lag: bool = False, head_variant: str = "gelu") -> None:
    print("\n" + "=" * 75)
    print("PHASE C: FULL 100% BLIND OUT-OF-SAMPLE TEST FORECAST (1996-2025)")
    print(f"Frequency = {champion_freq}-day | Variant = {head_variant} | Include Lag Yield = {include_lag}")
    print("=" * 75)

    chk = load_checkpoint()
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years = sorted(yield_df["year"].unique().tolist())
    test_years = [y for y in all_years if TEST_START <= y <= TEST_END]

    head_t = "linear" if head_variant == "linear" else ("prelu" if "prelu" in head_variant or "asym" in head_variant else "gelu")
    loss_t = "asym_huber" if "asym" in head_variant else "mse"

    for H in range(1, 12):
        cfg = champion_configs.get(H, champion_configs.get(str(H), champion_configs)) if isinstance(champion_configs, dict) else champion_configs
        h = cfg["hidden_size"]
        l = cfg["num_layers"]
        dr = cfg["dropout"]
        bs = cfg["batch_size"]
        print(f"\n--- Horizon H={H:02d} (Variant: {head_variant}, h={h}, l={l}, dr={dr}, bs={bs}) ---", flush=True)
        key = f"phase_c_test_{champion_freq}d_H{H:02d}_{head_variant}_{'lag' if include_lag else 'weather'}"
        legacy_key = f"phase_c_test_{champion_freq}d_H{H:02d}_{'lag' if include_lag else 'weather'}"

        if key in chk:
            print(f"  Cached Test RMSE = {chk[key]['rmse_pooled']:.4f}, R2 = {chk[key]['r2_pooled']:.4f}")
            continue
        elif head_variant == "gelu" and legacy_key in chk:
            chk[key] = chk[legacy_key]
            save_checkpoint(chk)
            print(f"  Cached Test RMSE = {chk[key]['rmse_pooled']:.4f}, R2 = {chk[key]['r2_pooled']:.4f}")
            continue

        X_seq, counties, years = get_sequence_slice(champion_freq, H)
        y_true_all = []
        y_pred_all = []
        y_years_all = []
        counties_all = []
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
                head_type=head_t, loss_type=loss_t,
            )
            y_true_all.extend(y_te)
            y_pred_all.extend(pred_te)
            y_years_all.extend([test_y] * len(y_te))
            counties_all.extend(counties[te_mask])
            if attn_te is not None:
                attn_all.append(attn_te.mean(axis=0))

        y_true_np = np.array(y_true_all)
        y_pred_np = np.array(y_pred_all)
        y_years_np = np.array(y_years_all)

        rmse_p = float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)))
        mae_p  = float(np.mean(np.abs(y_true_np - y_pred_np)))
        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
        ss_tot = np.sum(y_true_np ** 2)
        r2_p   = float(1.0 - ss_res / ss_tot)
        hit_rate = float(np.mean(np.sign(y_true_np) == np.sign(y_pred_np)))

        # Sub-cohort breakdown
        drought_2012_mask = (y_years_np == 2012)
        rmse_2012 = float(np.sqrt(np.mean((y_true_np[drought_2012_mask] - y_pred_np[drought_2012_mask]) ** 2))) if drought_2012_mask.any() else None
        bias_2012 = float(np.mean(y_pred_np[drought_2012_mask] - y_true_np[drought_2012_mask])) if drought_2012_mask.any() else None

        sigma_y = float(np.std(y_true_np))
        normal_mask = (np.abs(y_true_np) <= 1.5 * sigma_y)
        rmse_normal = float(np.sqrt(np.mean((y_true_np[normal_mask] - y_pred_np[normal_mask]) ** 2))) if normal_mask.any() else None

        mean_attn_vector = [float(v) for v in np.mean(attn_all, axis=0)] if len(attn_all) > 0 else []
        print(f"  H={H:02d} Final Test: RMSE={rmse_p:.4f} | R2_OOS={r2_p:.4f} | Normal RMSE={rmse_normal:.4f} | 2012 RMSE={rmse_2012:.4f}")
        chk[key] = {
            "rmse_pooled": rmse_p, "r2_pooled": r2_p, "mae_pooled": mae_p,
            "hit_rate": hit_rate, "n_obs": len(y_true_np),
            "rmse_normal": rmse_normal, "rmse_2012": rmse_2012, "bias_2012": bias_2012,
            "variant": head_variant,
            "mean_attention": mean_attn_vector,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint(chk)

        # Save individual prediction parquet
        pred_file = REPORT_DIR / f"preds_{champion_freq}d_H{H:02d}_{head_variant}_{'lag' if include_lag else 'weather'}.parquet"
        pd.DataFrame({
            "year": y_years_np,
            "county_fips": counties_all,
            "actual_anomaly": y_true_np,
            "pred_anomaly": y_pred_np,
            "horizon": H,
            "head_variant": head_variant,
            "include_lag": include_lag,
            "frequency": champion_freq,
        }).to_parquet(pred_file, index=False)


# ---------------------------------------------------------------------------
# Consolidated Export Helper
# ---------------------------------------------------------------------------
def export_consolidated_results(champion_freq: int) -> None:
    print("\n" + "=" * 75)
    print("EXPORTING CONSOLIDATED LSTM RESULTS & DIAGNOSTICS")
    print("=" * 75)

    chk = load_checkpoint()
    rows = []

    # 1. Base Naive reference at H=12
    rows.append({
        "model": "Naive",
        "variant": "secular_trend",
        "include_lag": False,
        "frequency": champion_freq,
        "H": 12,
        "RMSE_OOS_pooled": 5.9442,
        "R2_OOS_pooled": 0.0,
        "MAE_OOS_pooled": 4.7675,
        "Skill_Score": 0.0,
        "Hit_Rate": 0.5,
        "RMSE_2012": 6.84,
        "Bias_2012": 3.42,
        "RMSE_normal": 4.25,
    })

    # 2. Extract Phase C results
    for key, data in chk.items():
        if not key.startswith("phase_c_test_"):
            continue
        parts = key.split("_")
        h_part = [p for p in parts if p.startswith("H") and p[1:].isdigit()]
        if not h_part:
            continue
        H = int(h_part[0][1:])
        variant = data.get("variant", "gelu")
        include_lag = key.endswith("_lag")
        rmse_p = data.get("rmse_pooled", float("nan"))
        r2_p   = data.get("r2_pooled", float("nan"))
        mae_p  = data.get("mae_pooled", float("nan"))
        skill  = 1.0 - (rmse_p / 5.9442) if rmse_p else float("nan")

        rows.append({
            "model": "LSTM",
            "variant": variant,
            "include_lag": include_lag,
            "frequency": champion_freq,
            "H": H,
            "RMSE_OOS_pooled": rmse_p,
            "R2_OOS_pooled": r2_p,
            "MAE_OOS_pooled": mae_p,
            "Skill_Score": skill,
            "Hit_Rate": data.get("hit_rate", float("nan")),
            "RMSE_2012": data.get("rmse_2012", float("nan")),
            "Bias_2012": data.get("bias_2012", float("nan")),
            "RMSE_normal": data.get("rmse_normal", float("nan")),
        })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df.sort_values(by=["model", "variant", "include_lag", "H"], ascending=[False, True, True, False], inplace=True)
        out_csv = ROOT / "output" / "data" / "processed" / "evaluation_summary_lstm.csv"
        out_parquet = ROOT / "output" / "data" / "processed" / "evaluation_summary_lstm.parquet"
        summary_df.to_csv(out_csv, index=False)
        summary_df.to_parquet(out_parquet, index=False)
        print(f">> Saved evaluation summary to:\n   {out_csv}\n   {out_parquet}")

        # Build comparison: Weather vs Integrated (lag)
        comp_rows = []
        for var, grp in summary_df[summary_df["model"] == "LSTM"].groupby("variant"):
            w_df = grp[~grp["include_lag"]].set_index("H")
            l_df = grp[grp["include_lag"]].set_index("H")
            common_H = sorted(list(set(w_df.index).intersection(set(l_df.index))))
            for H in common_H:
                r2_w = w_df.loc[H, "R2_OOS_pooled"]
                r2_l = l_df.loc[H, "R2_OOS_pooled"]
                rmse_w = w_df.loc[H, "RMSE_OOS_pooled"]
                rmse_l = l_df.loc[H, "RMSE_OOS_pooled"]
                comp_rows.append({
                    "variant": var,
                    "H": H,
                    "R2_OOS_pooled_integrated": r2_l,
                    "RMSE_OOS_pooled_integrated": rmse_l,
                    "R2_OOS_pooled_weather_only": r2_w,
                    "RMSE_OOS_pooled_weather_only": rmse_w,
                    "delta_R2": r2_l - r2_w,
                    "delta_RMSE": rmse_l - rmse_w,
                })
        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            comp_csv = ROOT / "output" / "data" / "processed" / "diagnostics" / "comparison_weather_vs_integrated_lstm.csv"
            comp_csv.parent.mkdir(parents=True, exist_ok=True)
            comp_df.to_csv(comp_csv, index=False)
            print(f">> Saved weather vs integrated comparison to:\n   {comp_csv}")

        # Concatenate all predictions if available
        pred_files = list(REPORT_DIR.glob(f"preds_{champion_freq}d_*.parquet"))
        if pred_files:
            all_preds = pd.concat([pd.read_parquet(p) for p in pred_files], axis=0)
            pred_out = ROOT / "output" / "data" / "processed" / "predictions_lstm.parquet"
            all_preds.to_parquet(pred_out, index=False)
            print(f">> Saved consolidated predictions to:\n   {pred_out}")


# ---------------------------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Consolidated LSTM Pipeline with Multi-Head Support")
    parser.add_argument("--mode", choices=["phase_a", "microgrid", "phase_b", "phase_c", "overnight", "full_pipeline", "export_reports"], default="full_pipeline")
    parser.add_argument("--grid", choices=["44", "lean"], default="44", help="Grid type: 44 for exhaustive 44 configs, lean for 6 configs")
    parser.add_argument("--freqs", nargs="+", type=int, default=[30, 10, 5], help="Aggregation frequencies in days (default: 30 10 5)")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(range(1, 12)), help="Horizons to evaluate (default: 1..11)")
    parser.add_argument("--include_lag", action="store_true", default=False, help="Include lagged yield anomaly (ablation mode, default: False)")
    args = parser.parse_args()

    selected_grid = GRID_44 if args.grid == "44" else build_grid_lean()

    if args.mode == "export_reports":
        chk = load_checkpoint()
        champ = chk.get("champion", {})
        freq = champ.get("freq", args.freqs[0])
        export_consolidated_results(freq)
        return

    if args.mode == "phase_a":
        run_phase_a(horizons=args.horizons, frequencies=args.freqs, grid=selected_grid)
    elif args.mode == "microgrid":
        chk = load_checkpoint()
        champ = chk.get("champion", {})
        freq = champ.get("freq", args.freqs[0])
        opt_per_h = chk.get("optimal_params_per_horizon", {}).get(str(freq), {})
        run_phase_microgrid(freq, opt_per_h, horizons=[h for h in args.horizons if h <= 6])
    elif args.mode == "phase_b":
        chk = load_checkpoint()
        opt_per_h = chk.get("optimal_params_per_horizon", {})
        champ = chk.get("champion", {})
        freq = champ.get("freq", args.freqs[0])
        cfg_map = opt_per_h.get(str(freq), opt_per_h.get(freq, champ.get("config", selected_grid[0])))
        run_phase_b(freq, cfg_map, horizons=[h for h in args.horizons if h <= 6])
    elif args.mode == "phase_c":
        chk = load_checkpoint()
        opt_per_h = chk.get("optimal_params_per_horizon", {})
        champ = chk.get("champion", {})
        freq = champ.get("freq", args.freqs[0])
        cfg_map = opt_per_h.get(str(freq), opt_per_h.get(freq, champ.get("config", selected_grid[0])))
        run_phase_c(freq, cfg_map, include_lag=args.include_lag, head_variant="gelu")
        export_consolidated_results(freq)
    elif args.mode in ["overnight", "full_pipeline"]:
        # Step 1: Macro Grid Search across frequencies
        champion_freq, champion_configs_per_H = run_phase_a(horizons=args.horizons, frequencies=args.freqs, grid=selected_grid)
        # Step 2: Micro-Grid Local Search for the 4 Output Heads
        variant_champions = run_phase_microgrid(champion_freq, champion_configs_per_H, horizons=[h for h in args.horizons if h <= 6])
        # Step 3: Lagged Yield Anomaly Ablation
        run_phase_b(champion_freq, champion_configs_per_H, horizons=[h for h in args.horizons if h <= 6])
        # Step 4: Final Evaluation of the 4 Deep Learning Options: Weather-Only vs. With Past Yield Anomaly
        print("\n" + "=" * 75)
        print("FINAL EVALUATION ACROSS THE 4 DEEP LEARNING OPTIONS (WEATHER-ONLY vs. WITH PAST YIELD)")
        print("=" * 75)
        for v_name, v_data in variant_champions.items():
            final_cfg_map = {H: v_data["config"] for H in range(1, 12)}
            
            # (A) Final Model: Weather-Only
            print(f"\n>> [{v_name.upper()}] Final Out-of-Sample Evaluation (Weather-Only)...")
            run_phase_c(champion_freq, final_cfg_map, include_lag=False, head_variant=v_name)
            
            # (B) Final Model: With Past Yield Anomaly (Integrated)
            print(f"\n>> [{v_name.upper()}] Final Out-of-Sample Evaluation (With Past Yield Anomaly)...")
            run_phase_c(champion_freq, final_cfg_map, include_lag=True, head_variant=v_name)

        # Step 5: Export consolidated results and comparison files for Chapter 5
        export_consolidated_results(champion_freq)
        print("\n" + "=" * 75)
        print("FULL LSTM PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 75)


if __name__ == "__main__":
    main()

