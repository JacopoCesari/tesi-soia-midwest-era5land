"""LSTM test forecasts with complete continuous weather windows.

Keeps the author-selected frequency, head configurations and validation records.
Only the 2011--2025 expanding test forecasts are fitted by this entry point.
The training architecture and engine follow the existing project implementation;
Geron (2022) supports recurrent networks, AdamW and gradient clipping.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
TARGET_CSV = ROOT / "data/target/soybean_yield_1951_2025.csv"
WEATHER_DIR = ROOT / "data/weather/daily_counties"
FEAT_DIR = ROOT / "output/data/processed/model_datasets"
SEQ_DIR = ROOT / "output/data/processed/lstm_sequences"
REPORT_DIR = ROOT / "work/reports/experiments/lstm"
CHECKPOINT_PATH = REPORT_DIR / "checkpoint_lstm.json"
OUT_PREDS = ROOT / "output/data/processed/predictions_lstm.parquet"
RUN_PREDS = REPORT_DIR / "test_predictions_continuous.parquet"
RUN_MANIFEST = REPORT_DIR / "test_continuous_manifest.json"
WINDOW_VERSION = "continuous_daily_v1"
INPUT_SIZE = 19
LR = 3e-4
WEIGHT_DECAY = 1e-4

FLUX_COLS = ["total_precipitation", "surface_solar_radiation_downwards",
    "et0_fao56", "p_minus_et0", "growing_degree_days", "heat_day_30", "heat_day_35", "edd_30"]
STATE_COLS = ["air_temperature_mean", "air_temperature_maximum", "air_temperature_minimum",
    "dewpoint_temperature_mean", "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3", "vapor_pressure_deficit"]

def safe_replace(src: Path, dst: Path, retries: int = 10, delay: float = 0.5):
    for i in range(retries):
        try:
            src.replace(dst)
            return
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(delay)


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
            if any(k in name for k in ["lstm", "gru", "rnn", "attention"]):
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

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.tanh(self.w_a(hidden_states))
        scores = self.v_a(u).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(scores.shape[1], device=scores.device)[None, :] >= lengths[:, None]
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attn_weights

class RecurrentForecaster(nn.Module):
    """
    Production Recurrent Model (LSTM or GRU) with selectable output heads:
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
        cell_type: str = "lstm",
        include_lag_yield: bool = False,
    ):
        super().__init__()
        self.include_lag_yield = include_lag_yield
        self.head_type = head_type
        self.cell_type = cell_type.lower()
        if self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
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

    def forward(self, x: torch.Tensor, lag_y: torch.Tensor | None = None, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.rnn(x)
        out = self.layer_norm(out)
        context, attn = self.attention(out, lengths)
        if self.include_lag_yield and lag_y is not None:
            features = torch.cat([context, lag_y], dim=-1)
        else:
            features = context
        pred = self.head(features).squeeze(-1)
        return pred, attn

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
    seed: int = 88,
    head_type: str = "gelu",
    loss_type: str = "mse",
    cell_type: str = "lstm",
    lengths_tr=None, lengths_va=None, lengths_te=None,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    Fits model on X_tr, monitors early stopping on X_va, and predicts blind on X_te.
    Returns: (preds_test, attn_test, best_val_rmse)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    include_lag = (lag_tr is not None)
    model = RecurrentForecaster(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        head_type=head_type,
        cell_type=cell_type,
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

    train_lengths = torch.as_tensor(lengths_tr, dtype=torch.long)
    val_lengths = torch.as_tensor(lengths_va, dtype=torch.long)
    test_lengths = torch.as_tensor(lengths_te, dtype=torch.long)
    if include_lag:
        dataset = TensorDataset(t_X_tr, t_y_tr, train_lengths, t_lag_tr)
    else:
        dataset = TensorDataset(t_X_tr, t_y_tr, train_lengths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_rmse = float("inf")
    best_weights  = None
    p_left = patience

    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            if include_lag:
                xb, yb, lens, lb = batch
                pred, _ = model(xb, lb, lengths=lens)
            else:
                xb, yb, lens = batch
                pred, _ = model(xb, lengths=lens)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if include_lag:
                val_preds, _ = model(t_X_va, t_lag_va, lengths=val_lengths)
            else:
                val_preds, _ = model(t_X_va, lengths=val_lengths)
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
            test_preds, test_attn = model(t_X_te, t_lag_te, lengths=test_lengths)
        else:
            test_preds, test_attn = model(t_X_te, lengths=test_lengths)
        preds_np = test_preds.numpy()
        attn_np  = test_attn.numpy()

    return preds_np, attn_np, best_val_rmse


def campaign_windows(year, horizon, width):
    """Partition every available campaign date once, retaining a partial last block."""
    if horizon not in range(1, 12) or width not in (5, 10, 30):
        raise ValueError("Expected H=1..11 and width=5, 10 or 30")
    cutoff = pd.Timestamp(year, 11, 1) - pd.offsets.MonthEnd(horizon)
    dates = pd.date_range(f"{year-1}-11-01", cutoff)
    return [dates[start:start+width] for start in range(0, len(dates), width)]


def aggregate_daily_windows(daily, width):
    """Sum eight exposures and average nine states over actual days in each block."""
    if daily.ndim != 3 or daily.shape[2] != 17 or not np.isfinite(daily).all():
        raise ValueError("Daily weather must be finite with 17 channels")
    if width not in (5, 10, 30) or daily.shape[1] == 0:
        raise ValueError("Expected positive campaign length and width=5, 10 or 30")
    blocks = []
    for start in range(0, daily.shape[1], width):
        block = daily[:, start:start+width]
        blocks.append(np.concatenate([block[:, :, :8].sum(axis=1),
                                      block[:, :, 8:].mean(axis=1)], axis=1))
    return np.stack(blocks, axis=1).astype(np.float32)


def sequence_path(width, horizon):
    return SEQ_DIR / f"seq_{width}d_H{horizon:02d}_{WINDOW_VERSION}.npz"


def build_sequence_caches(width=10):
    """Build horizon-specific arrays; never slice a full-season partial block."""
    paths = [sequence_path(width, h) for h in range(1, 12)]
    if all(p.exists() for p in paths):
        return
    target = pd.read_csv(TARGET_CSV, dtype={"county_fips": str})
    fips = sorted(target.county_fips.str.zfill(5).unique())
    years = sorted(target.year.unique())
    coords = pd.read_parquet(FEAT_DIR / "features_H01.parquet",
                            columns=["county_fips", "lat_norm", "lon_norm"])
    coords["county_fips"] = coords.county_fips.astype(str).str.zfill(5)
    coords = coords.drop_duplicates().set_index("county_fips").reindex(fips)
    if not np.isfinite(coords.to_numpy()).all() or len(coords) != len(fips):
        raise ValueError("Incomplete spatial coordinates")
    buffers, lengths = {}, {}
    for h in range(1, 12):
        max_steps = max(len(campaign_windows(int(y), h, width)) for y in years)
        buffers[h] = np.zeros((len(years)*len(fips), max_steps, 19), dtype=np.float32)
        lengths[h] = np.zeros(len(years)*len(fips), dtype=np.int32)
    previous = None
    for yi, year in enumerate(years):
        if previous is None:
            previous = pd.read_parquet(WEATHER_DIR / f"county_daily_{year-1}.parquet")
        current = pd.read_parquet(WEATHER_DIR / f"county_daily_{year}.parquet")
        campaign = pd.concat([previous, current], ignore_index=True)
        previous = current
        campaign["date"] = pd.to_datetime(campaign.date)
        campaign["county_fips"] = campaign.county_fips.astype(str).str.zfill(5)
        dates = pd.date_range(f"{year-1}-11-01", f"{year}-10-31")
        campaign = campaign[campaign.date.isin(dates) & campaign.county_fips.isin(fips)].copy()
        if "edd_30" not in campaign:
            campaign["edd_30"] = np.maximum(0, campaign.air_temperature_maximum - 30)
        campaign["SM_root"] = (0.07*campaign.volumetric_soil_water_layer_1
                               + 0.21*campaign.volumetric_soil_water_layer_2
                               + 0.72*campaign.volumetric_soil_water_layer_3)
        if campaign.duplicated(["county_fips", "date"]).any():
            raise ValueError(f"Duplicate daily keys in campaign {year}")
        index = pd.MultiIndex.from_product([fips, dates], names=["county_fips", "date"])
        daily = campaign.set_index(["county_fips", "date"]).reindex(index)
        daily = daily[FLUX_COLS+STATE_COLS+["SM_root"]].to_numpy(dtype=np.float32)
        daily = daily.reshape(len(fips), len(dates), 17)
        if not np.isfinite(daily).all():
            raise ValueError(f"Missing/nonfinite weather in campaign {year}; no imputation permitted")
        rows = slice(yi*len(fips), (yi+1)*len(fips))
        for h in range(1, 12):
            available = sum(map(len, campaign_windows(int(year), h, width)))
            values = aggregate_daily_windows(daily[:, :available], width)
            steps = values.shape[1]
            buffers[h][rows, :steps, :17] = values
            buffers[h][rows, :steps, 17:] = coords.to_numpy()[:, None, :]
            lengths[h][rows] = steps
        print(f"Built complete {width}d campaign {year}: {len(dates)} days", flush=True)
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    for h in range(1, 12):
        path = sequence_path(width, h)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, X_seq=buffers[h], lengths=lengths[h],
                            counties=np.tile(fips, len(years)), years=np.repeat(years, len(fips)),
                            window_version=WINDOW_VERSION)
        temporary.replace(path)


@lru_cache(maxsize=1)
def get_sequence_slice(width, horizon):
    """Load complete windows ending at this horizon; lengths mask tensor padding."""
    path = sequence_path(width, horizon)
    if not path.exists():
        build_sequence_caches(width)
    with np.load(path, allow_pickle=False) as data:
        return data["X_seq"], data["counties"], data["years"], data["lengths"]


def scale_sequences(values, lengths, train_mask):
    """Fit scaling on real training steps only; padding carries no weather meaning."""
    valid = np.arange(values.shape[1])[None, :] < lengths[:, None]
    train = values[train_mask][valid[train_mask]]
    mu, sigma = train.mean(axis=0), train.std(axis=0)
    sigma[sigma == 0] = 1
    scaled = (values-mu)/sigma
    scaled[~valid] = 0
    return scaled


def test_fold(values, counties, years, lengths, target, config, head, lag, test_year, seed=88):
    """Preserve the existing expanding fold and three-year early-stopping rule."""
    history = sorted(target.loc[target.year < test_year, "year"].unique())
    inner = [test_year-3, test_year-2, test_year-1]
    train_years = [int(y) for y in history if y not in inner]
    params = fit_trends(target, history)
    anomalies = compute_anomalies(target, params, sorted(target.year.unique()))
    observed = np.array([anomalies[(f, int(y))] for f, y in zip(counties, years)], dtype=np.float32)
    train, validation, test = np.isin(years, train_years), np.isin(years, inner), years == test_year
    scaled = scale_sequences(values, lengths, train)
    if lag:
        first = int(target.year.min())
        # The existing training convention centers the unavailable first-panel lag at zero.
        prior = np.array([0.0 if int(y) == first else anomalies[(f, int(y)-1)]
                          for f, y in zip(counties, years)], dtype=np.float32).reshape(-1, 1)
        mu, sigma = prior[train].mean(), prior[train].std()
        if sigma <= 0:
            raise ValueError("Lagged yield has no training variance")
        prior = (prior-mu)/sigma
        lag_tr, lag_va, lag_te = prior[train], prior[validation], prior[test]
    else:
        lag_tr = lag_va = lag_te = None
    predictions, _, _ = fit_and_predict(
        scaled[train], observed[train], lag_tr, scaled[validation], observed[validation], lag_va,
        scaled[test], lag_te, hidden_size=config["hidden_size"], num_layers=config["num_layers"],
        dropout=config["dropout"], batch_size=config["batch_size"], seed=seed, head_type=head,
        lengths_tr=lengths[train], lengths_va=lengths[validation], lengths_te=lengths[test])
    if not np.isfinite(predictions).all() or len(predictions) != int(test.sum()):
        raise ValueError("Test fold produced incomplete or nonfinite predictions")
    return pd.DataFrame({"year": years[test], "county_fips": counties[test],
                         "actual_anomaly": observed[test], "pred_anomaly": predictions})


def fit_saved_fold(horizon, head, lag, year, config, seed=88):
    """Independent deterministic fold; the parent alone publishes completed results."""
    torch.set_num_threads(1)
    values, counties, years, lengths = get_sequence_slice(10, horizon)
    target = pd.read_csv(TARGET_CSV, dtype={"county_fips": str})
    target["county_fips"] = target.county_fips.str.zfill(5)
    start = time.monotonic()
    result = test_fold(values, counties, years, lengths, target, config, head, lag, year, seed=seed)
    result["horizon"], result["head_variant"] = horizon, head
    result["cell_type"], result["include_lag"], result["frequency"] = "lstm", lag, 10
    result["seed"] = seed
    return result, time.monotonic()-start


def run_single_seed(seed, horizons, heads, selected, fingerprint, completed_ref=None):
    """Run expanding test folds for a single random seed."""
    seed_preds_path = REPORT_DIR / f"test_predictions_seed_{seed}.parquet"
    if seed_preds_path.exists():
        completed = pd.read_parquet(seed_preds_path)
    elif completed_ref is not None and not completed_ref.empty and "seed" in completed_ref.columns and (completed_ref["seed"] == seed).all():
        completed = completed_ref.copy()
    else:
        completed = pd.DataFrame()

    with ProcessPoolExecutor(max_workers=6) as pool:
        for horizon in horizons:
            jobs = {}
            for head in heads:
                config = selected["variant_champions"][head]["config"]
                for lag in [False, True]:
                    for year in range(2011, 2026):
                        if not completed.empty:
                            existing = completed[completed.horizon.eq(horizon) & completed.head_variant.eq(head)
                                                 & completed.include_lag.eq(lag) & completed.year.eq(year)]
                            if len(existing) == 135 and existing.county_fips.nunique() == 135:
                                continue
                            if not existing.empty:
                                raise ValueError("Incomplete saved test fold")
                        future = pool.submit(fit_saved_fold, horizon, head, lag, year, config, seed)
                        jobs[future] = (horizon, head, lag, year)
            if jobs:
                print(f"[Seed {seed}] H={horizon:02d}: submitted {len(jobs)} independent folds", flush=True)
                for future in as_completed(jobs):
                    horizon, head, lag, year = jobs[future]
                    result, duration = future.result()
                    completed = pd.concat([completed, result], ignore_index=True)
                    temporary = seed_preds_path.with_suffix(".tmp.parquet")
                    completed.to_parquet(temporary, index=False)
                    safe_replace(temporary, seed_preds_path)
                    rmse = np.sqrt(np.mean((result.actual_anomaly - result.pred_anomaly)**2))
                    print(f"[Seed {seed}] Saved H={horizon:02d} {head} lag={lag} year={year}: RMSE={rmse:.4f}; {duration:.1f}s", flush=True)

    completed["seed"] = seed
    completed.to_parquet(seed_preds_path, index=False)
    return completed


def evaluate_multiseed_summary(seed_dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate individual seed metrics, distribution, and seed-ensemble."""
    all_seeds = sorted(seed_dfs.keys())
    combined = pd.concat([df.assign(seed=s) for s, df in seed_dfs.items()], ignore_index=True)

    records = []
    for (horizon, head, lag), grp in combined.groupby(["horizon", "head_variant", "include_lag"]):
        per_seed_r2 = {}
        per_seed_rmse = {}
        per_seed_mae = {}

        for s in all_seeds:
            sub_s = grp[grp["seed"] == s]
            if sub_s.empty:
                continue
            yt = sub_s["actual_anomaly"].values
            yp = sub_s["pred_anomaly"].values
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum(yt ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            mae = float(np.mean(np.abs(yt - yp)))
            per_seed_r2[s] = r2
            per_seed_rmse[s] = rmse
            per_seed_mae[s] = mae
            records.append({
                "type": "individual_seed",
                "seed": str(s),
                "horizon": horizon,
                "head_variant": head,
                "include_lag": lag,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "delta_r2_vs_ref": 0.0 if s == 42 else (r2 - per_seed_r2.get(42, r2)),
            })

        piv = grp.groupby(["year", "county_fips", "actual_anomaly"])["pred_anomaly"].mean().reset_index()
        yt_ens = piv["actual_anomaly"].values
        yp_ens = piv["pred_anomaly"].values
        ss_res_ens = float(np.sum((yt_ens - yp_ens) ** 2))
        ss_tot_ens = float(np.sum(yt_ens ** 2))
        r2_ens = float(1.0 - ss_res_ens / ss_tot_ens) if ss_tot_ens > 0 else np.nan
        rmse_ens = float(np.sqrt(np.mean((yt_ens - yp_ens) ** 2)))
        mae_ens = float(np.mean(np.abs(yt_ens - yp_ens)))

        r2_vals = list(per_seed_r2.values())
        rmse_vals = list(per_seed_rmse.values())
        r2_ref = per_seed_r2.get(42, r2_vals[0] if r2_vals else np.nan)
        rmse_ref = per_seed_rmse.get(42, rmse_vals[0] if rmse_vals else np.nan)

        records.append({
            "type": "seed_ensemble",
            "seed": f"Ensemble_{len(all_seeds)}seeds",
            "horizon": horizon,
            "head_variant": head,
            "include_lag": lag,
            "r2": r2_ens,
            "rmse": rmse_ens,
            "mae": mae_ens,
            "r2_mean_seeds": float(np.mean(r2_vals)),
            "r2_std_seeds": float(np.std(r2_vals)),
            "rmse_mean_seeds": float(np.mean(rmse_vals)),
            "rmse_std_seeds": float(np.std(rmse_vals)),
            "delta_r2_vs_ref": r2_ens - r2_ref,
            "delta_rmse_vs_ref": rmse_ens - rmse_ref,
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="LSTM test forecast expanding window training")
    parser.add_argument("--seeds", type=int, nargs="+", default=[88], help="Random seeds (default: [88])")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(range(11, 0, -1)), help="Horizons to run (default: 11..1)")
    parser.add_argument("--heads", type=str, nargs="+", default=["prelu", "linear"], help="Heads to run (default: prelu linear)")
    args = parser.parse_args()

    torch.set_num_threads(1)
    build_sequence_caches(10)
    validation_bytes = CHECKPOINT_PATH.read_bytes()
    selected = json.loads(validation_bytes)
    if selected["champion"]["freq"] != 10:
        raise ValueError("This run requires the author-fixed 10-day selection")
    fingerprint = {"window_version": WINDOW_VERSION,
                   "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
                   "selected_variants": selected["variant_champions"],
                   "test_start": 2011, "test_end": 2025}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    completed_ref = None
    if OUT_PREDS.exists():
        completed_ref = pd.read_parquet(OUT_PREDS)

    seed_dfs = {}
    for s in args.seeds:
        print(f"\n==========================================")
        print(f"Executing expanding LSTM test folds for SEED {s}")
        print(f"==========================================")
        seed_df = run_single_seed(s, args.horizons, args.heads, selected, fingerprint, completed_ref)
        seed_dfs[s] = seed_df

    primary_seed = args.seeds[0] if len(args.seeds) == 1 else None
    if primary_seed in seed_dfs:
        df_canon = seed_dfs[primary_seed]
        if len(df_canon) == 11 * 4 * 15 * 135 and not df_canon.duplicated(["horizon", "head_variant", "include_lag", "year", "county_fips"]).any():
            temporary = OUT_PREDS.with_suffix(".tmp.parquet")
            df_canon.to_parquet(temporary, index=False)
            safe_replace(temporary, OUT_PREDS)
            fingerprint["published_sha256"] = hashlib.sha256(OUT_PREDS.read_bytes()).hexdigest()
            fingerprint["canonical_seed"] = primary_seed
            temporary_manifest = RUN_MANIFEST.with_suffix(".tmp.json")
            temporary_manifest.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
            temporary_manifest.replace(RUN_MANIFEST)
            RUN_PREDS.unlink(missing_ok=True)
            print(f"Published canonical predictions_lstm.parquet for seed {primary_seed}", flush=True)

    if len(args.seeds) > 1:
        print("\n=== Multi-Seed Evaluation & Seed-Ensemble Analysis ===")
        summary_df = evaluate_multiseed_summary(seed_dfs)
        diag_dir = ROOT / "output" / "data" / "processed" / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        summary_path = diag_dir / "lstm_multiseed_evaluation.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved multi-seed evaluation to {summary_path}")

        # Display key horizons comparison
        ens_rows = summary_df[summary_df["type"] == "seed_ensemble"].sort_values(["horizon", "head_variant", "include_lag"])
        print("\nMulti-Seed Performance Summary (Key Horizons):")
        for _, row in ens_rows.iterrows():
            h, head, lag = int(row["horizon"]), row["head_variant"], row["include_lag"]
            spec = f"H={h:02d} {head} (lag={lag})"
            r2_ref = row["r2"] - row["delta_r2_vs_ref"]
            print(f"  {spec:25s} | Ref (seed 42): R2={r2_ref:.4f} | Seeds: {row['r2_mean_seeds']:.4f} +/- {row['r2_std_seeds']:.4f} | Seed-Ens: R2={row['r2']:.4f} ({row['delta_r2_vs_ref']:+.4f}) | RMSE={row['rmse']:.4f}")


if __name__ == "__main__":
    main()
