"""ML forecasting pipeline: tabular models only (no SVR, no LSTM).

Models:
  - Naive Trend baseline
  - ElasticNet (regularized linear)
  - RandomForest (bagging trees)
  - XGBoost (gradient boosted trees)

Validation & Testing Protocol:
  - Train-only OLS detrending per county per fold.
  - Train-only feature standardization.
  - Hyperparameter tuning via chronological expanding window across the 1985-1995 validation partition (11 folds).
  - Out-of-sample expanding-window evaluation: 1996-2025 (30 folds, 4,050 evaluations/horizon).
  - Primary metric: pooled R2_OOS. Secondary: year-averaged R2_OOS.

Outputs:
  output/data/processed/hyperparameters_optimal.json  -- best hyperparams per model x horizon
  output/data/processed/evaluation_metrics.parquet    -- all metrics per model x horizon x year
  output/data/processed/evaluation_summary.csv        -- consolidated metric table
  output/data/processed/evaluation_summary.parquet    -- parquet version of summary
  output/data/processed/predictions.parquet           -- raw predictions for SHAP and figures
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # repo root: Tesi/
FEAT_DIR  = ROOT / "output" / "data" / "processed" / "model_datasets"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
OUT_DIR   = ROOT / "output" / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Partition constants (consolidated 2026-09-29)
# ---------------------------------------------------------------------------
BASE_TRAIN_START = 1951
BASE_TRAIN_END   = 1984   # inclusive
VAL_START        = 1985
VAL_END          = 1995   # inclusive (11 years)
TEST_START       = 1996
TEST_END         = 2025   # inclusive (30 years)

# ---------------------------------------------------------------------------
# Hyperparameter grids (from Chapter 4, Table 4.3)
# ---------------------------------------------------------------------------
GRIDS: dict[str, list[dict]] = {
    "ElasticNet": [
        {"alpha": a, "l1_ratio": l1}
        for a, l1 in product([0.01, 0.05, 0.1, 0.5, 1.0, 5.0], [0.1, 0.3, 0.5, 0.7, 0.9])
    ],  # 30 combinations
    "RandomForest": [
        {"max_features": mf, "min_samples_leaf": ml, "max_depth": md}
        for mf, ml, md in product(["sqrt", 0.33, 0.5], [5, 20], [8, 14, None])
    ],  # 18 combinations
    "XGBoost": [
        {
            "learning_rate": lr,
            "max_depth": md,
            "colsample_bytree": cs,
            "subsample": ss,
            "reg_lambda": rl,
            "reg_alpha": ra,
            "min_child_weight": mcw,
        }
        for lr, md, cs, ss, rl, ra, mcw in product(
            [0.03, 0.06],      # 2 learning rates
            [3, 5],            # 2 tree depths
            [0.7],             # feature fraction per tree
            [0.8],             # stochastic row subsampling
            [1.0, 5.0],        # L2 ridge penalty
            [0.0, 0.5],        # L1 lasso penalty on leaves
            [3],               # minimum child weight to suppress leaf noise
        )
    ],  # 16 combinations
}

# ---------------------------------------------------------------------------
# Detrending (OLS, train-only per county)
# ---------------------------------------------------------------------------

def fit_county_trends(
    yield_df: pd.DataFrame, train_years: list[int]
) -> dict[str, tuple[float, float]]:
    """Fit linear OLS trend alpha + beta*year per county on train_years only."""
    train = yield_df[yield_df["year"].isin(train_years)]
    params = {}
    for fips, grp in train.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        # OLS via normal equations
        t_bar = t_vals.mean()
        y_bar = y_vals.mean()
        beta  = np.sum((t_vals - t_bar) * (y_vals - y_bar)) / np.sum((t_vals - t_bar) ** 2)
        alpha = y_bar - beta * t_bar
        params[str(fips)] = (alpha, beta)
    return params


def compute_anomalies(
    yield_df: pd.DataFrame, trend_params: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Compute yield anomaly epsilon = yield - trend for all rows in yield_df."""
    df = yield_df.copy()
    df["trend"] = df.apply(
        lambda r: trend_params[str(r["county_fips"])][0]
                  + trend_params[str(r["county_fips"])][1] * r["year"],
        axis=1,
    )
    df["anomaly"] = df["yield_bu_per_acre"] - df["trend"]
    return df


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_oos_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)   # denominator = sum of squares (anomaly, zero mean)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def skill_score(rmse_model: float, rmse_naive: float) -> float:
    if rmse_naive == 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_naive)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(name: str, params: dict, for_tuning: bool = False):
    if name == "ElasticNet":
        return ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )
    elif name == "RandomForest":
        n_trees = 60 if for_tuning else 150
        return RandomForestRegressor(
            n_estimators=n_trees,
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
            max_depth=params["max_depth"],
            random_state=42,
            n_jobs=-1,
        )
    elif name == "XGBoost":
        n_trees = 150 if for_tuning else 300
        return XGBRegressor(
            n_estimators=n_trees,
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            colsample_bytree=params["colsample_bytree"],
            subsample=params.get("subsample", 0.8),
            reg_lambda=params["reg_lambda"],
            reg_alpha=params.get("reg_alpha", 0.0),
            min_child_weight=params.get("min_child_weight", 3),
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Expanding-window grid search on validation partition (1985-1995)
# ---------------------------------------------------------------------------

def grid_search_validation_expanding(
    model_name: str,
    val_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    """Evaluate candidate parameter sets using chronological expanding window across the 11 validation folds.
    
    Returns parameter dict minimising pooled validation RMSE.
    """
    best_params = None
    best_rmse   = float("inf")

    for params in GRIDS[model_name]:
        preds_all = []
        trues_all = []
        try:
            for X_tr_sc, y_tr, X_va_sc, y_va in val_folds:
                model = make_model(model_name, params, for_tuning=True)
                model.fit(X_tr_sc, y_tr)
                y_hat = model.predict(X_va_sc)
                preds_all.append(y_hat)
                trues_all.append(y_va)
            
            y_pred_pooled = np.concatenate(preds_all)
            y_true_pooled = np.concatenate(trues_all)
            val_rmse = rmse(y_true_pooled, y_pred_pooled)

            if val_rmse < best_rmse:
                best_rmse   = val_rmse
                best_params = params
        except Exception as e:
            continue

    return best_params if best_params is not None else GRIDS[model_name][0]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(horizons: list[int] | None = None) -> None:
    t0 = time.time()
    print("=== ML Pipeline: Tabular Models (Naive, ElasticNet, RandomForest, XGBoost) ===", flush=True)

    # Load target
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years   = sorted(yield_df["year"].unique().tolist())
    panel_fips  = sorted(yield_df["county_fips"].unique().tolist())
    print(f"Panel: {len(panel_fips)} counties, {min(all_years)}-{max(all_years)}", flush=True)

    base_train_years = [y for y in all_years if BASE_TRAIN_START <= y <= BASE_TRAIN_END]
    val_years        = [y for y in all_years if VAL_START <= y <= VAL_END]
    test_years       = [y for y in all_years if TEST_START <= y <= TEST_END]
    print(f"Base train: {len(base_train_years)}yr | Val expanding: {len(val_years)}yr | Test expanding: {len(test_years)}yr", flush=True)

    MODEL_NAMES = ["ElasticNet", "RandomForest", "XGBoost"]
    if horizons is None:
        horizons = list(range(1, 13))

    # Storage
    all_optimal_params: dict = {}   # H -> model -> params
    all_metrics: list[dict] = []
    all_predictions: list[dict] = []

    for H in horizons:
        print(f"\n--- Horizon H={H} ---", flush=True)

        feat_path = FEAT_DIR / f"features_H{H:02d}.parquet"
        if not feat_path.exists():
            print(f"  SKIP: {feat_path} not found", flush=True)
            continue

        feat_df = pd.read_parquet(feat_path)
        feat_df["county_fips"] = feat_df["county_fips"].astype(str)
        feat_df = feat_df.rename(columns={"crop_year": "year"})

        # Merge features with yield
        panel = yield_df.merge(feat_df, on=["county_fips", "year"], how="inner")
        panel = panel.sort_values(["year", "county_fips"]).reset_index(drop=True)

        feature_cols = [c for c in feat_df.columns if c not in ("county_fips", "year")]

        # H=12: only lat_norm, lon_norm — no weather features
        if H == 12:
            print(f"  H=12: Naive baseline (0 weather features)", flush=True)
            for test_y in test_years:
                train_yrs = [y for y in all_years if y < test_y]
                tp = fit_county_trends(yield_df, train_yrs)
                test_rows = panel[panel["year"] == test_y]
                if test_rows.empty:
                    continue
                anom = compute_anomalies(test_rows, tp)
                y_true = anom["anomaly"].values
                y_pred_naive = np.zeros_like(y_true)
                for _, row in test_rows.iterrows():
                    all_predictions.append({
                        "model": "Naive",
                        "H": H,
                        "year": test_y,
                        "county_fips": row["county_fips"],
                        "y_true_anomaly": float(anom[anom["county_fips"] == row["county_fips"]]["anomaly"].values[0]),
                        "y_pred_anomaly": 0.0,
                        "trend": float(anom[anom["county_fips"] == row["county_fips"]]["trend"].values[0]),
                    })
                all_metrics.append({
                    "model": "Naive", "H": H, "year": test_y,
                    "RMSE": rmse(y_true, y_pred_naive),
                    "MAE": mae(y_true, y_pred_naive),
                    "R2_year": r2_oos_pooled(y_true, y_pred_naive),
                    "n_obs": len(y_true),
                })
            continue

        # --- For H < 12 ---

        # Step A: Build 11 validation folds using chronological expanding window
        print(f"  Building {len(val_years)} expanding validation folds (1985-1995)...", end=" ", flush=True)
        t_fold = time.time()
        val_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for y_val in val_years:
            train_yrs = [y for y in all_years if y < y_val]
            train_panel = panel[panel["year"].isin(train_yrs)]
            val_panel   = panel[panel["year"] == y_val]

            tp = fit_county_trends(yield_df, train_yrs)
            train_anom = compute_anomalies(train_panel, tp)
            val_anom   = compute_anomalies(val_panel,   tp)

            X_tr = train_anom[feature_cols].values.astype(float)
            y_tr = train_anom["anomaly"].values.astype(float)
            X_va = val_anom[feature_cols].values.astype(float)
            y_va = val_anom["anomaly"].values.astype(float)

            scaler_val = StandardScaler().fit(X_tr)
            val_folds.append((scaler_val.transform(X_tr), y_tr, scaler_val.transform(X_va), y_va))
        print(f"done ({time.time()-t_fold:.2f}s)", flush=True)

        # Step B: Expanding-window grid search across validation folds
        optimal_params: dict[str, dict] = {}
        for model_name in MODEL_NAMES:
            print(f"  Grid search {model_name} ({len(GRIDS[model_name])} configs x 11 folds)...", end=" ", flush=True)
            t_gs = time.time()
            best = grid_search_validation_expanding(model_name, val_folds)
            optimal_params[model_name] = best
            print(f"done ({time.time()-t_gs:.1f}s) | best: {best}", flush=True)

        all_optimal_params[H] = optimal_params

        # Step C: Expanding-window OOS evaluation (1996-2025, 30 years)
        print(f"  Evaluating expanding-window OOS across 1996-2025 (30 test years)...", end=" ", flush=True)
        t_test = time.time()

        for test_y in test_years:
            train_yrs = [y for y in all_years if y < test_y]
            train_panel = panel[panel["year"].isin(train_yrs)]
            test_panel  = panel[panel["year"] == test_y]
            if test_panel.empty or train_panel.empty:
                continue

            # Train-only detrending for this fold
            tp = fit_county_trends(yield_df, train_yrs)
            train_anom = compute_anomalies(train_panel, tp)
            test_anom  = compute_anomalies(test_panel,  tp)

            X_tr = train_anom[feature_cols].values.astype(float)
            y_tr = train_anom["anomaly"].values.astype(float)
            X_te = test_anom[feature_cols].values.astype(float)
            y_te = test_anom["anomaly"].values.astype(float)

            # Train-only scaler for this fold
            scaler_fold = StandardScaler()
            scaler_fold.fit(X_tr)
            X_tr_sc = scaler_fold.transform(X_tr)
            X_te_sc = scaler_fold.transform(X_te)

            # Naive prediction (anomaly=0 everywhere)
            y_naive = np.zeros_like(y_te)

            # Save naive metrics and predictions
            all_metrics.append({
                "model": "Naive", "H": H, "year": test_y,
                "RMSE": rmse(y_te, y_naive),
                "MAE": mae(y_te, y_naive),
                "R2_year": r2_oos_pooled(y_te, y_naive),
                "n_obs": len(y_te),
            })
            for i, row in enumerate(test_anom.itertuples(index=False)):
                all_predictions.append({
                    "model": "Naive", "H": H, "year": test_y,
                    "county_fips": row.county_fips,
                    "y_true_anomaly": float(y_te[i]),
                    "y_pred_anomaly": 0.0,
                    "trend": float(row.trend),
                })

            # ML models
            for model_name in MODEL_NAMES:
                params = optimal_params[model_name]
                model = make_model(model_name, params, for_tuning=False)
                try:
                    model.fit(X_tr_sc, y_tr)
                    y_pred = model.predict(X_te_sc)
                except Exception as e:
                    print(f"    WARNING: {model_name} H={H} y={test_y} failed: {e}", flush=True)
                    continue

                all_metrics.append({
                    "model": model_name, "H": H, "year": test_y,
                    "RMSE": rmse(y_te, y_pred),
                    "MAE": mae(y_te, y_pred),
                    "R2_year": r2_oos_pooled(y_te, y_pred),
                    "n_obs": len(y_te),
                })
                for i, row in enumerate(test_anom.itertuples(index=False)):
                    all_predictions.append({
                        "model": model_name, "H": H, "year": test_y,
                        "county_fips": row.county_fips,
                        "y_true_anomaly": float(y_te[i]),
                        "y_pred_anomaly": float(y_pred[i]),
                        "trend": float(row.trend),
                    })

        print(f"done ({time.time()-t_test:.1f}s)", flush=True)

    # ---------------------------------------------------------------------------
    # Compute pooled R2_OOS across all test years per (model, H)
    # ---------------------------------------------------------------------------
    print("\nComputing pooled R2_OOS and Skill Scores...", flush=True)
    metrics_df  = pd.DataFrame(all_metrics)
    pred_df     = pd.DataFrame(all_predictions)

    # Pooled R2_OOS: computed from all predictions at once per (model, H)
    pooled_records = []
    for (model_name, H), grp in pred_df.groupby(["model", "H"]):
        y_true = grp["y_true_anomaly"].values
        y_pred = grp["y_pred_anomaly"].values
        r2     = r2_oos_pooled(y_true, y_pred)
        rmse_v = rmse(y_true, y_pred)
        mae_v  = mae(y_true, y_pred)
        pooled_records.append({
            "model": model_name, "H": H,
            "R2_OOS_pooled": r2,
            "RMSE_OOS_pooled": rmse_v,
            "MAE_OOS_pooled": mae_v,
            "n_obs": len(y_true),
        })
    pooled_df = pd.DataFrame(pooled_records)

    # Year-averaged R2_OOS (secondary metric)
    year_avg = (
        metrics_df
        .groupby(["model", "H"])["R2_year"]
        .mean()
        .reset_index()
        .rename(columns={"R2_year": "R2_OOS_year_avg"})
    )

    # Skill score relative to Naive pooled RMSE per H
    naive_rmse_by_H = (
        pooled_df[pooled_df["model"] == "Naive"]
        .set_index("H")["RMSE_OOS_pooled"]
    )
    pooled_df["skill_RMSE"] = pooled_df.apply(
        lambda r: skill_score(r["RMSE_OOS_pooled"], naive_rmse_by_H.get(r["H"], float("nan"))),
        axis=1,
    )

    summary_df = pooled_df.merge(year_avg, on=["model", "H"], how="left")

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    metrics_path = OUT_DIR / "evaluation_metrics.parquet"
    metrics_df.to_parquet(metrics_path, index=False)

    summary_path = OUT_DIR / "evaluation_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)
    summary_df.to_csv(OUT_DIR / "evaluation_summary.csv", index=False)

    pred_path = OUT_DIR / "predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    hp_path = OUT_DIR / "hyperparameters_optimal.json"
    with open(hp_path, "w") as f:
        json.dump({str(H): v for H, v in all_optimal_params.items()}, f, indent=2)

    print(f"\n=== Pipeline complete in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"Outputs written to {OUT_DIR}", flush=True)
    print("\n--- Summary (pooled R2_OOS) ---", flush=True)
    pivot = summary_df.pivot(index="H", columns="model", values="R2_OOS_pooled").sort_index(ascending=False)
    print(pivot.to_string(), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        horizons_arg = [int(x) for x in sys.argv[1:]]
        run_pipeline(horizons=horizons_arg)
    else:
        run_pipeline()
