"""ML forecasting pipeline: tabular models only (no LSTM).

Models: Naive Trend baseline, ElasticNet, SVR (RBF), Random Forest, XGBoost.
Protocol:
  - Train-only OLS detrending per county per fold.
  - Train-only feature standardization.
  - Single-pass grid search on 1985-1995 validation partition.
  - Expanding-window OOS evaluation: 1996-2025 (30 folds, 4,050 evaluations/horizon).
  - Primary metric: pooled R2_OOS. Secondary: year-averaged R2_OOS.

Outputs:
  output/data/processed/hyperparameters_optimal.json  -- best hyperparams per model x horizon
  output/data/processed/evaluation_metrics.parquet    -- all metrics per model x horizon x year
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
from sklearn.svm import SVR
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
VAL_END          = 1995   # inclusive
TEST_START       = 1996
TEST_END         = 2025   # inclusive

# ---------------------------------------------------------------------------
# Hyperparameter grids (from Chapter 4, Table 4.3)
# ---------------------------------------------------------------------------
GRIDS: dict[str, list[dict]] = {
    "ElasticNet": [
        {"alpha": a, "l1_ratio": l1}
        for a, l1 in product([1e-3, 1e-2, 1e-1, 1.0, 10.0], [0.1, 0.3, 0.5, 0.7, 0.9])
    ],  # 25 combinations
    "SVR": [
        {"C": C, "epsilon": eps, "gamma_factor": gf}
        for C, eps, gf in product([0.5, 2.0, 10.0], [0.1, 0.5, 1.0], [0.1, 1.0, 5.0])
    ],  # 27 combinations
    "RandomForest": [
        {"max_features": mf, "min_samples_leaf": ml, "max_depth": md}
        for mf, ml, md in product(["sqrt", 0.33, 0.5], [5, 15, 30], [8, 12, None])
    ],  # 27 combinations
    "XGBoost": [
        {"learning_rate": lr, "max_depth": md, "colsample_bytree": cs, "reg_lambda": rl}
        for lr, md, cs, rl in product([0.03, 0.08], [3, 5], [0.6, 0.8], [1.0, 10.0])
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

def make_model(name: str, params: dict, gamma_scale: float = 1.0):
    if name == "ElasticNet":
        return ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            max_iter=5000,
            random_state=42,
        )
    elif name == "SVR":
        gamma = params["gamma_factor"] * gamma_scale
        return SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"], gamma=gamma)
    elif name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300,
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
            max_depth=params["max_depth"],
            random_state=42,
            n_jobs=-1,
        )
    elif name == "XGBoost":
        return XGBRegressor(
            n_estimators=400,
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            subsample=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Gamma scale helper for SVR
# ---------------------------------------------------------------------------

def compute_gamma_scale(X_train: np.ndarray) -> float:
    n_features = X_train.shape[1]
    var_mean = float(np.var(X_train, ddof=1)) if X_train.shape[0] > 1 else 1.0
    if n_features == 0 or var_mean == 0:
        return 1.0
    return 1.0 / (n_features * var_mean)


# ---------------------------------------------------------------------------
# Single-pass grid search on validation partition
# ---------------------------------------------------------------------------

def grid_search_validation(
    model_name: str,
    X_base_train: np.ndarray,
    y_base_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler,
) -> dict:
    """Search over pre-specified grid; return params minimising val RMSE."""
    gamma_scale = compute_gamma_scale(X_base_train)
    best_params = None
    best_rmse   = float("inf")

    X_base_sc = scaler.transform(X_base_train)
    X_val_sc  = scaler.transform(X_val)

    for params in GRIDS[model_name]:
        model = make_model(model_name, params, gamma_scale=gamma_scale)
        try:
            model.fit(X_base_sc, y_base_train)
            y_pred = model.predict(X_val_sc)
            val_rmse = rmse(y_val, y_pred)
            if val_rmse < best_rmse:
                best_rmse   = val_rmse
                best_params = {**params, "gamma_scale": gamma_scale}
        except Exception:
            continue

    return best_params if best_params is not None else GRIDS[model_name][0]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(horizons: list[int] | None = None) -> None:
    t0 = time.time()
    print("=== ML Pipeline (tabular models, no LSTM) ===", flush=True)

    # Load target
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    all_years   = sorted(yield_df["year"].unique().tolist())
    panel_fips  = sorted(yield_df["county_fips"].unique().tolist())
    print(f"Panel: {len(panel_fips)} counties, {min(all_years)}-{max(all_years)}", flush=True)

    base_train_years = [y for y in all_years if BASE_TRAIN_START <= y <= BASE_TRAIN_END]
    val_years        = [y for y in all_years if VAL_START <= y <= VAL_END]
    test_years       = [y for y in all_years if TEST_START <= y <= TEST_END]
    print(f"Base train: {len(base_train_years)}yr | Val: {len(val_years)}yr | Test: {len(test_years)}yr", flush=True)

    MODEL_NAMES = ["ElasticNet", "SVR", "RandomForest", "XGBoost"]
    if horizons is None:
        horizons = list(range(1, 13))

    # Storage
    all_optimal_params: dict = {}   # model -> H -> params
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
            # Naive baseline only: predict anomaly=0 everywhere
            print(f"  H=12: Naive baseline (0 weather features)", flush=True)
            # Compute naive metrics on test set
            trend_params_full = fit_county_trends(yield_df, all_years[:all_years.index(TEST_START)])
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

        # Step A: Fit scaler + best hyperparams on base-train -> validation
        base_train_panel = panel[panel["year"].isin(base_train_years)]
        val_panel        = panel[panel["year"].isin(val_years)]

        # Train-only detrending for base_train + val
        all_train_val_years = base_train_years + val_years
        trend_params_tv = fit_county_trends(yield_df, all_train_val_years)
        base_anom = compute_anomalies(base_train_panel, trend_params_tv)
        val_anom  = compute_anomalies(val_panel,        trend_params_tv)

        X_base = base_anom[feature_cols].values.astype(float)
        y_base = base_anom["anomaly"].values.astype(float)
        X_val  = val_anom[feature_cols].values.astype(float)
        y_val  = val_anom["anomaly"].values.astype(float)

        # Train-only scaler
        scaler_gs = StandardScaler()
        scaler_gs.fit(X_base)

        optimal_params: dict[str, dict] = {}
        for model_name in MODEL_NAMES:
            print(f"  Grid search {model_name} ({len(GRIDS[model_name])} configs)...", end=" ", flush=True)
            t_gs = time.time()
            best = grid_search_validation(model_name, X_base, y_base, X_val, y_val, scaler_gs)
            optimal_params[model_name] = best
            print(f"done ({time.time()-t_gs:.1f}s) | best: {best}", flush=True)

        all_optimal_params[H] = optimal_params

        # Step B: Expanding-window OOS evaluation (1996-2025)
        gamma_scale = compute_gamma_scale(X_base)

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
                model = make_model(model_name, params, gamma_scale=gamma_scale)
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

        print(f"  H={H} OOS evaluation complete.", flush=True)

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
    # Allow running specific horizons: python run_ml_pipeline.py 1 2 3
    if len(sys.argv) > 1:
        horizons_arg = [int(x) for x in sys.argv[1:]]
        run_pipeline(horizons=horizons_arg)
    else:
        run_pipeline()
