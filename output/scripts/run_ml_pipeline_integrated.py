"""ML forecasting pipeline: Integrated model (Weather + EDD30 + Lagged Yield Anomaly).

Models:
  - Naive Trend baseline
  - ElasticNet (regularized linear)
  - RandomForest (bagging trees)
  - XGBoost (gradient boosted trees)

Key Specification:
  - Feature set: 17 bioclimatic indicators per month (including EDD30) +
    spatial coordinates (lat_norm, lon_norm) +
    train-only detrended prior-year yield anomaly (anomaly_lag1 = epsilon_{c, t-1}).
  - Train-only OLS detrending per county per fold.
  - Train-only feature standardization.
  - Hyperparameter tuning via chronological expanding window across the 1985-1995 validation partition (11 folds).
  - Out-of-sample expanding-window evaluation: 1996-2025 (30 folds, 4,050 evaluations/horizon).
  - Primary metric: pooled R2_OOS. Secondary: year-averaged R2_OOS.

Outputs:
  output/data/processed/hyperparameters_optimal_integrated.json
  output/data/processed/evaluation_metrics_integrated.parquet
  output/data/processed/evaluation_summary_integrated.csv
  output/data/processed/evaluation_summary_integrated.parquet
  output/data/processed/predictions_integrated.parquet
  output/data/processed/diagnostics/comparison_weather_vs_integrated.csv
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
FEAT_DIR   = ROOT / "output" / "data" / "processed" / "model_datasets"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
OUT_DIR    = ROOT / "output" / "data" / "processed"
DIAG_DIR   = OUT_DIR / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Partition constants
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


def add_lagged_yield_anomaly(
    df: pd.DataFrame,
    yield_df: pd.DataFrame,
    trend_params: dict[str, tuple[float, float]],
    max_year: int,
) -> pd.DataFrame:
    """Compute prior-year detrended yield anomaly epsilon_{c, t-1} using train-only trends.
    
    For the first panel year (1951), epsilon_{c, 1950} = 0.0 (uninformative neutral prior).
    For any test year Y, Y-1 was in the training set, so epsilon_{c, Y-1} is 100% strictly available.
    """
    hist_yield = yield_df[yield_df["year"] <= max_year].copy()
    hist_anom = compute_anomalies(hist_yield, trend_params)
    
    # Map (county_fips, year) -> anomaly
    anom_lookup = hist_anom.set_index(["county_fips", "year"])["anomaly"].to_dict()

    df = df.copy()
    df["anomaly_lag1"] = df.apply(
        lambda r: anom_lookup.get((str(r["county_fips"]), int(r["year"]) - 1), 0.0),
        axis=1,
    )
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
    ss_tot = np.sum(y_true ** 2)
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
        except Exception:
            continue

    return best_params if best_params is not None else GRIDS[model_name][0]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(horizons: list[int] | None = None) -> None:
    t0 = time.time()
    print("=== Integrated ML Pipeline (Weather + EDD30 + Lagged Yield Anomaly) ===", flush=True)

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

    all_optimal_params: dict = {}
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

        # Merge base features with yield
        panel = yield_df.merge(feat_df, on=["county_fips", "year"], how="inner")
        panel = panel.sort_values(["year", "county_fips"]).reset_index(drop=True)

        # Feature columns include weather indicators + lat_norm, lon_norm + anomaly_lag1
        weather_cols = [c for c in feat_df.columns if c not in ("county_fips", "year")]
        feature_cols = weather_cols + ["anomaly_lag1"]

        # Step A: Build 11 validation folds using chronological expanding window
        print(f"  Building {len(val_years)} expanding validation folds (1985-1995)...", end=" ", flush=True)
        t_fold = time.time()
        val_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for y_val in val_years:
            train_yrs = [y for y in all_years if y < y_val]
            train_panel = panel[panel["year"].isin(train_yrs)].copy()
            val_panel   = panel[panel["year"] == y_val].copy()

            tp = fit_county_trends(yield_df, train_yrs)
            train_anom = compute_anomalies(train_panel, tp)
            val_anom   = compute_anomalies(val_panel,   tp)

            # Add train-only lagged yield anomaly
            train_anom = add_lagged_yield_anomaly(train_anom, yield_df, tp, max_year=y_val - 1)
            val_anom   = add_lagged_yield_anomaly(val_anom,   yield_df, tp, max_year=y_val)

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
            train_panel = panel[panel["year"].isin(train_yrs)].copy()
            test_panel  = panel[panel["year"] == test_y].copy()
            if test_panel.empty or train_panel.empty:
                continue

            # Train-only detrending
            tp = fit_county_trends(yield_df, train_yrs)
            train_anom = compute_anomalies(train_panel, tp)
            test_anom  = compute_anomalies(test_panel,  tp)

            # Add train-only lagged yield anomaly
            train_anom = add_lagged_yield_anomaly(train_anom, yield_df, tp, max_year=test_y - 1)
            test_anom  = add_lagged_yield_anomaly(test_anom,   yield_df, tp, max_year=test_y)

            X_tr = train_anom[feature_cols].values.astype(float)
            y_tr = train_anom["anomaly"].values.astype(float)
            X_te = test_anom[feature_cols].values.astype(float)
            y_te = test_anom["anomaly"].values.astype(float)

            # Train-only scaler
            scaler_fold = StandardScaler().fit(X_tr)
            X_tr_sc = scaler_fold.transform(X_tr)
            X_te_sc = scaler_fold.transform(X_te)

            # Naive prediction (anomaly = 0)
            y_naive = np.zeros_like(y_te)
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
    # Consolidated Metrics
    # ---------------------------------------------------------------------------
    print("\nComputing pooled R2_OOS and Skill Scores...", flush=True)
    metrics_df  = pd.DataFrame(all_metrics)
    pred_df     = pd.DataFrame(all_predictions)

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

    year_avg = (
        metrics_df
        .groupby(["model", "H"])["R2_year"]
        .mean()
        .reset_index()
        .rename(columns={"R2_year": "R2_OOS_year_avg"})
    )

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
    metrics_path = OUT_DIR / "evaluation_metrics_integrated.parquet"
    metrics_df.to_parquet(metrics_path, index=False)

    summary_path = OUT_DIR / "evaluation_summary_integrated.parquet"
    summary_df.to_parquet(summary_path, index=False)
    summary_df.to_csv(OUT_DIR / "evaluation_summary_integrated.csv", index=False)

    pred_path = OUT_DIR / "predictions_integrated.parquet"
    pred_df.to_parquet(pred_path, index=False)

    hp_path = OUT_DIR / "hyperparameters_optimal_integrated.json"
    with open(hp_path, "w") as f:
        json.dump({str(H): v for H, v in all_optimal_params.items()}, f, indent=2)

    print(f"\n=== Integrated Pipeline complete in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"Outputs written to {OUT_DIR}", flush=True)
    print("\n--- Integrated Summary (pooled R2_OOS) ---", flush=True)
    pivot = summary_df.pivot(index="H", columns="model", values="R2_OOS_pooled").sort_index(ascending=False)
    print(pivot.to_string(), flush=True)

    # ---------------------------------------------------------------------------
    # Direct Comparison Table: Weather-Only vs Integrated (Weather + Lag1)
    # ---------------------------------------------------------------------------
    weather_only_path = OUT_DIR / "evaluation_summary.csv"
    if weather_only_path.exists():
        try:
            wo_df = pd.read_csv(weather_only_path)
            comp = summary_df[["model", "H", "R2_OOS_pooled", "RMSE_OOS_pooled"]].merge(
                wo_df[["model", "H", "R2_OOS_pooled", "RMSE_OOS_pooled"]],
                on=["model", "H"],
                suffixes=("_integrated", "_weather_only"),
            )
            comp["delta_R2"] = comp["R2_OOS_pooled_integrated"] - comp["R2_OOS_pooled_weather_only"]
            comp["delta_RMSE"] = comp["RMSE_OOS_pooled_integrated"] - comp["RMSE_OOS_pooled_weather_only"]
            comp_csv = DIAG_DIR / "comparison_weather_vs_integrated.csv"
            comp.to_csv(comp_csv, index=False)
            print(f"\nSaved direct ablation comparison to {comp_csv}", flush=True)
            print("\n--- Ablation Comparison: Delta R2_OOS (Integrated - Weather_Only) ---", flush=True)
            piv_comp = comp.pivot(index="H", columns="model", values="delta_R2").sort_index(ascending=False)
            print(piv_comp.to_string(), flush=True)
        except Exception as e:
            print(f"Warning: could not generate comparison table: {e}", flush=True)

    # Automatically run econometric diagnostics on predictions_integrated
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.append(str(scripts_dir))
        from analyze_model_diagnostics import run_diagnostics
        run_diagnostics(pred_path)
    except Exception as e:
        print(f"Warning: diagnostics failed to run: {e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        horizons_arg = [int(x) for x in sys.argv[1:]]
        run_pipeline(horizons=horizons_arg)
    else:
        run_pipeline()
