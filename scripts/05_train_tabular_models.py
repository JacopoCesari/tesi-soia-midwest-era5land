"""
05_train_tabular_models.py

Expanding-window out-of-sample machine learning forecasting pipeline
for county soybean yield anomalies across 12 monthly horizons (H=12 down to H=1).

Models:
- Naive Secular Baseline: zero anomaly (predicts historical linear trend)
- ElasticNet: L1/L2 regularized linear regression
- Random Forest: bagging decision tree ensemble
- XGBoost: gradient boosted decision trees

Econometric constraints:
1. Strictly chronological expanding window: train on 1951..(t-1), test on t.
2. Train-only county-specific OLS detrending to prevent lookahead bias.
3. Train-only feature scaling (StandardScaler).
4. Predicts yield anomaly epsilon_{c,t}; gross yield reconstructed as trend + anomaly.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "output" / "data" / "processed" / "model_datasets"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
HYPERPARAMS_JSON = ROOT / "output" / "data" / "processed" / "hyperparameters_optimal_integrated.json"

OUT_PREDS_PARQUET = ROOT / "output" / "data" / "processed" / "predictions.parquet"
OUT_SUMMARY_CSV = ROOT / "output" / "data" / "processed" / "evaluation_summary.csv"

# Evaluation test window
TEST_START = 2011
TEST_END = 2025


def fit_county_trends(yield_df, train_years):
    """Estimate county-specific linear trend y_t = alpha_c + beta_c * t on train years only."""
    sub = yield_df[yield_df["year"].isin(train_years)]
    trends = {}
    for fips, grp in sub.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        t_mean, y_mean = t_vals.mean(), y_vals.mean()
        beta = np.sum((t_vals - t_mean) * (y_vals - y_mean)) / np.sum((t_vals - t_mean) ** 2)
        alpha = y_mean - beta * t_mean
        trends[str(fips)] = (alpha, beta)
    return trends


def compute_anomalies(df, trends):
    """Compute yield anomalies epsilon = yield - trend."""
    out = df.copy()
    out["trend"] = [trends[str(r["county_fips"])][0] + trends[str(r["county_fips"])][1] * r["year"] for _, r in out.iterrows()]
    out["anomaly"] = out["yield_bu_per_acre"] - out["trend"]
    return out


def add_lagged_yield_anomaly(df, yield_df, trends, max_year):
    """Add detrended prior-year yield anomaly (epsilon_{c, t-1}) strictly using historical data."""
    hist = yield_df[yield_df["year"] <= max_year].copy()
    hist_anom = compute_anomalies(hist, trends)
    lookup = hist_anom.set_index(["county_fips", "year"])["anomaly"].to_dict()

    out = df.copy()
    out["anomaly_lag1"] = [lookup.get((str(r["county_fips"]), int(r["year"]) - 1), 0.0) for _, r in out.iterrows()]
    return out


def build_model(name, params):
    """Instantiate model with validated hyperparameter specifications."""
    if name == "ElasticNet":
        return ElasticNet(alpha=params.get("alpha", 0.05), l1_ratio=params.get("l1_ratio", 0.5), max_iter=1000, random_state=42)
    elif name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100,
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_depth=params.get("max_depth", 14),
            random_state=42,
            n_jobs=-1,
        )
    elif name == "XGBoost":
        return XGBRegressor(
            n_estimators=150,
            learning_rate=params.get("learning_rate", 0.06),
            max_depth=params.get("max_depth", 5),
            colsample_bytree=params.get("colsample_bytree", 0.7),
            subsample=params.get("subsample", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            min_child_weight=params.get("min_child_weight", 3),
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model name: {name}")


def main():
    print("Starting tabular ML expanding-window evaluation pipeline...")
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str).str.zfill(5)
    all_years = sorted(yield_df["year"].unique())
    test_years = [y for y in all_years if TEST_START <= y <= TEST_END]

    # Load pre-tuned optimal hyperparameters
    params_dict = {}
    if HYPERPARAMS_JSON.exists():
        with open(HYPERPARAMS_JSON, "r") as f:
            params_dict = json.load(f)

    all_predictions = []
    summary_records = []

    # Iterate through all 12 forecast horizons
    for H in range(12, 0, -1):
        feat_path = FEAT_DIR / f"features_H{H:02d}.parquet"
        if not feat_path.exists():
            print(f"Skipping H={H:02d}: {feat_path.name} not found.")
            continue

        feat_df = pd.read_parquet(feat_path)
        feat_df["county_fips"] = feat_df["county_fips"].astype(str).str.zfill(5)
        feat_df = feat_df.rename(columns={"crop_year": "year"})

        panel = yield_df.merge(feat_df, on=["county_fips", "year"]).sort_values(["year", "county_fips"]).reset_index(drop=True)
        feature_cols = [c for c in feat_df.columns if c not in ("county_fips", "year")] + ["anomaly_lag1"]

        h_params = params_dict.get(str(H), {})
        models = {
            "Naive": None,
            "ElasticNet": build_model("ElasticNet", h_params.get("ElasticNet", {})),
            "RandomForest": build_model("RandomForest", h_params.get("RandomForest", {})),
            "XGBoost": build_model("XGBoost", h_params.get("XGBoost", {})),
        }

        print(f"Evaluating Horizon H={H:02d} across {len(test_years)} out-of-sample years ({TEST_START}-{TEST_END})...")

        h_preds = {m: [] for m in models}
        y_trues = []
        y_trends = []
        fips_records = []
        year_records = []

        for y_test in test_years:
            train_years = [y for y in all_years if y < y_test]
            train_panel = panel[panel["year"].isin(train_years)]
            test_panel = panel[panel["year"] == y_test]

            # Train-only trend estimation and detrending
            trends = fit_county_trends(yield_df, train_years)
            train_anom = compute_anomalies(train_panel, trends)
            test_anom = compute_anomalies(test_panel, trends)

            # Lagged yield anomaly
            train_anom = add_lagged_yield_anomaly(train_anom, yield_df, trends, max_year=y_test - 1)
            test_anom = add_lagged_yield_anomaly(test_anom, yield_df, trends, max_year=y_test)

            X_tr = train_anom[feature_cols].values
            y_tr = train_anom["anomaly"].values
            X_te = test_anom[feature_cols].values
            y_te = test_anom["anomaly"].values

            # Standardize features using train statistics only
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            y_trues.extend(y_te)
            y_trends.extend(test_anom["trend"].values)
            fips_records.extend(test_anom["county_fips"].values)
            year_records.extend([y_test] * len(y_te))

            # Naive baseline predicts zero anomaly
            h_preds["Naive"].extend(np.zeros_like(y_te))

            # Fit machine learning models
            for m_name in ["ElasticNet", "RandomForest", "XGBoost"]:
                model = build_model(m_name, h_params.get(m_name, {}))
                model.fit(X_tr_scaled, y_tr)
                h_preds[m_name].extend(model.predict(X_te_scaled))

        # Store predictions
        y_true_arr = np.array(y_trues)
        ss_tot = np.sum(y_true_arr ** 2)

        for m_name in models:
            pred_arr = np.array(h_preds[m_name])
            ss_res = np.sum((y_true_arr - pred_arr) ** 2)
            r2_oos = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse_val = np.sqrt(np.mean((y_true_arr - pred_arr) ** 2))
            mae_val = np.mean(np.abs(y_true_arr - pred_arr))

            summary_records.append({
                "H": H,
                "model": m_name,
                "R2_OOS_pooled": r2_oos,
                "RMSE_OOS_pooled": rmse_val,
                "MAE_OOS_pooled": mae_val,
            })

            for fips, yr, yt, yp, tr in zip(fips_records, year_records, y_trues, pred_arr, y_trends):
                all_predictions.append({
                    "model": m_name,
                    "H": H,
                    "year": yr,
                    "county_fips": fips,
                    "y_true_anomaly": yt,
                    "y_pred_anomaly": yp,
                    "trend": tr,
                })

    # Save out-of-sample predictions and metrics
    OUT_PREDS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_preds = pd.DataFrame(all_predictions)
    df_preds.to_parquet(OUT_PREDS_PARQUET, index=False)

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print("Tabular model predictions and evaluation summary saved successfully.")


if __name__ == "__main__":
    main()
