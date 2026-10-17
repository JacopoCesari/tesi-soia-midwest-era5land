"""
train_tabular_full_detrend.py

Trains tabular models (Naive, ElasticNet, RandomForest, XGBoost)
under Full-Sample (In-Sample 1951-2025) OLS Detrending across 12 horizons (H=12..1)
and 15 expanding test years (2011-2025).

Both Weather-Only and Integrated (+Lag) specifications are fitted.
"""

from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEAT_DIR = ROOT / "output" / "data" / "processed" / "model_datasets"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
HPARAMS_WO_JSON = ROOT / "output" / "data" / "processed" / "hyperparameters_optimal.json"
HPARAMS_IN_JSON = ROOT / "output" / "data" / "processed" / "hyperparameters_optimal_integrated.json"
OUT_PARQUET = ROOT / "work" / "experiments" / "full_sample_detrending" / "data" / "predictions_tabular_full_detrend.parquet"

TEST_START = 2011
TEST_END = 2025


def compute_full_sample_trends_and_anomalies(yield_df):
    """Compute county-specific linear trend y_t = alpha_c + beta_c * t on the entire 1951-2025 sample."""
    df = yield_df.copy()
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    
    trends = {}
    for fips, grp in df.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        b = np.sum((t_vals - t_vals.mean()) * (y_vals - y_vals.mean())) / np.sum((t_vals - t_vals.mean()) ** 2)
        a = y_vals.mean() - b * t_vals.mean()
        trends[str(fips)] = (a, b)
        
    df["trend_full"] = [trends[str(r["county_fips"])][0] + trends[str(r["county_fips"])][1] * r["year"] for _, r in df.iterrows()]
    df["anomaly_full"] = df["yield_bu_per_acre"] - df["trend_full"]
    
    # Prior-year lag lookup
    lookup = df.set_index(["county_fips", "year"])["anomaly_full"].to_dict()
    df["anomaly_lag1"] = [lookup.get((str(r["county_fips"]), int(r["year"]) - 1), 0.0) for _, r in df.iterrows()]
    
    return df, trends


def build_model(m_type, params):
    if m_type == "ElasticNet":
        return ElasticNet(
            alpha=params.get("alpha", 0.05),
            l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=2000,
            random_state=42,
        )
    elif m_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100,
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_depth=params.get("max_depth", 14),
            random_state=42,
            n_jobs=2,
        )
    elif m_type == "XGBoost":
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
            n_jobs=2,
        )
    raise ValueError(f"Unknown model: {m_type}")


def process_single_horizon(H, yield_df, hparams_wo, hparams_in):
    t0 = time.time()
    feat_path = FEAT_DIR / f"features_H{H:02d}.parquet"
    if not feat_path.exists():
        return []

    feat_df = pd.read_parquet(feat_path)
    feat_df["county_fips"] = feat_df["county_fips"].astype(str).str.zfill(5)
    feat_df = feat_df.rename(columns={"crop_year": "year"})

    panel = yield_df.merge(feat_df, on=["county_fips", "year"]).sort_values(["year", "county_fips"]).reset_index(drop=True)
    raw_feature_cols = [c for c in feat_df.columns if c not in ("county_fips", "year")]
    
    test_years = [y for y in sorted(yield_df["year"].unique()) if TEST_START <= y <= TEST_END]
    
    hp_wo = hparams_wo.get(str(H), {})
    hp_in = hparams_in.get(str(H), {})

    records = []

    for y_test in test_years:
        train_sub = panel[panel["year"] < y_test]
        test_sub = panel[panel["year"] == y_test]

        y_tr = train_sub["anomaly_full"].values
        y_te = test_sub["anomaly_full"].values
        trends_te = test_sub["trend_full"].values
        fips_te = test_sub["county_fips"].values

        # 1. Weather-Only features
        scaler_wo = StandardScaler()
        X_tr_wo = scaler_wo.fit_transform(train_sub[raw_feature_cols].values)
        X_te_wo = scaler_wo.transform(test_sub[raw_feature_cols].values)

        # 2. Integrated (+Lag) features
        feat_cols_in = raw_feature_cols + ["anomaly_lag1"]
        scaler_in = StandardScaler()
        X_tr_in = scaler_in.fit_transform(train_sub[feat_cols_in].values)
        X_te_in = scaler_in.transform(test_sub[feat_cols_in].values)

        # Baseline Naive (predicts 0.0)
        p_naive = np.zeros_like(y_te)
        for f, yt, tr, yp in zip(fips_te, y_te, trends_te, p_naive):
            records.append({
                "model": "Naive",
                "spec": "Naive",
                "H": H,
                "year": y_test,
                "county_fips": f,
                "y_true_anomaly": yt,
                "y_pred_anomaly": yp,
                "trend": tr,
            })

        # Fit Weather-Only models
        for m_name in ["ElasticNet", "RandomForest", "XGBoost"]:
            model = build_model(m_name, hp_wo.get(m_name, {}))
            model.fit(X_tr_wo, y_tr)
            preds = model.predict(X_te_wo)
            for f, yt, tr, yp in zip(fips_te, y_te, trends_te, preds):
                records.append({
                    "model": m_name,
                    "spec": "Weather-Only",
                    "H": H,
                    "year": y_test,
                    "county_fips": f,
                    "y_true_anomaly": yt,
                    "y_pred_anomaly": yp,
                    "trend": tr,
                })

        # Fit Integrated (+Lag) models
        for m_name in ["ElasticNet", "RandomForest", "XGBoost"]:
            model = build_model(m_name, hp_in.get(m_name, {}))
            model.fit(X_tr_in, y_tr)
            preds = model.predict(X_te_in)
            for f, yt, tr, yp in zip(fips_te, y_te, trends_te, preds):
                records.append({
                    "model": m_name,
                    "spec": "Integrated",
                    "H": H,
                    "year": y_test,
                    "county_fips": f,
                    "y_true_anomaly": yt,
                    "y_pred_anomaly": yp,
                    "trend": tr,
                })

    print(f"Horizon H={H:02d} completed in {time.time() - t0:.1f}s ({len(records)} records).")
    return records


def run_all():
    print("Computing full-sample 1951-2025 linear trends and anomalies...")
    yield_df_raw = pd.read_csv(TARGET_CSV)
    yield_df, trends = compute_full_sample_trends_and_anomalies(yield_df_raw)

    hparams_wo = {}
    if HPARAMS_WO_JSON.exists():
        with open(HPARAMS_WO_JSON) as f:
            hparams_wo = json.load(f)

    hparams_in = {}
    if HPARAMS_IN_JSON.exists():
        with open(HPARAMS_IN_JSON) as f:
            hparams_in = json.load(f)

    all_records = []
    horizons = list(range(12, 0, -1))

    # Run with ProcessPoolExecutor across horizons
    print(f"Executing tabular model training for H={horizons} with 4 parallel worker processes...")
    t_start = time.time()
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_horizon, H, yield_df, hparams_wo, hparams_in) for H in horizons]
        for f in futures:
            all_records.extend(f.result())

    df_out = pd.DataFrame(all_records)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nAll tabular predictions saved to {OUT_PARQUET} ({len(df_out)} rows in {time.time() - t_start:.1f}s).")


if __name__ == "__main__":
    run_all()
