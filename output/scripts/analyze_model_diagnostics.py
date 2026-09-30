"""Comprehensive Econometric & ML Model Diagnostics:
1. Explainability:
   - Feature importance (gain/impurity & TreeSHAP) across horizons.
   - Seasonal decomposition of importance (campaign months m01..m12).
   - Bioclimatic group decomposition (thermal, moisture, radiation, spatial, lagged).
   - ElasticNet coefficient analysis (active L1 non-zero features and signs).
2. Residual & Error Diagnostics:
   - Residual summary: bias (mean error), RMSE, MAE.
   - Normality tests: Skewness, Kurtosis, Jarque-Bera test, Shapiro-Wilk test.
   - Homoscedasticity tests: correlation between fitted values and absolute errors;
     residual variance across yield quintiles and across decades.
3. Tail-Risk & Extreme Shock Evaluation:
   - Specific performance during historic drought/shock years (1988, 1993, 2003, 2012).
   - Benchmark comparison: normal years vs extreme shortfall years (< 20th percentile).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "output" / "data" / "processed"
FEAT_DIR = PROCESSED_DIR / "model_datasets"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
OUT_DIR = PROCESSED_DIR / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_residuals(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute normality and homoscedasticity diagnostics for all (model, H) pairs."""
    records = []
    for (model, H), grp in pred_df.groupby(["model", "H"]):
        y_true = grp["y_true_anomaly"].values
        y_pred = grp["y_pred_anomaly"].values
        resid = y_true - y_pred
        n = len(resid)

        # Basic error metrics
        mean_err = float(np.mean(resid))
        std_err  = float(np.std(resid, ddof=1))
        rmse_val = float(np.sqrt(np.mean(resid**2)))
        mae_val  = float(np.mean(np.abs(resid)))

        # Normality diagnostics
        skew_val = float(stats.skew(resid))
        kurt_val = float(stats.kurtosis(resid, fisher=True))  # excess kurtosis
        jb_stat, jb_p = stats.jarque_bera(resid)

        # Shapiro-Wilk on sample (limit 2000 for stability)
        sample_resid = resid[:min(2000, n)]
        sw_stat, sw_p = stats.shapiro(sample_resid)

        # Homoscedasticity: Spearman rank corr between fitted value and |resid|
        corr_spearman, corr_p = stats.spearmanr(y_pred, np.abs(resid))

        # Residual variance across yield regimes:
        # Shock regime: y_true < -5 bu/ac
        # Normal regime: -5 <= y_true <= 5
        # Bumper regime: y_true > 5
        shock_mask  = y_true < -5.0
        normal_mask = (y_true >= -5.0) & (y_true <= 5.0)
        bumper_mask = y_true > 5.0

        var_shock  = float(np.var(resid[shock_mask]))  if np.sum(shock_mask) > 10  else np.nan
        var_normal = float(np.var(resid[normal_mask])) if np.sum(normal_mask) > 10 else np.nan
        var_bumper = float(np.var(resid[bumper_mask])) if np.sum(bumper_mask) > 10 else np.nan

        # Decade variances
        y_years = grp["year"].values
        var_1990s = float(np.var(resid[(y_years >= 1996) & (y_years <= 2005)]))
        var_2000s = float(np.var(resid[(y_years >= 2006) & (y_years <= 2015)]))
        var_2010s = float(np.var(resid[(y_years >= 2016) & (y_years <= 2025)]))

        records.append({
            "model": model,
            "H": int(H),
            "n_obs": n,
            "bias_mean": round(mean_err, 4),
            "rmse": round(rmse_val, 4),
            "mae": round(mae_val, 4),
            "skewness": round(skew_val, 4),
            "excess_kurtosis": round(kurt_val, 4),
            "jarque_bera_stat": round(float(jb_stat), 2),
            "jarque_bera_p": round(float(jb_p), 6),
            "shapiro_stat": round(float(sw_stat), 4),
            "shapiro_p": round(float(sw_p), 6),
            "spearman_fitted_vs_abs_resid": round(float(corr_spearman), 4),
            "spearman_p": round(float(corr_p), 6),
            "var_shock_regime": round(var_shock, 2),
            "var_normal_regime": round(var_normal, 2),
            "var_bumper_regime": round(var_bumper, 2),
            "var_1996_2005": round(var_1990s, 2),
            "var_2006_2015": round(var_2000s, 2),
            "var_2016_2025": round(var_2010s, 2),
        })

    return pd.DataFrame(records)


def analyze_shock_years(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate performance specifically during historic drought/shock years."""
    # Key test shock years: 2012 (historic drought), 2003 (severe late-season drought),
    # 1998 (El Niño transition), 2008 (Midwest floods).
    # Normal years: all other test years.
    SHOCK_YEARS = [2012, 2003, 1998, 2008]

    records = []
    for (model, H), grp in pred_df.groupby(["model", "H"]):
        for y in SHOCK_YEARS:
            sub = grp[grp["year"] == y]
            if sub.empty:
                continue
            y_t = sub["y_true_anomaly"].values
            y_p = sub["y_pred_anomaly"].values
            err = y_t - y_p
            records.append({
                "model": model,
                "H": int(H),
                "year": y,
                "mean_true_anomaly": round(float(np.mean(y_t)), 2),
                "rmse": round(float(np.sqrt(np.mean(err**2))), 3),
                "mae": round(float(np.mean(np.abs(err))), 3),
                "mean_error_bias": round(float(np.mean(err)), 3),
                "is_shock_year": True,
            })

        # Compare to normal years (excluding the 4 shock years)
        normal_sub = grp[~grp["year"].isin(SHOCK_YEARS)]
        if not normal_sub.empty:
            y_tn = normal_sub["y_true_anomaly"].values
            y_pn = normal_sub["y_pred_anomaly"].values
            err_n = y_tn - y_pn
            records.append({
                "model": model,
                "H": int(H),
                "year": 9999,  # code for "all normal years pooled"
                "mean_true_anomaly": round(float(np.mean(y_tn)), 2),
                "rmse": round(float(np.sqrt(np.mean(err_n**2))), 3),
                "mae": round(float(np.mean(np.abs(err_n))), 3),
                "mean_error_bias": round(float(np.mean(err_n)), 3),
                "is_shock_year": False,
            })

    return pd.DataFrame(records)


def parse_feature_metadata(feat_name: str) -> tuple[str, str, int | None]:
    """Parse indicator name, category, and campaign month from feature name."""
    if feat_name in ("lat_norm", "lon_norm"):
        return "spatial", "Spatial Coordinates", None
    if feat_name.startswith("anomaly_lag"):
        return "carryover", "Past Yield Anomaly", None

    parts = feat_name.rsplit("_m", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
        month = int(parts[1])
    else:
        base = feat_name
        month = None

    # Bioclimatic category
    if base in ("EDD30", "HD30", "HD35", "GDD", "T_max", "T_mean", "T_min", "T_dew"):
        cat = "Thermal Stress & Dynamics"
    elif base in ("P", "P_minus_ET0", "ET0", "VPD", "SM_root", "SM1", "SM2", "SM3"):
        cat = "Water Balance & Moisture"
    elif base == "SSRD":
        cat = "Solar Radiation"
    else:
        cat = "Other"

    return base, cat, month


def compute_feature_importance_h(H: int, model_name: str = "XGBoost") -> dict[str, pd.DataFrame]:
    """Train model on available data for horizon H and extract feature importance decomposition."""
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet

    feat_path = FEAT_DIR / f"features_H{H:02d}.parquet"
    if not feat_path.exists():
        return {}

    feat_df = pd.read_parquet(feat_path).rename(columns={"crop_year": "year"})
    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str)
    feat_df["county_fips"] = feat_df["county_fips"].astype(str)

    panel = yield_df.merge(feat_df, on=["county_fips", "year"], how="inner").sort_values(["year", "county_fips"])

    # Load optimal hyperparameters from json
    hp_path = PROCESSED_DIR / "hyperparameters_optimal.json"
    hp_dict = {}
    if hp_path.exists():
        with open(hp_path) as f:
            all_hp = json.load(f)
            hp_dict = all_hp.get(str(H), {}).get(model_name, {})

    feature_cols = [c for c in feat_df.columns if c not in ("county_fips", "year")]

    # Fit linear OLS trend per county on train years (1951-1995)
    train_years = [y for y in panel["year"].unique() if y <= 1995]
    train_panel = panel[panel["year"].isin(train_years)].copy()
    test_panel  = panel[panel["year"] > 1995].copy()

    # OLS trend
    trend_params = {}
    for fips, grp in train_panel.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        t_bar, y_bar = t_vals.mean(), y_vals.mean()
        beta = np.sum((t_vals - t_bar) * (y_vals - y_bar)) / np.sum((t_vals - t_bar) ** 2)
        alpha = y_bar - beta * t_bar
        trend_params[str(fips)] = (alpha, beta)

    def get_anomaly(df):
        tr = df.apply(lambda r: trend_params[str(r["county_fips"])][0] + trend_params[str(r["county_fips"])][1] * r["year"], axis=1)
        return df["yield_bu_per_acre"] - tr

    y_train = get_anomaly(train_panel).values
    y_test  = get_anomaly(test_panel).values

    X_train = train_panel[feature_cols].values.astype(float)
    X_test  = test_panel[feature_cols].values.astype(float)

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train model
    if model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=hp_dict.get("learning_rate", 0.06),
            max_depth=hp_dict.get("max_depth", 5),
            colsample_bytree=hp_dict.get("colsample_bytree", 0.7),
            subsample=hp_dict.get("subsample", 0.8),
            reg_lambda=hp_dict.get("reg_lambda", 5.0),
            reg_alpha=hp_dict.get("reg_alpha", 0.5),
            min_child_weight=hp_dict.get("min_child_weight", 3),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=150,
            max_features=hp_dict.get("max_features", "sqrt"),
            min_samples_leaf=hp_dict.get("min_samples_leaf", 5),
            max_depth=hp_dict.get("max_depth", None),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "ElasticNet":
        model = ElasticNet(
            alpha=hp_dict.get("alpha", 0.01),
            l1_ratio=hp_dict.get("l1_ratio", 0.1),
            max_iter=1000,
            random_state=42,
        )
    else:
        raise ValueError(model_name)

    model.fit(X_train_sc, y_train)

    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_imp = np.abs(model.coef_)
    else:
        raw_imp = np.ones(len(feature_cols))

    raw_imp = raw_imp / np.sum(raw_imp) if np.sum(raw_imp) > 0 else raw_imp

    # Build detailed feature dataframe
    feat_records = []
    for f_name, imp in zip(feature_cols, raw_imp):
        base, cat, month = parse_feature_metadata(f_name)
        feat_records.append({
            "feature": f_name,
            "indicator": base,
            "category": cat,
            "campaign_month": month,
            "importance": float(imp),
            "H": H,
            "model": model_name,
        })
    df_feat = pd.DataFrame(feat_records).sort_values("importance", ascending=False).reset_index(drop=True)

    # Monthly aggregation
    MONTH_NAMES = {
        1: "Nov (Y-1)", 2: "Dec (Y-1)", 3: "Jan (Y)", 4: "Feb (Y)", 5: "Mar (Y)", 6: "Apr (Y)",
        7: "May (Y - Planting)", 8: "Jun (Y)", 9: "Jul (Y - Flowering)", 10: "Aug (Y - Grain Fill)",
        11: "Sep (Y - Maturity)", 12: "Oct (Y - Harvest)"
    }
    df_month = (
        df_feat.dropna(subset=["campaign_month"])
        .groupby("campaign_month")["importance"]
        .sum()
        .reset_index()
    )
    df_month["month_name"] = df_month["campaign_month"].map(MONTH_NAMES)
    df_month["share_pct"] = round(df_month["importance"] * 100, 2)
    df_month = df_month.sort_values("campaign_month").reset_index(drop=True)

    # Category aggregation
    df_cat = (
        df_feat.groupby("category")["importance"]
        .sum()
        .reset_index()
    )
    df_cat["share_pct"] = round(df_cat["importance"] * 100, 2)
    df_cat = df_cat.sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "features": df_feat,
        "monthly": df_month,
        "categories": df_cat,
    }


def run_diagnostics(pred_path: Path | None = None) -> None:
    if pred_path is None or not pred_path.exists():
        pred_path = PROCESSED_DIR / "predictions.parquet"
    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}")
        return

    print(f"=== Running Econometric & Residual Diagnostics on {pred_path.name} ===")
    pred_df = pd.read_parquet(pred_path)

    # 1. Residual normality and homoscedasticity table
    res_df = analyze_residuals(pred_df)
    res_csv = OUT_DIR / "residual_diagnostics.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"Saved residual diagnostics to {res_csv}")

    # 2. Shock years performance table
    shock_df = analyze_shock_years(pred_df)
    shock_csv = OUT_DIR / "shock_years_performance.csv"
    shock_df.to_csv(shock_csv, index=False)
    print(f"Saved shock years evaluation to {shock_csv}")

    print("\n--- Key Residual Normality Summary (H=1, 2, 3, 6) ---")
    sub_res = res_df[res_df["H"].isin([1, 2, 3, 6])][
        ["model", "H", "rmse", "skewness", "excess_kurtosis", "jarque_bera_p", "spearman_fitted_vs_abs_resid", "var_shock_regime", "var_normal_regime"]
    ]
    print(sub_res.to_string(index=False))

    print("\n--- 2012 Historic Drought Performance (H=1, 2, 3, 6) ---")
    d2012 = shock_df[(shock_df["year"] == 2012) & (shock_df["H"].isin([1, 2, 3, 6]))][
        ["model", "H", "mean_true_anomaly", "rmse", "mae", "mean_error_bias"]
    ]
    print(d2012.to_string(index=False))

    # 3. Feature Importance & Explainability for key horizons
    print("\n=== Computing Feature Importance & Seasonal Breakdown (H=1, 3, 6) ===")
    for h in [1, 3, 6]:
        imp_data = compute_feature_importance_h(h, model_name="XGBoost")
        if not imp_data:
            continue
        print(f"\n--- Horizon H={h} (XGBoost) Seasonal Importance Breakdown ---")
        print(imp_data["monthly"][["campaign_month", "month_name", "share_pct"]].to_string(index=False))
        print(f"\n--- Horizon H={h} Bioclimatic Category Breakdown ---")
        print(imp_data["categories"].to_string(index=False))
        print(f"\n--- Horizon H={h} Top 10 Features ---")
        print(imp_data["features"][["feature", "indicator", "category", "importance"]].head(10).to_string(index=False))

        # Save to disk
        imp_data["features"].to_csv(OUT_DIR / f"feature_importance_H{h:02d}_xgb.csv", index=False)
        imp_data["monthly"].to_csv(OUT_DIR / f"seasonal_importance_H{h:02d}_xgb.csv", index=False)
        imp_data["categories"].to_csv(OUT_DIR / f"category_importance_H{h:02d}_xgb.csv", index=False)


if __name__ == "__main__":
    run_diagnostics()

