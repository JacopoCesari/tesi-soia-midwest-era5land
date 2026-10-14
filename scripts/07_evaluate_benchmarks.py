"""
07_evaluate_benchmarks.py

Computes master econometric evaluation metrics and reproduces Chapter 5 tables:
1. Table 5.1: Multi-Horizon Out-of-Sample Performance Matrix (R2_OOS, RMSE, MAE, Skill Score).
2. Table 5.3: Extreme Weather Shock Diagnostics (2012 Flash Drought vs 2016 Bumper Harvest).
3. Statistical Inference: Diebold-Mariano tests for predictive superiority vs baselines.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PREDS_ML_PARQUET = ROOT / "output" / "data" / "processed" / "predictions.parquet"
PREDS_LSTM_PARQUET = ROOT / "output" / "data" / "processed" / "predictions_lstm.parquet"

# Evaluation test window
TEST_YEARS = list(range(2011, 2026))
ACTIVE_HORIZONS = list(range(11, 0, -1))

MONTH_NAMES = {
    11: "Dec Y-1 (H=11)",
    10: "Jan Y (H=10)",
    9: "Feb Y (H=9)",
    8: "Mar Y (H=8)",
    7: "Apr Y (H=7, Pre-Planting)",
    6: "May Y (H=6, Emergence)",
    5: "Jun Y (H=5, Vegetative)",
    4: "Jul Y (H=4, Flowering)",
    3: "Aug Y (H=3, Pod-Filling)",
    2: "Sep Y (H=2, Maturity)",
    1: "Oct Y (H=1, Harvest)",
}


def load_combined_predictions():
    """Load and unify out-of-sample predictions across tabular and recurrent models."""
    dfs = []
    if PREDS_ML_PARQUET.exists():
        df_ml = pd.read_parquet(PREDS_ML_PARQUET)
        df_ml = df_ml[df_ml["year"].isin(TEST_YEARS)]
        dfs.append(df_ml)

    if PREDS_LSTM_PARQUET.exists():
        df_lstm = pd.read_parquet(PREDS_LSTM_PARQUET)
        df_lstm = df_lstm[df_lstm["year"].isin(TEST_YEARS)]
        # Map LSTM variants to model labels
        df_lstm["model"] = np.where(df_lstm["include_lag"], "LSTM (+Lag)", "LSTM (Weather-Only)")
        df_lstm = df_lstm.rename(columns={
            "actual_anomaly": "y_true_anomaly",
            "pred_anomaly": "y_pred_anomaly",
            "horizon": "H",
        })
        dfs.append(df_lstm[["model", "H", "year", "county_fips", "y_true_anomaly", "y_pred_anomaly"]])

    if not dfs:
        raise FileNotFoundError("Prediction files not found. Run training scripts first.")

    combined = pd.concat(dfs, ignore_index=True)

    # Construct Linear-Recurrent Ensemble: 50/50 convex blend of ElasticNet (+Lag) and LSTM (+Lag)
    enet = combined[combined["model"] == "ElasticNet"].rename(columns={"y_pred_anomaly": "yp_enet"})
    lstm = combined[combined["model"] == "LSTM (+Lag)"].rename(columns={"y_pred_anomaly": "yp_lstm"})
    if not enet.empty and not lstm.empty:
        ens = pd.merge(
            enet[["H", "year", "county_fips", "y_true_anomaly", "yp_enet"]],
            lstm[["H", "year", "county_fips", "yp_lstm"]],
            on=["H", "year", "county_fips"],
        )
        ens["y_pred_anomaly"] = 0.5 * (ens["yp_enet"] + ens["yp_lstm"])
        ens["model"] = "Ensemble (ENet + LSTM)"
        combined = pd.concat(
            [combined, ens[["model", "H", "year", "county_fips", "y_true_anomaly", "y_pred_anomaly"]]],
            ignore_index=True,
        )

    return combined


def apply_minnesota_phenological_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply state-dependent harvest closure horizon H*(s).
    
    For Minnesota (FIPS starting with '27'), physiological maturity (R7) and harvest
    occur by early October (Maturity Group I, Sections 3.2 and 4.1). At harvest horizon H=1
    (October 31), predictions are set to the September 30 (H=2) forecast to eliminate
    post-harvest precipitation noise contamination. For other states (IA, IL, IN, MO, OH),
    H=1 retains the full October weather aggregates.
    """
    df_out = df.copy()
    h2_lookup = df_out[df_out["H"] == 2].set_index(["model", "year", "county_fips"])["y_pred_anomaly"].to_dict()
    is_mn_h1 = (df_out["H"] == 1) & (df_out["county_fips"].astype(str).str.startswith("27"))
    for idx in df_out[is_mn_h1].index:
        key = (df_out.at[idx, "model"], df_out.at[idx, "year"], df_out.at[idx, "county_fips"])
        if key in h2_lookup:
            df_out.at[idx, "y_pred_anomaly"] = h2_lookup[key]
    return df_out


def diebold_mariano_test(loss_a, loss_b):
    """Diebold-Mariano test for equal predictive accuracy."""
    d = loss_a - loss_b
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / len(d)
    if var_d <= 0:
        return 0.0, 1.0
    dm_stat = mean_d / np.sqrt(var_d)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_val)


def main():
    print("Loading predictions for out-of-sample evaluation (2011-2025)...")
    preds_raw = load_combined_predictions()
    preds = apply_minnesota_phenological_rule(preds_raw)

    models = [
        "Naive",
        "ElasticNet",
        "RandomForest",
        "XGBoost",
        "LSTM (Weather-Only)",
        "LSTM (+Lag)",
        "Ensemble (ENet + LSTM)",
    ]
    available_models = [m for m in models if m in preds["model"].unique()]

    # =========================================================================
    # 1. TABLE 5.1: Multi-Horizon Performance Matrix (R2_OOS)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TABLE 5.1: Multi-Horizon Out-of-Sample Performance Matrix (R2_OOS, 2011-2025)")
    print("=" * 90)

    matrix_rows = []
    for H in ACTIVE_HORIZONS:
        row = {"Horizon": MONTH_NAMES[H]}
        sub_h = preds[preds["H"] == H]

        for m in available_models:
            sub_m = sub_h[sub_h["model"] == m]
            if sub_m.empty:
                row[m] = "---"
                continue
            yt = sub_m["y_true_anomaly"].values
            yp = sub_m["y_pred_anomaly"].values
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum(yt ** 2)
            r2 = 1.0 - (ss_res / ss_tot)
            row[m] = f"{r2:+.3f}"

        matrix_rows.append(row)

    df_table_5_1 = pd.DataFrame(matrix_rows)
    print(df_table_5_1.to_string(index=False))

    # =========================================================================
    # 2. TABLE 5.3: Extreme Weather Shock Diagnostics (2012 vs 2016)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TABLE 5.3: Extreme Shock Resolution (RMSE in bu/ac with Mean Bias in parentheses)")
    print("=" * 90)

    shock_years = {"Panel A: 2012 Flash Drought (Severe Stress)": 2012, "Panel B: 2016 Bumper Crop (Record Favorable)": 2016}
    key_horizons = [7, 4, 2, 1]  # Pre-planting, Post-flowering, Maturity, Harvest

    for panel_title, sy in shock_years.items():
        print(f"\n{panel_title} (Year: {sy}):")
        shock_rows = []
        for H in key_horizons:
            row = {"Milestone": MONTH_NAMES[H]}
            sub_sy = preds[(preds["year"] == sy) & (preds["H"] == H)]

            for m in available_models:
                sub_m = sub_sy[sub_sy["model"] == m]
                if sub_m.empty:
                    row[m] = "---"
                    continue
                yt = sub_m["y_true_anomaly"].values
                yp = sub_m["y_pred_anomaly"].values
                rmse = np.sqrt(np.mean((yt - yp) ** 2))
                bias = np.mean(yp - yt)
                row[m] = f"{rmse:.2f} ({bias:+.1f})"

            shock_rows.append(row)

        df_shock = pd.DataFrame(shock_rows)
        print(df_shock.to_string(index=False))

    # =========================================================================
    # 3. DIEBOLD-MARIANO INFERENTIAL TESTS AT HARVEST (H=1)
    # =========================================================================
    print("\n" + "=" * 90)
    print("INFERENTIAL TESTS: Diebold-Mariano Loss Differential vs Naive Baseline (H=1)")
    print("=" * 90)

    sub_h1 = preds[preds["H"] == 1]
    sub_naive = sub_h1[sub_h1["model"] == "Naive"]
    if not sub_naive.empty:
        # Group squared error by year
        se_naive = sub_naive.groupby("year").apply(lambda g: np.mean((g["y_true_anomaly"] - g["y_pred_anomaly"]) ** 2))

        for m in available_models:
            if m == "Naive":
                continue
            sub_m = sub_h1[sub_h1["model"] == m]
            if sub_m.empty:
                continue
            se_m = sub_m.groupby("year").apply(lambda g: np.mean((g["y_true_anomaly"] - g["y_pred_anomaly"]) ** 2))
            dm_stat, p_val = diebold_mariano_test(se_naive.values, se_m.values)
            sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else "ns"))
            print(f"  {m:<25}: DM-Stat = {dm_stat:+.2f}, p-value = {p_val:.4f} [{sig}]")

    print("\nBenchmark analysis completed successfully.")


if __name__ == "__main__":
    main()
