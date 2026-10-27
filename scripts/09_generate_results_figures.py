"""
09_generate_results_figures.py

Generates all official empirical results figures for Chapter 5 (Results) and Appendix A:
1. Figure 5.1: Multi-Horizon Progression Curves across Phenological Stages (fig_r2_horizon_progression.pdf / .png)
2. Macro-Regional Portfolio Horizon Progression (fig_macro_horizon_progression.pdf / .png)
3. Figure 5.2: Lead-Time Shock Bias and Extreme Anomaly Scatter (fig_shock_bias_profiles.pdf / .png)
4. Figure 5.3: Decile Reliability Diagram and Residual Diagnostics (fig_decile_calibration_and_residuals.pdf / .png)
5. Figure 5.4: Grouped Permutation Feature Importance Faceted Bars (fig54_pfi_faceted_bars.pdf / .png)
6. Figure 5.5: GPFI Macro-Domain Physiological Sensitivity Bars (fig55_pfi_domain_bars.pdf / .png)
7. Figure 5.6: Spatial Choropleth of Out-of-Sample County R2 (fig_county_r2_map.pdf / .png)

Outputs:
- output/thesis/figures/fig_r2_horizon_progression.pdf / .png
- output/thesis/figures/fig_macro_horizon_progression.pdf / .png
- output/thesis/figures/fig_shock_bias_profiles.pdf / .png
- output/thesis/figures/fig_decile_calibration_and_residuals.pdf / .png
- output/thesis/figures/fig54_pfi_faceted_bars.pdf / .png
- output/thesis/figures/fig55_pfi_domain_bars.pdf / .png
- output/thesis/figures/fig_county_r2_map.pdf / .png
"""

import json
import shutil
from scipy.stats import gaussian_kde, pearsonr
import statsmodels.api as sm
import importlib.util
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "output" / "data" / "processed"
FEATURES_H01_PATH = PROCESSED_DIR / "model_datasets" / "features_H01.parquet"
HPARAMS_PATH = PROCESSED_DIR / "hyperparameters_optimal_integrated.json"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
SHAPEFILE = ROOT / "data" / "raw" / "census_counties" / "cb_2020_us_county_500k.shp"
FIGURES_DIR = ROOT / "output" / "thesis" / "figures"
TABLES_DIR = ROOT / "output" / "thesis" / "tables"
EXP_DIR = ROOT / "work" / "experiments" / "full_sample_detrending"
DATA_DIR = EXP_DIR / "data"
FIG_OUT = FIGURES_DIR
TAB_OUT = TABLES_DIR

# Test window & horizons
TEST_YEARS = list(range(2011, 2026))
HORIZONS = list(range(11, 0, -1))

MONTH_LABELS = {
    11: "Dec Y-1\n(H=11)", 10: "Jan Y\n(H=10)", 9: "Feb Y\n(H=9)", 8: "Mar Y\n(H=8)",
    7: "Apr Y\n(H=7)", 6: "May Y\n(H=6)", 5: "Jun Y\n(H=5)", 4: "Jul Y\n(H=4)",
    3: "Aug Y\n(H=3)", 2: "Sep Y\n(H=2)", 1: "Oct Y\n(H=1)"
}

COLORS = {
    "Naive": "#555555",
    "ElasticNet": "#1f77b4",
    "RandomForest": "#ff7f0e",
    "ElasticNet (+Lag)": "#1f77b4",
    "RandomForest (+Lag)": "#ff7f0e",
    "XGBoost (+Lag)": "#d62728",
    "XGBoost": "#d62728",
    "LSTM (Weather-Only)": "#2ca02c",
    "LSTM (+Lag)": "#8e44ad",
    "Ensemble (ENet + LSTM)": "#16a085",
}


def load_data():
    """Use the same identified forecasts and H* rule as the evidence tables."""
    spec = importlib.util.spec_from_file_location("benchmark_evaluation", Path(__file__).with_name("07_evaluate_benchmarks.py"))
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    return evaluation.apply_minnesota_phenological_rule(evaluation.load_combined_predictions())


# ==============================================================================
# FIGURE 5.1: Multi-Horizon Progression Curves
# ==============================================================================
def plot_figure_5_1(preds):
    print("Generating Figure 5.1: R2 and RMSE progression across horizons...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4), sharex=True)
    x_pos = np.arange(len(HORIZONS))

    for ax in (ax1, ax2):
        ax.axvspan(-0.5, 5.5, color="#f2f4f8", alpha=0.75)
        ax.axvspan(5.5, 10.5, color="#fdfbf2", alpha=0.85)
        ax.axvline(5.5, color="#c0392b", linestyle=":", linewidth=1.5, alpha=0.85)

    ax1.plot(x_pos, np.zeros(len(HORIZONS)), label="Naive Secular Trend", color=COLORS["Naive"], linestyle="-", marker="s", markersize=5)

    models_to_plot = ["ElasticNet", "RandomForest", "XGBoost", "LSTM (Weather-Only)", "ElasticNet (+Lag)", "RandomForest (+Lag)", "XGBoost (+Lag)", "LSTM (+Lag)", "Ensemble (ENet + LSTM)"]
    for m in models_to_plot:
        sub_m = preds[preds["model"] == m]
        if sub_m.empty:
            continue
        r2_vals, rmse_vals = [], []
        for h in HORIZONS:
            sh = sub_m[sub_m["H"] == h]
            if sh.empty:
                r2_vals.append(np.nan)
                rmse_vals.append(np.nan)
            else:
                yt, yp = sh["y_true_anomaly"].values, sh["y_pred_anomaly"].values
                ss_res, ss_tot = np.sum((yt - yp) ** 2), np.sum(yt ** 2)
                r2_vals.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
                rmse_vals.append(np.sqrt(np.mean((yt - yp) ** 2)))

        ls = "--" if "+Lag" in m else ("-." if "Ensemble" in m else "-")
        lw = 2.2 if "Ensemble" in m else 1.8
        ax1.plot(x_pos, r2_vals, label=m, color=COLORS[m], linestyle=ls, marker="o", markersize=5, linewidth=lw)
        ax2.plot(x_pos, rmse_vals, label=m, color=COLORS[m], linestyle=ls, marker="o", markersize=5, linewidth=lw)

    ax1.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.set_ylabel(r"Pooled Out-of-Sample $R^2_{\mathrm{OOS}}$")
    ax1.set_title(r"(a) Multi-Horizon Explained Anomaly Variance ($R^2_{\mathrm{OOS}}$)", loc="left", fontweight="bold", fontsize=11)
    ax1.set_ylim(min(-0.35, min(ax1.get_ylim())), max(0.42, max(ax1.get_ylim())))
    ax1.legend(loc="lower left", frameon=True, fontsize=8.0, ncol=2)

    reference = preds[(preds["model"] == "Naive") & (preds["H"] == 1)]
    naive_rmse = np.sqrt(np.mean(reference["y_true_anomaly"] ** 2))
    ax2.plot(x_pos, np.full(len(HORIZONS), naive_rmse), label=f"Naive Baseline ({naive_rmse:.2f} bu/ac)", color=COLORS["Naive"], linestyle="-", marker="s", markersize=5)
    ax2.set_ylabel("Out-of-Sample RMSE (bu/acre)")
    ax2.set_title("(b) Forecast Error Magnitude (RMSE)", loc="left", fontweight="bold", fontsize=11)
    ax2.set_ylim(min(4.80, min(ax2.get_ylim())), max(7.20, max(ax2.get_ylim())))
    ax2.legend(loc="upper right", frameon=True, fontsize=8.0, ncol=2)

    for ax in (ax1, ax2):
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MONTH_LABELS[h] for h in HORIZONS])
        ax.set_xlabel("Forecast Horizon & Campaign Month")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_r2_horizon_progression.pdf")
    fig.savefig(FIGURES_DIR / "fig_r2_horizon_progression.png")
    plt.close(fig)


# ==============================================================================
# MACRO-REGIONAL HORIZON PROGRESSION (PORTFOLIO MEAN OVER 135 COUNTIES)
# ==============================================================================
def plot_macro_horizon_progression(preds):
    print("Generating Figure: Macro-regional portfolio progression across horizons...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4), sharex=True)
    x_pos = np.arange(len(HORIZONS))

    for ax in (ax1, ax2):
        ax.axvspan(-0.5, 5.5, color="#f2f4f8", alpha=0.75)
        ax.axvspan(5.5, 10.5, color="#fdfbf2", alpha=0.85)
        ax.axvline(5.5, color="#c0392b", linestyle=":", linewidth=1.5, alpha=0.85)

    ax1.plot(x_pos, np.zeros(len(HORIZONS)), label="Naive Secular Trend", color=COLORS["Naive"], linestyle="-", marker="s", markersize=5)

    # Reference naive macro RMSE
    yearly_naive = preds[preds["H"] == 1].groupby(["year", "county_fips"])["y_true_anomaly"].first().groupby("year").mean()
    naive_rmse = float(np.sqrt(np.mean(yearly_naive.values ** 2)))
    ax2.plot(x_pos, np.full(len(HORIZONS), naive_rmse), label=f"Naive Trend ({naive_rmse:.2f} bu/ac)", color=COLORS["Naive"], linestyle="-", marker="s", markersize=5)

    models_to_plot = [
        "ElasticNet (+Lag)",
        "RandomForest (+Lag)",
        "XGBoost (+Lag)",
        "LSTM (+Lag)",
        "Ensemble (ENet + LSTM)",
    ]

    for m in models_to_plot:
        sub_m = preds[preds["model"] == m]
        if sub_m.empty:
            continue
        r2_vals, rmse_vals = [], []
        for h in HORIZONS:
            sh = sub_m[sub_m["H"] == h]
            if sh.empty:
                r2_vals.append(np.nan)
                rmse_vals.append(np.nan)
            else:
                yearly = sh.groupby("year")[["y_true_anomaly", "y_pred_anomaly"]].mean()
                yt = yearly["y_true_anomaly"].values
                yp = yearly["y_pred_anomaly"].values
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum(yt ** 2)
                r2_vals.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
                rmse_vals.append(np.sqrt(np.mean((yt - yp) ** 2)))

        ls = "--" if "+Lag" in m else ("-." if "Ensemble" in m else "-")
        lw = 2.2 if "Ensemble" in m else 1.8
        color_val = COLORS.get(m, "#333333")
        ax1.plot(x_pos, r2_vals, label=m, color=color_val, linestyle=ls, marker="o", markersize=5, linewidth=lw)
        ax2.plot(x_pos, rmse_vals, label=m, color=color_val, linestyle=ls, marker="o", markersize=5, linewidth=lw)

    ax1.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.set_ylabel(r"Macro-Regional Explained Variance ($R^2_{\mathrm{macro}}$)")
    ax1.set_title(r"(a) Macro-Regional Anomaly Skill ($R^2_{\mathrm{macro}}$)", loc="left", fontweight="bold", fontsize=11)
    ax1.set_ylim(-0.75, 0.75)
    ax1.legend(loc="lower left", frameon=True, fontsize=8.0, ncol=2)

    ax2.set_ylabel("Macro-Regional RMSE (bu/acre)")
    ax2.set_title(r"(b) Macro-Regional Error Magnitude (RMSE)", loc="left", fontweight="bold", fontsize=11)
    ax2.set_ylim(1.8, 5.2)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.0, ncol=2)

    for ax in (ax1, ax2):
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MONTH_LABELS[h] for h in HORIZONS])
        ax.set_xlabel("Forecast Horizon & Campaign Month")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_macro_horizon_progression.pdf")
    fig.savefig(FIGURES_DIR / "fig_macro_horizon_progression.png")
    plt.close(fig)


# ==============================================================================
# ==============================================================================
# FIGURE 5.2: Lead-Time Shock Bias Profiles & Scatter
# ==============================================================================
def plot_figure_5_2(preds):
    print("Generating Figure 5.2: Shock bias profiles and scatter resolution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.2))
    active_h = [5, 4, 3, 2, 1]
    x_pos = np.arange(len(active_h))

    # Panel (a): Lead-Time Bias Trajectories including Ensemble
    models_panel_a = [
        ("ElasticNet (+Lag)", "ElasticNet (+Lag)", COLORS["ElasticNet"]),
        ("XGBoost (+Lag)", "XGBoost (+Lag)", COLORS["XGBoost"]),
        ("LSTM (+Lag)", "LSTM", COLORS["LSTM (+Lag)"]),
        ("Ensemble (ENet + LSTM)", "Ensemble", COLORS["Ensemble (ENet + LSTM)"]),
    ]

    for yr, name, ls in [(2012, "2012 Drought", "-"), (2016, "2016 Bumper", "--")]:
        sub_yr = preds[preds["year"] == yr]
        for m_key, m_lbl, col in models_panel_a:
            sub_m = sub_yr[sub_yr["model"] == m_key]
            if sub_m.empty:
                continue
            biases = [sub_m[sub_m["H"] == h]["y_pred_anomaly"].mean() - sub_m[sub_m["H"] == h]["y_true_anomaly"].mean() for h in active_h]
            ax1.plot(x_pos, biases, label=f"{m_lbl} ({name})", color=col, linestyle=ls, marker="o", markersize=4.8, linewidth=1.8)

    ax1.axhline(0, color="black", linestyle=":", linewidth=1.2, label="Zero Bias")
    ax1.set_ylabel(r"Mean Anomaly Bias ($\bar{\hat{\epsilon}} - \bar{\epsilon}$, bu/acre)")
    ax1.set_title("(a) Regional forecast bias", loc="left", fontweight="bold", fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r"$H^*(s)$" if h == 1 else f"H={h}" for h in active_h])
    ax1.set_xlabel("Forecast Horizon")
    ax1.legend(loc="lower left", frameon=True, fontsize=7.8, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel (b): Scatter of Predicted vs Observed 2012 Drought Anomalies with OLS Trendlines
    sub_2012 = preds[(preds["year"] == 2012) & (preds["model"] == "LSTM (+Lag)")]
    colors_h = {5: "#e67e22", 3: "#2980b9", 1: "#27ae60"}
    ax2.plot([-20, 10], [-20, 10], "k--", linewidth=1.1, label="1:1 Reference Line", zorder=1)

    for h in [5, 3, 1]:
        sh = sub_2012[sub_2012["H"] == h]
        if not sh.empty:
            xt = sh["y_true_anomaly"].values
            yp = sh["y_pred_anomaly"].values
            slope, intercept, r_val, p_val, std_err = stats.linregress(xt, yp)
            
            # Scatter points
            ax2.scatter(xt, yp, alpha=0.55, color=colors_h[h], s=26, zorder=3)
            
            # OLS regression line
            x_line = np.linspace(xt.min(), xt.max(), 50)
            ax2.plot(x_line, slope * x_line + intercept, color=colors_h[h], linewidth=1.8,
                     label=(r"$H^*(s)$" if h == 1 else f"H={h}")+f" (slope: {slope:.2f})", zorder=4)

    ax2.set_xlabel("Observed 2012 County Anomaly (bu/acre)")
    ax2.set_ylabel("Predicted Anomaly (bu/acre)")
    ax2.set_title("(b) 2012 drought forecasts (LSTM)", loc="left", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.0)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_shock_bias_profiles.pdf")
    fig.savefig(FIGURES_DIR / "fig_shock_bias_profiles.png")
    plt.close(fig)


# ==============================================================================
# FIGURE 5.3: Decile Reliability Diagram and Residual Diagnostics
# ==============================================================================
def plot_figure_5_3(preds):
    print("Generating Figure 5.3: Decile Reliability and Residual Diagnostics...")
    sub_h1 = preds[preds["H"] == 1].copy()
    if sub_h1.empty:
        return

    # Clean selection of candidate models
    eval_models = [
        ("Naive", "Naive", COLORS["Naive"], "s"),
        ("ElasticNet (+Lag)", "ElasticNet (+Lag)", COLORS["ElasticNet"], "o"),
        ("RandomForest (+Lag)", "Random Forest (+Lag)", COLORS["RandomForest"], "^"),
        ("XGBoost (+Lag)", "XGBoost (+Lag)", COLORS["XGBoost"], "D"),
        ("LSTM (+Lag)", "LSTM", COLORS["LSTM (+Lag)"], "X"),
        ("Ensemble (ENet + LSTM)", "Ensemble", COLORS["Ensemble (ENet + LSTM)"], "P"),
    ]

    # Compute Deciles on observed test anomalies
    obs_sample = sub_h1[sub_h1["model"] == "Naive"][["year", "county_fips", "y_true_anomaly"]].drop_duplicates()
    obs_sample["decile"] = pd.qcut(obs_sample["y_true_anomaly"], 10, labels=False) + 1
    decile_map = obs_sample.set_index(["year", "county_fips"])["decile"].to_dict()
    sub_h1["decile"] = sub_h1.set_index(["year", "county_fips"]).index.map(decile_map)

    decile_stats = sub_h1.groupby(["decile", "model"]).agg(
        obs_mean=("y_true_anomaly", "mean"),
        pred_mean=("y_pred_anomaly", "mean")
    ).reset_index()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Panel (a): Decile Reliability Diagram
    obs_dec = decile_stats[decile_stats["model"] == "Naive"].sort_values("decile")["obs_mean"].values
    min_val, max_val = min(obs_dec.min(), -12), max(obs_dec.max(), 12)
    ax1.plot([min_val - 2, max_val + 2], [min_val - 2, max_val + 2],
             color="#111111", linestyle="--", linewidth=1.1, alpha=0.75, label="1:1 Calibration", zorder=1)

    for m_key, m_lbl, m_col, m_mk in eval_models:
        dm = decile_stats[decile_stats["model"] == m_key].sort_values("decile")
        if not dm.empty:
            ax1.plot(dm["obs_mean"], dm["pred_mean"], label=m_lbl, color=m_col,
                     marker=m_mk, markersize=5.5, linewidth=1.7, zorder=3)

    ax1.set_xlabel("Observed decile mean (bu/ac)")
    ax1.set_ylabel("Predicted decile mean (bu/ac)")
    ax1.set_title("(a) Decile calibration", loc="left", fontsize=11, fontweight="bold")
    ax1.set_xlim(min_val - 1.5, max_val + 1.5)
    ax1.set_ylim(-6.0, 5.0)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.0)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel (b): Normal Q-Q Plot of Standardized Residuals
    theo_q = np.linspace(-3.5, 3.5, 100)
    ax2.plot(theo_q, theo_q, color="#111111", linestyle="--", linewidth=1.1, alpha=0.75, label="Theoretical Normal", zorder=1)

    for m_key, m_lbl, m_col in [("ElasticNet (+Lag)", "ElasticNet (+Lag)", COLORS["ElasticNet"]),
                                 ("XGBoost (+Lag)", "XGBoost (+Lag)", COLORS["XGBoost"]),
                                 ("LSTM (+Lag)", "LSTM", COLORS["LSTM (+Lag)"])]:
        sm = sub_h1[sub_h1["model"] == m_key]
        if not sm.empty:
            e = (sm["y_true_anomaly"] - sm["y_pred_anomaly"]).values
            std_e = (e - np.mean(e)) / np.std(e)
            osm, osr = stats.probplot(std_e, dist="norm", fit=False)
            ax2.scatter(osm[::8], osr[::8], color=m_col, label=f"{m_lbl} Residuals", alpha=0.6, s=16, edgecolors="none", zorder=3)

    ax2.set_xlabel("Normal quantile")
    ax2.set_ylabel("Standardized residual")
    ax2.set_title("(b) Residual Q-Q plot", loc="left", fontsize=11, fontweight="bold")
    ax2.set_xlim(-3.6, 3.6)
    ax2.set_ylim(-4.2, 3.8)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.0)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel (c): Residual Variance vs Fitted Values (Champion LSTM)
    champ_m = "LSTM (+Lag)" if "LSTM (+Lag)" in sub_h1["model"].values else "ElasticNet"
    sub_foc = sub_h1[sub_h1["model"] == champ_m]
    y_fit = sub_foc["y_pred_anomaly"].values
    y_res = (sub_foc["y_true_anomaly"] - sub_foc["y_pred_anomaly"]).values
    col_foc = COLORS.get(champ_m, "#8e44ad")

    ax3.scatter(y_fit[::3], y_res[::3], color=col_foc, alpha=0.30, s=14, edgecolors="none", label="LSTM (Subsampled)")
    ax3.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)

    grid_x = np.linspace(y_fit.min(), y_fit.max(), 120)
    bandwidth = (y_fit.max() - y_fit.min()) / 10.0
    weights = np.exp(-0.5 * ((y_fit[:, None] - grid_x[None, :]) / bandwidth) ** 2)
    w_sum = weights.sum(axis=0)
    mu_smooth = (weights * y_res[:, None]).sum(axis=0) / w_sum
    var_smooth = (weights * (y_res[:, None] - mu_smooth[None, :]) ** 2).sum(axis=0) / w_sum
    sigma_smooth = np.sqrt(var_smooth)

    ax3.plot(grid_x, mu_smooth, color="#111111", linewidth=1.8, label="Smoothed Mean Bias")
    ax3.fill_between(grid_x, mu_smooth - sigma_smooth, mu_smooth + sigma_smooth, color=col_foc, alpha=0.18, label=r"$\pm 1\sigma$ Residual Band")

    ax3.set_xlabel("Fitted anomaly (bu/ac)")
    ax3.set_ylabel("Residual (bu/ac)")
    ax3.set_title("(c) Residuals vs fitted (LSTM)", loc="left", fontsize=11, fontweight="bold")
    ax3.set_xlim(-10.5, 6.0)
    ax3.set_ylim(-18, 16)
    ax3.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=8.0)
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_decile_calibration_and_residuals.pdf")
    fig.savefig(FIGURES_DIR / "fig_decile_calibration_and_residuals.png")
    plt.close(fig)


# ==============================================================================
# Helper for Figures 5.4 & 5.5: Train Models & Setup PFI Datasets
# ==============================================================================
def prepare_pfi_data():
    """Fixed-2010 diagnostic fits, evaluated at state-specific harvest closure.

    The LSTM surrogate approximates refreshed forecasts in the same fixed-trend
    space as these diagnostic fits. This is descriptive attribution, not another
    expanding-window accuracy comparison.
    """
    forecasts = load_data()
    recurrent = forecasts[forecasts.model.eq("LSTM (+Lag)")]
    target = pd.read_csv(TARGET_CSV, dtype={"county_fips": str})
    target["county_fips"] = target.county_fips.str.zfill(5)
    params = json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))
    segments, fidelity = [], []
    for horizon in [1, 2]:
        features = pd.read_parquet(FEATURES_H01_PATH.with_name(f"features_H{horizon:02d}.parquet"))
        features = features.rename(columns={"crop_year": "year"})
        features["county_fips"] = features.county_fips.astype(str).str.zfill(5)
        panel = target.merge(features, on=["county_fips", "year"], validate="one_to_one")
        trends = {}
        for fips, group in panel[panel.year.le(2010)].groupby("county_fips"):
            trends[fips] = np.polyfit(group.year, group.yield_bu_per_acre, 1)
        panel["trend"] = [trends[f][0]*y+trends[f][1] for f,y in zip(panel.county_fips,panel.year)]
        panel["anomaly"] = panel.yield_bu_per_acre-panel.trend
        panel = panel.sort_values(["county_fips", "year"])
        panel["lag_anomaly"] = panel.groupby("county_fips").anomaly.shift(1)
        first = panel.year.eq(panel.year.min())
        panel.loc[first, "lag_anomaly"] = 0.0
        train = panel[panel.year.le(2010)]
        test = panel[panel.year.isin(TEST_YEARS)]
        minnesota = test.county_fips.str.startswith("27")
        test = test[minnesota if horizon == 2 else ~minnesota].copy()
        cols = [c for c in features if c not in ["county_fips", "year"]]+["lag_anomaly"]
        forecast = recurrent[recurrent.H.eq(horizon)][["year", "county_fips", "y_pred_anomaly", "trend"]]
        test = test.merge(forecast, on=["year", "county_fips"], how="left",
                          suffixes=("", "_forecast"), validate="one_to_one")
        required = cols+["anomaly", "y_pred_anomaly", "trend_forecast"]
        if not np.isfinite(train[cols+["anomaly"]].to_numpy()).all() or not np.isfinite(test[required].to_numpy()).all():
            raise ValueError("Incomplete attribution panel")
        X_train, X_test = train[cols].to_numpy(), test[cols].to_numpy()
        y_train, y_test = train.anomaly.to_numpy(), test.anomaly.to_numpy()
        scaler = StandardScaler().fit(X_train)
        models = {
            "enet": ElasticNet(**params[str(horizon)]["ElasticNet"], max_iter=2000, tol=1e-3, random_state=42),
            "xgb": XGBRegressor(**params[str(horizon)]["XGBoost"], n_estimators=300, random_state=42, n_jobs=1),
            "lstm": XGBRegressor(**params[str(horizon)]["XGBoost"], n_estimators=300, random_state=42, n_jobs=1)}
        models["enet"].fit(scaler.transform(X_train), y_train)
        models["xgb"].fit(X_train, y_train)
        # Translate from expanding-trend forecasts to the diagnostic fixed-2010 trend.
        surrogate_target = (test.y_pred_anomaly+test.trend_forecast-test.trend).to_numpy()
        models["lstm"].fit(X_test, surrogate_target)
        fitted = models["lstm"].predict(X_test)
        fidelity.append({"H": horizon, "n_obs": len(test), "scope": "in_sample_surrogate_fit",
                         "RMSE": np.sqrt(np.mean((fitted-surrogate_target)**2)),
                         "R2": 1-np.sum((fitted-surrogate_target)**2)/np.sum((surrogate_target-surrogate_target.mean())**2)})
        segments.append({"feature_cols": cols, "X_test": X_test, "y_test": y_test,
                         "scaler": scaler, "models": models})
    report = PROCESSED_DIR / "diagnostics"
    pd.DataFrame(fidelity).to_csv(report / "lstm_surrogate_fidelity.csv", index=False)
    return segments


def attribution_columns(columns, prefixes):
    """Match complete weather prefixes; precipitation excludes water-balance columns."""
    selected = set()
    for prefix in prefixes:
        if prefix == "spatial":
            selected.update(c for c in columns if c in ["lat_norm", "lon_norm"])
        elif prefix == "lag_anomaly":
            selected.update(c for c in columns if c == prefix)
        else:
            selected.update(c for c in columns if c.startswith(prefix+"_m")
                            and c[len(prefix)+2:].isdigit())
    if not selected:
        raise ValueError(f"No attribution columns for {prefixes}")
    return [i for i,c in enumerate(columns) if c in selected]


def attribution_losses(segments, prefixes, repeats):
    """Joint permutations within the two closure strata, with one pooled denominator."""
    observed = np.concatenate([part["y_test"] for part in segments])
    denominator = np.sum((observed-observed.mean())**2)
    baseline = {name: 0.0 for name in ["enet", "xgb", "lstm"]}
    for part in segments:
        for name, model in part["models"].items():
            X = part["X_test"]
            fitted = model.predict(part["scaler"].transform(X) if name == "enet" else X)
            baseline[name] += np.sum((part["y_test"]-fitted)**2)
    random = np.random.default_rng(42)
    losses = {name: [] for name in baseline}
    for _ in range(repeats):
        permuted_sse = {name: 0.0 for name in baseline}
        for part in segments:
            indices = attribution_columns(part["feature_cols"], prefixes)
            X = part["X_test"].copy()
            permutation = random.permutation(len(X))
            X[:, indices] = X[permutation][:, indices]
            for name, model in part["models"].items():
                fitted = model.predict(part["scaler"].transform(X) if name == "enet" else X)
                permuted_sse[name] += np.sum((part["y_test"]-fitted)**2)
        for name in losses:
            losses[name].append(max(0.0,(permuted_sse[name]-baseline[name])/denominator))
    return {"dr2_"+name: float(np.mean(values)) for name,values in losses.items()}



# ==============================================================================
# FIGURE 5.4: Grouped Permutation Feature Importance Faceted Bars
# ==============================================================================
def plot_figure_5_4(pfi_data):
    print("Generating Figure 5.4: PFI Faceted Bars...")
    if pfi_data is None:
        return

    canonical_domains = [
        {"domain": "Atmospheric Demand & Heat", "color": "#e66101", "vars": [
            ("VPD", "Vapor Pressure Deficit (VPD)"), ("ET0", "Reference Evapotranspiration (ET0)"),
            ("HD30", "Moderate Heat Days (>=30C)"), ("HD35", "Extreme Heat Days (>=35C)"), ("EDD30", "Extreme Degree Days (>30C)")
        ]},
        {"domain": "Hydrological Supply", "color": "#1f78b4", "vars": [
            ("P", "Cumulative Precipitation (P)"), ("P_minus_ET0", "Climatic Water Balance (P - ET0)")
        ]},
        {"domain": "Soil Moisture Dynamics", "color": "#8c510a", "vars": [
            ("SM1", "Topsoil Moisture (0-7 cm)"), ("SM2", "Mid Subsoil Moisture (7-28 cm)"),
            ("SM3", "Deep Subsoil Moisture (28-100 cm)"), ("SM_root", "Root-Zone Soil Moisture")
        ]},
        {"domain": "Thermal & Radiation Regimes", "color": "#e7298a", "vars": [
            ("T_mean", "Mean Temperature (T_mean)"), ("T_max", "Max Temperature (T_max)"),
            ("T_min", "Min Temperature (T_min)"), ("GDD", "Growing Degree Days (GDD)"),
            ("T_dew", "Dewpoint Temperature (T_dew)"), ("SSRD", "Surface Solar Radiation (SSRD)")
        ]},
        {"domain": "Memory & Spatial Baseline", "color": "#4d4d4d", "vars": [
            ("lag_anomaly", r"Lagged Yield Memory ($\epsilon_{t-1}$)"), ("spatial", "Spatial Coordinates (Lat / Lon)")
        ]}
    ]

    records = []
    for domain in canonical_domains:
        for prefix,label in domain["vars"]:
            row = {"domain":domain["domain"], "color":domain["color"], "prefix":prefix, "label":label}
            row.update(attribution_losses(pfi_data, [prefix], 5))
            records.append(row)
    res_df = pd.DataFrame(records)
    res_df["share_enet"] = res_df["dr2_enet"] / res_df["dr2_enet"].sum() * 100.0
    res_df["share_xgb"] = res_df["dr2_xgb"] / res_df["dr2_xgb"].sum() * 100.0
    res_df["share_lstm"] = res_df["dr2_lstm"] / res_df["dr2_lstm"].sum() * 100.0
    res_df.to_csv(PROCESSED_DIR / "diagnostics/forecast_pfi_groups.csv", index=False)
    # Grouped by class/domain in natural order (inverted so Atmospheric Demand is at top)
    df_plot = res_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df_plot))
    bar_h = 0.65

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 9.8), sharey=True)

    # Subtle horizontal dividers between physical domains
    curr_dom = None
    for i in range(len(df_plot)):
        d_name = df_plot.iloc[i]["domain"]
        if curr_dom is not None and d_name != curr_dom:
            for ax in (ax1, ax2, ax3):
                ax.axhline(i - 0.5, color="#d0d0d0", linewidth=0.8, linestyle="-")
        curr_dom = d_name

    panels = [
        (ax1, "share_enet", "(a) ENet (+Lag)"),
        (ax2, "share_xgb", "(b) XGB (+Lag)"),
        (ax3, "share_lstm", "(c) LSTM (Surrogate)"),
    ]

    for ax, col_name, title in panels:
        bars = ax.barh(y_pos, df_plot[col_name], color=df_plot["color"], alpha=0.90, edgecolor="none", height=bar_h)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Information Share (%)", fontsize=10.0)
        ax.set_xlim(0, max(32, df_plot[col_name].max()+5))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
        ax.grid(axis="x", color="#e8e8e8", linestyle=":", linewidth=0.7)
        ax.yaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        for b, val in zip(bars, df_plot[col_name]):
            if val >= 0.2:
                ax.text(b.get_width() + 0.45, b.get_y() + b.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.5, color="#333333")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_plot["label"], fontsize=9.2)
    ax1.tick_params(axis="y", length=0)

    legend_handles = [Patch(facecolor=d["color"], edgecolor="none", label=d["domain"]) for d in canonical_domains]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=5, frameon=False, fontsize=9.0)

    plt.suptitle("Grouped predictor importance", fontsize=12.0, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(FIGURES_DIR / "fig54_pfi_faceted_bars.pdf")
    fig.savefig(FIGURES_DIR / "fig54_pfi_faceted_bars.png", dpi=300)
    plt.close(fig)


# ==============================================================================
# ==============================================================================
# FIGURE 5.5: GPFI Macro-Domain Physiological Sensitivity Bars (100% Stacked)
# ==============================================================================
def plot_figure_5_5(pfi_data):
    print("Generating Figure 5.5: GPFI Macro-Domain 100% Stacked Bars...")
    if pfi_data is None:
        return

    canonical_domains = [
        {"domain": "Atmospheric Demand & Heat", "color": "#e66101", "prefixes": ["VPD", "ET0", "HD30", "HD35", "EDD30"]},
        {"domain": "Hydrological Supply", "color": "#1f78b4", "prefixes": ["P", "P_minus_ET0"]},
        {"domain": "Soil Moisture Dynamics", "color": "#8c510a", "prefixes": ["SM1", "SM2", "SM3", "SM_root"]},
        {"domain": "Thermal & Radiation Regimes", "color": "#e7298a", "prefixes": ["T_mean", "T_max", "T_min", "GDD", "T_dew", "SSRD"]},
        {"domain": "Memory & Spatial Baseline", "color": "#7570b3", "prefixes": ["lag_anomaly", "spatial"]}
    ]

    records = []
    for domain in canonical_domains:
        row = {"domain":domain["domain"], "color":domain["color"]}
        row.update(attribution_losses(pfi_data, domain["prefixes"], 10))
        records.append(row)
    dom_df = pd.DataFrame(records)
    dom_df.to_csv(PROCESSED_DIR / "diagnostics/forecast_pfi_domains.csv", index=False)
    models = ["(a) ElasticNet (+Lag)", "(b) XGBoost (+Lag)", "(c) LSTM (Surrogate)"]
    model_cols = ["dr2_enet", "dr2_xgb", "dr2_lstm"]

    matrix_vals = np.array([dom_df[c].values for c in model_cols])
    totals = matrix_vals.sum(axis=1, keepdims=True)
    shares_matrix = (matrix_vals / totals) * 100.0

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    y_pos = np.arange(len(models))[::-1]
    height = 0.48
    lefts = np.zeros(len(models))

    domain_names = dom_df["domain"].tolist()
    domain_colors = dom_df["color"].tolist()

    for i in range(len(domain_names)):
        s = shares_matrix[:, i]
        bars = ax.barh(
            y_pos, s, left=lefts, color=domain_colors[i],
            alpha=0.92, height=height, edgecolor="white", linewidth=1.5
        )
        for j, b in enumerate(bars):
            pct = s[j]
            if pct >= 4.5:
                ax.text(
                    lefts[j] + pct / 2, b.get_y() + b.get_height() / 2,
                    f"{pct:.1f}%", ha="center", va="center",
                    fontsize=10.0, color="white", fontweight="bold"
                )
        lefts += s

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Macro-Domain Relative Information Share (%)", fontsize=11.0)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    ax.grid(axis="x", color="#e8e8e8", linestyle=":", linewidth=0.7)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    legend_handles = [Patch(facecolor=c, edgecolor="none", label=d) for d, c in zip(domain_names, domain_colors)]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5, frameon=False, fontsize=9.2)

    plt.suptitle("Macro-domain information share across forecasting specifications", fontsize=12.0, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    fig.savefig(FIGURES_DIR / "fig55_pfi_domain_bars.pdf")
    fig.savefig(FIGURES_DIR / "fig55_pfi_domain_bars.png", dpi=300)
    plt.close(fig)


# ==============================================================================
# FIGURE 5.6: Spatial Choropleth of Out-of-Sample County R2
# ==============================================================================
def plot_figure_5_6(preds):
    print("Generating Figure 5.6: County R2 Spatial Map with Minnesota Rule Highlight...")
    if not SHAPEFILE.exists():
        return

    sub = preds[(preds["model"] == "LSTM (+Lag)") & (preds["H"] == 1)]
    if sub.empty:
        raise ValueError("The county map requires the verified LSTM (+Lag) forecasts")

    county_r2 = []
    for fips, grp in sub.groupby("county_fips"):
        yt, yp = grp["y_true_anomaly"].values, grp["y_pred_anomaly"].values
        ss_res, ss_tot = np.sum((yt - yp) ** 2), np.sum(yt ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        county_r2.append({"GEOID": str(fips).zfill(5), "r2_oos": r2})

    df_r2 = pd.DataFrame(county_r2)
    color_min = min(-0.05, float(df_r2.r2_oos.min()))
    color_max = max(0.65, float(df_r2.r2_oos.max()))
    gdf = gpd.read_file(SHAPEFILE)
    midwest_fips = ["17", "18", "19", "27", "29", "39"]
    gdf_mw = gdf[gdf["STATEFP"].isin(midwest_fips)].to_crs(epsg=5070)
    gdf_merged = gdf_mw.merge(df_r2, on="GEOID", how="left")

    fig, ax = plt.subplots(figsize=(10.5, 7.8))
    gdf_mw.plot(ax=ax, color="#f2f2f2", edgecolor="#d9d9d9", linewidth=0.4)
    gdf_merged.dropna(subset=["r2_oos"]).plot(
        column="r2_oos", ax=ax, cmap="YlGnBu", legend=True,
        vmin=color_min, vmax=color_max,
        legend_kwds={
            "label": r"Trend-relative $R^2_{\mathrm{OOS}}$ at harvest closure $H^*(s)$",
            "orientation": "horizontal", "pad": 0.04, "shrink": 0.65, "aspect": 25
        },
        edgecolor="#333333", linewidth=0.5
    )
    gdf_states = gdf_mw.dissolve(by="STATE_NAME")
    gdf_states.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.2)

    # Highlight Minnesota boundary with distinct red dashed contour
    gdf_mn = gdf_mw[gdf_mw["STATEFP"] == "27"].dissolve(by="STATEFP")
    gdf_mn.plot(ax=ax, facecolor="none", edgecolor="#99000d", linewidth=2.0, linestyle="--")

    ax.set_title(r"County-level forecast skill at harvest closure $H^*(s)$ (2011--2025)", fontsize=11.5, fontweight="bold", pad=12)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_county_r2_map.pdf")
    fig.savefig(FIGURES_DIR / "fig_county_r2_map.png", dpi=300)
    plt.close(fig)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generating All Official Chapter 5 Results Figures (5.1 to 5.6) ===")
    preds = load_data()

    plot_figure_5_1(preds)
    plot_macro_horizon_progression(preds)
    plot_figure_5_2(preds)
    plot_figure_5_3(preds)

    pfi_data = prepare_pfi_data()
    plot_figure_5_4(pfi_data)
    plot_figure_5_5(pfi_data)

    plot_figure_5_6(preds)
    print("All Chapter 5 figures generated successfully into output/thesis/figures/!")

    print("=== Generating Appendix A Robustness Figures and Tables ===")
    generate_table_a1_and_figure_a2()
    generate_table_a2_and_figure_a3()
    generate_table_a3()
    copy_figure_a4()
    print("All Appendix A figures and tables generated successfully!")


# ==============================================================================
# APPENDIX A ARTIFACT GENERATION
# ==============================================================================

def generate_table_a1_and_figure_a2():
    print("Generating Table A.1 and Figure A.2 (Decile Anomaly Distributions)...")
    df_yield = pd.read_csv(TARGET_CSV)
    df_yield["county_fips"] = df_yield["county_fips"].astype(str).str.zfill(5)
    
    counties = df_yield["county_fips"].unique()
    records_exp = []
    records_cub = []
    
    for c in counties:
        sub = df_yield[df_yield["county_fips"] == c].sort_values("year")
        y_c = sub["yield_bu_per_acre"].values
        yrs_c = sub["year"].values
        t_c = yrs_c - yrs_c[0]
        
        # Full-sample cubic
        m_cub = sm.OLS(y_c, sm.add_constant(np.column_stack([t_c, t_c**2, t_c**3]))).fit()
        pred_cub = m_cub.predict(sm.add_constant(np.column_stack([t_c, t_c**2, t_c**3])))
        anom_cub = y_c - pred_cub
        
        for yr, a_cub, y_act in zip(yrs_c, anom_cub, y_c):
            records_cub.append({"county_fips": c, "year": yr, "anomaly": a_cub, "yield": y_act})
            
        # Expanding linear: 1951-1980 base (30 years)
        m_lin_base = sm.OLS(y_c[:30], sm.add_constant(t_c[:30])).fit()
        for i in range(30):
            p = m_lin_base.predict([1, t_c[i]])[0]
            records_exp.append({"county_fips": c, "year": yrs_c[i], "anomaly": y_c[i] - p})
            
        for i in range(30, len(yrs_c)):
            m_lin = sm.OLS(y_c[:i], sm.add_constant(t_c[:i])).fit()
            p = m_lin.predict([1, t_c[i]])[0]
            records_exp.append({"county_fips": c, "year": yrs_c[i], "anomaly": y_c[i] - p})
            
    df_exp = pd.DataFrame(records_exp)
    df_cub = pd.DataFrame(records_cub)
    
    s_hist_exp = df_exp["anomaly"].values
    s_hist_cub = df_cub["anomaly"].values
    
    s_test_exp = df_exp[df_exp["year"] >= 2011]["anomaly"].values
    s_test_cub = df_cub[df_cub["year"] >= 2011]["anomaly"].values
    
    q_hist_exp = np.percentile(s_hist_exp, np.linspace(0, 100, 11))
    q_hist_cub = np.percentile(s_hist_cub, np.linspace(0, 100, 11))
    q_test_exp = np.percentile(s_test_exp, np.linspace(0, 100, 11))
    q_test_cub = np.percentile(s_test_cub, np.linspace(0, 100, 11))
    
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\small",
        r"\caption{Empirical yield anomaly distributions by decile ($D_1 \dots D_{10}$): Baseline Expanding Linear Trend vs Full-Sample Cubic Trend.}",
        r"\label{tab:appendix_decile_distribution}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l cc cc @{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Baseline (Expanding Linear Trend)}} & \multicolumn{2}{c}{\textbf{Robustness (Full-Sample Cubic Trend)}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Decile} & \textbf{Anomaly Range (bu/ac)} & \textbf{\% Negative ($\epsilon < 0$)} & \textbf{Anomaly Range (bu/ac)} & \textbf{\% Negative ($\epsilon < 0$)} \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{\textbf{Panel A: Complete Historical Panel (1951--2025, $N = 10{,}125$)}}} \\",
        r"\midrule",
    ]
    
    for d in range(10):
        low_e, high_e = q_hist_exp[d], q_hist_exp[d+1]
        mask_e = (s_hist_exp >= low_e) & (s_hist_exp <= high_e)
        pct_neg_e = np.mean(s_hist_exp[mask_e] < 0) * 100
        
        low_c, high_c = q_hist_cub[d], q_hist_cub[d+1]
        mask_c = (s_hist_cub >= low_c) & (s_hist_cub <= high_c)
        pct_neg_c = np.mean(s_hist_cub[mask_c] < 0) * 100
        
        lbl = f"$D_{{{d+1}}}$" + (" (Shortfall)" if d == 0 else (" (Peak)" if d == 9 else ""))
        tex.append(f"{lbl} & $[{low_e:+.1f}, {high_e:+.1f}]$ & {pct_neg_e:.0f}\\% & $[{low_c:+.1f}, {high_c:+.1f}]$ & {pct_neg_c:.0f}\\% \\\\")
        
    tot_neg_e = np.mean(s_hist_exp < 0) * 100
    tot_neg_c = np.mean(s_hist_cub < 0) * 100
    tex.append(r"\midrule")
    tex.append(f"\\textbf{{Historical Total}} & $[{s_hist_exp.min():+.1f}, {s_hist_exp.max():+.1f}]$ & \\textbf{{{tot_neg_e:.1f}\\%}} & $[{s_hist_cub.min():+.1f}, {s_hist_cub.max():+.1f}]$ & \\textbf{{{tot_neg_c:.1f}\\%}} \\\\")
    
    tex.append(r"\midrule[\heavyrulewidth]")
    tex.append(r"\multicolumn{5}{l}{\textit{\textbf{Panel B: Out-of-Sample Test Set (2011--2025, $N = 2{,}025$)}}} \\",)
    tex.append(r"\midrule")
    
    for d in range(10):
        low_e, high_e = q_test_exp[d], q_test_exp[d+1]
        mask_e = (s_test_exp >= low_e) & (s_test_exp <= high_e)
        pct_neg_e = np.mean(s_test_exp[mask_e] < 0) * 100
        
        low_c, high_c = q_test_cub[d], q_test_cub[d+1]
        mask_c = (s_test_cub >= low_c) & (s_test_cub <= high_c)
        pct_neg_c = np.mean(s_test_cub[mask_c] < 0) * 100
        
        lbl = f"$D_{{{d+1}}}$" + (" (Shortfall)" if d == 0 else (" (Peak)" if d == 9 else ""))
        tex.append(f"{lbl} & $[{low_e:+.1f}, {high_e:+.1f}]$ & {pct_neg_e:.0f}\\% & $[{low_c:+.1f}, {high_c:+.1f}]$ & {pct_neg_c:.0f}\\% \\\\")
        
    tot_test_neg_e = np.mean(s_test_exp < 0) * 100
    tot_test_neg_c = np.mean(s_test_cub < 0) * 100
    tex.append(r"\midrule")
    tex.append(f"\\textbf{{Test Total}} & $[{s_test_exp.min():+.1f}, {s_test_exp.max():+.1f}]$ & \\textbf{{{tot_test_neg_e:.1f}\\%}} & $[{s_test_cub.min():+.1f}, {s_test_cub.max():+.1f}]$ & \\textbf{{{tot_test_neg_c:.1f}\\%}} \\\\")
    
    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{0.2em}",
        r"{\footnotesize \textit{Source: Author's calculation on USDA NASS county soybean yields (1951--2025). The baseline expanding-window trend shifts upward in the test set (mean $+2.41$ bu/ac), whereas the full-sample cubic trend remains centered at $-0.06$ bu/ac.}}",
        r"\end{table}",
    ])
    
    t1_path = TAB_OUT / "tab_appendix_decile_distribution.tex"
    t1_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {t1_path}")
    
    # Figure A.2
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.0))
    color_exp = "#1f77b4"
    color_cub = "#d62728"
    
    # (a) Hist KDE
    ax = axes[0, 0]
    gx = np.linspace(-26, 24, 300)
    kde_he = gaussian_kde(s_hist_exp)
    kde_hc = gaussian_kde(s_hist_cub)
    ax.plot(gx, kde_he(gx), label=f"Expanding Linear (Mean {s_hist_exp.mean():+.2f})", color=color_exp, linestyle="--", linewidth=1.8)
    ax.plot(gx, kde_hc(gx), label=f"Full-Sample Cubic (Mean {s_hist_cub.mean():+.2f})", color=color_cub, linestyle="-", linewidth=1.8)
    ax.axvline(s_hist_exp.mean(), color=color_exp, linestyle="--", alpha=0.6, linewidth=1.0)
    ax.axvline(s_hist_cub.mean(), color=color_cub, linestyle="-", alpha=0.6, linewidth=1.0)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.7, linewidth=0.9)
    ax.set_title("(a) Anomaly Density: Historical Panel 1951--2025 ($N=10{,}125$)", loc="left", fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Yield Anomaly $\\epsilon$ (bu/acre)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    # (b) Test KDE
    ax = axes[0, 1]
    gx_t = np.linspace(-26, 24, 300)
    kde_te = gaussian_kde(s_test_exp)
    kde_tc = gaussian_kde(s_test_cub)
    ax.plot(gx_t, kde_te(gx_t), label=f"Baseline Expanding (Mean {s_test_exp.mean():+.2f})", color=color_exp, linestyle="--", linewidth=1.8)
    ax.plot(gx_t, kde_tc(gx_t), label=f"Full-Sample Cubic (Mean {s_test_cub.mean():+.2f})", color=color_cub, linestyle="-", linewidth=1.8)
    ax.axvline(s_test_exp.mean(), color=color_exp, linestyle="--", alpha=0.6, linewidth=1.0)
    ax.axvline(s_test_cub.mean(), color=color_cub, linestyle="-", alpha=0.6, linewidth=1.0)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.7, linewidth=0.9)
    ax.set_title("(b) Anomaly Density: Out-of-Sample Test 2011--2025 ($N=2{,}025$)", loc="left", fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Yield Anomaly $\\epsilon$ (bu/acre)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    # (c) Hist Decile Means
    ax = axes[1, 0]
    xd = np.arange(1, 11)
    w = 0.35
    means_he = [np.mean(s_hist_exp[(s_hist_exp >= q_hist_exp[i]) & (s_hist_exp <= q_hist_exp[i+1])]) for i in range(10)]
    means_hc = [np.mean(s_hist_cub[(s_hist_cub >= q_hist_cub[i]) & (s_hist_cub <= q_hist_cub[i+1])]) for i in range(10)]
    ax.bar(xd - w/2, means_he, width=w, label="Expanding Linear", color=color_exp, alpha=0.85)
    ax.bar(xd + w/2, means_hc, width=w, label="Full-Sample Cubic", color=color_cub, alpha=0.85)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=0.9)
    ax.set_xticks(xd)
    ax.set_xticklabels([f"$D_{{{i}}}$" for i in xd])
    ax.set_title("(c) Mean Decile Anomaly: Historical Panel 1951--2025", loc="left", fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Yield Decile")
    ax.set_ylabel("Mean Anomaly (bu/acre)")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    # (d) Test Decile Means
    ax = axes[1, 1]
    means_te = [np.mean(s_test_exp[(s_test_exp >= q_test_exp[i]) & (s_test_exp <= q_test_exp[i+1])]) for i in range(10)]
    means_tc = [np.mean(s_test_cub[(s_test_cub >= q_test_cub[i]) & (s_test_cub <= q_test_cub[i+1])]) for i in range(10)]
    ax.bar(xd - w/2, means_te, width=w, label="Baseline Expanding", color=color_exp, alpha=0.85)
    ax.bar(xd + w/2, means_tc, width=w, label="Full-Sample Cubic", color=color_cub, alpha=0.85)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=0.9)
    ax.set_xticks(xd)
    ax.set_xticklabels([f"$D_{{{i}}}$" for i in xd])
    ax.set_title("(d) Mean Decile Anomaly: Out-of-Sample Test 2011--2025", loc="left", fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Yield Decile")
    ax.set_ylabel("Mean Anomaly (bu/acre)")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    f2_pdf = FIG_OUT / "fig_decile_distribution_comparison.pdf"
    f2_png = FIG_OUT / "fig_decile_distribution_comparison.png"
    fig.savefig(f2_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(f2_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f2_pdf} and {f2_png}")

def load_baseline():
    df = load_data()
    mapping = {
        "Naive": "Naive",
        "ElasticNet (+Lag)": "ElasticNet",
        "XGBoost (+Lag)": "XGBoost",
        "LSTM (+Lag)": "LSTM",
        "Ensemble (ENet + LSTM)": "Ensemble",
    }
    df_clean = df[df["model"].isin(mapping.keys())].copy()
    df_clean["model"] = df_clean["model"].map(mapping)
    return df_clean

def generate_table_a2_and_figure_a3():
    print("Generating Table A.2 and Figure A.3 (Decile Performance & Calibration)...")
    df_b = load_baseline()
    df_c = pd.read_parquet(DATA_DIR / "predictions_cubic_full_detrend.parquet")
    
    horizons = [
        (5, r"\multicolumn{13}{l}{\textit{\textbf{Panel A: Early Vegetative Horizon ($H = 5$, June 30)}}} \\"),
        (3, r"\multicolumn{13}{l}{\textit{\textbf{Panel B: Pod-Filling Reproductive Horizon ($H = 3$, August 31)}}} \\"),
        (1, r"\multicolumn{13}{l}{\textit{\textbf{Panel C: Harvest Closure ($H = 1^*$, October 31 / September 30 MN)}}} \\"),
    ]
    models = ["ElasticNet", "XGBoost", "LSTM", "Ensemble"]
    
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\caption{Out-of-sample decile mean predicted anomaly ($\bar{\hat{\epsilon}}$, bu/ac) vs observed anomaly ($\bar{\epsilon}$) across key lead times ($H = 5, 3, 1^*$). Comparison between \textbf{Base} (expanding linear trend) and \textbf{Full} (full-sample cubic trend). In bold the prediction closest to the respective observed decile mean.}",
        r"\label{tab:appendix_decile_error_performance}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l cc cc cc cc cc cc @{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Observed } $\bar{\epsilon}$} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} & \multicolumn{2}{c}{\textbf{Model Mean}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
        r"\textbf{Decile} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
        r"\midrule",
    ]
    
    for H, panel_header in horizons:
        tex.append(panel_header)
        tex.append(r"\midrule")
        
        sub_b = df_b[df_b["H"] == H].drop_duplicates(subset=["model", "year", "county_fips"])
        sub_c = df_c[df_c["H"] == H].drop_duplicates(subset=["model", "year", "county_fips"])
        
        ref_b = sub_b[sub_b["model"] == "ElasticNet"][["year", "county_fips", "y_true_anomaly"]].copy()
        ref_b["decile"] = pd.qcut(ref_b["y_true_anomaly"], 10, labels=range(1, 11))
        
        ref_c = sub_c[sub_c["model"] == "ElasticNet"][["year", "county_fips", "y_true_anomaly"]].copy()
        ref_c["decile"] = pd.qcut(ref_c["y_true_anomaly"], 10, labels=range(1, 11))
        
        piv_b = sub_b.pivot(index=["year", "county_fips"], columns="model", values="y_pred_anomaly").reset_index()
        piv_b["Model Mean"] = piv_b[models].mean(axis=1)
        piv_b = pd.merge(ref_b, piv_b, on=["year", "county_fips"])
        
        piv_c = sub_c.pivot(index=["year", "county_fips"], columns="model", values="y_pred_anomaly").reset_index()
        piv_c["Model Mean"] = piv_c[models].mean(axis=1)
        piv_c = pd.merge(ref_c, piv_c, on=["year", "county_fips"])
        
        for d in range(1, 11):
            db = piv_b[piv_b["decile"] == d]
            dc = piv_c[piv_c["decile"] == d]
            
            yb = db["y_true_anomaly"].mean()
            yc = dc["y_true_anomaly"].mean()
            
            row_items = [f"$D_{{{d}}}$" + (" (Shortfall)" if d == 1 else (" (Peak)" if d == 10 else ""))]
            row_items.append(f"{yb:+.2f}")
            row_items.append(f"{yc:+.2f}")
            
            for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]:
                pb = db[m].mean()
                pc = dc[m].mean()
                
                diff_b = abs(pb - yb)
                diff_c = abs(pc - yc)
                
                if diff_b < diff_c:
                    row_items.append(f"\\textbf{{{pb:+.2f}}}")
                    row_items.append(f"{pc:+.2f}")
                elif diff_c < diff_b:
                    row_items.append(f"{pb:+.2f}")
                    row_items.append(f"\\textbf{{{pc:+.2f}}}")
                else:
                    row_items.append(f"{pb:+.2f}")
                    row_items.append(f"{pc:+.2f}")
                    
            tex.append(" & ".join(row_items) + r" \\")
            
        if H == 1:
            tex.append(r"\midrule")
            piv_b_all = piv_b
            piv_c_all = piv_c
            
            # Mean Bias
            b_bias = [f"\\textbf{{{piv_b_all[m].mean() - piv_b_all['y_true_anomaly'].mean():+.2f}}}" if abs(piv_b_all[m].mean() - piv_b_all['y_true_anomaly'].mean()) < abs(piv_c_all[m].mean() - piv_c_all['y_true_anomaly'].mean()) else f"{piv_b_all[m].mean() - piv_b_all['y_true_anomaly'].mean():+.2f}" for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]]
            c_bias = [f"\\textbf{{{piv_c_all[m].mean() - piv_c_all['y_true_anomaly'].mean():+.2f}}}" if abs(piv_c_all[m].mean() - piv_c_all['y_true_anomaly'].mean()) < abs(piv_b_all[m].mean() - piv_b_all['y_true_anomaly'].mean()) else f"{piv_c_all[m].mean() - piv_c_all['y_true_anomaly'].mean():+.2f}" for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]]
            bias_pairs = []
            for bb, cb in zip(b_bias, c_bias):
                bias_pairs.extend([bb, cb])
            tex.append(r"\textbf{Mean Signed Bias} & --- & --- & " + " & ".join(bias_pairs) + r" \\")
            
            # MAE
            b_mae = [f"\\textbf{{{np.mean(np.abs(piv_b_all[m] - piv_b_all['y_true_anomaly'])):.2f}}}" if np.mean(np.abs(piv_b_all[m] - piv_b_all['y_true_anomaly'])) < np.mean(np.abs(piv_c_all[m] - piv_c_all['y_true_anomaly'])) else f"{np.mean(np.abs(piv_b_all[m] - piv_b_all['y_true_anomaly'])):.2f}" for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]]
            c_mae = [f"\\textbf{{{np.mean(np.abs(piv_c_all[m] - piv_c_all['y_true_anomaly'])):.2f}}}" if np.mean(np.abs(piv_c_all[m] - piv_c_all['y_true_anomaly'])) else f"{np.mean(np.abs(piv_c_all[m] - piv_c_all['y_true_anomaly'])):.2f}" for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]]
            mae_pairs = []
            for bm, cm in zip(b_mae, c_mae):
                mae_pairs.extend([bm, cm])
            tex.append(r"\textbf{MAE (bu/ac)} & --- & --- & " + " & ".join(mae_pairs) + r" \\")
            
            # RMSE
            b_rmse = [f"\\textbf{{{np.sqrt(np.mean((piv_b_all[m] - piv_b_all['y_true_anomaly'])**2)):.2f}}}" if np.sqrt(np.mean((piv_b_all[m] - piv_b_all['y_true_anomaly'])**2)) < np.sqrt(np.mean((piv_c_all[m] - piv_c_all['y_true_anomaly'])**2)) else f"{np.sqrt(np.mean((piv_b_all[m] - piv_b_all['y_true_anomaly'])**2)):.2f}" for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]]
            c_rmse = [f"\\textbf{{{np.sqrt(np.mean((piv_c_all[m] - piv_c_all['y_true_anomaly'])**2)):.2f}}}" if np.sqrt(np.mean((piv_c_all[m] - piv_c_all['y_true_anomaly'])**2)) < np.sqrt(np.mean((piv_b_all[m] - piv_b_all['y_true_anomaly'])**2)) else f"{np.sqrt(np.mean((piv_c_all[m] - piv_c_all['y_true_anomaly'])**2)):.2f}" for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]]
            rmse_pairs = []
            for br, cr in zip(b_rmse, c_rmse):
                rmse_pairs.extend([br, cr])
            tex.append(r"\textbf{RMSE (bu/ac)} & --- & --- & " + " & ".join(rmse_pairs) + r" \\")
            
            # Pearson r
            b_r = [f"\\textbf{{{pearsonr(piv_b_all[m], piv_b_all['y_true_anomaly'])[0]:.3f}}}" if pearsonr(piv_b_all[m], piv_b_all['y_true_anomaly'])[0] > pearsonr(piv_c_all[m], piv_c_all['y_true_anomaly'])[0] else f"{pearsonr(piv_b_all[m], piv_b_all['y_true_anomaly'])[0]:.3f}" for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]]
            c_r = [f"\\textbf{{{pearsonr(piv_c_all[m], piv_c_all['y_true_anomaly'])[0]:.3f}}}" if pearsonr(piv_c_all[m], piv_c_all['y_true_anomaly'])[0] > pearsonr(piv_b_all[m], piv_b_all['y_true_anomaly'])[0] else f"{pearsonr(piv_c_all[m], piv_c_all['y_true_anomaly'])[0]:.3f}" for m in ["ElasticNet", "XGBoost", "LSTM", "Ensemble", "Model Mean"]]
            r_pairs = []
            for br_, cr_ in zip(b_r, c_r):
                r_pairs.extend([br_, cr_])
            tex.append(r"\textbf{Pearson $r$} & --- & --- & " + " & ".join(r_pairs) + r" \\")
        else:
            tex.append(r"\midrule")
            
    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{0.2em}",
        r"{\footnotesize \textit{Source: Author's calculation on USDA NASS and ECMWF ERA5-Land data ($2{,}025$ evaluations per horizon). In decile $D_1$ (severe shortfall), under-prediction persists under both trend models (residual $+6.74$ bu/ac under Base, $+8.38$ bu/ac under Full), proving that tail attenuation is governed by $L_2$ conditional shrinkage rather than detrending lag.}}",
        r"\end{table}",
    ])
    
    t2_path = TAB_OUT / "tab_appendix_decile_error_performance.tex"
    t2_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {t2_path}")
    
    # Figure A.3: Decile Calibration and Signed Error
    sb_h1 = df_b[df_b["H"] == 1].drop_duplicates(subset=["model", "year", "county_fips"])
    sc_h1 = df_c[df_c["H"] == 1].drop_duplicates(subset=["model", "year", "county_fips"])
    
    ref_b = sb_h1[sb_h1["model"] == "ElasticNet"][["year", "county_fips", "y_true_anomaly"]].copy()
    ref_b["decile"] = pd.qcut(ref_b["y_true_anomaly"], 10, labels=range(1, 11))
    sb_h1 = pd.merge(sb_h1, ref_b[["year", "county_fips", "decile"]], on=["year", "county_fips"])
    
    ref_c = sc_h1[sc_h1["model"] == "ElasticNet"][["year", "county_fips", "y_true_anomaly"]].copy()
    ref_c["decile"] = pd.qcut(ref_c["y_true_anomaly"], 10, labels=range(1, 11))
    sc_h1 = pd.merge(sc_h1, ref_c[["year", "county_fips", "decile"]], on=["year", "county_fips"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    deciles = np.arange(1, 11)
    
    models_plot = [
        ("ElasticNet", "#1f77b4", "ElasticNet"),
        ("XGBoost", "#d62728", "XGBoost"),
        ("LSTM", "#8e44ad", "LSTM (+Lag)"),
        ("Ensemble", "#16a085", "Ensemble"),
    ]
    
    for m_key, col, lbl in models_plot:
        obs_b = [sb_h1[(sb_h1["decile"] == d) & (sb_h1["model"] == m_key)]["y_true_anomaly"].mean() for d in deciles]
        pred_b = [sb_h1[(sb_h1["decile"] == d) & (sb_h1["model"] == m_key)]["y_pred_anomaly"].mean() for d in deciles]
        ax1.plot(obs_b, pred_b, linestyle="--", marker="s", markersize=4.0, color=col, alpha=0.75, label=f"{lbl} [Base]")
        
        obs_c = [sc_h1[(sc_h1["decile"] == d) & (sc_h1["model"] == m_key)]["y_true_anomaly"].mean() for d in deciles]
        pred_c = [sc_h1[(sc_h1["decile"] == d) & (sc_h1["model"] == m_key)]["y_pred_anomaly"].mean() for d in deciles]
        ax1.plot(obs_c, pred_c, linestyle="-", marker="o", markersize=4.5, color=col, label=f"{lbl} [Full Cubic]")
        
    ax1.plot([-12, 12], [-12, 12], color="black", linestyle=":", linewidth=1.2, label="1:1 Calibration")
    ax1.set_xlabel("Mean Observed Anomaly in Decile (bu/acre)")
    ax1.set_ylabel("Mean Predicted Anomaly (bu/acre)")
    ax1.set_title(r"(a) Decile Calibration Diagram ($D_1 \dots D_{10}$ at Harvest $H=1^*$)", loc="left", fontweight="bold", fontsize=9.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=7.2, ncol=2)
    ax1.set_xlim(-12, 13)
    ax1.set_ylim(-6.5, 7.5)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Panel (b): Signed Residuals
    for m_key, col, lbl in [("ElasticNet", "#1f77b4", "ElasticNet"), ("LSTM", "#8e44ad", "LSTM (+Lag)"), ("Ensemble", "#16a085", "Ensemble")]:
        errs_b = [sb_h1[(sb_h1["decile"] == d) & (sb_h1["model"] == m_key)]["y_pred_anomaly"].mean() - sb_h1[(sb_h1["decile"] == d) & (sb_h1["model"] == m_key)]["y_true_anomaly"].mean() for d in deciles]
        errs_c = [sc_h1[(sc_h1["decile"] == d) & (sc_h1["model"] == m_key)]["y_pred_anomaly"].mean() - sc_h1[(sc_h1["decile"] == d) & (sc_h1["model"] == m_key)]["y_true_anomaly"].mean() for d in deciles]
        
        ax2.plot(deciles, errs_b, linestyle="--", marker="s", markersize=4.0, color=col, alpha=0.75, label=f"{lbl} [Base]")
        ax2.plot(deciles, errs_c, linestyle="-", marker="o", markersize=4.5, color=col, label=f"{lbl} [Full Cubic]")
        
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.set_xlabel("Yield Decile ($D_1 = $ Severe Shortfall, $D_{10} = $ Bumper Harvest)")
    ax2.set_ylabel(r"Mean Signed Error $\bar{\hat{\epsilon}} - \bar{\epsilon}$ (bu/acre)")
    ax2.set_title(r"(b) Mean Signed Forecast Error across Deciles", loc="left", fontweight="bold", fontsize=9.5)
    ax2.set_xticks(deciles)
    ax2.set_xticklabels([f"$D_{{{d}}}$" for d in deciles])
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=7.5, ncol=2)
    ax2.set_ylim(-10.5, 9.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    f3_pdf = FIG_OUT / "fig_decile_calibration_and_residuals_comparison.pdf"
    f3_png = FIG_OUT / "fig_decile_calibration_and_residuals_comparison.png"
    fig.savefig(f3_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(f3_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f3_pdf} and {f3_png}")

def generate_table_a3():
    print("Generating Table A.3 (Macro-Regional Harvest Yields)...")
    df_b = load_baseline()
    df_c = pd.read_parquet(DATA_DIR / "predictions_cubic_full_detrend.parquet")
    
    ydf = pd.read_csv(TARGET_CSV)
    ydf["county_fips"] = ydf["county_fips"].astype(str).str.zfill(5)
    mean_yields = ydf[ydf["year"] >= 2011].groupby("year")["yield_bu_per_acre"].mean().to_dict()
    
    sb_h1 = df_b[df_b["H"] == 1].drop_duplicates(subset=["model", "year", "county_fips"]).copy()
    sc_h1 = df_c[df_c["H"] == 1].drop_duplicates(subset=["model", "year", "county_fips"]).copy()
    
    # Reconstruct raw yield pred for df_b: raw_yield_pred = trend + y_pred_anomaly
    sb_h1["raw_yield_pred"] = sb_h1["trend"] + sb_h1["y_pred_anomaly"]
    
    year_diffs = {}
    for y, y_obs in mean_yields.items():
        sub_c = sc_h1[sc_h1["year"] == y]
        tau_c = sub_c[sub_c["model"] == "ElasticNet"]["trend"].mean()
        year_diffs[y] = (y_obs - tau_c, tau_c)
        
    sorted_years = [y for y, _ in sorted(year_diffs.items(), key=lambda x: x[1][0])]
    
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{Macro-regional aggregate yield forecasts ($\bar{\hat{y}}$, bu/ac) at harvest closure ($H = 1^*$) across test years (2011--2025) ordered by realized cubic anomaly ($\bar{\epsilon}_{\text{cub}}$). In bold the specification closest to actual yield.}",
        r"\label{tab:appendix_yearly_raw_yields}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c cc cc cc cc cc cc @{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Secular Trend } $\bar{\tau}$} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} & \multicolumn{2}{c}{\textbf{Model Mean}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
        r"\textbf{Year} & \textbf{Actual ($\bar{\epsilon}_{\text{cub}}$)} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
        r"\midrule",
    ]
    
    records = []
    for y in sorted_years:
        y_obs = mean_yields[y]
        diff_c, tau_c = year_diffs[y]
        
        sub_b = sb_h1[sb_h1["year"] == y]
        sub_c = sc_h1[sc_h1["year"] == y]
        
        tau_b = sub_b[sub_b["model"] == "Naive"]["raw_yield_pred"].mean()
        
        en_b = sub_b[sub_b["model"] == "ElasticNet"]["raw_yield_pred"].mean()
        en_c = sub_c[sub_c["model"] == "ElasticNet"]["raw_yield_pred"].mean()
        
        xgb_b = sub_b[sub_b["model"] == "XGBoost"]["raw_yield_pred"].mean()
        xgb_c = sub_c[sub_c["model"] == "XGBoost"]["raw_yield_pred"].mean()
        
        ls_b = sub_b[sub_b["model"] == "LSTM"]["raw_yield_pred"].mean()
        ls_c = sub_c[sub_c["model"] == "LSTM"]["raw_yield_pred"].mean()
        
        ens_b = sub_b[sub_b["model"] == "Ensemble"]["raw_yield_pred"].mean()
        ens_c = sub_c[sub_c["model"] == "Ensemble"]["raw_yield_pred"].mean()
        
        ml_b = np.mean([en_b, xgb_b, ls_b, ens_b])
        ml_c = np.mean([en_c, xgb_c, ls_c, ens_c])
        
        records.append({
            "year": y, "y_obs": y_obs,
            "tau_b": tau_b, "tau_c": tau_c,
            "en_b": en_b, "en_c": en_c,
            "xgb_b": xgb_b, "xgb_c": xgb_c,
            "ls_b": ls_b, "ls_c": ls_c,
            "ens_b": ens_b, "ens_c": ens_c,
            "ml_b": ml_b, "ml_c": ml_c,
        })
        
        def pair_fmt(vb, vc):
            db = abs(vb - y_obs)
            dc = abs(vc - y_obs)
            if db < dc:
                return f"\\textbf{{{vb:.2f}}}", f"{vc:.2f}"
            elif dc < db:
                return f"{vb:.2f}", f"\\textbf{{{vc:.2f}}}"
            return f"{vb:.2f}", f"{vc:.2f}"
            
        t_b, t_c = pair_fmt(tau_b, tau_c)
        e_b, e_c = pair_fmt(en_b, en_c)
        x_b, x_c = pair_fmt(xgb_b, xgb_c)
        l_b, l_c = pair_fmt(ls_b, ls_c)
        ens_b_s, ens_c_s = pair_fmt(ens_b, ens_c)
        ml_b_s, ml_c_s = pair_fmt(ml_b, ml_c)
        
        y_label_str = f"{y_obs:.2f} {{\\tiny ({diff_c:+.2f})}}"
        tex.append(f"{y} & {y_label_str} & {t_b} & {t_c} & {e_b} & {e_c} & {x_b} & {x_c} & {l_b} & {l_c} & {ens_b_s} & {ens_c_s} & {ml_b_s} & {ml_c_s} \\\\")
        
    tex.append(r"\midrule")
    df_rec = pd.DataFrame(records)
    
    # 1. Bias
    bias_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = np.mean(df_rec[f"{tag}_b"] - df_rec["y_obs"])
        vc = np.mean(df_rec[f"{tag}_c"] - df_rec["y_obs"])
        if abs(vb) < abs(vc):
            bias_strs.extend([f"\\textbf{{{vb:+.2f}}}", f"{vc:+.2f}"])
        elif abs(vc) < abs(vb):
            bias_strs.extend([f"{vb:+.2f}", f"\\textbf{{{vc:+.2f}}}"])
        else:
            bias_strs.extend([f"{vb:+.2f}", f"{vc:+.2f}"])
    tex.append(r"\textbf{Mean Bias} & --- & " + " & ".join(bias_strs) + r" \\")
    
    # 2. MAE
    mae_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = np.mean(np.abs(df_rec[f"{tag}_b"] - df_rec["y_obs"]))
        vc = np.mean(np.abs(df_rec[f"{tag}_c"] - df_rec["y_obs"]))
        if vb < vc:
            mae_strs.extend([f"\\textbf{{{vb:.2f}}}", f"{vc:.2f}"])
        elif vc < vb:
            mae_strs.extend([f"{vb:.2f}", f"\\textbf{{{vc:.2f}}}"])
        else:
            mae_strs.extend([f"{vb:.2f}", f"{vc:.2f}"])
    tex.append(r"\textbf{MAE (bu/ac)} & --- & " + " & ".join(mae_strs) + r" \\")
    
    # 3. RMSE
    rmse_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = np.sqrt(np.mean((df_rec[f"{tag}_b"] - df_rec["y_obs"])**2))
        vc = np.sqrt(np.mean((df_rec[f"{tag}_c"] - df_rec["y_obs"])**2))
        if vb < vc:
            rmse_strs.extend([f"\\textbf{{{vb:.2f}}}", f"{vc:.2f}"])
        elif vc < vb:
            rmse_strs.extend([f"{vb:.2f}", f"\\textbf{{{vc:.2f}}}"])
        else:
            rmse_strs.extend([f"{vb:.2f}", f"{vc:.2f}"])
    tex.append(r"\textbf{RMSE (bu/ac)} & --- & " + " & ".join(rmse_strs) + r" \\")
    
    # 4. Pearson r
    r_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        rb, _ = pearsonr(df_rec[f"{tag}_b"], df_rec["y_obs"])
        rc, _ = pearsonr(df_rec[f"{tag}_c"], df_rec["y_obs"])
        if rb > rc:
            r_strs.extend([f"\\textbf{{{rb:.3f}}}", f"{rc:.3f}"])
        elif rc > rb:
            r_strs.extend([f"{rb:.3f}", f"\\textbf{{{rc:.3f}}}"])
        else:
            r_strs.extend([f"{rb:.3f}", f"{rc:.3f}"])
    tex.append(r"\textbf{Pearson $r$} & --- & " + " & ".join(r_strs) + r" \\")
    
    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{0.2em}",
        r"{\footnotesize \textit{Source: Author's calculation on USDA NASS and ECMWF ERA5-Land data ($135$ counties, 2011--2025). The full-sample cubic trend eliminates secular drift in the test set (bias $+0.06$ vs $-2.41$ bu/ac), improving macro MAE from $1.87$ to $1.46$ bu/ac while capturing the 2012 drought shock with high fidelity ($43.71$ vs $43.55$ actual).}}",
        r"\end{table}",
    ])
    
    t3_path = TAB_OUT / "tab_appendix_yearly_raw_yields.tex"
    t3_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {t3_path}")

def copy_figure_a4():
    print("Copying Figure A.4 (Progression comparison)...")
    src_pdf = EXP_DIR / "figures" / "fig_r2_horizon_progression_comparison.pdf"
    src_png = EXP_DIR / "figures" / "fig_r2_horizon_progression_comparison.png"
    if src_pdf.exists():
        shutil.copy(src_pdf, FIG_OUT / "fig_r2_horizon_progression_comparison.pdf")
        print(f"Copied {src_pdf} to {FIG_OUT}")
    if src_png.exists():
        shutil.copy(src_png, FIG_OUT / "fig_r2_horizon_progression_comparison.png")
        print(f"Copied {src_png} to {FIG_OUT}")


if __name__ == "__main__":
    main()


