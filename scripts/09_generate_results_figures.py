"""
09_generate_results_figures.py

Generates all 6 official empirical results figures for Chapter 5 (Results):
1. Figure 5.1: Multi-Horizon Progression Curves across Phenological Stages (fig_r2_horizon_progression.pdf / .png)
2. Figure 5.2: Lead-Time Shock Bias and Extreme Anomaly Scatter (fig_shock_bias_profiles.pdf / .png)
3. Figure 5.3: Decile Reliability Diagram and Residual Diagnostics (fig_decile_calibration_and_residuals.pdf / .png)
4. Figure 5.4: Grouped Permutation Feature Importance Faceted Bars (fig54_pfi_faceted_bars.pdf / .png)
5. Figure 5.5: GPFI Macro-Domain Physiological Sensitivity Bars (fig55_pfi_domain_bars.pdf / .png)
6. Figure 5.6: Spatial Choropleth of Out-of-Sample County R2 (fig_county_r2_map.pdf / .png)

Outputs:
- output/thesis/figures/fig_r2_horizon_progression.pdf / .png
- output/thesis/figures/fig_shock_bias_profiles.pdf / .png
- output/thesis/figures/fig_decile_calibration_and_residuals.pdf / .png
- output/thesis/figures/fig54_pfi_faceted_bars.pdf / .png
- output/thesis/figures/fig55_pfi_domain_bars.pdf / .png
- output/thesis/figures/fig_county_r2_map.pdf / .png
"""

import json
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
    plot_figure_5_2(preds)
    plot_figure_5_3(preds)

    pfi_data = prepare_pfi_data()
    plot_figure_5_4(pfi_data)
    plot_figure_5_5(pfi_data)

    plot_figure_5_6(preds)
    print("All Chapter 5 figures generated successfully into output/thesis/figures/!")


if __name__ == "__main__":
    main()
