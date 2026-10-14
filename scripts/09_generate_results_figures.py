"""
09_generate_results_figures.py

Generates empirical results figures for Chapter 5 (Results):
1. Figure 5.1: Multi-Horizon Progression Curves across Phenological Stages (fig_r2_horizon_progression.pdf).
2. Figure 5.2: Lead-Time Shock Bias and Extreme Anomaly Scatter (fig_shock_bias_profiles.pdf).
3. Figure 5.5: Spatial Choropleth of Out-of-Sample County R2 (fig_county_r2_map.pdf).

Outputs:
- output/thesis/figures/fig_r2_horizon_progression.pdf / .png
- output/thesis/figures/fig_shock_bias_profiles.pdf / .png
- output/thesis/figures/fig_county_r2_map.pdf / .png
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREDS_ML_PARQUET = ROOT / "output" / "data" / "processed" / "predictions.parquet"
PREDS_LSTM_PARQUET = ROOT / "output" / "data" / "processed" / "predictions_lstm.parquet"
SHAPEFILE = ROOT / "data" / "raw" / "census_counties" / "cb_2020_us_county_500k.shp"
FIGURES_DIR = ROOT / "output" / "thesis" / "figures"

# Test window
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
    "XGBoost": "#d62728",
    "LSTM (Weather-Only)": "#2ca02c",
    "LSTM (+Lag)": "#8e44ad",
}


def load_data():
    """Load and unify out-of-sample predictions for test years 2011-2025."""
    dfs = []
    if PREDS_ML_PARQUET.exists():
        df_ml = pd.read_parquet(PREDS_ML_PARQUET)
        df_ml = df_ml[df_ml["year"].isin(TEST_YEARS)]
        dfs.append(df_ml)

    if PREDS_LSTM_PARQUET.exists():
        df_lstm = pd.read_parquet(PREDS_LSTM_PARQUET)
        df_lstm = df_lstm[df_lstm["year"].isin(TEST_YEARS)]
        df_lstm["model"] = np.where(df_lstm["include_lag"], "LSTM (+Lag)", "LSTM (Weather-Only)")
        df_lstm = df_lstm.rename(columns={
            "actual_anomaly": "y_true_anomaly",
            "pred_anomaly": "y_pred_anomaly",
            "horizon": "H",
        })
        dfs.append(df_lstm[["model", "H", "year", "county_fips", "y_true_anomaly", "y_pred_anomaly"]])

    return pd.concat(dfs, ignore_index=True)


def plot_figure_5_1(preds):
    """Figure 5.1: Multi-Horizon Progression Curves."""
    print("Generating Figure 5.1: R2 and RMSE progression across horizons...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4), sharex=True)
    x_pos = np.arange(len(HORIZONS))

    # Highlight Noise Regime (H=11..6) vs Active Signal Regime (H=5..1)
    for ax in (ax1, ax2):
        ax.axvspan(-0.5, 5.5, color="#f2f4f8", alpha=0.75)
        ax.axvspan(5.5, 10.5, color="#fdfbf2", alpha=0.85)
        ax.axvline(5.5, color="#c0392b", linestyle=":", linewidth=1.5, alpha=0.85)

    # Panel (a): R2_OOS Pooled
    ax1.plot(x_pos, np.zeros(len(HORIZONS)), label="Naive Secular Trend", color=COLORS["Naive"], linestyle="-", marker="s", markersize=5)

    models_to_plot = ["ElasticNet", "RandomForest", "XGBoost", "LSTM (Weather-Only)", "LSTM (+Lag)"]
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

        ls = "--" if "+Lag" in m else "-"
        ax1.plot(x_pos, r2_vals, label=m, color=COLORS.get(m, "#333333"), linestyle=ls, marker="o", markersize=5, linewidth=1.8)
        ax2.plot(x_pos, rmse_vals, label=m, color=COLORS.get(m, "#333333"), linestyle=ls, marker="o", markersize=5, linewidth=1.8)

    ax1.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.set_ylabel(r"Pooled Out-of-Sample $R^2_{\mathrm{OOS}}$")
    ax1.set_title(r"(a) Multi-Horizon Explained Anomaly Variance ($R^2_{\mathrm{OOS}}$)", loc="left", fontweight="bold", fontsize=11)
    ax1.set_ylim(-0.35, 0.42)
    ax1.legend(loc="lower left", frameon=True, fontsize=8.0, ncol=2)

    # Panel (b): Out-of-sample RMSE
    naive_rmse = 5.944
    ax2.plot(x_pos, np.full(len(HORIZONS), naive_rmse), label="Naive Baseline (5.94 bu/ac)", color=COLORS["Naive"], linestyle="-", marker="s", markersize=5)
    ax2.set_ylabel("Out-of-Sample RMSE (bu/acre)")
    ax2.set_title("(b) Forecast Error Magnitude (RMSE)", loc="left", fontweight="bold", fontsize=11)
    ax2.set_ylim(4.80, 7.20)
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


def plot_figure_5_2(preds):
    """Figure 5.2: Lead-time shock bias profiles and scatter."""
    print("Generating Figure 5.2: Shock bias profiles...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    active_h = [5, 4, 3, 2, 1]
    x_pos = np.arange(len(active_h))

    for yr, name, ls in [(2012, "2012 Drought", "-"), (2016, "2016 Bumper", "--")]:
        sub_yr = preds[preds["year"] == yr]
        for m, col in [("XGBoost", "#d62728"), ("ElasticNet", "#1f77b4"), ("LSTM (+Lag)", "#8e44ad")]:
            sub_m = sub_yr[sub_yr["model"] == m]
            if sub_m.empty:
                continue
            biases = [sub_m[sub_m["H"] == h]["y_pred_anomaly"].mean() - sub_m[sub_m["H"] == h]["y_true_anomaly"].mean() for h in active_h]
            ax1.plot(x_pos, biases, label=f"{m} ({name})", color=col, linestyle=ls, marker="o", markersize=5, linewidth=1.8)

    ax1.axhline(0, color="black", linestyle=":", linewidth=1.2, label="Zero Bias")
    ax1.set_ylabel(r"Mean Anomaly Bias ($\bar{\hat{\epsilon}} - \bar{\epsilon}$, bu/acre)")
    ax1.set_title(r"(a) Lead-Time Shock Bias Trajectory Across Active Horizons", loc="left", fontweight="bold", fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"H={h}" for h in active_h])
    ax1.set_xlabel("Forecast Horizon")
    ax1.legend(loc="lower left", frameon=True, fontsize=8.0, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel (b): Scatter of Predicted vs Observed 2012 Drought Anomalies
    sub_2012 = preds[(preds["year"] == 2012) & (preds["model"] == "LSTM (+Lag)")]
    colors_h = {5: "#e67e22", 3: "#2980b9", 1: "#27ae60"}
    for h in [5, 3, 1]:
        sh = sub_2012[sub_2012["H"] == h]
        if not sh.empty:
            ax2.scatter(sh["y_true_anomaly"], sh["y_pred_anomaly"], alpha=0.65, color=colors_h[h], s=30, label=f"H={h}")

    ax2.plot([-20, 10], [-20, 10], "k--", linewidth=1.1, label="1:1 Reference Line")
    ax2.set_xlabel("Observed 2012 County Anomaly (bu/acre)")
    ax2.set_ylabel("Predicted Anomaly (bu/acre)")
    ax2.set_title(r"(b) 2012 Flash Drought: Trajectory from Blind to Resolved", loc="left", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_shock_bias_profiles.pdf")
    fig.savefig(FIGURES_DIR / "fig_shock_bias_profiles.png")
    plt.close(fig)


def plot_figure_5_5(preds):
    """Figure 5.5: Spatial Choropleth of County R2_OOS."""
    print("Generating Figure 5.5: County R2 Spatial Map...")
    if not SHAPEFILE.exists():
        return

    # Evaluate champion model (LSTM (+Lag)) at harvest horizon H=1
    sub = preds[(preds["model"] == "LSTM (+Lag)") & (preds["H"] == 1)]
    if sub.empty:
        # Fallback to XGBoost if LSTM not present
        sub = preds[(preds["model"] == "XGBoost") & (preds["H"] == 1)]
        if sub.empty:
            return

    county_r2 = []
    for fips, grp in sub.groupby("county_fips"):
        yt, yp = grp["y_true_anomaly"].values, grp["y_pred_anomaly"].values
        ss_res, ss_tot = np.sum((yt - yp) ** 2), np.sum(yt ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        county_r2.append({"GEOID": str(fips).zfill(5), "r2_oos": r2})

    df_r2 = pd.DataFrame(county_r2)
    gdf = gpd.read_file(SHAPEFILE)
    midwest_fips = ["17", "18", "19", "27", "29", "39"]
    gdf_mw = gdf[gdf["STATEFP"].isin(midwest_fips)].to_crs(epsg=5070)
    gdf_merged = gdf_mw.merge(df_r2, on="GEOID", how="left")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    gdf_mw.plot(ax=ax, color="#f2f2f2", edgecolor="#d9d9d9", linewidth=0.4)
    gdf_merged.dropna(subset=["r2_oos"]).plot(
        column="r2_oos", ax=ax, cmap="YlGnBu", legend=True,
        vmin=-0.05, vmax=0.65,
        legend_kwds={
            "label": r"Out-of-Sample Explained Variance ($R^2_{\mathrm{OOS}}$ at Harvest Horizon $H=1$)",
            "orientation": "horizontal", "pad": 0.04, "shrink": 0.65, "aspect": 25
        },
        edgecolor="#333333", linewidth=0.5
    )
    gdf_mw.dissolve(by="STATE_NAME").plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.1)
    ax.set_title(r"Spatial Heterogeneity of Out-of-Sample Predictive Skill ($R^2_{\mathrm{OOS}}$, $H=1$, 2011--2025)", fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_county_r2_map.pdf")
    fig.savefig(FIGURES_DIR / "fig_county_r2_map.png")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_data()
    plot_figure_5_1(preds)
    plot_figure_5_2(preds)
    plot_figure_5_5(preds)
    print("Chapter 5 figures generated successfully.")


if __name__ == "__main__":
    main()
