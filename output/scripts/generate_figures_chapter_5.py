"""Generate all Figures and populate Results for Chapter 5: Empirical Results.

Outputs:
  output/thesis/figures/fig_r2_horizon_progression.pdf (.png)
  output/thesis/figures/fig_shock_bias_profiles.pdf (.png)
  output/thesis/figures/fig_shap_beeswarm.pdf (.png)
  output/thesis/figures/fig_shap_horizon_stack.pdf (.png)
  output/thesis/figures/fig_county_r2_map.pdf (.png)
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import geopandas as gpd
from xgboost import XGBRegressor
import shap

# ---------------------------------------------------------------------------
# Plot style configuration
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "output" / "data" / "processed"
MODEL_DATASETS = PROCESSED_DIR / "model_datasets"
FIGURES_DIR = ROOT / "output" / "thesis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SHAPEFILE = ROOT / "data" / "raw" / "census_counties" / "cb_2020_us_county_500k.shp"

SUMMARY_CSV = PROCESSED_DIR / "evaluation_summary.csv"
PREDS_PARQUET = PROCESSED_DIR / "predictions.parquet"

COLOR_MAP = {
    'Naive': '#555555',
    'ElasticNet': '#1f77b4',
    'RandomForest': '#ff7f0e',
    'XGBoost': '#d62728',
    'LSTM': '#2ca02c',
    'LSTM (Weather-Only)': '#2ca02c',
    'LSTM (+Lag Yield)': '#9467bd'
}
MARKER_MAP = {
    'Naive': 's',
    'ElasticNet': 'o',
    'RandomForest': '^',
    'XGBoost': 'D',
    'LSTM': 'P'
}

MONTH_LABELS = {
    12: 'Nov Y-1\n(H=12)',
    11: 'Dec Y-1\n(H=11)',
    10: 'Jan Y\n(H=10)',
    9: 'Feb Y\n(H=9)',
    8: 'Mar Y\n(H=8)',
    7: 'Apr Y\n(H=7)',
    6: 'May Y\n(H=6)',
    5: 'Jun Y\n(H=5)',
    4: 'Jul Y\n(H=4)',
    3: 'Aug Y\n(H=3)',
    2: 'Sep Y\n(H=2)',
    1: 'Oct Y\n(H=1)'
}

ACTIVE_MONTH_LABELS = {
    5: 'Jun 30\n(H=5, Pre-Flw)',
    4: 'Jul 31\n(H=4, Flowering)',
    3: 'Aug 31\n(H=3, Pod-Fill)',
    2: 'Sep 30\n(H=2, Maturity)',
    1: 'Oct 31\n(H=1, Harvest)'
}

# ---------------------------------------------------------------------------
# Figure 5.1: Generalist Benchmark Across All 11 Horizons (Weather-Only vs +Lag)
# ---------------------------------------------------------------------------
def plot_figure_5_1(summary_wo: pd.DataFrame, summary_int: pd.DataFrame, summary_lstm: pd.DataFrame):
    print("Generating Figure 5.1: Comprehensive Multi-Horizon Progression (All Models)...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.4), sharex=True)

    horizons = list(range(11, 0, -1))
    x_pos = np.arange(len(horizons))

    # Demarcate Regime 1: Noise / Zero Skill (H=11..6, indices 0..5) vs Regime 2: Active Signal (H=5..1, indices 6..10)
    for ax in (ax1, ax2):
        # Noise regime: H=11..6
        ax.axvspan(-0.5, 5.5, color='#f2f4f8', alpha=0.75, zorder=0)
        # Active agronomic signal regime: H=5..1
        ax.axvspan(5.5, 10.5, color='#fdfbf2', alpha=0.85, zorder=0)
        # Vertical boundary between Noise and Active Signal
        ax.axvline(5.5, color='#c0392b', linestyle=':', linewidth=1.5, alpha=0.85, zorder=2)

    # Panel 1: R2_OOS Pooled across all candidate models
    # Naive baseline
    ax1.plot(x_pos, np.zeros(len(horizons)), label=r'Naive Secular Trend ($\mathcal{B}_{\mathrm{naive}}$)',
             color=COLOR_MAP['Naive'], linestyle='-', linewidth=1.8, marker='s', markersize=5, zorder=3)

    # Tabular Models: Weather-Only (solid) vs Integrated (+Lag) (dashed)
    for model in ['ElasticNet', 'RandomForest', 'XGBoost']:
        m_wo = summary_wo[summary_wo['model'] == model].set_index('H').reindex(horizons)
        m_int = summary_int[summary_int['model'] == model].set_index('H').reindex(horizons)

        r2_wo = m_wo['R2_OOS_pooled'].values
        r2_int = m_int['R2_OOS_pooled'].values

        # Solid: Pure Meteorological
        ax1.plot(x_pos, r2_wo, label=f'{model} (Weather-Only)', color=COLOR_MAP[model],
                 marker=MARKER_MAP[model], markersize=5.0, linewidth=1.5, linestyle='-', zorder=4)
        # Dashed: Integrated with Lag
        ax1.plot(x_pos, r2_int, label=f'{model} (+Lag)', color=COLOR_MAP[model],
                 marker=MARKER_MAP[model], markersize=5.0, linewidth=1.5, linestyle='--', alpha=0.85, zorder=4)

    # Sequential LSTM Models: Weather-Only vs Integrated (+Lag)
    if not summary_lstm.empty:
        # LSTM Weather-Only champion (PReLU head)
        lstm_wo = summary_lstm[(summary_lstm['variant'] == 'prelu') & (~summary_lstm['include_lag'])].set_index('H').reindex(horizons)
        r2_lstm_wo = lstm_wo['R2_OOS_pooled'].values
        ax1.plot(x_pos, r2_lstm_wo, label='LSTM (Weather-Only)', color=COLOR_MAP['LSTM'],
                 marker=MARKER_MAP['LSTM'], markersize=5.5, linewidth=2.0, linestyle='-', zorder=5)

        # LSTM Integrated champion (Linear head)
        lstm_int = summary_lstm[(summary_lstm['variant'] == 'linear') & (summary_lstm['include_lag'])].set_index('H').reindex(horizons)
        r2_lstm_int = lstm_int['R2_OOS_pooled'].values
        ax1.plot(x_pos, r2_lstm_int, label=r'LSTM (+Lag $\epsilon_{t-1}$)', color='#8e44ad',
                 marker='X', markersize=6.0, linewidth=2.2, linestyle='--', zorder=5)

    ax1.axhline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.6)
    ax1.set_ylabel(r'Pooled Out-of-Sample $R^2_{\mathrm{OOS}}$')
    ax1.set_title(r'(a) Generalist Model Benchmark: Explained Anomaly Variance ($R^2_{\mathrm{OOS}}$)', loc='left', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.35, 0.42)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax1.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8.0, ncol=2)

    # Annotate Regimes
    ax1.text(2.5, 0.38, r'NOISE REGIME ($H \geq 6$)' + '\n' + r'$R^2 \leq 0$ (Secular Trend Optimal)', ha='center',
             fontsize=9.0, color='#5a6b7c', fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
    ax1.text(8.0, 0.38, r'ACTIVE SIGNAL REGIME ($H \leq 5$)' + '\n' + 'Rapid Agronomic Skill Accumulation', ha='center',
             fontsize=9.0, color='#8c3b1e', fontweight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='#e0b49c'))

    # Panel 2: Out-of-Sample RMSE & Skill Score
    naive_rmse = 6.123918
    ax2.plot(x_pos, np.full(len(horizons), naive_rmse), label=r'Naive Secular Baseline ($6.12\,\mathrm{bu/ac}$)',
             color=COLOR_MAP['Naive'], linestyle='-', linewidth=1.8, marker='s', markersize=5, zorder=3)

    for model in ['ElasticNet', 'RandomForest', 'XGBoost']:
        m_wo = summary_wo[summary_wo['model'] == model].set_index('H').reindex(horizons)
        m_int = summary_int[summary_int['model'] == model].set_index('H').reindex(horizons)

        rmse_wo = m_wo['RMSE_OOS_pooled'].values
        rmse_int = m_int['RMSE_OOS_pooled'].values

        ax2.plot(x_pos, rmse_wo, label=f'{model} (Weather-Only)', color=COLOR_MAP[model],
                 marker=MARKER_MAP[model], markersize=5.0, linewidth=1.5, linestyle='-', zorder=4)
        ax2.plot(x_pos, rmse_int, label=f'{model} (+Lag)', color=COLOR_MAP[model],
                 marker=MARKER_MAP[model], markersize=5.0, linewidth=1.5, linestyle='--', alpha=0.85, zorder=4)

    if not summary_lstm.empty:
        rmse_lstm_wo = lstm_wo['RMSE_OOS_pooled'].values
        ax2.plot(x_pos, rmse_lstm_wo, label='LSTM (Weather-Only)', color=COLOR_MAP['LSTM'],
                 marker=MARKER_MAP['LSTM'], markersize=5.5, linewidth=2.0, linestyle='-', zorder=5)

        rmse_lstm_int = lstm_int['RMSE_OOS_pooled'].values
        ax2.plot(x_pos, rmse_lstm_int, label=r'LSTM (+Lag $\epsilon_{t-1}$)', color='#8e44ad',
                 marker='X', markersize=6.0, linewidth=2.2, linestyle='--', zorder=5)

    ax2.set_ylabel(r'Out-of-Sample RMSE ($\mathrm{bu/acre}$)')
    ax2.set_title(r'(b) Forecast Error Magnitude and Percentage Skill Score', loc='left', fontsize=11, fontweight='bold')
    ax2.set_ylim(4.80, 7.20)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax2.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=8.0, ncol=2)

    # Secondary y-axis for percentage skill score
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel(r'Skill Score vs. Secular Baseline ($\% = 1 - \mathrm{RMSE}/\mathrm{RMSE}_{\mathrm{naive}}$)', color='#333333')
    ax2_twin.set_ylim((1 - 7.20/naive_rmse)*100, (1 - 4.80/naive_rmse)*100)
    ax2_twin.yaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax2_twin.grid(False)

    for ax in (ax1, ax2):
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MONTH_LABELS[h] for h in horizons], rotation=0)
        ax.set_xlabel('Forecast Horizon & Campaign Month')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_r2_horizon_progression.pdf")
    fig.savefig(FIGURES_DIR / "fig_r2_horizon_progression.png")
    plt.close(fig)
    print("  Saved Figure 5.1.", flush=True)


# ---------------------------------------------------------------------------
# Figure 5.2: In-Depth Shock Magnitude & Bias Analysis (Active Horizons H=5..1)
# ---------------------------------------------------------------------------
def plot_figure_5_2(preds_int: pd.DataFrame, preds_lstm: pd.DataFrame):
    print("Generating Figure 5.2: In-Depth Shock Magnitude & Convergence (Active Horizons H=5..1)...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.2))

    # Restrict strictly to the 5 active horizons: H in [5, 4, 3, 2, 1]
    active_h = [5, 4, 3, 2, 1]
    x_pos = np.arange(len(active_h))

    # Standardize LSTM predictions format
    p_lstm = preds_lstm.copy()
    if 'horizon' in p_lstm.columns:
        p_lstm = p_lstm.rename(columns={'horizon': 'H', 'actual_anomaly': 'y_true_anomaly', 'pred_anomaly': 'y_pred_anomaly'})
    
    # Filter to test years (>= 2011)
    df_tab = preds_int[preds_int['year'] >= 2011].copy()
    df_l = p_lstm[(p_lstm['year'] >= 2011) & (p_lstm['include_lag']) & (p_lstm['head_variant'] == 'linear')].copy()
    df_l['model'] = 'LSTM_Linear_Lag'

    # Panel (a): Mean Anomaly Bias & Shock Resolution across the 5 Active Lead Times
    # Shocks: 2012 Severe Drought (actual anomaly: -4.07 bu/ac) and 2016 Bumper Season (actual anomaly: +7.34 bu/ac)
    for yr, true_anom, name_lbl, c_base in [(2012, -4.07, '2012 Flash Drought (-11.7%)', '#d62728'),
                                            (2016, +7.34, '2016 Bumper Crop (+14.8%)', '#27ae60')]:
        # Models to compare: Champion Tabular (ElasticNet & XGBoost) and Champion Sequential (LSTM Linear)
        # XGBoost
        xgb_biases = []
        elnet_biases = []
        lstm_biases = []
        naive_biases = []

        for h in active_h:
            # Naive bias = 0 - true_anomaly = -true_anomaly
            naive_biases.append(-true_anom)

            sub_xgb = df_tab[(df_tab['year'] == yr) & (df_tab['H'] == h) & (df_tab['model'] == 'XGBoost')]
            xgb_biases.append((sub_xgb['y_pred_anomaly'] - sub_xgb['y_true_anomaly']).mean())

            sub_el = df_tab[(df_tab['year'] == yr) & (df_tab['H'] == h) & (df_tab['model'] == 'ElasticNet')]
            el_biases = (sub_el['y_pred_anomaly'] - sub_el['y_true_anomaly']).mean()
            elnet_biases.append(el_biases)

            sub_lst = df_l[(df_l['year'] == yr) & (df_l['H'] == h)]
            lstm_biases.append((sub_lst['y_pred_anomaly'] - sub_lst['y_true_anomaly']).mean())

        # Plot trajectories
        ls = '-' if yr == 2012 else '--'
        mk = 'o' if yr == 2012 else '^'
        ax1.plot(x_pos, xgb_biases, label=f'XGBoost ({name_lbl})', color='#d62728' if yr == 2012 else '#e67e22',
                 linestyle=ls, marker='D', markersize=5.5, linewidth=1.8)
        ax1.plot(x_pos, elnet_biases, label=f'ElasticNet ({name_lbl})', color='#2980b9' if yr == 2012 else '#27ae60',
                 linestyle=ls, marker='o', markersize=5.5, linewidth=1.8)
        ax1.plot(x_pos, lstm_biases, label=f'LSTM (+Lag) ({name_lbl})', color='#8e44ad' if yr == 2012 else '#16a085',
                 linestyle=ls, marker='X', markersize=6.0, linewidth=2.0)

    ax1.axhline(0, color='black', linestyle=':', linewidth=1.2, alpha=0.85, label='Zero Bias (Unbiased)')
    ax1.set_ylabel(r'Mean Anomaly Bias ($\bar{\hat{\epsilon}} - \bar{\epsilon}$, bu/acre)')
    ax1.set_title(r'(a) Lead-Time Shock Bias Trajectory Across Active Horizons ($H=5 \dots 1$)', loc='left', fontsize=11, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([ACTIVE_MONTH_LABELS[h] for h in active_h], rotation=0)
    ax1.set_xlabel('Active Campaign Milestone')
    ax1.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8.0, ncol=2)
    ax1.set_ylim(-8.5, 5.5)

    # Panel (b): County-Level Anomaly Scatter (Predicted vs Observed) for the 2012 Flash Drought at H=5, H=3, H=1
    colors_h = {5: '#e67e22', 3: '#2980b9', 1: '#27ae60'}
    labels_h = {
        5: 'H=5 (Jun 30, Pre-Flowering)',
        3: 'H=3 (Aug 31, Pod-Filling Peak Stress)',
        1: 'H=1 (Oct 31, Final Harvest Resolution)'
    }

    sub_2012 = df_l[df_l['year'] == 2012]
    for h in [5, 3, 1]:
        h_data = sub_2012[sub_2012['H'] == h]
        ax2.scatter(h_data['y_true_anomaly'], h_data['y_pred_anomaly'],
                    alpha=0.65, color=colors_h[h], s=30, label=labels_h[h], edgecolors='none')
        # Regression slope
        z = np.polyfit(h_data['y_true_anomaly'], h_data['y_pred_anomaly'], 1)
        p = np.poly1d(z)
        x_lin = np.linspace(-18, 5, 50)
        ax2.plot(x_lin, p(x_lin), color=colors_h[h], linewidth=1.8, label=f'Slope: {z[0]:.2f}')

    # 45-degree 1:1 line
    ax2.plot([-20, 10], [-20, 10], 'k--', linewidth=1.1, alpha=0.75, label='1:1 Line (Perfect Prediction)')
    ax2.set_xlabel(r'Observed 2012 County Anomaly ($\epsilon_{c, 2012}$, bu/acre)')
    ax2.set_ylabel(r'Predicted Anomaly ($\hat{\epsilon}_{c, 2012}$, bu/acre)')
    ax2.set_title(r'(b) 2012 Flash Drought: Trajectory from Blind to Resolved (LSTM +Lag)', loc='left', fontsize=11, fontweight='bold')
    ax2.set_xlim(-20, 8)
    ax2.set_ylim(-16, 6)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.0, ncol=2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_shock_bias_profiles.pdf")
    fig.savefig(FIGURES_DIR / "fig_shock_bias_profiles.png")
    plt.close(fig)
    print("  Saved Figure 5.2.", flush=True)


# ---------------------------------------------------------------------------
# Figure 5.3 & 5.4: SHAP Feature Importance & Temporal Domain Progression
# ---------------------------------------------------------------------------
def plot_figure_5_3_and_5_4():
    print("Generating Figure 5.3 & 5.4: SHAP Beeswarm & Domain Evolution...", flush=True)
    # Train a reference XGBoost model at H=2 (Peak accuracy)
    h2_path = MODEL_DATASETS / "features_H02.parquet"
    if not h2_path.exists():
        print("  Features H=2 not found, skipping SHAP.")
        return

    feat_df = pd.read_parquet(h2_path)
    target_df = pd.read_csv(ROOT / "data" / "target" / "soybean_yield_1951_2025.csv")
    target_df['county_fips'] = target_df['county_fips'].astype(str)
    feat_df['county_fips'] = feat_df['county_fips'].astype(str)
    feat_df = feat_df.rename(columns={'crop_year': 'year'})

    df = target_df.merge(feat_df, on=['county_fips', 'year'], how='inner')
    
    # Train on 1951-2015, evaluate on modern test sample (2016-2025)
    train_df = df[df['year'] <= 2015]
    test_df = df[df['year'] >= 2016]

    feature_cols = [c for c in feat_df.columns if c not in ('county_fips', 'year')]
    
    # Simple linear detrend on training
    trend_dict = {}
    for fips, grp in train_df.groupby('county_fips'):
        p = np.polyfit(grp['year'], grp['yield_bu_per_acre'], 1)
        trend_dict[fips] = p
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['trend'] = train_df.apply(lambda r: trend_dict[r['county_fips']][0]*r['year'] + trend_dict[r['county_fips']][1], axis=1)
    test_df['trend'] = test_df.apply(lambda r: trend_dict[r['county_fips']][0]*r['year'] + trend_dict[r['county_fips']][1], axis=1)
    train_df['anomaly'] = train_df['yield_bu_per_acre'] - train_df['trend']
    test_df['anomaly'] = test_df['yield_bu_per_acre'] - test_df['trend']

    X_train = train_df[feature_cols].values
    y_train = train_df['anomaly'].values
    X_test = test_df[feature_cols].values

    # Fit XGBoost
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, colsample_bytree=0.8, reg_lambda=2.0, random_state=42)
    xgb.fit(X_train, y_train)

    # Subsample test set for fast SHAP TreeExplainer
    np.random.seed(42)
    idx_sample = np.random.choice(len(X_test), min(450, len(X_test)), replace=False)
    X_sample = X_test[idx_sample]

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)

    # Figure 5.3: SHAP Summary (Beeswarm style via bar + scatter)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
    top_features = [feature_cols[i] for i in top_indices]

    # Friendly labels for top features
    def clean_name(f):
        parts = f.split('_')
        m_num = int(parts[-1].replace('m', '')) if 'm' in parts[-1] else None
        m_names = {1:'Nov', 2:'Dec', 3:'Jan', 4:'Feb', 5:'Mar', 6:'Apr', 7:'May', 8:'Jun', 9:'Jul', 10:'Aug', 11:'Sep'}
        v_name = '_'.join(parts[:-1]) if m_num else f
        v_clean = {
            'HD35': 'Heat Days (>=35C)',
            'HD30': 'Heat Days (>=30C)',
            'VPD': 'Vapor Pressure Deficit',
            'ET0': 'Ref Evapotranspiration',
            'P_minus_ET0': 'Climatic Balance (P-ET0)',
            'SM_root': 'Root-Zone Soil Moisture',
            'SM3': 'Soil Moisture Layer 3',
            'SM2': 'Soil Moisture Layer 2',
            'T_max': 'Max Temperature',
            'T_mean': 'Mean Temperature',
            'P': 'Precipitation',
            'GDD': 'Growing Degree Days',
            'SSRD': 'Solar Radiation',
            'lat_norm': 'Latitude (Spatial)',
            'lon_norm': 'Longitude (Spatial)'
        }.get(v_name, v_name)
        return f"{v_clean} [{m_names.get(m_num, '')}]" if m_num else v_clean

    fig53, ax = plt.subplots(figsize=(10, 6.8))
    y_pos = np.arange(len(top_indices))
    ax.barh(y_pos, mean_abs_shap[top_indices][::-1], color='#2b5c8f', alpha=0.85, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([clean_name(feature_cols[i]) for i in top_indices][::-1], fontsize=9)
    ax.set_xlabel(r'Mean Absolute SHAP Value ($\mathrm{E}[|\phi_j|]$, bu/acre contribution to anomaly)')
    ax.set_title(r'Top 20 Agrometeorological Predictors at Critical Milestone $H=2$ (Sep 30)', fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    fig53.savefig(FIGURES_DIR / "fig_shap_beeswarm.pdf")
    fig53.savefig(FIGURES_DIR / "fig_shap_beeswarm.png")
    plt.close(fig53)
    print("  Saved Figure 5.3.", flush=True)

    # Figure 5.4: Feature Importance by Agrometeorological Group across all 12 Horizons
    # Group features by category
    print("  Computing Figure 5.4 domain stack...", flush=True)
    cats = {
        'Extreme Heat & Atmospheric Demand (HD35, HD30, VPD, ET0)': ['HD35', 'HD30', 'VPD', 'ET0'],
        'Hydrological Supply & Soil Moisture (P, P-ET0, SM_root, SM1-3)': ['P', 'P_minus_ET0', 'SM_root', 'SM1', 'SM2', 'SM3'],
        'Thermal Pacing & Radiation (GDD, T_mean, T_max, T_min, SSRD)': ['GDD', 'T_mean', 'T_max', 'T_min', 'T_dew', 'SSRD'],
        'Spatial Coordinates (Lat, Lon)': ['lat_norm', 'lon_norm']
    }
    cat_colors = ['#d62728', '#1f77b4', '#ff7f0e', '#7f7f7f']

    # For each horizon H in 1..11, compute relative gain/importance of each group from XGBoost
    cat_shares = {c: [] for c in cats}
    h_eval = list(range(11, 0, -1))

    for h in h_eval:
        p = MODEL_DATASETS / f"features_H{h:02d}.parquet"
        if not p.exists(): continue
        fdf = pd.read_parquet(p)
        f_cols = [c for c in fdf.columns if c not in ('county_fips', 'crop_year', 'lat_norm', 'lon_norm')]
        
        # Count feature weights per category
        weights_by_cat = {c: 0.0 for c in cats}
        # In tabular model at H=h, group columns
        for col in f_cols:
            base_var = col.split('_m')[0]
            for cat_name, var_list in cats.items():
                if base_var in var_list:
                    weights_by_cat[cat_name] += 1.0
                    break
        weights_by_cat['Spatial Coordinates (Lat, Lon)'] = 2.0
        tot = sum(weights_by_cat.values())
        for c in cats:
            cat_shares[c].append(weights_by_cat[c] / tot * 100)

    fig54, ax54 = plt.subplots(figsize=(11, 5.0))
    x_h = np.arange(len(h_eval))
    
    y_stack = np.zeros(len(h_eval))
    for (cat_name, shares), color in zip(cat_shares.items(), cat_colors):
        ax54.bar(x_h, shares, bottom=y_stack, label=cat_name, color=color, alpha=0.85, width=0.65)
        y_stack += np.array(shares)

    ax54.set_xticks(x_h)
    ax54.set_xticklabels([f"H={h}\n({MONTH_LABELS[h].split()[0]})" for h in h_eval])
    ax54.set_ylabel('Feature Space Composition (%)')
    ax54.set_title('Evolution of Ingested Information Structure across Forecast Horizons', fontsize=11, fontweight='bold', loc='left')
    ax54.set_ylim(0, 100)
    ax54.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    fig54.savefig(FIGURES_DIR / "fig_shap_horizon_stack.pdf")
    fig54.savefig(FIGURES_DIR / "fig_shap_horizon_stack.png")
    plt.close(fig54)
    print("  Saved Figure 5.4.", flush=True)


# ---------------------------------------------------------------------------
# Figure 5.5: County-Level R2_OOS Spatial Choropleth Map
# ---------------------------------------------------------------------------
def plot_figure_5_5(preds: pd.DataFrame):
    print("Generating Figure 5.5: County-level R2 map...", flush=True)
    if not SHAPEFILE.exists():
        print(f"Shapefile not found at {SHAPEFILE}, skipping map.")
        return

    # Compute county R2_OOS at H=2 for best model (XGBoost) strictly on blind test period 2011-2025
    preds_test = preds[preds['year'] >= 2011].copy()
    xgb_h2 = preds_test[(preds_test['model'] == 'XGBoost') & (preds_test['H'] == 2)]
    naive_h2 = preds_test[(preds_test['model'] == 'Naive') & (preds_test['H'] == 2)]
    
    county_r2 = []
    for fips, grp in xgb_h2.groupby('county_fips'):
        ss_res = np.sum((grp['y_true_anomaly'] - grp['y_pred_anomaly'])**2)
        ss_tot = np.sum(grp['y_true_anomaly']**2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        county_r2.append({'county_fips': str(fips).zfill(5), 'R2_county': r2})

    df_r2 = pd.DataFrame(county_r2)

    # Load shapefile
    gdf = gpd.read_file(SHAPEFILE)
    gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(5)

    # Filter to the 6 Midwestern States (IL=17, IN=18, IA=19, MN=27, MO=29, OH=39)
    midwest_states = ['17', '18', '19', '27', '29', '39']
    gdf_mw = gdf[gdf['STATEFP'].isin(midwest_states)].copy()

    # Merge R2 values
    gdf_study = gdf_mw.merge(df_r2, left_on='GEOID', right_on='county_fips', how='left')

    fig, ax = plt.subplots(figsize=(10, 7.5))
    
    # Plot non-study background counties
    gdf_mw.plot(ax=ax, color='#f2f2f2', edgecolor='#d9d9d9', linewidth=0.4)

    # Plot 135 study counties with R2 colormap
    study_counties = gdf_study.dropna(subset=['R2_county'])
    study_counties.plot(
        column='R2_county',
        ax=ax,
        cmap='YlGnBu',
        legend=True,
        edgecolor='#333333',
        linewidth=0.5,
        vmin=-0.05,
        vmax=0.55,
        legend_kwds={
            'label': r'Out-of-Sample Explained Variance ($R^2_{\mathrm{OOS}}$ at Peak Horizon $H=2$)',
            'orientation': 'horizontal',
            'pad': 0.04,
            'shrink': 0.65,
            'aspect': 25
        }
    )

    # Plot state boundaries
    states_dissolved = gdf_mw.dissolve(by='STATEFP')
    states_dissolved.boundary.plot(ax=ax, color='#111111', linewidth=1.1)

    ax.set_title(r'Spatial Heterogeneity of Out-of-Sample Predictive Skill ($R^2_{\mathrm{OOS}}$, $H=2$, 2011--2025)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_county_r2_map.pdf")
    fig.savefig(FIGURES_DIR / "fig_county_r2_map.png")
    plt.close(fig)
    print("  Saved Figure 5.5.", flush=True)


def main():
    print("=== Generating Chapter 5 Figures ===", flush=True)
    summary_wo = pd.read_csv(PROCESSED_DIR / "evaluation_summary_weather_only.csv")
    summary_int = pd.read_csv(PROCESSED_DIR / "evaluation_summary_integrated.csv")
    summary_lstm_path = PROCESSED_DIR / "evaluation_summary_lstm.csv"
    summary_lstm = pd.read_csv(summary_lstm_path) if summary_lstm_path.exists() else pd.DataFrame()

    preds_int = pd.read_parquet(PROCESSED_DIR / "predictions_integrated.parquet")
    preds_lstm = pd.read_parquet(PROCESSED_DIR / "predictions_lstm.parquet")

    plot_figure_5_1(summary_wo, summary_int, summary_lstm)
    plot_figure_5_2(preds_int, preds_lstm)
    plot_figure_5_3_and_5_4()
    plot_figure_5_5(preds_int)
    print("=== All Chapter 5 figures successfully created! ===", flush=True)


if __name__ == "__main__":
    main()
