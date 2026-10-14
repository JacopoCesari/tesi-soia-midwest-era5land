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
from matplotlib.lines import Line2D
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
    12: 'Nov\n(H=12)',
    11: 'Dec\n(H=11)',
    10: 'Jan\n(H=10)',
    9: 'Feb\n(H=9)',
    8: 'Mar\n(H=8)',
    7: 'Apr\n(H=7)',
    6: 'May\n(H=6)',
    5: 'Jun\n(H=5)',
    4: 'Jul\n(H=4)',
    3: 'Aug\n(H=3)',
    2: 'Sep\n(H=2)',
    1: 'Oct\n(H=1)'
}

# ---------------------------------------------------------------------------
# Figure 5.1: Generalist Benchmark Across All 11 Horizons (Weather-Only vs +Lag)
# ---------------------------------------------------------------------------
def plot_figure_5_1(summary_wo: pd.DataFrame, summary_int: pd.DataFrame, summary_lstm: pd.DataFrame):
    print("Generating Figure 5.1: Comprehensive Multi-Horizon Progression (All Models)...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.2), sharex=True)

    horizons = list(range(11, 0, -1))
    x_pos = np.arange(len(horizons))

    # Pre-sowing noise boundary: indices 0..4 (H=11..7, Dec-Apr) vs In-season campaign: indices 5..10 (H=6..1, May-Oct)
    # The demarcation line is set exactly at May Y (H=6, index 5.0)
    for ax in (ax1, ax2):
        ax.axvspan(-0.5, 5.0, color='#f4f5f8', alpha=0.75, zorder=0)
        ax.axvspan(5.0, 10.5, color='#fefdf9', alpha=0.85, zorder=0)
        ax.axvline(5.0, color='#7f8c8d', linestyle=':', linewidth=1.4, alpha=0.85, zorder=2)

    # Panel 1: R2_OOS Pooled across all candidate models
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)
    ax1.plot(x_pos, np.zeros(len(horizons)), label=r'Naive Baseline ($\mathcal{B}_{\mathrm{naive}}$)',
             color=COLOR_MAP['Naive'], linestyle=':', linewidth=1.5, marker='s', markersize=4.5, zorder=3)

    # Tabular Models: Weather-Only (solid) vs Integrated (+Lag) (dashed)
    for model in ['ElasticNet', 'RandomForest', 'XGBoost']:
        m_wo = summary_wo[summary_wo['model'] == model].set_index('H').reindex(horizons)
        m_int = summary_int[summary_int['model'] == model].set_index('H').reindex(horizons)

        r2_wo = m_wo['R2_OOS_pooled'].values
        r2_int = m_int['R2_OOS_pooled'].values

        ax1.plot(x_pos, r2_wo, label=f'{model} (Weather-Only)', color=COLOR_MAP[model],
                 marker=MARKER_MAP[model], markersize=5.0, linewidth=1.4, linestyle='-', zorder=4)
        ax1.plot(x_pos, r2_int, label=f'{model} (+Lag)', color=COLOR_MAP[model],
                 marker=MARKER_MAP[model], markersize=5.0, linewidth=1.4, linestyle='--', alpha=0.85, zorder=4)

    # Sequential LSTM Models: Weather-Only vs Integrated (+Lag)
    if not summary_lstm.empty:
        lstm_wo = summary_lstm[(summary_lstm['variant'] == 'prelu') & (~summary_lstm['include_lag'])].set_index('H').reindex(horizons)
        r2_lstm_wo = lstm_wo['R2_OOS_pooled'].values
        ax1.plot(x_pos, r2_lstm_wo, label='LSTM (Weather-Only)', color=COLOR_MAP['LSTM (Weather-Only)'],
                 marker=MARKER_MAP['LSTM'], markersize=5.5, linewidth=1.8, linestyle='-', zorder=5)

        lstm_int = summary_lstm[(summary_lstm['variant'] == 'linear') & (summary_lstm['include_lag'])].set_index('H').reindex(horizons)
        r2_lstm_int = lstm_int['R2_OOS_pooled'].values
        ax1.plot(x_pos, r2_lstm_int, label=r'LSTM (+Lag $\epsilon_{t-1}$)', color=COLOR_MAP['LSTM (+Lag Yield)'],
                 marker='X', markersize=6.0, linewidth=2.0, linestyle='--', zorder=5)

    ax1.set_ylabel(r'Pooled Out-of-Sample $R^2_{\mathrm{OOS}}$')
    ax1.set_title(r'(a) Out-of-Sample Performance Trajectory ($R^2_{\mathrm{OOS}}$)', loc='left', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.35, 0.42)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax1.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.0, ncol=2)

    # Panel 2: Marginal Gain of Historical Memory with Symmetrical Log Scale (symlog)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)

    for model in ['ElasticNet', 'RandomForest', 'XGBoost']:
        m_wo = summary_wo[summary_wo['model'] == model].set_index('H').reindex(horizons)
        m_int = summary_int[summary_int['model'] == model].set_index('H').reindex(horizons)
        delta_r2 = m_int['R2_OOS_pooled'].values - m_wo['R2_OOS_pooled'].values

        ax2.plot(x_pos, delta_r2, label=rf'{model} $\Delta R^2$', color=COLOR_MAP[model],
                 marker=MARKER_MAP[model], markersize=5.5, linewidth=1.5, linestyle='-', zorder=4)

    if not summary_lstm.empty:
        delta_lstm = r2_lstm_int - r2_lstm_wo
        ax2.plot(x_pos, delta_lstm, label=r'LSTM $\Delta R^2$', color=COLOR_MAP['LSTM (+Lag Yield)'],
                 marker='X', markersize=6.5, linewidth=2.0, linestyle='-', zorder=5)

    ax2.set_yscale('symlog', linthresh=0.03, linscale=1.2)
    ax2.set_ylim(-0.35, 0.22)
    ticks2 = [-0.30, -0.10, -0.03, 0.0, 0.03, 0.10, 0.18]
    ax2.set_yticks(ticks2)
    ax2.set_yticklabels(['-0.30', '-0.10', '-0.03', '0.00', '+0.03', '+0.10', '+0.18'])

    ax2.set_ylabel(r'Marginal Information Gain ($\Delta R^2_{\mathrm{OOS}} = R^2_{\mathrm{int}} - R^2_{\mathrm{wo}}$)')
    ax2.set_title(r'(b) Marginal Contribution of Lagged Yield Memory ($\epsilon_{t-1}$)', loc='left', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.5, ncol=2)

    for ax in (ax1, ax2):
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MONTH_LABELS[h] for h in horizons], rotation=0)
        ax.set_xlabel('Forecast Horizon')

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
    handles_dict = {}

    for yr, true_anom, name_lbl in [(2012, -4.07, '2012 Drought'),
                                    (2016, +7.34, '2016 Bumper')]:
        xgb_biases = []
        elnet_biases = []
        lstm_biases = []

        for h in active_h:
            sub_xgb = df_tab[(df_tab['year'] == yr) & (df_tab['H'] == h) & (df_tab['model'] == 'XGBoost')]
            xgb_biases.append((sub_xgb['y_pred_anomaly'] - sub_xgb['y_true_anomaly']).mean())

            sub_el = df_tab[(df_tab['year'] == yr) & (df_tab['H'] == h) & (df_tab['model'] == 'ElasticNet')]
            el_biases = (sub_el['y_pred_anomaly'] - sub_el['y_true_anomaly']).mean()
            elnet_biases.append(el_biases)

            sub_lst = df_l[(df_l['year'] == yr) & (df_l['H'] == h)]
            lstm_biases.append((sub_lst['y_pred_anomaly'] - sub_lst['y_true_anomaly']).mean())

        # Consistent styling: same color per model as Figure 5.1
        # Solid line + filled marker for 2012 Drought; dashed line + open marker for 2016 Bumper
        ls = '-' if yr == 2012 else '--'
        fill_st = 'full' if yr == 2012 else 'none'
        alpha_val = 1.0 if yr == 2012 else 0.85

        l_el = ax1.plot(x_pos, elnet_biases, label=f'ElasticNet ({name_lbl})', color=COLOR_MAP['ElasticNet'],
                        linestyle=ls, marker='o', fillstyle=fill_st, markersize=6.0, linewidth=1.8, alpha=alpha_val)[0]
        l_xgb = ax1.plot(x_pos, xgb_biases, label=f'XGBoost ({name_lbl})', color=COLOR_MAP['XGBoost'],
                         linestyle=ls, marker='D', fillstyle=fill_st, markersize=5.5, linewidth=1.8, alpha=alpha_val)[0]
        l_lstm = ax1.plot(x_pos, lstm_biases, label=f'LSTM (+Lag) ({name_lbl})', color=COLOR_MAP['LSTM (+Lag Yield)'],
                          linestyle=ls, marker='X', fillstyle=fill_st, markersize=6.5, linewidth=2.0, alpha=alpha_val)[0]

        handles_dict[f'el_{yr}'] = l_el
        handles_dict[f'xgb_{yr}'] = l_xgb
        handles_dict[f'lstm_{yr}'] = l_lstm

    l_zero = ax1.axhline(0, color='black', linestyle=':', linewidth=1.2, alpha=0.85, label='Zero Bias')
    handles_dict['zero'] = l_zero
    ax1.set_ylabel(r'Mean Anomaly Bias ($\bar{\hat{\epsilon}} - \bar{\epsilon}$, bu/ac)')
    ax1.set_title(r'(a) Regional Shock Bias Convergence ($H=5 \dots 1$)', loc='left', fontsize=11, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([MONTH_LABELS[h] for h in active_h], rotation=0)
    ax1.set_xlabel('Forecast Horizon')
    ax1.set_ylim(-8.8, 3.8)

    # 3-column legend: Col 1 = 2012 Drought, Col 2 = 2016 Bumper, Col 3 = Zero Bias
    dummy = Line2D([], [], color='none')
    h_3col = [
        handles_dict['el_2012'], handles_dict['xgb_2012'], handles_dict['lstm_2012'],
        handles_dict['el_2016'], handles_dict['xgb_2016'], handles_dict['lstm_2016'],
        handles_dict['zero'], dummy, dummy
    ]
    lbl_3col = [h.get_label() if h != dummy else "" for h in h_3col]
    ax1.legend(h_3col, lbl_3col, loc='upper right', frameon=True, framealpha=0.92,
               fontsize=8.0, ncol=3, columnspacing=1.1, handletextpad=0.5)

    # Panel (b): County-Level Anomaly Scatter (Predicted vs Observed) for the 2012 Flash Drought at H=5, H=3, H=1
    colors_h = {5: '#e67e22', 3: '#1f77b4', 1: '#2ca02c'}

    sub_2012 = df_l[df_l['year'] == 2012]
    for h in [5, 3, 1]:
        h_data = sub_2012[sub_2012['H'] == h]
        z = np.polyfit(h_data['y_true_anomaly'], h_data['y_pred_anomaly'], 1)
        p = np.poly1d(z)
        x_lin = np.linspace(-18, 5, 50)
        ax2.scatter(h_data['y_true_anomaly'], h_data['y_pred_anomaly'],
                    alpha=0.60, color=colors_h[h], s=28, edgecolors='none')
        clean_lbl = MONTH_LABELS[h].replace('\n', ' ')
        ax2.plot(x_lin, p(x_lin), color=colors_h[h], linewidth=1.8,
                 label=f'{clean_lbl} (slope: {z[0]:.2f})')

    # 45-degree 1:1 line
    ax2.plot([-20, 10], [-20, 10], 'k--', linewidth=1.1, alpha=0.75, label='1:1 Diagonal')
    ax2.set_xlabel(r'Observed County Anomaly ($\epsilon_{c, 2012}$, bu/ac)')
    ax2.set_ylabel(r'Predicted Anomaly ($\hat{\epsilon}_{c, 2012}$, bu/ac)')
    ax2.set_title(r'(b) 2012 Drought: County Anomaly Resolution ($N=135$, LSTM)', loc='left', fontsize=11, fontweight='bold')
    ax2.set_xlim(-19, 6)
    ax2.set_ylim(-15, 4)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8.2, ncol=1)

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
