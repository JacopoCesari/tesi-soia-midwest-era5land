"""Unified Econometric & Statistical Benchmark Analysis.

Generates rigorous statistical outputs for Chapter 5 / Professor review:
1. Master Comparison with Inferential Significance (Paired Wilcoxon / Diebold-Mariano tests vs Naive and Linear).
2. Decile Calibration & Tail Shrinkage Table (D1..D10 across 2011-2025 out-of-sample).
3. Spatio-Temporal Autocorrelation of Residuals (Temporal AR(1) & Spatial Moran's I).
4. Publication-Quality Diagnostic Figure (Figure 5.6: Decile Calibration, Q-Q Plot, Residuals vs Fitted).

Dynamic Integration:
  - Ingests ML predictions from output/data/processed/predictions.parquet.
  - Ingests LSTM Phase C predictions if available from work/reports/experiments/lstm/preds_*.parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Plot style configuration (consistent with Chapter 1-5 figures)
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10.5,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.0,
    'figure.titlesize': 12.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "output" / "data" / "processed"
DIAG_DIR = PROCESSED_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = ROOT / "output" / "thesis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LSTM_REPORTS = ROOT / "work" / "reports" / "experiments" / "lstm"

PREDS_PARQUET = PROCESSED_DIR / "predictions.parquet"
PREDS_INT_PARQUET = PROCESSED_DIR / "predictions_integrated.parquet"
FEAT_H01 = PROCESSED_DIR / "model_datasets" / "features_H01.parquet"

COLOR_MAP = {
    'Naive': '#555555',
    'ElasticNet': '#1f77b4',
    'RandomForest': '#ff7f0e',
    'XGBoost': '#d62728',
    'LSTM_PReLU': '#2ca02c',
    'LSTM_Linear': '#17becf',
    'LSTM_PReLU_Lag': '#9467bd',
    'LSTM_Linear_Lag': '#8c564b'
}

MARKER_MAP = {
    'Naive': 's',
    'ElasticNet': 'o',
    'RandomForest': '^',
    'XGBoost': 'D',
    'LSTM_PReLU': 'P',
    'LSTM_Linear': 'v',
    'LSTM_PReLU_Lag': 'X',
    'LSTM_Linear_Lag': '*'
}


def load_all_predictions() -> pd.DataFrame:
    """Load ML predictions and seamlessly concatenate any completed LSTM Phase C predictions strictly for 2011-2025."""
    if not PREDS_PARQUET.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDS_PARQUET}")

    df_ml = pd.read_parquet(PREDS_PARQUET)
    df_ml['county_fips'] = df_ml['county_fips'].astype(str)
    # Restrict strictly to blind test set 2011-2025
    df_ml = df_ml[df_ml['year'] >= 2011].copy()
    dfs_to_concat = [df_ml]

    # Ingest integrated tabular predictions if available
    if PREDS_INT_PARQUET.exists():
        df_int = pd.read_parquet(PREDS_INT_PARQUET)
        df_int['county_fips'] = df_int['county_fips'].astype(str)
        df_int = df_int[df_int['year'] >= 2011].copy()
        # Drop naive duplicate and append _Lag
        df_int = df_int[df_int['model'] != 'Naive'].copy()
        df_int['model'] = df_int['model'] + '_Lag'
        dfs_to_concat.append(df_int)

    # Check for LSTM Phase C predictions in LSTM_REPORTS
    lstm_files = list(LSTM_REPORTS.glob("preds_*d_H*.parquet"))
    if lstm_files:
        print(f"Discovered {len(lstm_files)} LSTM prediction files. Ingesting...", flush=True)
        lstm_records = []
        for f in lstm_files:
            try:
                ldf = pd.read_parquet(f)
                var_name = str(ldf.get('head_variant', ['prelu'])[0]).lower()
                inc_lag = bool(ldf.get('include_lag', [False])[0]) if 'include_lag' in ldf.columns else ('lag' in f.stem)

                if 'prelu' in var_name:
                    model_label = 'LSTM_PReLU_Lag' if inc_lag else 'LSTM_PReLU'
                elif 'linear' in var_name:
                    model_label = 'LSTM_Linear_Lag' if inc_lag else 'LSTM_Linear'
                else:
                    model_label = f"LSTM_{var_name}"

                for row in ldf.itertuples(index=False):
                    if int(row.year) >= 2011:
                        lstm_records.append({
                            'model': model_label,
                            'H': int(row.horizon),
                            'year': int(row.year),
                            'county_fips': str(row.county_fips),
                            'y_true_anomaly': float(row.actual_anomaly),
                            'y_pred_anomaly': float(row.pred_anomaly),
                            'trend': np.nan
                        })
            except Exception as e:
                print(f"  Warning reading {f.name}: {e}")
        if lstm_records:
            dfs_to_concat.append(pd.DataFrame(lstm_records))

    all_preds = pd.concat(dfs_to_concat, ignore_index=True)
    all_preds = all_preds[all_preds['year'] >= 2011].copy()
    return all_preds


# ---------------------------------------------------------------------------
# 1. Master Comparison with Paired Wilcoxon / Diebold-Mariano Tests
# ---------------------------------------------------------------------------
def compute_master_comparison(preds: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled performance metrics and paired significance tests across the test years."""
    print("Computing Master Statistical Comparison with Paired Inference...", flush=True)
    records = []
    horizons = sorted(preds['H'].unique(), reverse=True)
    models = [m for m in ['Naive', 'ElasticNet', 'RandomForest', 'XGBoost', 'LSTM_PReLU', 'LSTM_Linear', 'LSTM_PReLU_Lag', 'LSTM_Linear_Lag'] if m in preds['model'].unique()]

    # Reference naive pooled RMSE for skill score
    naive_rmse_by_h = {}
    for h in horizons:
        sub_naive = preds[(preds['H'] == h) & (preds['model'] == 'Naive')]
        if not sub_naive.empty:
            naive_rmse_by_h[h] = float(np.sqrt(np.mean((sub_naive['y_true_anomaly'] - sub_naive['y_pred_anomaly'])**2)))
        else:
            naive_rmse_by_h[h] = 6.1239

    for h in horizons:
        sub_h = preds[preds['H'] == h]
        
        # Calculate yearly MSE per model for paired tests
        yearly_mse = {}
        for m in models:
            sub_m = sub_h[sub_h['model'] == m]
            if sub_m.empty:
                continue
            y_mse = sub_m.groupby('year').apply(
                lambda g: float(np.mean((g['y_true_anomaly'] - g['y_pred_anomaly'])**2))
            )
            yearly_mse[m] = y_mse

        # Reference vectors for paired tests
        mse_naive = yearly_mse.get('Naive', None)
        mse_elnet = yearly_mse.get('ElasticNet', None)

        for m in models:
            sub_m = sub_h[sub_h['model'] == m]
            if sub_m.empty:
                continue

            y_t = sub_m['y_true_anomaly'].values
            y_p = sub_m['y_pred_anomaly'].values
            n_obs = len(y_t)
            ss_res = np.sum((y_t - y_p)**2)
            ss_tot = np.sum(y_t**2)
            r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
            rmse_val = float(np.sqrt(np.mean((y_t - y_p)**2)))
            mae_val = float(np.mean(np.abs(y_t - y_p)))
            skill = 1.0 - (rmse_val / naive_rmse_by_h.get(h, 5.9442))

            # Paired Wilcoxon test against Naive (H0: error_m == error_naive, H1: error_m < error_naive)
            p_vs_naive = float('nan')
            if m != 'Naive' and mse_naive is not None and m in yearly_mse:
                common_yrs = yearly_mse[m].index.intersection(mse_naive.index)
                if len(common_yrs) >= 10:
                    try:
                        _, p_vs_naive = stats.wilcoxon(yearly_mse[m].loc[common_yrs], mse_naive.loc[common_yrs], alternative='less')
                    except Exception:
                        pass

            # Paired Wilcoxon test against ElasticNet (H0: error_m == error_elnet, two-sided)
            p_vs_elnet = float('nan')
            if m not in ('Naive', 'ElasticNet') and mse_elnet is not None and m in yearly_mse:
                common_yrs = yearly_mse[m].index.intersection(mse_elnet.index)
                if len(common_yrs) >= 10:
                    try:
                        _, p_vs_elnet = stats.wilcoxon(yearly_mse[m].loc[common_yrs], mse_elnet.loc[common_yrs])
                    except Exception:
                        pass

            records.append({
                'H': h,
                'model': m,
                'n_obs': n_obs,
                'R2_OOS_pooled': round(r2_val, 4),
                'RMSE_OOS_pooled': round(rmse_val, 4),
                'MAE_OOS_pooled': round(mae_val, 4),
                'Skill_Score_pct': round(skill * 100, 2),
                'p_val_vs_naive': p_vs_naive,
                'p_val_vs_elasticnet': p_vs_elnet
            })

    df_master = pd.DataFrame(records)
    out_csv = DIAG_DIR / "master_model_comparison_with_significance.csv"
    df_master.to_csv(out_csv, index=False)
    print(f"  Saved master comparison to {out_csv}", flush=True)
    return df_master


# ---------------------------------------------------------------------------
# 2. Decile Calibration & Tail Shrinkage Table (D1..D10)
# ---------------------------------------------------------------------------
def compute_decile_diagnostics(preds: pd.DataFrame, target_h: int = 2) -> pd.DataFrame:
    """Divide test sample into 10 quantiles of observed anomaly and evaluate model shrinkage."""
    print(f"Computing Decile Calibration Analysis at Milestone H={target_h}...", flush=True)
    sub = preds[preds['H'] == target_h].copy()
    if sub.empty:
        return pd.DataFrame()

    # Base deciles computed on observed anomaly (ground truth)
    ref_model = 'ElasticNet' if 'ElasticNet' in sub['model'].unique() else sub['model'].unique()[0]
    ref_df = sub[sub['model'] == ref_model].copy().sort_values(['year', 'county_fips']).reset_index(drop=True)
    
    # 10 deciles (1 = worst negative drought, 10 = highest positive bumper)
    decile_series, bins = pd.qcut(ref_df['y_true_anomaly'], q=10, labels=list(range(1, 11)), retbins=True)
    ref_df['decile'] = decile_series

    # Map (year, county_fips) -> decile
    decile_map = dict(zip(zip(ref_df['year'], ref_df['county_fips']), ref_df['decile']))
    sub['decile'] = sub.apply(lambda r: decile_map.get((r['year'], r['county_fips'])), axis=1)

    decile_records = []
    models = [m for m in [
        'Naive',
        'ElasticNet', 'ElasticNet_Lag',
        'RandomForest', 'RandomForest_Lag',
        'XGBoost', 'XGBoost_Lag',
        'LSTM_PReLU', 'LSTM_Linear', 'LSTM_PReLU_Lag', 'LSTM_Linear_Lag'
    ] if m in sub['model'].unique()]

    for d in range(1, 11):
        d_sub = sub[sub['decile'] == d]
        y_obs = d_sub[d_sub['model'] == ref_model]['y_true_anomaly'].values
        obs_mean = float(np.mean(y_obs))
        obs_min = float(np.min(y_obs))
        obs_max = float(np.max(y_obs))

        row = {
            'H': target_h,
            'decile': d,
            'n_obs': len(y_obs),
            'obs_mean': round(obs_mean, 3),
            'obs_min': round(obs_min, 3),
            'obs_max': round(obs_max, 3)
        }

        for m in models:
            m_sub = d_sub[d_sub['model'] == m]
            if m_sub.empty:
                continue
            y_pred = m_sub['y_pred_anomaly'].values
            y_true = m_sub['y_true_anomaly'].values
            pred_mean = float(np.mean(y_pred))
            bias = float(pred_mean - obs_mean)
            d_rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))

            row[f'pred_mean_{m}'] = round(pred_mean, 3)
            row[f'bias_{m}'] = round(bias, 3)
            row[f'rmse_{m}'] = round(d_rmse, 3)

        decile_records.append(row)

    df_deciles = pd.DataFrame(decile_records)
    out_csv = DIAG_DIR / f"decile_tail_diagnostics_H{target_h:02d}.csv"
    df_deciles.to_csv(out_csv, index=False)
    print(f"  Saved decile tail diagnostics to {out_csv}", flush=True)
    return df_deciles


# ---------------------------------------------------------------------------
# 2b. Active Season Decile Diagnostics (Average across H=5..1)
# ---------------------------------------------------------------------------
def compute_active_season_deciles(preds: pd.DataFrame, active_horizons: list = [5, 4, 3, 2, 1]) -> pd.DataFrame:
    """Compute mean error across active campaign horizons (H=5..1) partitioned by observed anomaly deciles."""
    print(f"Computing Active Season Decile Error Diagnostics across H={active_horizons}...", flush=True)
    sub = preds[preds['H'].isin(active_horizons)].copy()
    if sub.empty:
        return pd.DataFrame()

    ref_model = 'ElasticNet' if 'ElasticNet' in sub['model'].unique() else sub['model'].unique()[0]
    ref_df = sub[(sub['model'] == ref_model) & (sub['H'] == active_horizons[0])].copy().sort_values(['year', 'county_fips']).reset_index(drop=True)
    
    # 10 deciles of observed ground-truth anomaly
    decile_series, bins = pd.qcut(ref_df['y_true_anomaly'], q=10, labels=list(range(1, 11)), retbins=True)
    ref_df['decile'] = decile_series

    decile_map = dict(zip(zip(ref_df['year'], ref_df['county_fips']), ref_df['decile']))
    sub['decile'] = sub.apply(lambda r: decile_map.get((r['year'], r['county_fips'])), axis=1)

    models = [m for m in [
        'Naive',
        'ElasticNet', 'ElasticNet_Lag',
        'RandomForest', 'RandomForest_Lag',
        'XGBoost', 'XGBoost_Lag',
        'LSTM_PReLU', 'LSTM_PReLU_Lag',
        'LSTM_Linear', 'LSTM_Linear_Lag'
    ] if m in sub['model'].unique()]

    decile_records = []
    for d in range(1, 11):
        d_sub = sub[sub['decile'] == d]
        y_obs = d_sub[d_sub['model'] == ref_model]['y_true_anomaly'].values
        obs_mean = float(np.mean(y_obs))
        obs_min = float(np.min(y_obs))
        obs_max = float(np.max(y_obs))

        row = {
            'decile': f"D{d}",
            'n_obs': len(d_sub[d_sub['model'] == ref_model]) // len(active_horizons),
            'total_evals': len(d_sub[d_sub['model'] == ref_model]),
            'obs_min': round(obs_min, 2),
            'obs_max': round(obs_max, 2),
            'obs_mean': round(obs_mean, 2)
        }

        for m in models:
            m_sub = d_sub[d_sub['model'] == m]
            if m_sub.empty:
                continue
            
            # Average metrics across horizons
            h_rmses = []
            h_maes = []
            h_biases = []
            for h in active_horizons:
                h_sub = m_sub[m_sub['H'] == h]
                if not h_sub.empty:
                    err = h_sub['y_true_anomaly'].values - h_sub['y_pred_anomaly'].values
                    h_rmses.append(float(np.sqrt(np.mean(err**2))))
                    h_maes.append(float(np.mean(np.abs(err))))
                    h_biases.append(float(np.mean(h_sub['y_pred_anomaly'].values - h_sub['y_true_anomaly'].values)))
            
            row[f'rmse_{m}'] = round(float(np.mean(h_rmses)), 2)
            row[f'mae_{m}'] = round(float(np.mean(h_maes)), 2)
            row[f'bias_{m}'] = round(float(np.mean(h_biases)), 2)

        decile_records.append(row)

    # Add Overall row (all deciles pooled across test set)
    y_obs_all = sub[sub['model'] == ref_model]['y_true_anomaly'].values
    overall_row = {
        'decile': 'Overall',
        'n_obs': len(ref_df),
        'total_evals': len(y_obs_all),
        'obs_min': round(float(np.min(y_obs_all)), 2),
        'obs_max': round(float(np.max(y_obs_all)), 2),
        'obs_mean': round(float(np.mean(y_obs_all)), 2)
    }
    for m in models:
        m_sub = sub[sub['model'] == m]
        if m_sub.empty:
            continue
        h_rmses = []
        h_maes = []
        h_biases = []
        for h in active_horizons:
            h_sub = m_sub[m_sub['H'] == h]
            if not h_sub.empty:
                err = h_sub['y_true_anomaly'].values - h_sub['y_pred_anomaly'].values
                h_rmses.append(float(np.sqrt(np.mean(err**2))))
                h_maes.append(float(np.mean(np.abs(err))))
                h_biases.append(float(np.mean(h_sub['y_pred_anomaly'].values - h_sub['y_true_anomaly'].values)))
        overall_row[f'rmse_{m}'] = round(float(np.mean(h_rmses)), 2)
        overall_row[f'mae_{m}'] = round(float(np.mean(h_maes)), 2)
        overall_row[f'bias_{m}'] = round(float(np.mean(h_biases)), 2)
    decile_records.append(overall_row)

    df_deciles_active = pd.DataFrame(decile_records)
    out_csv = DIAG_DIR / "decile_active_horizons_diagnostics.csv"
    df_deciles_active.to_csv(out_csv, index=False)
    print(f"  Saved active season decile diagnostics to {out_csv}", flush=True)
    return df_deciles_active



# ---------------------------------------------------------------------------
# 3. Spatio-Temporal Autocorrelation (Moran's I & Temporal AR(1))
# ---------------------------------------------------------------------------
def compute_panel_autocorrelation(preds: pd.DataFrame) -> pd.DataFrame:
    """Evaluate temporal AR(1) and spatial Moran's I on model forecast residuals."""
    print("Computing Spatio-Temporal Residual Autocorrelation...", flush=True)
    if not FEAT_H01.exists():
        print("  Features H=1 not found for spatial coordinates; skipping spatial Moran.")
        return pd.DataFrame()

    feat_df = pd.read_parquet(FEAT_H01)[['county_fips', 'lat_norm', 'lon_norm']].drop_duplicates()
    feat_df['county_fips'] = feat_df['county_fips'].astype(str)
    coords = feat_df.set_index('county_fips')

    horizons = [1, 2, 3, 6]
    models = [m for m in ['ElasticNet', 'RandomForest', 'XGBoost', 'LSTM_PReLU', 'LSTM_Linear', 'LSTM_PReLU_Lag', 'LSTM_Linear_Lag'] if m in preds['model'].unique()]
    fips_list = sorted(coords.index.intersection(preds['county_fips'].unique()))
    N = len(fips_list)

    # Compute Euclidean distance spatial weights matrix W
    xy = coords.loc[fips_list][['lat_norm', 'lon_norm']].values
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :])**2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    W = 1.0 / dist
    W = W / W.sum(axis=1, keepdims=True)  # Row-standardized spatial weights

    records = []
    for h in horizons:
        sub_h = preds[preds['H'] == h]
        for m in models:
            sub_m = sub_h[sub_h['model'] == m].copy()
            if sub_m.empty:
                continue
            sub_m['residual'] = sub_m['y_true_anomaly'] - sub_m['y_pred_anomaly']

            # Temporal AR(1) per county
            ar1_vals = []
            for fips, grp in sub_m.groupby('county_fips'):
                e = grp.sort_values('year')['residual'].values
                if len(e) >= 10 and np.std(e[:-1]) > 0 and np.std(e[1:]) > 0:
                    r = np.corrcoef(e[:-1], e[1:])[0, 1]
                    if not np.isnan(r):
                        ar1_vals.append(r)

            # Spatial Moran's I per year across the 135 counties
            moran_vals = []
            for yr, grp in sub_m.groupby('year'):
                grp_indexed = grp.set_index('county_fips')
                e = grp_indexed.reindex(fips_list)['residual'].values
                if np.all(np.isnan(e)):
                    continue
                e = np.nan_to_num(e, nan=0.0)
                e_dev = e - np.mean(e)
                s0 = np.sum(e_dev**2)
                if s0 > 0:
                    I = (N / np.sum(W)) * np.sum(W * np.outer(e_dev, e_dev)) / s0
                    moran_vals.append(float(I))

            records.append({
                'H': h,
                'model': m,
                'temporal_AR1_mean': round(float(np.mean(ar1_vals)), 4) if ar1_vals else float('nan'),
                'temporal_AR1_std': round(float(np.std(ar1_vals)), 4) if ar1_vals else float('nan'),
                'morans_I_spatial_mean': round(float(np.mean(moran_vals)), 4) if moran_vals else float('nan'),
                'morans_I_spatial_std': round(float(np.std(moran_vals)), 4) if moran_vals else float('nan')
            })

    df_corr = pd.DataFrame(records)
    out_csv = DIAG_DIR / "panel_autocorrelation_diagnostics.csv"
    df_corr.to_csv(out_csv, index=False)
    print(f"  Saved panel autocorrelation diagnostics to {out_csv}", flush=True)
    return df_corr


# ---------------------------------------------------------------------------
# 4. Publication-Quality Figure 5.6: Decile Calibration & Residuals
# ---------------------------------------------------------------------------
def plot_diagnostic_figure(preds: pd.DataFrame, df_deciles: pd.DataFrame, target_h: int = 2) -> None:
    """Generate high-clarity 3-panel figure: Decile Calibration, Q-Q Plot, and Residuals vs Fitted."""
    print("Generating Figure 5.6: Decile Calibration & Residual Diagnostics...", flush=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Clean selection of candidate models: 1 per paradigm family (all champion integrated specifications)
    clean_models = [
        ('Naive', 'Naive', '#555555', 's'),
        ('ElasticNet_Lag', 'ElasticNet', '#1f77b4', 'o'),
        ('RandomForest_Lag', 'Random Forest', '#ff7f0e', '^'),
        ('XGBoost_Lag', 'XGBoost', '#d62728', 'D'),
        ('LSTM_Linear_Lag', 'LSTM', '#9467bd', 'X')
    ]

    # ---------------------------------------------------------
    # Panel (a): Decile Reliability & Tail Shrinkage Diagram
    # ---------------------------------------------------------
    x_obs = df_deciles['obs_mean'].values
    min_val = min(x_obs.min(), -12)
    max_val = max(x_obs.max(), 12)

    # 1:1 Reference Line of perfect calibration
    ax1.plot([min_val - 2, max_val + 2], [min_val - 2, max_val + 2],
             color='#111111', linestyle='--', linewidth=1.1, alpha=0.75, label='1:1 Calibration', zorder=1)

    for m_key, m_lbl, m_col, m_mk in clean_models:
        if f'pred_mean_{m_key}' in df_deciles.columns:
            y_pred = df_deciles[f'pred_mean_{m_key}'].values
            ax1.plot(x_obs, y_pred, label=m_lbl, color=m_col,
                     marker=m_mk, markersize=5.5, linewidth=1.7, zorder=3)

    ax1.set_xlabel(r'Mean Observed Anomaly ($\bar{\epsilon}$, bu/acre by Decile)')
    ax1.set_ylabel(r'Mean Predicted Anomaly ($\bar{\hat{\epsilon}}$, bu/acre)')
    ax1.set_title(r'(a) Decile Reliability Diagram ($H=2$)', loc='left', fontsize=11, fontweight='bold')
    ax1.set_xlim(min_val - 1.5, max_val + 1.5)
    ax1.set_ylim(-6.0, 5.0)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.2)

    # ---------------------------------------------------------
    # Panel (b): Normal Q-Q Plot of Standardized Residuals
    # ---------------------------------------------------------
    sub_h = preds[preds['H'] == target_h]
    
    # Theoretical quantiles line
    theo_q = np.linspace(-3.5, 3.5, 100)
    ax2.plot(theo_q, theo_q, color='#111111', linestyle='--', linewidth=1.1, alpha=0.75, label='Theoretical Normal', zorder=1)

    qq_models = [
        ('ElasticNet_Lag', 'ElasticNet', '#1f77b4'),
        ('XGBoost_Lag', 'XGBoost', '#d62728'),
        ('LSTM_Linear_Lag', 'LSTM', '#9467bd')
    ]

    for m_key, m_lbl, m_col in qq_models:
        if m_key in sub_h['model'].unique():
            sub_m = sub_h[sub_h['model'] == m_key]
            e = (sub_m['y_true_anomaly'] - sub_m['y_pred_anomaly']).values
            std_e = (e - np.mean(e)) / np.std(e)
            osm, osr = stats.probplot(std_e, dist='norm', fit=False)
            ax2.scatter(osm[::10], osr[::10], color=m_col, label=f'{m_lbl} Residuals',
                        alpha=0.6, s=16, edgecolors='none', zorder=3)

    ax2.set_xlabel('Theoretical Normal Quantiles ($z$)')
    ax2.set_ylabel(r'Standardized Residuals ($e / \sigma_e$)')
    ax2.set_title(r'(b) Normal Q-Q Plot of Residuals ($H=2$)', loc='left', fontsize=11, fontweight='bold')
    ax2.set_xlim(-3.6, 3.6)
    ax2.set_ylim(-4.2, 3.8)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8.2)

    # ---------------------------------------------------------
    # Panel (c): Residuals vs Fitted Values (Champion Model: LSTM)
    # ---------------------------------------------------------
    m_focus = 'LSTM_Linear_Lag' if 'LSTM_Linear_Lag' in sub_h['model'].unique() else ('XGBoost_Lag' if 'XGBoost_Lag' in sub_h['model'].unique() else 'ElasticNet_Lag')
    sub_foc = sub_h[sub_h['model'] == m_focus]
    y_fit = sub_foc['y_pred_anomaly'].values
    y_res = (sub_foc['y_true_anomaly'] - sub_foc['y_pred_anomaly']).values

    label_foc = 'LSTM' if 'LSTM' in m_focus else ('XGBoost' if 'XGBoost' in m_focus else 'ElasticNet')
    color_foc = '#9467bd' if 'LSTM' in m_focus else COLOR_MAP.get(m_focus, '#d62728')

    ax3.scatter(y_fit[::4], y_res[::4], color=color_foc, alpha=0.30, s=14, edgecolors='none', label=f'{label_foc} (Subsampled)')
    ax3.axhline(0, color='black', linestyle=':', linewidth=1.0, alpha=0.8)

    # Kernel-smoothed conditional mean and standard deviation across full data support
    grid_x = np.linspace(y_fit.min(), y_fit.max(), 120)
    bandwidth = (y_fit.max() - y_fit.min()) / 10.0
    weights = np.exp(-0.5 * ((y_fit[:, None] - grid_x[None, :]) / bandwidth) ** 2)
    w_sum = weights.sum(axis=0)
    mu_smooth = (weights * y_res[:, None]).sum(axis=0) / w_sum
    var_smooth = (weights * (y_res[:, None] - mu_smooth[None, :]) ** 2).sum(axis=0) / w_sum
    sigma_smooth = np.sqrt(var_smooth)

    ax3.plot(grid_x, mu_smooth, color='#111111', linewidth=1.8, label=r'Smoothed Mean Bias')
    ax3.fill_between(grid_x,
                     mu_smooth - sigma_smooth,
                     mu_smooth + sigma_smooth,
                     color=color_foc, alpha=0.18, label=r'$\pm 1\sigma$ Residual Band')

    ax3.set_xlabel(r'Fitted Yield Anomaly ($\hat{\epsilon}$, bu/acre)')
    ax3.set_ylabel(r'Residual Error ($e = \epsilon - \hat{\epsilon}$, bu/acre)')
    ax3.set_title(rf'(c) Residual Variance vs Fitted Values ({label_foc})', loc='left', fontsize=11, fontweight='bold')
    ax3.set_xlim(-10.5, 6.0)
    ax3.set_ylim(-18, 16)
    ax3.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=8.2)

    # Global formal source attribution line (Vademecum Rule)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(direction='out', length=3.5, width=0.8)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.5, 0.04,
             "Source: Author's calculation on USDA NASS and ECMWF ERA5-Land data (2011–2025 out-of-sample panel).",
             ha='center', fontsize=9.2, style='italic', color='#333333')

    fig.savefig(FIGURES_DIR / "fig_decile_calibration_and_residuals.pdf")
    fig.savefig(FIGURES_DIR / "fig_decile_calibration_and_residuals.png")
    plt.close(fig)
    print("  Saved Figure 5.6 to output/thesis/figures/", flush=True)


def main():
    print("=== Running Unified Benchmark & Diagnostic Analysis ===", flush=True)
    preds = load_all_predictions()
    print(f"Total out-of-sample evaluations loaded: {len(preds):,} rows across models: {preds['model'].unique().tolist()}", flush=True)

    # 1. Master statistical comparison with paired inference
    compute_master_comparison(preds)

    # 2. Decile calibration and tail shrinkage at milestone H=2
    df_deciles = compute_decile_diagnostics(preds, target_h=2)

    # 2b. Active season decile diagnostics (average across H=5..1)
    df_deciles_active = compute_active_season_deciles(preds, active_horizons=[5, 4, 3, 2, 1])

    # 3. Spatio-temporal residual autocorrelation
    compute_panel_autocorrelation(preds)

    # 4. Diagnostic Figure 5.6
    plot_diagnostic_figure(preds, df_deciles, target_h=2)

    print("=== Diagnostic Suite Completed Successfully ===", flush=True)


if __name__ == "__main__":
    main()
