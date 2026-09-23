#!/usr/bin/env python3
"""
analyze_climate_extremes.py

Dual-scale agro-climatic analysis of extreme soybean yield years (+/- 10% threshold)
in the Midwestern panel (1951-2025):
  1. Multi-Year Secular Climatology (1951-2025):
     Contrasts positive bumper harvests (>= +10%) and catastrophic negative shocks (<= -10%)
     against annual heat stress (HD30) and summer evaporative demand (VPD).
     Saves: output/thesis/figures/fig_weather_climatology.pdf / .png
  2. Intra-Campaign Phenological Progression (April 1 to October 31):
     Traces daily progression of extreme years for Tmax/HD30, VPD, cumulative P-ET0,
     and root-zone soil moisture (SM_root), distinguishing each annual trajectory.
     Saves: output/thesis/figures/fig_intra_seasonal_extremes.pdf / .png

Also preserves trial copies in work/reports/exploratory_analysis/candidates/.
"""

import os
import glob
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from datetime import datetime

# Setup publication matplotlib styling
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.0,
    'figure.titlesize': 13,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

def load_panel_yield_extremes(yield_csv, threshold_pct=10.0):
    df_yield = pd.read_csv(yield_csv)
    panel = df_yield.groupby('year')['yield_bu_per_acre'].mean().reset_index()
    
    # Linear trend estimation
    years = panel['year'].values
    y = panel['yield_bu_per_acre'].values
    z = np.polyfit(years, y, 1)
    trend = np.polyval(z, years)
    anom = y - trend
    rel_anom = (anom / trend) * 100.0
    
    panel['trend'] = trend
    panel['anomaly'] = anom
    panel['rel_anomaly'] = rel_anom
    
    # Identify extreme years
    negative_shocks = panel[panel['rel_anomaly'] <= -threshold_pct].sort_values('rel_anomaly')
    positive_bumpers = panel[panel['rel_anomaly'] >= threshold_pct].sort_values('rel_anomaly', ascending=False)
    
    return panel, negative_shocks, positive_bumpers, z

def load_weather_data(weather_dir):
    files = glob.glob(os.path.join(weather_dir, 'county_daily_*.parquet'))
    year_map = {}
    for f in files:
        m = re.search(r'county_daily_(\d+)\.parquet', os.path.basename(f))
        if m:
            year_map[int(m.group(1))] = f
    print(f"Found {len(year_map)} weather parquet files: {min(year_map.keys())} to {max(year_map.keys())}")
    return year_map

def compute_annual_weather_metrics(year_map):
    records = []
    for y, f in sorted(year_map.items()):
        df = pd.read_parquet(f)
        df['date'] = pd.to_datetime(df['date'])
        
        # Growing season (Apr 1 - Oct 31)
        mask_season = (df['date'].dt.month >= 4) & (df['date'].dt.month <= 10)
        df_season = df[mask_season]
        
        # Summer (Jul - Aug)
        mask_summer = (df['date'].dt.month >= 7) & (df['date'].dt.month <= 8)
        df_summer = df[mask_summer]
        
        # Annual / season metrics per county
        county_heat_days = df_season.groupby('county_fips')['heat_day_30'].sum().mean()
        county_heat_days_35 = df_season.groupby('county_fips')['heat_day_35'].sum().mean()
        summer_vpd = df_summer['vapor_pressure_deficit'].mean()
        season_p = df_season.groupby('county_fips')['total_precipitation'].sum().mean()
        season_et0 = df_season.groupby('county_fips')['et0_fao56'].sum().mean()
        season_balance = df_season.groupby('county_fips')['p_minus_et0'].sum().mean()
        
        records.append({
            'year': y,
            'heat_days_30': county_heat_days,
            'heat_days_35': county_heat_days_35,
            'summer_vpd': summer_vpd,
            'season_p': season_p,
            'season_et0': season_et0,
            'season_balance': season_balance
        })
    return pd.DataFrame(records)

def extract_daily_trajectories(year_map, target_years):
    trajectories = {}
    climatology_days = []
    
    for y, f in sorted(year_map.items()):
        df = pd.read_parquet(f)
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'].dt.month >= 4) & (df['date'].dt.month <= 10)
        sub = df[mask].copy()
        sub['doy'] = sub['date'].dt.dayofyear
        
        # Depth-weighted root zone soil moisture (0-100cm)
        sub['sm_root'] = (0.07 * sub['volumetric_soil_water_layer_1'] +
                          0.21 * sub['volumetric_soil_water_layer_2'] +
                          0.72 * sub['volumetric_soil_water_layer_3'])
        
        # Cross-sectional county mean per day
        daily = sub.groupby('doy')[['air_temperature_maximum', 'vapor_pressure_deficit',
                                    'heat_day_30', 'p_minus_et0', 'total_precipitation',
                                    'et0_fao56', 'sm_root']].mean().reset_index()
        daily['year'] = y
        climatology_days.append(daily)
        
        if y in target_years:
            # Compute cumulative metrics
            daily['cum_hd30'] = daily['heat_day_30'].cumsum()
            daily['cum_p'] = daily['total_precipitation'].cumsum()
            daily['cum_et0'] = daily['et0_fao56'].cumsum()
            daily['cum_balance'] = daily['p_minus_et0'].cumsum()
            # 15-day rolling average for smooth presentation
            daily['vpd_roll15'] = daily['vapor_pressure_deficit'].rolling(15, center=True, min_periods=1).mean()
            daily['tmax_roll7'] = daily['air_temperature_maximum'].rolling(7, center=True, min_periods=1).mean()
            trajectories[y] = daily
            
    # Compute full climatology
    df_clim = pd.concat(climatology_days, ignore_index=True)
    clim_summary = df_clim.groupby('doy').agg({
        'air_temperature_maximum': ['median', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)],
        'vapor_pressure_deficit': ['median', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)],
        'heat_day_30': ['median', 'mean'],
        'p_minus_et0': ['median', 'mean'],
        'total_precipitation': ['median', 'mean'],
        'et0_fao56': ['median', 'mean'],
        'sm_root': ['median', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)],
    })
    clim_summary.columns = ['_'.join(c).strip() for c in clim_summary.columns]
    clim_summary['cum_balance_median'] = clim_summary['p_minus_et0_median'].cumsum()
    clim_summary['vpd_roll15_median'] = clim_summary['vapor_pressure_deficit_median'].rolling(15, center=True, min_periods=1).mean()
    
    return trajectories, clim_summary

def plot_multiyear_climatology(panel_yield, weather_annual, negative_shocks, positive_bumpers, out_pdf, out_png):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True, gridspec_kw={'height_ratios': [1.1, 1.0]})
    
    # Panel (a): Detrended Relative Yield Anomalies
    years = panel_yield['year'].values
    rel_anom = panel_yield['rel_anomaly'].values
    
    # Classify colors
    bar_colors = []
    for y_val, r_val in zip(years, rel_anom):
        if r_val <= -10.0:
            bar_colors.append('#a51d24') # Deep crimson shock
        elif r_val >= 10.0:
            bar_colors.append('#1e7e34') # Emerald bumper
        elif r_val < 0:
            bar_colors.append('#7ca8cf') # Light blue
        else:
            bar_colors.append('#4a88c7') # Steel blue
            
    bars = ax1.bar(years, rel_anom, color=bar_colors, width=0.75, edgecolor='#333333', linewidth=0.4, alpha=0.9, zorder=3)
    ax1.axhline(0, color='black', linewidth=0.9, zorder=4)
    
    # Threshold lines (+/- 10%)
    ax1.axhline(10.0, color='#1e7e34', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4, label='Bumper Harvest Threshold (+10%)')
    ax1.axhline(-10.0, color='#a51d24', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4, label='Severe Shortfall Threshold (-10%)')
    
    # Annotate extreme negative shocks
    for _, row in negative_shocks.iterrows():
        yr = int(row['year'])
        val = row['rel_anomaly']
        ax1.annotate(f"{yr}\n({val:.1f}%)", (yr, val), textcoords="offset points", xytext=(0, -22),
                     ha='center', fontsize=7.5, fontweight='bold', color='#721c24')
                     
    # Annotate extreme positive bumpers
    for _, row in positive_bumpers.iterrows():
        yr = int(row['year'])
        val = row['rel_anomaly']
        ax1.annotate(f"{yr}\n(+{val:.1f}%)", (yr, val), textcoords="offset points", xytext=(0, 6),
                     ha='center', fontsize=7.5, fontweight='bold', color='#155724')
                     
    ax1.set_title('(a) Multi-Decadal Relative Yield Anomalies & Extreme Shock Regimes (1951--2025)', fontweight='bold', loc='left')
    ax1.set_ylabel('Detrended Yield Anomaly (%)')
    ax1.set_ylim(-32, 20)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='lower right', fontsize=8.5)
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    # Panel (b): Heat Stress Days and Summer VPD
    w_years = weather_annual['year'].values
    hd30 = weather_annual['heat_days_30'].values
    vpd = weather_annual['summer_vpd'].values
    
    ax2.bar(w_years, hd30, color='#e67e22', width=0.75, alpha=0.75, edgecolor='#b95d0e', linewidth=0.4,
            label=r'Annual Heat Days ($HD_{30}$, $T_{\mathrm{max}} \geq 30^{\circ}\mathrm{C}$/county)', zorder=3)
    ax2.set_ylabel('Heat Days per County ($HD_{30}$, days)', color='#b95d0e')
    ax2.tick_params(axis='y', labelcolor='#b95d0e')
    ax2.set_ylim(0, 65)
    ax2.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    # Secondary axis: summer VPD
    ax2_twin = ax2.twinx()
    ax2_twin.plot(w_years, vpd, color='#6f42c1', linewidth=2.2, linestyle='-', marker='s', markersize=3.5,
                  label='Summer Atmospheric Demand ($VPD$, Jul--Aug)', zorder=5)
    ax2_twin.set_ylabel('July--August Mean $VPD$ (kPa)', color='#5a2c91')
    ax2_twin.tick_params(axis='y', labelcolor='#5a2c91')
    ax2_twin.set_ylim(0.5, 1.55)
    
    # Vertical connecting bands for notable extreme years
    # Red for severe heat/drought shocks
    drought_years = [1988, 1983, 1953]
    for dy in drought_years:
        if dy in years:
            ax1.axvspan(dy-0.45, dy+0.45, color='#ffcccc', alpha=0.35, zorder=2)
            ax2.axvspan(dy-0.45, dy+0.45, color='#ffcccc', alpha=0.35, zorder=2)
            
    # Cyan for flood year (1993)
    if 1993 in years:
        ax1.axvspan(1993-0.45, 1993+0.45, color='#ccf2ff', alpha=0.35, zorder=2)
        ax2.axvspan(1993-0.45, 1993+0.45, color='#ccf2ff', alpha=0.35, zorder=2)
        
    # Purple for temporal planting/frost shock (1974)
    if 1974 in years:
        ax1.axvspan(1974-0.45, 1974+0.45, color='#e6ccff', alpha=0.35, zorder=2)
        ax2.axvspan(1974-0.45, 1974+0.45, color='#e6ccff', alpha=0.35, zorder=2)
        
    # Green for bumper harvests (1994, 1961, 1952)
    bumper_highlight = [1994, 1961, 1952]
    for by in bumper_highlight:
        if by in years:
            ax1.axvspan(by-0.45, by+0.45, color='#ccffcc', alpha=0.35, zorder=2)
            ax2.axvspan(by-0.45, by+0.45, color='#ccffcc', alpha=0.35, zorder=2)
            
    ax2.set_title('(b) Synchronized Atmospheric Heat Stress & Evaporative Demand (1950--Present)', fontweight='bold', loc='left')
    ax2.set_xlabel('Crop Year')
    ax2.set_xlim(1950, 2026)
    
    # Combined legend for panel (b)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right', fontsize=8.5)
    
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Climatology: {out_pdf} and {out_png}")

def plot_intra_seasonal_extremes(trajectories, clim_summary, out_pdf, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True)
    
    # Doy to Calendar Month mapping
    # DOY 91 = Apr 1, DOY 121 = May 1, DOY 152 = Jun 1, DOY 182 = Jul 1, DOY 213 = Aug 1, DOY 244 = Sep 1, DOY 274 = Oct 1, DOY 304 = Oct 31
    month_ticks = [91, 121, 152, 182, 213, 244, 274, 304]
    month_labels = ['Apr 1', 'May 1', 'Jun 1', 'Jul 1', 'Aug 1', 'Sep 1', 'Oct 1', 'Oct 31']
    
    # Define style dictionary for distinct extreme years
    # Palette designed for color clarity and visual distinction
    styles = {
        # Negative Shocks
        1988: {'color': '#d62728', 'linewidth': 2.4, 'linestyle': '-', 'label': '1988: Severe Heat & Drought (-25.6%)'},
        1983: {'color': '#ff7f0e', 'linewidth': 2.0, 'linestyle': '-', 'label': '1983: Pod-Fill Thermal Drought (-11.4%)'},
        1974: {'color': '#9467bd', 'linewidth': 2.0, 'linestyle': '--', 'label': '1974: Wet Sowing & Early Frost (-19.7%)'},
        1993: {'color': '#17becf', 'linewidth': 2.2, 'linestyle': '-.', 'label': '1993: Great Midwest Flood (-10.5%)'},
        2003: {'color': '#8c564b', 'linewidth': 1.8, 'linestyle': ':', 'label': '2003: Drought & Aphid Outbreak (-21.6%)'},
        # Positive Bumper Crops
        1994: {'color': '#2ca02c', 'linewidth': 2.4, 'linestyle': '-', 'label': '1994: Climatic Optimum Bumper (+10.7%)'},
        1961: {'color': '#006400', 'linewidth': 2.0, 'linestyle': '--', 'label': '1961: Benign Moisture Bumper (+10.1%)'},
        1952: {'color': '#1f77b4', 'linewidth': 1.8, 'linestyle': '-.', 'label': '1952: Early Favorable Bumper (+11.9%)'},
    }
    
    # Phenological stage bands
    def add_pheno_bands(ax):
        ax.axvspan(91, 152, color='#e9ecef', alpha=0.35, zorder=0)  # Pre-sowing & Sowing
        ax.axvspan(152, 182, color='#d8f3dc', alpha=0.35, zorder=0) # Vegetative
        ax.axvspan(182, 213, color='#fff3cd', alpha=0.40, zorder=0) # Flowering R1-R3
        ax.axvspan(213, 244, color='#ffe5d9', alpha=0.45, zorder=0) # Pod-filling R4-R6
        ax.axvspan(244, 304, color='#f8edeb', alpha=0.35, zorder=0) # Maturity & Harvest
        
    doys = clim_summary.index.values
    
    # ----------------------------------------------------
    # Panel (a): Cumulative Heat Stress Days (HD30)
    # ----------------------------------------------------
    ax_a = axes[0, 0]
    add_pheno_bands(ax_a)
    
    # Plot extreme years
    for yr, sty in styles.items():
        if yr in trajectories:
            df_y = trajectories[yr]
            ax_a.plot(df_y['doy'], df_y['cum_hd30'], color=sty['color'],
                      linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                      label=sty['label'], zorder=4)
                      
    ax_a.set_title(r'(a) Cumulative Extreme Heat Stress Exposure ($\sum HD_{30}$)', fontweight='bold', loc='left')
    ax_a.set_ylabel(r'Cumulative Heat Days ($T_{\mathrm{max}} \geq 30^{\circ}\mathrm{C}$)')
    ax_a.set_ylim(-2, 68)
    ax_a.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax_a.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper left', fontsize=7.8)
    
    # Annotate phenological labels at top
    ax_a.text(121, 63.5, 'Sowing', ha='center', fontsize=8, color='#555555', fontstyle='italic')
    ax_a.text(167, 63.5, 'Vegetative', ha='center', fontsize=8, color='#555555', fontstyle='italic')
    ax_a.text(197, 63.5, 'Flowering (R1-R3)', ha='center', fontsize=8, color='#b8860b', fontweight='bold')
    ax_a.text(228, 63.5, 'Pod-Fill (R4-R6)', ha='center', fontsize=8, color='#c0392b', fontweight='bold')
    ax_a.text(274, 63.5, 'Maturity (R7-R8)', ha='center', fontsize=8, color='#555555', fontstyle='italic')
    
    # ----------------------------------------------------
    # Panel (b): Atmospheric Evaporative Demand (VPD, 15-day rolling)
    # ----------------------------------------------------
    ax_b = axes[0, 1]
    add_pheno_bands(ax_b)
    
    # Climatological envelope
    vpd_med = clim_summary['vapor_pressure_deficit_median']
    vpd_p10 = clim_summary['vapor_pressure_deficit_<lambda_0>']
    vpd_p90 = clim_summary['vapor_pressure_deficit_<lambda_1>']
    ax_b.fill_between(doys, vpd_p10, vpd_p90, color='#cccccc', alpha=0.35, label='Normal Envelope (10th--90th %ile)', zorder=2)
    ax_b.plot(doys, vpd_med, color='#555555', linewidth=1.5, linestyle=':', label='Climatological Median', zorder=3)
    
    # Plot extreme years
    for yr, sty in styles.items():
        if yr in trajectories:
            df_y = trajectories[yr]
            ax_b.plot(df_y['doy'], df_y['vpd_roll15'], color=sty['color'],
                      linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                      label=sty['label'], zorder=4)
                      
    ax_b.axhline(1.5, color='#a51d24', linestyle='--', linewidth=1.0, alpha=0.7, label='Elevated Atmospheric Demand (1.5 kPa)')
    ax_b.set_title(r'(b) Atmospheric Evaporative Demand (15-Day Rolling Mean $VPD$)', fontweight='bold', loc='left')
    ax_b.set_ylabel('Vapor Pressure Deficit ($VPD$, kPa)')
    ax_b.set_ylim(0.3, 2.3)
    ax_b.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax_b.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper left', fontsize=7.8)
    
    # ----------------------------------------------------
    # Panel (c): Cumulative Climatic Water Deficit / Surplus (\\sum (P - ET0))
    # ----------------------------------------------------
    ax_c = axes[1, 0]
    add_pheno_bands(ax_c)
    
    ax_c.axhline(0, color='black', linewidth=0.9, linestyle='-', zorder=2)
    
    # Climatological median water deficit
    ax_c.plot(doys, clim_summary['cum_balance_median'], color='#555555', linewidth=1.8, linestyle=':',
              label='Climatological Normal Deficit', zorder=3)
              
    # Plot extreme years
    for yr, sty in styles.items():
        if yr in trajectories:
            df_y = trajectories[yr]
            ax_c.plot(df_y['doy'], df_y['cum_balance'], color=sty['color'],
                      linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                      label=sty['label'], zorder=4)
                      
    ax_c.set_title(r'(c) Cumulative Climatic Water Budget ($\sum (P - ET_0)$)', fontweight='bold', loc='left')
    ax_c.set_ylabel('Cumulative Water Deficit / Surplus (mm)')
    ax_c.set_ylim(-500, 220)
    ax_c.set_xticks(month_ticks)
    ax_c.set_xticklabels(month_labels)
    ax_c.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax_c.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='lower left', fontsize=7.8)
    
    # ----------------------------------------------------
    # Panel (d): Root-Zone Soil Moisture Depletion (SM_root, 0-100cm)
    # ----------------------------------------------------
    ax_d = axes[1, 1]
    add_pheno_bands(ax_d)
    
    sm_med = clim_summary['sm_root_median']
    sm_p10 = clim_summary['sm_root_<lambda_0>']
    sm_p90 = clim_summary['sm_root_<lambda_1>']
    ax_d.fill_between(doys, sm_p10, sm_p90, color='#cccccc', alpha=0.35, label='Normal Envelope (10th--90th %ile)', zorder=2)
    ax_d.plot(doys, sm_med, color='#555555', linewidth=1.5, linestyle=':', label='Climatological Median', zorder=3)
    
    # Soil physical reference levels
    ax_d.axhline(0.38, color='#17becf', linestyle='--', linewidth=0.9, alpha=0.8, label=r'Regional Field Capacity ($\approx 0.38\ \mathrm{m}^3/\mathrm{m}^3$)')
    ax_d.axhline(0.18, color='#d62728', linestyle='--', linewidth=0.9, alpha=0.8, label=r'Incipient Wilting Stress ($\approx 0.18\ \mathrm{m}^3/\mathrm{m}^3$)')
    
    # Plot extreme years
    for yr, sty in styles.items():
        if yr in trajectories:
            df_y = trajectories[yr]
            ax_d.plot(df_y['doy'], df_y['sm_root'], color=sty['color'],
                      linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                      label=sty['label'], zorder=4)
                      
    ax_d.set_title(r'(d) Root-Zone Soil Moisture Depletion (0--100 cm $SM_{\mathrm{root}}$)', fontweight='bold', loc='left')
    ax_d.set_ylabel(r'Volumetric Soil Moisture ($\mathrm{m}^3/\mathrm{m}^3$)')
    ax_d.set_ylim(0.15, 0.44)
    ax_d.set_xticks(month_ticks)
    ax_d.set_xticklabels(month_labels)
    ax_d.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax_d.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='lower left', fontsize=7.8)
    
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Intra-Seasonal Extremes: {out_pdf} and {out_png}")

def main():
    print("=== STARTING CLIMATE EXTREMES ANALYSIS ===")
    yield_csv = 'output/data/target/soybean_yield_1951_2025.csv'
    weather_dir = 'work/data/interim/county_daily_weather/production'
    cand_dir = 'work/reports/exploratory_analysis/candidates'
    
    out_clim_pdf = 'output/thesis/figures/fig_weather_climatology.pdf'
    out_clim_png = 'output/thesis/figures/fig_weather_climatology.png'
    out_intra_pdf = 'output/thesis/figures/fig_intra_seasonal_extremes.pdf'
    out_intra_png = 'output/thesis/figures/fig_intra_seasonal_extremes.png'
    
    # 1. Load Yield Extremes (+/- 10%)
    panel_yield, neg_shocks, pos_bumpers, trend_fit = load_panel_yield_extremes(yield_csv, threshold_pct=10.0)
    print(f"\nIdentified {len(neg_shocks)} Negative Shocks (<= -10%):")
    print(neg_shocks[['year', 'yield_bu_per_acre', 'rel_anomaly']].to_string(index=False))
    print(f"\nIdentified {len(pos_bumpers)} Positive Bumper Harvests (>= +10%):")
    print(pos_bumpers[['year', 'yield_bu_per_acre', 'rel_anomaly']].to_string(index=False))
    
    # 2. Load Weather Data
    year_map = load_weather_data(weather_dir)
    
    # 3. Compute Multi-Year Weather Aggregates
    print("\nComputing multi-year weather indicators...")
    weather_annual = compute_annual_weather_metrics(year_map)
    
    # 4. Generate Multi-Year Climatology
    print("\nGenerating Multi-Year Macro Climatology (fig_weather_climatology)...")
    plot_multiyear_climatology(panel_yield, weather_annual, neg_shocks, pos_bumpers, out_clim_pdf, out_clim_png)
    # Save candidate preservation
    shutil.copyfile(out_clim_pdf, os.path.join(cand_dir, 'fig_weather_climatology_extremes_candidate.pdf'))
    shutil.copyfile(out_clim_png, os.path.join(cand_dir, 'fig_weather_climatology_extremes_candidate.png'))
    
    # 5. Extract Daily Trajectories for Extreme Years
    target_extreme_years = [1988, 1983, 1974, 1993, 2003, 1994, 1961, 1952, 2012, 2016, 2021]
    print(f"\nExtracting daily phenological trajectories for extreme years: {target_extreme_years}...")
    trajectories, clim_summary = extract_daily_trajectories(year_map, target_extreme_years)
    
    # 6. Generate Intra-Seasonal Extremes Figure
    print("\nGenerating Intra-Seasonal Trajectories (fig_intra_seasonal_extremes)...")
    plot_intra_seasonal_extremes(trajectories, clim_summary, out_intra_pdf, out_intra_png)
    # Save candidate preservation
    shutil.copyfile(out_intra_pdf, os.path.join(cand_dir, 'fig_intra_seasonal_extremes_candidate.pdf'))
    shutil.copyfile(out_intra_png, os.path.join(cand_dir, 'fig_intra_seasonal_extremes_candidate.png'))
    
    print("\n=== CLIMATE EXTREMES ANALYSIS COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
