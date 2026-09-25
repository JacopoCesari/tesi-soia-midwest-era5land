"""
Revised script to generate pristine, publication-grade weather figures:
1. fig_intra_seasonal_extremes.pdf / .png (Cumulative 2x2 with phenological background)
2. fig_weather_shocks_footprint.pdf / .png (Non-cumulative 3x1 for Top 3 Best vs Top 3 Worst)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

ROOT_DIR = r"c:\Users\JacopoCesari-Aretésr\Desktop\Tesi"
WEATHER_DIR = os.path.join(ROOT_DIR, "work", "data", "interim", "county_daily_weather", "production")
FIGURES_DIR = os.path.join(ROOT_DIR, "output", "thesis", "figures")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.0,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

parquet_files = sorted(glob.glob(os.path.join(WEATHER_DIR, "county_daily_*.parquet")))
years_available = [int(os.path.basename(f).replace("county_daily_", "").replace(".parquet", "")) for f in parquet_files]

all_target_years = [1988, 2003, 1974, 1983, 1993, 1952, 2021, 2016, 1994]

daily_by_year = {}
climatology_list = []

for yr in years_available:
    fpath = os.path.join(WEATHER_DIR, f"county_daily_{yr}.parquet")
    df_yr = pd.read_parquet(fpath, columns=[
        "date", "air_temperature_maximum", "total_precipitation", "et0_fao56", "p_minus_et0",
        "vapor_pressure_deficit", "heat_day_30",
        "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3"
    ])
    df_yr["doy"] = df_yr["date"].dt.dayofyear
    df_season = df_yr[(df_yr["doy"] >= 91) & (df_yr["doy"] <= 304)].copy()
    
    # Depth-weighted SM_root (Eq 226)
    df_season["sm_root"] = (0.07 * df_season["volumetric_soil_water_layer_1"] +
                            0.21 * df_season["volumetric_soil_water_layer_2"] +
                            0.72 * df_season["volumetric_soil_water_layer_3"])
    
    daily_mean = df_season.groupby("doy").agg({
        "air_temperature_maximum": "mean",
        "total_precipitation": "mean",
        "et0_fao56": "mean",
        "p_minus_et0": "mean",
        "vapor_pressure_deficit": "mean",
        "heat_day_30": "mean",
        "sm_root": "mean"
    }).reset_index()
    daily_mean["year"] = yr
    
    # Cumulative curves
    daily_mean["cum_p"] = daily_mean["total_precipitation"].cumsum()
    daily_mean["cum_et0"] = daily_mean["et0_fao56"].cumsum()
    daily_mean["cum_balance"] = daily_mean["p_minus_et0"].cumsum()
    daily_mean["cum_hd30"] = daily_mean["heat_day_30"].cumsum()
    
    # Rolling curves (15-day)
    daily_mean["vpd_roll15"] = daily_mean["vapor_pressure_deficit"].rolling(15, center=True, min_periods=1).mean()
    daily_mean["tmax_roll15"] = daily_mean["air_temperature_maximum"].rolling(15, center=True, min_periods=1).mean()
    
    climatology_list.append(daily_mean)
    if yr in all_target_years:
        daily_by_year[yr] = daily_mean

df_all = pd.concat(climatology_list, ignore_index=True)

clim_doy = df_all.groupby("doy").agg({
    "cum_p": "median",
    "cum_et0": "median",
    "cum_balance": "median",
    "cum_hd30": "median",
    "vpd_roll15": ["median", lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)],
    "sm_root": ["median", lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)],
    "tmax_roll15": ["median", lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)]
})
clim_doy.columns = ['_'.join(c).strip() for c in clim_doy.columns]
clim_doy = clim_doy.reset_index()

ref_dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(clim_doy["doy"] - 1, unit="D")

month_ticks = [pd.to_datetime(f"2021-{m:02d}-01") for m in range(4, 11)] + [pd.to_datetime("2021-10-31")]
month_labels = ['Apr 1', 'May 1', 'Jun 1', 'Jul 1', 'Aug 1', 'Sep 1', 'Oct 1', 'Oct 31']

stages = [
    ("Pre-sowing & Sowing", pd.to_datetime("2021-04-01"), pd.to_datetime("2021-05-31"), "#f0f0f0"),
    ("Vegetative Growth", pd.to_datetime("2021-06-01"), pd.to_datetime("2021-06-30"), "#e5f5e0"),
    ("Flowering (R1-R3)", pd.to_datetime("2021-07-01"), pd.to_datetime("2021-07-31"), "#fee6ce"),
    ("Pod-Filling (R4-R6)", pd.to_datetime("2021-08-01"), pd.to_datetime("2021-08-31"), "#fdd0a2"),
    ("Maturity & Harvest", pd.to_datetime("2021-09-01"), pd.to_datetime("2021-10-31"), "#f0f0f0")
]

# ==============================================================================
# 1. FIGURE 3.7: CUMULATIVE 2x2 WITH CLEAN TOP LABELS & UNIFIED BOTTOM LEGEND
# ==============================================================================
print("Generating Figure 3.7 (Cumulative 2x2)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), sharex=True)

def add_pheno_clean(ax, show_labels=False):
    for name, start, end, col in stages:
        ax.axvspan(start, end, color=col, alpha=0.40, zorder=0)
    if show_labels:
        # Add labels just above the plot frame or at y_max
        y_lim = ax.get_ylim()
        y_pos = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.94
        for name, start, end, _ in stages:
            mid = start + (end - start) / 2
            ax.text(mid, y_pos, name, ha="center", va="center", fontsize=8.0, fontweight="bold",
                    color="#444444", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))

styles_cum = {
    # Negative shocks
    1988: {"label": "1988: Severe Heat & Drought (-25.6%)", "color": "#d73027", "lw": 2.2, "ls": "-"},
    2003: {"label": "2003: Late-Season Deficit (-21.6%)", "color": "#fc8d59", "lw": 1.8, "ls": "--"},
    1974: {"label": "1974: Wet Sowing & Freeze (-19.7%)", "color": "#762a83", "lw": 1.8, "ls": "-."},
    1983: {"label": "1983: Pod-Fill Thermal Drought (-11.4%)", "color": "#e08214", "lw": 1.6, "ls": ":"},
    1993: {"label": "1993: Great Midwest Flood (-10.5%)", "color": "#4575b4", "lw": 2.2, "ls": "-"},
    # Positive bumpers
    1952: {"label": "1952: Early Favorable Bumper (+11.9%)", "color": "#1a9850", "lw": 1.8, "ls": "-."},
    2021: {"label": "2021: Modern Favorable Bumper (+10.8%)", "color": "#006837", "lw": 2.0, "ls": "--"},
    2016: {"label": "2016: Modern Benchmark Bumper (+10.8%)", "color": "#41ab5d", "lw": 2.0, "ls": "-"},
    1994: {"label": "1994: Climatic Optimum Bumper (+10.7%)", "color": "#238443", "lw": 1.8, "ls": ":"}
}

years_plot_cum = [1988, 2003, 1974, 1983, 1993, 1952, 2021, 2016, 1994]

# (a) Cumulative Heat Stress Days
ax_a = axes[0, 0]
line_clim = ax_a.plot(ref_dates, clim_doy["cum_hd30_median"], color="black", lw=2.2, ls=":", label="Climatological Normal", zorder=3)[0]
for yr in years_plot_cum:
    if yr in daily_by_year:
        s = styles_cum[yr]
        ax_a.plot(ref_dates, daily_by_year[yr]["cum_hd30"], color=s["color"], lw=s["lw"], ls=s["ls"], label=s["label"], zorder=4)
ax_a.set_ylabel(r"Cumulative Heat Days ($T_{\text{max}} \geq 30^\circ\text{C}$)")
ax_a.set_title(r"(a) Cumulative Extreme Heat Stress Exposure ($\sum HD_{30}$)", loc="left", fontweight="bold")
ax_a.set_ylim(-2, 70)
add_pheno_clean(ax_a, show_labels=True)

# (b) Cumulative Precipitation
ax_b = axes[0, 1]
ax_b.plot(ref_dates, clim_doy["cum_p_median"], color="black", lw=2.2, ls=":", zorder=3)
for yr in years_plot_cum:
    if yr in daily_by_year:
        s = styles_cum[yr]
        ax_b.plot(ref_dates, daily_by_year[yr]["cum_p"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
ax_b.set_ylabel("Cumulative Precipitation (mm)")
ax_b.set_title(r"(b) Cumulative Growing Season Precipitation ($\sum P$)", loc="left", fontweight="bold")
ax_b.set_ylim(0, 1050)
add_pheno_clean(ax_b, show_labels=True)

# (c) Cumulative Reference Evapotranspiration
ax_c = axes[1, 0]
ax_c.plot(ref_dates, clim_doy["cum_et0_median"], color="black", lw=2.2, ls=":", zorder=3)
for yr in years_plot_cum:
    if yr in daily_by_year:
        s = styles_cum[yr]
        ax_c.plot(ref_dates, daily_by_year[yr]["cum_et0"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
ax_c.set_ylabel(r"Cumulative $ET_0$ (mm)")
ax_c.set_title(r"(c) Cumulative Reference Evapotranspiration ($\sum ET_0$)", loc="left", fontweight="bold")
ax_c.set_ylim(0, 1050)
add_pheno_clean(ax_c, show_labels=False)

# (d) Cumulative Climatic Water Budget
ax_d = axes[1, 1]
ax_d.axhline(0, color="black", lw=0.9, ls="-", zorder=2)
ax_d.plot(ref_dates, clim_doy["cum_balance_median"], color="black", lw=2.2, ls=":", zorder=3)
for yr in years_plot_cum:
    if yr in daily_by_year:
        s = styles_cum[yr]
        ax_d.plot(ref_dates, daily_by_year[yr]["cum_balance"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
ax_d.set_ylabel(r"Cumulative Water Balance $\sum(P - ET_0)$ (mm)")
ax_d.set_title(r"(d) Cumulative Climatic Water Budget ($\sum (P - ET_0)$)", loc="left", fontweight="bold")
ax_d.set_ylim(-520, 260)
add_pheno_clean(ax_d, show_labels=False)

for ax in axes.flat:
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.set_xlim(ref_dates.iloc[0], ref_dates.iloc[-1])

# Single clean unified legend at bottom
handles, labels = ax_a.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5,
           frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.2)

plt.tight_layout()
out_cum_pdf = os.path.join(FIGURES_DIR, "fig_intra_seasonal_extremes.pdf")
out_cum_png = os.path.join(FIGURES_DIR, "fig_intra_seasonal_extremes.png")
plt.savefig(out_cum_pdf, bbox_inches="tight")
plt.savefig(out_cum_png, bbox_inches="tight")
plt.close()
print(f"Saved Figure 3.7 to {out_cum_pdf}")


# ==============================================================================
# 2. FIGURE 3.8: NON-CUMULATIVE DYNAMIC STATE DYNAMICS: TOP 3 BEST vs TOP 3 WORST
# ==============================================================================
print("Generating Figure 3.8 (Non-Cumulative 3x1: Top 3 Best vs Top 3 Worst)...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)

styles_shocks = {
    # Top 3 Worst
    1988: {"label": "1988: Severe Heat & Drought (-25.6%)", "color": "#d73027", "lw": 2.2, "ls": "-"},
    2003: {"label": "2003: Late-Season Moisture Deficit (-21.6%)", "color": "#e08214", "lw": 2.0, "ls": "--"},
    1974: {"label": "1974: Wet Sowing Delay & Autumn Freeze (-19.7%)", "color": "#762a83", "lw": 2.0, "ls": "-."},
    # Top 3 Best
    1952: {"label": "1952: Early Favorable Optimum (+11.9%)", "color": "#1a9850", "lw": 2.0, "ls": "-"},
    2021: {"label": "2021: Modern Climatic Optimum (+10.8%)", "color": "#006837", "lw": 2.0, "ls": "--"},
    2016: {"label": "2016: Benign Thermal-Hydrological Buffer (+10.8%)", "color": "#31a354", "lw": 1.8, "ls": "-."}
}

target_6 = [1988, 2003, 1974, 1952, 2021, 2016]

# (a) Atmospheric Evaporative Demand (VPD, 15-day rolling)
ax0 = axes[0]
ax0.fill_between(ref_dates, clim_doy["vpd_roll15_<lambda_0>"], clim_doy["vpd_roll15_<lambda_1>"],
                 color="#cccccc", alpha=0.35, label="Normal Climatological Envelope (10th–90th %ile)", zorder=2)
ax0.plot(ref_dates, clim_doy["vpd_roll15_median"], color="black", lw=2.0, ls=":", label="Climatological Median", zorder=3)
for yr in target_6:
    s = styles_shocks[yr]
    ax0.plot(ref_dates, daily_by_year[yr]["vpd_roll15"], color=s["color"], lw=s["lw"], ls=s["ls"], label=s["label"], zorder=4)
ax0.axhline(1.5, color="#b2182b", ls="--", lw=1.2, alpha=0.8)
ax0.text(ref_dates[5], 1.54, r"Elevated Atmospheric Demand / Stomatal Resistance ($VPD \geq 1.5\text{ kPa}$)",
         color="#b2182b", fontsize=8.2, fontweight="bold")
ax0.set_ylabel(r"15-Day Rolling $VPD$ (kPa)")
ax0.set_title(r"(a) Atmospheric Evaporative Demand: 15-Day Rolling Mean Vapor Pressure Deficit ($VPD$)", loc="left", fontweight="bold")
ax0.set_ylim(0.4, 2.3)
ax0.legend(loc="upper left", frameon=True, fontsize=8.0, ncol=2, facecolor="white", edgecolor="#cccccc")

# (b) Root-Zone Soil Moisture
ax1 = axes[1]
ax1.fill_between(ref_dates, clim_doy["sm_root_<lambda_0>"], clim_doy["sm_root_<lambda_1>"],
                 color="#cccccc", alpha=0.35, label="Normal Climatological Envelope (10th–90th %ile)", zorder=2)
ax1.plot(ref_dates, clim_doy["sm_root_median"], color="black", lw=2.0, ls=":", label="Climatological Median", zorder=3)
for yr in target_6:
    s = styles_shocks[yr]
    ax1.plot(ref_dates, daily_by_year[yr]["sm_root"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
ax1.axhline(0.38, color="#2b83ba", ls="--", lw=1.0, alpha=0.8)
ax1.text(pd.to_datetime("2021-08-20"), 0.385, r"Regional Field Capacity ($\approx 0.38\ \mathrm{m}^3/\mathrm{m}^3$)",
         color="#2b83ba", fontsize=8.2, fontweight="bold", ha="right")
ax1.axhline(0.18, color="#d73027", ls="--", lw=1.0, alpha=0.8)
ax1.text(pd.to_datetime("2021-05-15"), 0.185, r"Incipient Wilting Stress Threshold ($\approx 0.18\ \mathrm{m}^3/\mathrm{m}^3$)",
         color="#d73027", fontsize=8.2, fontweight="bold")
ax1.set_ylabel(r"Soil Moisture ($\text{m}^3/\text{m}^3$)")
ax1.set_title(r"(b) Integrated Root-Zone Soil Moisture Dynamics ($SM_{\text{root}}$, $0$--$100\text{ cm}$ Column)", loc="left", fontweight="bold")
ax1.set_ylim(0.15, 0.44)

# (c) Maximum Temperature (15-day rolling)
ax2 = axes[2]
ax2.fill_between(ref_dates, clim_doy["tmax_roll15_<lambda_0>"], clim_doy["tmax_roll15_<lambda_1>"],
                 color="#cccccc", alpha=0.35, label="Normal Climatological Envelope (10th–90th %ile)", zorder=2)
ax2.plot(ref_dates, clim_doy["tmax_roll15_median"], color="black", lw=2.0, ls=":", label="Climatological Median", zorder=3)
for yr in target_6:
    s = styles_shocks[yr]
    ax2.plot(ref_dates, daily_by_year[yr]["tmax_roll15"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
ax2.axhline(30, color="#d73027", ls="--", lw=1.2, alpha=0.8)
ax2.text(pd.to_datetime("2021-05-15"), 30.5, r"Reproductive Thermal Stress Ceiling ($30^\circ\text{C}$)",
         color="#d73027", fontsize=8.2, fontweight="bold")
ax2.set_ylabel(r"15-Day Rolling $T_{\text{max}}$ ($^\circ$C)")
ax2.set_xlabel("Calendar Date Across Growing Season")
ax2.set_title(r"(c) Thermal Regimes: 15-Day Rolling Maximum Temperature ($T_{\text{max}}$)", loc="left", fontweight="bold")
ax2.set_ylim(10, 36)

# Highlight critical reproductive window (July-August) across all 3 panels
jul_1 = pd.to_datetime("2021-07-01")
aug_31 = pd.to_datetime("2021-08-31")
for ax in axes:
    ax.axvspan(jul_1, aug_31, color="#fee8c8", alpha=0.35, zorder=0)

axes[0].text(pd.to_datetime("2021-08-01"), 2.18, "Critical Reproductive Peak (July--August, R1--R6)",
             ha="center", fontsize=8.8, fontweight="bold", color="#b2182b",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#e34a33", alpha=0.85, lw=0.8))

for ax in axes:
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.set_xlim(ref_dates.iloc[0], ref_dates.iloc[-1])

plt.tight_layout()
out_shocks_pdf = os.path.join(FIGURES_DIR, "fig_weather_shocks_footprint.pdf")
out_shocks_png = os.path.join(FIGURES_DIR, "fig_weather_shocks_footprint.png")
plt.savefig(out_shocks_pdf, bbox_inches="tight")
plt.savefig(out_shocks_png, bbox_inches="tight")
plt.close()
print(f"Saved Figure 3.8 to {out_shocks_pdf}")

print("All figures generated successfully!")
