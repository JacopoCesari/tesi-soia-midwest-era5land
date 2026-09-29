"""
Script to generate all Weather EDA tables and figures for Chapter 3: Data and Study Area.
Outputs:
- Tables in output/thesis/tables/
  - tab_weather_variables.tex
- Figures in output/thesis/figures/
  - fig_weather_seasonality.pdf and .png
  - fig_weather_climatology.pdf and .png (Figure 3.6: Multi-Decadal Stress & Yield Shocks Nexus)
  - fig_intra_seasonal_extremes.pdf and .png (Figure 3.7: Cumulative Intra-Seasonal Phenology)
  - fig_weather_shocks_footprint.pdf and .png (Figure 3.8: Non-Cumulative Dynamic States)
  - fig_climate_yield_sensitivity.pdf and .png (Figure 3.10: Monthly Climate-Yield Sensitivity)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# High-quality publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

from pathlib import Path
_script_path = Path(__file__).resolve()
ROOT_DIR = _script_path.parents[2] if (_script_path.parents[1].name == "output") else _script_path.parents[1]
OUTPUT_DIR_PATH = ROOT_DIR / "output" if (ROOT_DIR / "output").exists() else ROOT_DIR

WEATHER_DIR = str(ROOT_DIR / "data" / "weather" / "daily_counties")
TARGET_CSV = str(ROOT_DIR / "data" / "target" / "soybean_yield_1951_2025.csv")
COUNTIES_CSV = str(ROOT_DIR / "data" / "auxiliary" / "counties.csv")
TABLES_DIR = str(OUTPUT_DIR_PATH / "thesis" / "tables")
FIGURES_DIR = str(OUTPUT_DIR_PATH / "thesis" / "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
FIGURES_DIR_EXISTS = os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD AVAILABLE WEATHER DATA & YIELDS
# ------------------------------------------------------------------------------
print("Scanning available weather parquet files...")
parquet_files = sorted(glob.glob(os.path.join(WEATHER_DIR, "county_daily_*.parquet")))
years_available = []
for f in parquet_files:
    basename = os.path.basename(f)
    yr = int(basename.replace("county_daily_", "").replace(".parquet", ""))
    years_available.append(yr)

print(f"Found {len(years_available)} weather years: from {min(years_available)} to {max(years_available)}")

# Load target yield data and compute linear detrended yield anomalies
df_yield = pd.read_csv(TARGET_CSV)
df_yield["county_fips"] = df_yield["county_fips"].astype(str).str.zfill(5)
df_counties = pd.read_csv(COUNTIES_CSV)
df_counties["county_fips"] = df_counties["county_fips"].astype(str).str.zfill(5)

# County-level detrending
df_yield_detrended = []
for fips, group in df_yield.groupby("county_fips"):
    group = group.sort_values("year").copy()
    slope, intercept, _, _, _ = stats.linregress(group["year"], group["yield_bu_per_acre"])
    group["yield_trend"] = intercept + slope * group["year"]
    group["yield_anomaly"] = group["yield_bu_per_acre"] - group["yield_trend"]
    group["yield_anomaly_pct"] = (group["yield_anomaly"] / group["yield_trend"]) * 100.0
    df_yield_detrended.append(group)

df_yields = pd.concat(df_yield_detrended, ignore_index=True)

# Regional panel mean yield and detrending
pmean = df_yield.groupby("year")["yield_bu_per_acre"].mean().reset_index()
z = np.polyfit(pmean["year"], pmean["yield_bu_per_acre"], 1)
pmean["trend"] = np.polyval(z, pmean["year"])
pmean["anomaly"] = pmean["yield_bu_per_acre"] - pmean["trend"]
pmean["anomaly_pct"] = (pmean["anomaly"] / pmean["trend"]) * 100.0
pmean["yoy_change_pct"] = (pmean["yield_bu_per_acre"].diff() / pmean["yield_bu_per_acre"].shift(1)) * 100.0


# ==============================================================================
# TABLE 3.5: Reanalysis and Agrometeorological Variables Specification
# ==============================================================================
def generate_table_3_5():
    print("Generating Table 3.5 (Weather Variables)...")
    tex_content = r"""% Table 3.5: ERA5-Land surface meteorological variables and derived daily agrometeorological indicators
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\caption{ERA5-Land surface meteorological variables and derived daily agrometeorological indicators.}
\label{tab:weather_variables}
\begin{tabularx}{\textwidth}{p{2.4cm} p{2.6cm} p{1.3cm} >{\raggedright\arraybackslash}X}
\toprule
\textbf{Category} & \textbf{Symbol} & \textbf{Unit} & \textbf{Physical Description and Agronomic Rationale} \\
\midrule
\multicolumn{4}{l}{\textit{Panel A: Primary Atmospheric and Land-Surface Reanalysis Fields (ECMWF ERA5-Land)}} \\
Thermal Dynamics & $T_{\text{mean}}, T_{\text{max}}, T_{\text{min}}$ & $^\circ$C & Daily mean, maximum, and minimum 2\,m air temperature; thermal accumulation and heat stress. \\
& $T_{\text{dew}}$ & $^\circ$C & Daily mean 2\,m dewpoint temperature; near-surface moisture baseline. \\
Hydrological Fluxes & $P$ & mm/d & Total daily precipitation; surface water supply, recharge, and waterlogging. \\
& $PE$ & mm/d & Potential evaporation; atmospheric evaporative capacity from the surface. \\
Radiative Forcing & $SSRD$ & $\text{MJ}/\text{m}^2/\text{d}$ & Surface downward solar radiation; photosynthetically active radiation driving biomass assimilation. \\
Pedological Moisture & $SM_1, SM_2, SM_3$ & $\text{m}^3/\text{m}^3$ & Volumetric soil water across depths: $0$--$7\text{ cm}$ (seedbed), $7$--$28\text{ cm}$ (upper root zone), and $28$--$100\text{ cm}$ (deep root reservoir). \\
\midrule
\multicolumn{4}{l}{\textit{Panel B: Derived Agrometeorological and Bioclimatic Indicators}} \\
Evaporative Demand & $VPD$ & kPa & Vapor pressure deficit $e_s(T_{\text{mean}}) - e_a(T_{\text{dew}})$; atmospheric drying power driving stomatal closure. \\
Crop Water Demand & $ET_0$ & mm/d & FAO-56 Penman-Monteith reference evapotranspiration \citep{Allen1998}. \\
Climatic Balance & $P - ET_0$ & mm/d & Climatic water balance; net daily atmospheric moisture deficit or surplus. \\
Thermal Pacing & $GDD$ & $^\circ$C$\cdot$d & Growing Degree Days ($10$--$30^\circ$C); vegetative and reproductive phenological progression pacing. \\
Extreme Heat Stress & $HD_{30}, HD_{35}$ & binary & Daily heat day exceedances $\mathbb{I}(T_{\text{max}} \ge 30^\circ\text{C})$ and $\ge 35^\circ\text{C}$ \citep{Schlenker2009}. \\
Root-Zone Reservoir & $SM_{\text{root}}$ & $\text{m}^3/\text{m}^3$ & Depth-weighted soil water column ($0$--$100\text{ cm}$, Eq.~\eqref{eq:sm_root}); multi-month hydrological buffer. \\
\bottomrule
\end{tabularx}
\end{table}
"""
    with open(os.path.join(TABLES_DIR, "tab_weather_variables.tex"), "w", encoding="utf-8") as f:
        f.write(tex_content)
    print("Table 3.5 saved.")


# ==============================================================================
# FIGURE 3.6: Multi-Decadal Stress Evolution & Shock/Bumper Yield Nexus
# ==============================================================================
def generate_figure_3_6():
    print("Generating Figure 3.6 (Multi-Decadal Stress Evolution: Shocks and Bumpers)...")
    
    annual_metrics = []
    for yr in years_available:
        fpath = os.path.join(WEATHER_DIR, f"county_daily_{yr}.parquet")
        df_yr = pd.read_parquet(fpath, columns=[
            "county_fips", "date", "air_temperature_maximum", "total_precipitation",
            "vapor_pressure_deficit", "heat_day_30"
        ])
        df_yr["month"] = df_yr["date"].dt.month
        summer = df_yr[df_yr["month"].isin([7, 8])]
        vpd_summer = summer["vapor_pressure_deficit"].mean()
        hd30_annual = df_yr.groupby("county_fips")["heat_day_30"].sum().mean()
        
        annual_metrics.append({
            "year": yr,
            "vpd_summer": vpd_summer,
            "hd30_annual": hd30_annual
        })
        
    df_clim = pd.DataFrame(annual_metrics).sort_values("year").reset_index(drop=True)
    df_merged = pd.merge(pmean, df_clim, on="year", how="left")
    
    # Identify shocks (<= -10%) and bumpers (>= +10%)
    shocks = df_merged[df_merged["anomaly_pct"] <= -10.0].sort_values("year")
    bumpers = df_merged[df_merged["anomaly_pct"] >= 10.0].sort_values("year")
    shock_years = set(shocks["year"].tolist())
    bumper_years = set(bumpers["year"].tolist())
    
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.0), sharex=True, gridspec_kw={'height_ratios': [1.15, 1.15]})
    
    # --- PANEL (a): Yield Detrended Anomalies (%) ---
    ax0 = axes[0]
    for _, row in df_merged.iterrows():
        yr = int(row["year"])
        val = row["anomaly_pct"]
        if yr in shock_years:
            ax0.bar(yr, val, width=0.85, color="#b2182b", alpha=0.90, edgecolor="#67001f", lw=1.1, zorder=3)
        elif yr in bumper_years:
            ax0.bar(yr, val, width=0.85, color="#1b7837", alpha=0.90, edgecolor="#00441b", lw=1.1, zorder=3)
        elif val >= 0:
            ax0.bar(yr, val, width=0.85, color="#4393c3", alpha=0.65, edgecolor="none", zorder=2)
        else:
            ax0.bar(yr, val, width=0.85, color="#92c5de", alpha=0.65, edgecolor="none", zorder=2)
            
    ax0.axhline(0, color="#333333", lw=1.0, zorder=3)
    ax0.axhline(-10, color="#b2182b", ls=":", lw=1.2, alpha=0.75, zorder=2)
    ax0.axhline(10, color="#1b7837", ls=":", lw=1.2, alpha=0.75, zorder=2)
    
    # Annotate Negative Shocks (below bars)
    for _, row in shocks.iterrows():
        yr = int(row["year"])
        val = row["anomaly_pct"]
        # Stagger years close together
        offset = -4.0 if val < -15 else -3.2
        if yr == 1983:
            ax0.text(yr - 0.4, val - 3.8, f"{yr}\n({val:.1f}%)", ha="center", va="top",
                     fontsize=7.8, fontweight="bold", color="#67001f")
        elif yr == 1984:
            ax0.text(yr + 0.5, val - 1.8, f"{yr}\n({val:.1f}%)", ha="center", va="top",
                     fontsize=7.8, fontweight="bold", color="#67001f")
        else:
            ax0.text(yr, val + offset, f"{yr}\n({val:.1f}%)", ha="center", va="top",
                     fontsize=7.8, fontweight="bold", color="#67001f")
            
    # Annotate Positive Bumpers (above bars)
    for _, row in bumpers.iterrows():
        yr = int(row["year"])
        val = row["anomaly_pct"]
        ax0.text(yr, val + 1.2, f"{yr}\n(+{val:.1f}%)", ha="center", va="bottom",
                 fontsize=7.8, fontweight="bold", color="#00441b")
        
    ax0.set_ylabel(r"Detrended Anomaly (%)")
    ax0.set_title(r"(a) Midwestern County Soybean Yield Detrended Anomalies (1951--2025): Benchmark Shocks ($\leq -10\%$) vs. Bumpers ($\geq +10\%$)",
                  loc="left", fontweight="bold")
    ax0.set_ylim(-35, 23)
    
    from matplotlib.patches import Patch
    legend_elements_a = [
        Patch(facecolor="#1b7837", alpha=0.90, label=r"Bumper Harvests ($\geq +10\%$)"),
        Patch(facecolor="#4393c3", alpha=0.65, label="Positive Yield Anomaly (0% to +10%)"),
        Patch(facecolor="#92c5de", alpha=0.65, label="Moderate Negative Anomaly (0% to -10%)"),
        Patch(facecolor="#b2182b", alpha=0.90, label=r"Severe Downside Shocks ($\leq -10\%$)")
    ]
    ax0.legend(handles=legend_elements_a, loc="upper right", frameon=True, fontsize=8.8, ncol=2)
    
    # --- PANEL (b): Heat Stress Days (HD30) & Summer VPD ---
    ax1 = axes[1]
    
    # Subtle vertical guide bands linking both panels
    for yr in shock_years:
        ax0.axvspan(yr - 0.45, yr + 0.45, color="#fee8c8", alpha=0.20, zorder=0)
        ax1.axvspan(yr - 0.45, yr + 0.45, color="#fee8c8", alpha=0.20, zorder=0)
    for yr in bumper_years:
        ax0.axvspan(yr - 0.45, yr + 0.45, color="#e5f5e0", alpha=0.20, zorder=0)
        ax1.axvspan(yr - 0.45, yr + 0.45, color="#e5f5e0", alpha=0.20, zorder=0)
        
    # Heat Days bars with conditional coloring
    for _, row in df_merged.iterrows():
        yr = int(row["year"])
        hd = row["hd30_annual"]
        if yr in shock_years:
            ax1.bar(yr, hd, width=0.85, color="#b2182b", alpha=0.85, edgecolor="#67001f", lw=0.9, zorder=3)
        elif yr in bumper_years:
            ax1.bar(yr, hd, width=0.85, color="#1b7837", alpha=0.85, edgecolor="#00441b", lw=0.9, zorder=3)
        else:
            ax1.bar(yr, hd, width=0.85, color="#cbd5e1", alpha=0.70, edgecolor="#94a3b8", lw=0.4, zorder=2)
    
    # Secondary Y-axis for summer VPD
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_merged["year"], df_merged["vpd_summer"], color="#4a148c", lw=1.8, ls="-", marker="o", markersize=3.0,
                  alpha=0.90, label=r"Summer $VPD$ (July--August Mean, kPa)")
    
    ax1.set_ylabel(r"Heat Days / Year ($T_{\text{max}} \geq 30^\circ$C)")
    ax1_twin.set_ylabel(r"Summer $VPD$ (July--August, kPa)")
    ax1.set_xlabel("Crop Year")
    ax1.set_title(r"(b) Synchronized Agro-Climatic Forcing: Heat Stress Days ($HD_{30}$) and Atmospheric Evaporative Demand ($VPD$)",
                  loc="left", fontweight="bold")
    ax1.set_ylim(0, 75)
    ax1_twin.set_ylim(0.65, 1.40)
    ax1_twin.grid(False)
    
    from matplotlib.lines import Line2D
    legend_elements_b = [
        Patch(facecolor="#b2182b", alpha=0.85, edgecolor="#67001f", label=r"Heat Days: Severe Shocks ($\leq -10\%$)"),
        Patch(facecolor="#1b7837", alpha=0.85, edgecolor="#00441b", label=r"Heat Days: Bumper Years ($\geq +10\%$)"),
        Patch(facecolor="#cbd5e1", alpha=0.70, edgecolor="#94a3b8", label=r"Heat Days: Baseline Years"),
        Line2D([0], [0], color="#4a148c", lw=1.8, marker="o", markersize=3.0, label=r"Summer $VPD$ (July–August, kPa)")
    ]
    ax1.legend(handles=legend_elements_b, loc="upper left", frameon=True, fontsize=8.5, ncol=2)
    ax1.set_xlim(1950, 2026)
    
    plt.tight_layout()
    pdf_out = os.path.join(FIGURES_DIR, "fig_weather_climatology.pdf")
    png_out = os.path.join(FIGURES_DIR, "fig_weather_climatology.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out)
    plt.close()
    print(f"Figure 3.6 (Climatology & Yield Shocks) saved to {pdf_out}")


# ==============================================================================
# FIGURE 3.7: Cumulative Intra-Seasonal Phenological Progression (Balanced 6 Years)
# ==============================================================================
def generate_figure_intra_seasonal_cumulative():
    print("Generating Cumulative Intra-Seasonal Trajectories (fig_intra_seasonal_extremes)...")
    
    stages = [
        ("Sowing", pd.to_datetime("2021-04-01"), pd.to_datetime("2021-05-31"), "#f0f0f0"),
        ("Vegetative", pd.to_datetime("2021-06-01"), pd.to_datetime("2021-06-30"), "#e5f5e0"),
        ("Flowering", pd.to_datetime("2021-07-01"), pd.to_datetime("2021-07-31"), "#fee6ce"),
        ("Pod-Filling", pd.to_datetime("2021-08-01"), pd.to_datetime("2021-08-31"), "#fdd0a2"),
        ("Harvest", pd.to_datetime("2021-09-01"), pd.to_datetime("2021-10-31"), "#f0f0f0")
    ]
    
    # Balanced 4 Archetype Years: 2 Adverse Shocks + 2 Bumper Harvests
    target_years = [2012, 1993, 2016, 1994]
    
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
        
        # Depth-weighted root-zone soil moisture
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
        
        daily_mean["cum_p"] = daily_mean["total_precipitation"].cumsum()
        daily_mean["cum_et0"] = daily_mean["et0_fao56"].cumsum()
        daily_mean["cum_balance"] = daily_mean["p_minus_et0"].cumsum()
        daily_mean["cum_hd30"] = daily_mean["heat_day_30"].cumsum()
        daily_mean["vpd_roll15"] = daily_mean["vapor_pressure_deficit"].rolling(15, center=True, min_periods=1).mean()
        daily_mean["tmax_roll15"] = daily_mean["air_temperature_maximum"].rolling(15, center=True, min_periods=1).mean()
        
        climatology_list.append(daily_mean)
        if yr in target_years:
            daily_by_year[yr] = daily_mean
            
    df_all = pd.concat(climatology_list, ignore_index=True)
    def p10(x): return np.percentile(x, 10)
    def p90(x): return np.percentile(x, 90)
    
    clim_doy = df_all.groupby("doy").agg({
        "cum_p": ["median", p10, p90],
        "cum_et0": ["median", p10, p90],
        "cum_balance": ["median", p10, p90],
        "cum_hd30": ["median", p10, p90],
        "vpd_roll15": ["median", p10, p90],
        "sm_root": ["median", p10, p90],
        "tmax_roll15": ["median", p10, p90]
    })
    clim_doy.columns = ['_'.join(c).strip() for c in clim_doy.columns]
    clim_doy = clim_doy.reset_index()
    
    ref_dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(clim_doy["doy"] - 1, unit="D")
    month_ticks = [pd.to_datetime(f"2021-{m:02d}-01") for m in range(4, 11)]
    month_labels = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), sharex=True)
    
    def add_pheno_clean(ax, show_labels=False):
        for name, start, end, col in stages:
            ax.axvspan(start, end, color=col, alpha=0.35, zorder=0)
        if show_labels:
            y_lim = ax.get_ylim()
            y_pos = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.93
            for name, start, end, _ in stages:
                mid = start + (end - start) / 2
                ax.text(mid, y_pos, name, ha="center", va="center", fontsize=8.0, fontweight="bold",
                        color="#334155", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#cbd5e1", alpha=0.90, lw=0.6))
                
    # Unified high-contrast palettes: Shocks (warm) vs Bumpers (cool) - all solid lines
    styles_cum = {
        # Shocks (Warm/contrasting palette)
        2012: {"label": "2012 Shock: Flash Drought (-11.7%)", "color": "#d73027", "lw": 2.2, "ls": "-"},
        1993: {"label": "1993 Shock: Midwest Flood (-10.5%)", "color": "#762a83", "lw": 2.0, "ls": "-"},
        # Bumpers (Cool/harmonious palette)
        2016: {"label": "2016 Bumper: Optimum Buffer (+10.8%)", "color": "#006837", "lw": 2.2, "ls": "-"},
        1994: {"label": "1994 Bumper: Ideal Balance (+10.7%)", "color": "#2171b5", "lw": 2.0, "ls": "-"}
    }
    target_ordered = [2012, 1993, 2016, 1994]
    
    # (a) Cumulative Heat Stress Days
    ax_a = axes[0, 0]
    h_range = ax_a.fill_between(ref_dates, clim_doy["cum_hd30_p10"], clim_doy["cum_hd30_p90"],
                                color="#cbd5e1", alpha=0.50, label="Climatological Range (10th–90th %ile)", zorder=2)
    h_med = ax_a.plot(ref_dates, clim_doy["cum_hd30_median"], color="black", lw=2.0, ls="--", label="Climatological Median", zorder=3)[0]
    
    line_handles = {}
    for yr in target_ordered:
        if yr in daily_by_year:
            s = styles_cum[yr]
            line_handles[yr] = ax_a.plot(ref_dates, daily_by_year[yr]["cum_hd30"], color=s["color"], lw=s["lw"], ls=s["ls"], label=s["label"], zorder=4)[0]
    ax_a.set_ylabel(r"Cumulative Heat Days ($T_{\text{max}} \geq 30^\circ\text{C}$)")
    ax_a.set_title(r"(a) Cumulative Extreme Heat Days ($\sum HD_{30}$)", loc="left", fontweight="bold")
    ax_a.set_ylim(-2, 70)
    add_pheno_clean(ax_a, show_labels=True)
    
    # (b) Cumulative Precipitation
    ax_b = axes[0, 1]
    ax_b.fill_between(ref_dates, clim_doy["cum_p_p10"], clim_doy["cum_p_p90"],
                      color="#cbd5e1", alpha=0.50, zorder=2)
    ax_b.plot(ref_dates, clim_doy["cum_p_median"], color="black", lw=2.0, ls="--", zorder=3)
    for yr in target_ordered:
        if yr in daily_by_year:
            s = styles_cum[yr]
            ax_b.plot(ref_dates, daily_by_year[yr]["cum_p"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax_b.set_ylabel("Cumulative Precipitation (mm)")
    ax_b.set_title(r"(b) Cumulative Precipitation ($\sum P$)", loc="left", fontweight="bold")
    ax_b.set_ylim(0, 1050)
    add_pheno_clean(ax_b, show_labels=True)
    
    # (c) Cumulative Reference Evapotranspiration
    ax_c = axes[1, 0]
    ax_c.fill_between(ref_dates, clim_doy["cum_et0_p10"], clim_doy["cum_et0_p90"],
                      color="#cbd5e1", alpha=0.50, zorder=2)
    ax_c.plot(ref_dates, clim_doy["cum_et0_median"], color="black", lw=2.0, ls="--", zorder=3)
    for yr in target_ordered:
        if yr in daily_by_year:
            s = styles_cum[yr]
            ax_c.plot(ref_dates, daily_by_year[yr]["cum_et0"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax_c.set_ylabel(r"Cumulative $ET_0$ (mm)")
    ax_c.set_title(r"(c) Cumulative Evapotranspiration ($\sum ET_0$)", loc="left", fontweight="bold")
    ax_c.set_ylim(0, 1050)
    add_pheno_clean(ax_c, show_labels=False)
    
    # (d) Cumulative Climatic Water Budget
    ax_d = axes[1, 1]
    ax_d.axhline(0, color="black", lw=0.9, ls="-", zorder=2)
    ax_d.fill_between(ref_dates, clim_doy["cum_balance_p10"], clim_doy["cum_balance_p90"],
                      color="#cbd5e1", alpha=0.50, zorder=2)
    ax_d.plot(ref_dates, clim_doy["cum_balance_median"], color="black", lw=2.0, ls="--", zorder=3)
    for yr in target_ordered:
        if yr in daily_by_year:
            s = styles_cum[yr]
            ax_d.plot(ref_dates, daily_by_year[yr]["cum_balance"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax_d.set_ylabel(r"Cumulative Water Balance (mm)")
    ax_d.set_title(r"(d) Cumulative Climatic Water Budget ($\sum (P - ET_0)$)", loc="left", fontweight="bold")
    ax_d.set_ylim(-380, 240)
    add_pheno_clean(ax_d, show_labels=False)
    
    for ax in axes.flat:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(ref_dates.iloc[0], ref_dates.iloc[-1])
        
    # Harmonized 2-row legend (column-first ordering):
    # Col 0: Climatological Range, Climatological Median
    # Col 1: 2012 Shock, 1993 Shock
    # Col 2: 2016 Bumper, 1994 Bumper
    legend_handles = [
        h_range, h_med,
        line_handles[2012], line_handles[1993],
        line_handles[2016], line_handles[1994]
    ]
    legend_labels = [h.get_label() for h in legend_handles]
    
    fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3,
               frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.2)
    
    plt.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig_intra_seasonal_extremes.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig_intra_seasonal_extremes.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Figure 3.7 (Cumulative Extremes) saved to {pdf_path}")
    return daily_by_year, clim_doy, ref_dates, month_ticks, month_labels


# ==============================================================================
# FIGURE 3.8: Non-Cumulative Dynamic States (Harmonized 4 Archetypes)
# ==============================================================================
def generate_figure_shocks_noncumulative(daily_by_year, clim_doy, ref_dates, month_ticks, month_labels):
    print("Generating Figure 3.8 (Non-Cumulative 3x1: Harmonized 4 Archetypes)...")
    
    stages = [
        ("Sowing", pd.to_datetime("2021-04-01"), pd.to_datetime("2021-05-31"), "#f0f0f0"),
        ("Vegetative", pd.to_datetime("2021-06-01"), pd.to_datetime("2021-06-30"), "#e5f5e0"),
        ("Flowering", pd.to_datetime("2021-07-01"), pd.to_datetime("2021-07-31"), "#fee6ce"),
        ("Pod-Filling", pd.to_datetime("2021-08-01"), pd.to_datetime("2021-08-31"), "#fdd0a2"),
        ("Harvest", pd.to_datetime("2021-09-01"), pd.to_datetime("2021-10-31"), "#f0f0f0")
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    
    styles_shocks = {
        # Shocks (Warm/contrasting palette)
        2012: {"label": "2012 Shock: Flash Drought (-11.7%)", "color": "#d73027", "lw": 2.2, "ls": "-"},
        1993: {"label": "1993 Shock: Midwest Flood (-10.5%)", "color": "#762a83", "lw": 2.0, "ls": "-"},
        # Bumpers (Cool/harmonious palette)
        2016: {"label": "2016 Bumper: Optimum Buffer (+10.8%)", "color": "#006837", "lw": 2.2, "ls": "-"},
        1994: {"label": "1994 Bumper: Ideal Balance (+10.7%)", "color": "#2171b5", "lw": 2.0, "ls": "-"}
    }
    target_ordered = [2012, 1993, 2016, 1994]
    
    def add_pheno_subtle(ax):
        for name, start, end, col in stages:
            ax.axvspan(start, end, color=col, alpha=0.30, zorder=0)
            
    # (a) Atmospheric Evaporative Demand (VPD, 15-day rolling)
    ax0 = axes[0]
    add_pheno_subtle(ax0)
    h_range_0 = ax0.fill_between(ref_dates, clim_doy["vpd_roll15_p10"], clim_doy["vpd_roll15_p90"],
                                 color="#cbd5e1", alpha=0.50, label="Climatological Range (10th–90th %ile)", zorder=2)
    h_med_0 = ax0.plot(ref_dates, clim_doy["vpd_roll15_median"], color="black", lw=2.0, ls="--", label="Climatological Median", zorder=3)[0]
    
    line_handles_0 = {}
    for yr in target_ordered:
        s = styles_shocks[yr]
        line_handles_0[yr] = ax0.plot(ref_dates, daily_by_year[yr]["vpd_roll15"], color=s["color"], lw=s["lw"], ls=s["ls"], label=s["label"], zorder=4)[0]
    ax0.axhline(1.5, color="#b2182b", ls=":", lw=1.2, alpha=0.85)
    ax0.text(ref_dates[4], 1.54, r"Elevated Evaporative Demand Threshold ($1.5\text{ kPa}$)",
             color="#b2182b", fontsize=8.5, fontweight="bold")
    ax0.set_ylabel(r"15-Day Rolling $VPD$ (kPa)")
    ax0.set_title(r"(a) Atmospheric Evaporative Demand (15-Day Rolling $VPD$)", loc="left", fontweight="bold")
    ax0.set_ylim(0.35, 2.15)
    
    # (b) Root-Zone Soil Moisture
    ax1 = axes[1]
    add_pheno_subtle(ax1)
    ax1.fill_between(ref_dates, clim_doy["sm_root_p10"], clim_doy["sm_root_p90"],
                     color="#cbd5e1", alpha=0.50, label="Climatological Range (10th–90th %ile)", zorder=2)
    ax1.plot(ref_dates, clim_doy["sm_root_median"], color="black", lw=2.0, ls="--", label="Climatological Median", zorder=3)
    for yr in target_ordered:
        s = styles_shocks[yr]
        ax1.plot(ref_dates, daily_by_year[yr]["sm_root"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax1.axhline(0.38, color="#2b83ba", ls=":", lw=1.1, alpha=0.85)
    ax1.text(pd.to_datetime("2021-08-20"), 0.385, r"Regional Field Capacity ($0.38\ \mathrm{m}^3/\mathrm{m}^3$)",
             color="#2b83ba", fontsize=8.5, fontweight="bold", ha="right")
    ax1.axhline(0.18, color="#d73027", ls=":", lw=1.1, alpha=0.85)
    ax1.text(pd.to_datetime("2021-05-15"), 0.185, r"Incipient Wilting Stress Threshold ($0.18\ \mathrm{m}^3/\mathrm{m}^3$)",
             color="#d73027", fontsize=8.5, fontweight="bold")
    ax1.set_ylabel(r"Soil Moisture ($\text{m}^3/\text{m}^3$)")
    ax1.set_title(r"(b) Integrated Root-Zone Soil Moisture Dynamics ($SM_{\text{root}}$, $0$–$100\text{ cm}$)", loc="left", fontweight="bold")
    ax1.set_ylim(0.16, 0.44)
    
    # (c) Maximum Temperature (15-day rolling)
    ax2 = axes[2]
    add_pheno_subtle(ax2)
    ax2.fill_between(ref_dates, clim_doy["tmax_roll15_p10"], clim_doy["tmax_roll15_p90"],
                     color="#cbd5e1", alpha=0.50, label="Climatological Range (10th–90th %ile)", zorder=2)
    ax2.plot(ref_dates, clim_doy["tmax_roll15_median"], color="black", lw=2.0, ls="--", label="Climatological Median", zorder=3)
    for yr in target_ordered:
        s = styles_shocks[yr]
        ax2.plot(ref_dates, daily_by_year[yr]["tmax_roll15"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax2.axhline(30, color="#d73027", ls=":", lw=1.2, alpha=0.85)
    ax2.text(pd.to_datetime("2021-05-15"), 30.5, r"Reproductive Thermal Stress Ceiling ($30^\circ\text{C}$)",
             color="#d73027", fontsize=8.5, fontweight="bold")
    ax2.set_ylabel(r"15-Day Rolling $T_{\text{max}}$ ($^\circ$C)")
    ax2.set_title(r"(c) Thermal Regimes (15-Day Rolling Maximum Temperature)", loc="left", fontweight="bold")
    ax2.set_ylim(10, 36)
    
    for ax in axes:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(ref_dates.iloc[0], ref_dates.iloc[-1])
        
    # Harmonized 2-row bottom legend (column-first ordering):
    # Col 0: Climatological Range, Climatological Median
    # Col 1: 2012 Shock, 1993 Shock
    # Col 2: 2016 Bumper, 1994 Bumper
    legend_handles_shocks = [
        h_range_0, h_med_0,
        line_handles_0[2012], line_handles_0[1993],
        line_handles_0[2016], line_handles_0[1994]
    ]
    legend_labels_shocks = [h.get_label() for h in legend_handles_shocks]
    
    fig.legend(legend_handles_shocks, legend_labels_shocks, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3,
               frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.2)
        
    plt.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig_weather_shocks_footprint.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig_weather_shocks_footprint.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Figure 3.8 (Dynamic Shock Profiles) saved to {pdf_path}")


# ==============================================================================
# FIGURE 3.10: Phenological Climate-Yield Sensitivity / Correlations
# ==============================================================================
def generate_figure_climate_yield_sensitivity():
    print("Generating Figure 3.10 (Monthly Climate-Yield Correlations)...")
    
    monthly_records = []
    for yr in years_available:
        if yr < 1951:
            continue
        fpath = os.path.join(WEATHER_DIR, f"county_daily_{yr}.parquet")
        df_yr = pd.read_parquet(fpath, columns=[
            "county_fips", "date", "air_temperature_maximum", "total_precipitation",
            "vapor_pressure_deficit", "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3"
        ])
        df_yr["year"] = yr
        df_yr["month"] = df_yr["date"].dt.month
        df_season = df_yr[(df_yr["month"] >= 4) & (df_yr["month"] <= 10)].copy()
        df_season["rootzone_sm"] = (0.07 * df_season["volumetric_soil_water_layer_1"] +
                                    0.21 * df_season["volumetric_soil_water_layer_2"] +
                                    0.72 * df_season["volumetric_soil_water_layer_3"])
        
        df_mo = df_season.groupby(["county_fips", "year", "month"]).agg({
            "air_temperature_maximum": "mean",
            "total_precipitation": "sum",
            "vapor_pressure_deficit": "mean",
            "rootzone_sm": "mean"
        }).reset_index()
        monthly_records.append(df_mo)
        
    df_weather_monthly = pd.concat(monthly_records, ignore_index=True)
    
    df_merged = df_weather_monthly.merge(
        df_yields[["county_fips", "year", "yield_anomaly_pct"]],
        on=["county_fips", "year"],
        how="inner"
    )
    
    clim = df_merged.groupby(["county_fips", "month"])[[
        "air_temperature_maximum", "total_precipitation", "vapor_pressure_deficit", "rootzone_sm"
    ]].transform("mean")
    
    df_merged["tmax_anom"] = df_merged["air_temperature_maximum"] - clim["air_temperature_maximum"]
    df_merged["precip_anom"] = df_merged["total_precipitation"] - clim["total_precipitation"]
    df_merged["vpd_anom"] = df_merged["vapor_pressure_deficit"] - clim["vapor_pressure_deficit"]
    df_merged["sm_anom"] = df_merged["rootzone_sm"] - clim["rootzone_sm"]
    
    months = [4, 5, 6, 7, 8, 9, 10]
    month_names = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    
    corr_results = {var: [] for var in ["tmax", "precip", "vpd", "sm"]}
    
    for m in months:
        sub = df_merged[df_merged["month"] == m]
        r_tmax, _ = stats.pearsonr(sub["tmax_anom"], sub["yield_anomaly_pct"])
        r_prec, _ = stats.pearsonr(sub["precip_anom"], sub["yield_anomaly_pct"])
        r_vpd, _ = stats.pearsonr(sub["vpd_anom"], sub["yield_anomaly_pct"])
        r_sm, _ = stats.pearsonr(sub["sm_anom"], sub["yield_anomaly_pct"])
        
        corr_results["tmax"].append(r_tmax)
        corr_results["precip"].append(r_prec)
        corr_results["vpd"].append(r_vpd)
        corr_results["sm"].append(r_sm)
        
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    x = np.arange(len(months))
    w = 0.20
    
    rects1 = ax.bar(x - 1.5*w, corr_results["tmax"], w, label=r"Max Temperature ($T_{\text{max}}$)", color="#d95f02", alpha=0.90)
    rects2 = ax.bar(x - 0.5*w, corr_results["vpd"], w, label=r"Vapor Pressure Deficit ($VPD$)", color="#e7298a", alpha=0.85)
    rects3 = ax.bar(x + 0.5*w, corr_results["precip"], w, label="Precipitation ($P$)", color="#2b83ba", alpha=0.90)
    rects4 = ax.bar(x + 1.5*w, corr_results["sm"], w, label=r"Root-Zone Soil Moisture ($SM_{\text{root}}$)", color="#1b9e77", alpha=0.90)
    
    ax.axhline(0, color="black", lw=1.0, ls="-")
    ax.set_ylabel("Pearson Correlation ($r$) with Detrended Yield Anomaly", fontsize=10.5)
    ax.set_title("Exploratory Climate-Yield Sensitivity: Monthly Bivariate Correlations across Phenological Cycle", loc="left", fontweight="bold", fontsize=12.0)
    ax.set_xticks(x)
    ax.set_xticklabels(month_names, fontweight="bold", fontsize=10.5)
    ax.set_ylim(-0.58, 0.58)
    ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=9.2, facecolor="white", edgecolor="#cbd5e1")
    
    # Prominent, high-contrast phenological stage callout badges
    ax.text(0.5, -0.49, "Sowing Window\nExcess rain delay", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color="#334155",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.8))
    ax.text(2.0, -0.49, "Vegetative\nCanopy closure", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color="#334155",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.8))
    ax.text(3.0, -0.49, "Flowering (R1–R3)\nHeat penalty", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef2f2", edgecolor="#fca5a5", lw=1.0))
    ax.text(4.0, 0.45, "Pod-Filling (R4–R6)\nMoisture critical", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color="#065f46",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fdf4", edgecolor="#86efac", lw=1.0))
    ax.text(6.0, -0.49, "Harvest\nDesiccation window", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color="#334155",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.8))
    
    plt.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig_climate_yield_sensitivity.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig_climate_yield_sensitivity.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()
    print(f"Figure 3.10 (Sensitivity) saved to {pdf_path}")


if __name__ == "__main__":
    generate_table_3_5()
    # Note: generate_figure_3_6() excluded to streamline Chapter 3 narrative
    # generate_figure_3_6()
    daily_by_year, clim_doy, ref_dates, month_ticks, month_labels = generate_figure_intra_seasonal_cumulative()
    generate_figure_shocks_noncumulative(daily_by_year, clim_doy, ref_dates, month_ticks, month_labels)
    generate_figure_climate_yield_sensitivity()
    print("All Chapter 3 weather artifacts generated successfully!")
