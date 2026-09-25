"""
Script to generate all Weather EDA tables and figures for Chapter 3: Data and Study Area.
Dynamically handles all available years (1950 to 2025).

Outputs:
- Tables in output/thesis/tables/
  - tab_weather_variables.tex
- Figures in output/thesis/figures/
  - fig_weather_seasonality.pdf and .png (Intra-Annual Agro-Climatic Cycle & Water Deficit)
  - fig_weather_climatology.pdf and .png (Multi-Decadal Climate Stress & 10 Worst Yield Shocks Nexus)
  - fig_weather_shocks_footprint.pdf and .png (Multi-Hazard Shock Signatures across 5 archetypes)
  - fig_climate_yield_sensitivity.pdf and .png (Monthly Climate-Yield Sensitivity)
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

ROOT_DIR = r"c:\Users\JacopoCesari-Aretésr\Desktop\Tesi"
WEATHER_DIR = os.path.join(ROOT_DIR, "work", "data", "interim", "county_daily_weather", "production")
TARGET_CSV = os.path.join(ROOT_DIR, "output", "data", "target", "soybean_yield_1951_2025.csv")
COUNTIES_CSV = os.path.join(ROOT_DIR, "output", "data", "auxiliary", "counties.csv")
TABLES_DIR = os.path.join(ROOT_DIR, "output", "thesis", "tables")
FIGURES_DIR = os.path.join(ROOT_DIR, "output", "thesis", "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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
# TABLE 3.4: Reanalysis and Agrometeorological Variables Specification
# ==============================================================================
def generate_table_3_4():
    print("Generating Table 3.4 (Weather Variables)...")
    tex_content = r"""\begin{table}[htbp]
\centering
\small
\caption{ERA5-Land surface meteorological variables and derived daily agrometeorological indicators.}
\label{tab:weather_variables}
\begin{tabularx}{\textwidth}{llllX}
\toprule
\textbf{Variable Category} & \textbf{Symbol} & \textbf{Physical Variable} & \textbf{Unit} & \textbf{Agronomic Rationale and Formulation} \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: Primary Atmospheric Reanalysis Fields (ECMWF ERA5-Land)}} \\
Thermal Dynamics & $T_{\text{mean}}$ & Daily mean 2m temperature & $^\circ$C & Thermal accumulation and phenological progression pacing. \\
& $T_{\text{max}}$ & Daily maximum 2m temperature & $^\circ$C & Acute heat stress; canopy overheating ($>30^\circ$C). \\
& $T_{\text{min}}$ & Daily minimum 2m temperature & $^\circ$C & Nighttime respiration losses and autumnal frost detection. \\
& $T_{\text{dew}}$ & Daily mean 2m dewpoint & $^\circ$C & Atmospheric moisture content; vapor pressure baseline. \\
Hydrological Fluxes & $P$ & Total daily precipitation & mm/day & Surface water supply, recharge, and waterlogging events. \\
& $PE$ & Potential evaporation & mm/day & Atmospheric drying capacity from land surface. \\
Radiative Forcing & $SSRD$ & Surface solar radiation downwards & $\text{MJ}/\text{m}^2/\text{d}$ & Photosynthetically active radiation driving biomass assimilation. \\
Pedological Moisture & $SM_1$ & Volumetric soil water (0--7 cm) & $\text{m}^3/\text{m}^3$ & Topsoil moisture controlling seed emergence and early root set. \\
& $SM_2$ & Volumetric soil water (7--28 cm) & $\text{m}^3/\text{m}^3$ & Upper root-zone moisture buffering early vegetative growth. \\
& $SM_3$ & Volumetric soil water (28--100 cm) & $\text{m}^3/\text{m}^3$ & Deep root-zone reservoir sustaining pod-fill during dry spells. \\
\midrule
\multicolumn{5}{l}{\textit{Panel B: Derived Agrometeorological and Bioclimatic Indicators}} \\
Evaporative Demand & $VPD$ & Vapor pressure deficit & kPa & $e_s(T_{\text{mean}}) - e_a(T_{\text{dew}})$; atmospheric thirst driving stomatal closure. \\
Crop Water Demand & $ET_0$ & FAO-56 Penman-Monteith & mm/day & Standardized grass reference evapotranspiration \citep{Allen1998}. \\
Climatic Balance & $P - ET_0$ & Effective atmospheric water balance & mm/day & Net daily atmospheric moisture deficit or surplus. \\
Thermal Pacing & $GDD$ & Growing Degree Days ($10$--$30^\circ$C) & $^\circ$C$\cdot$d & $\max(0, \min(T_{\text{mean}}, 30) - 10)$; vegetative and reproductive pacing. \\
Extreme Heat Stress & $HD_{30}$ & Severe heat day indicator & binary & $\mathbb{I}(T_{\text{max}} \ge 30^\circ\text{C})$; threshold for floral abortion and pollen sterility. \\
Extreme Heat Stress & $HD_{35}$ & Acute heat day indicator & binary & $\mathbb{I}(T_{\text{max}} \ge 35^\circ\text{C})$; acute leaf desiccation and irreversible pod drop. \\
\bottomrule
\end{tabularx}
\end{table}
"""
    with open(os.path.join(TABLES_DIR, "tab_weather_variables.tex"), "w", encoding="utf-8") as f:
        f.write(tex_content)
    print("Table 3.4 saved.")


# ==============================================================================
# FIGURE 3.5: Intra-Annual Agro-Climatic Seasonality & Water Deficit
# ==============================================================================
def generate_figure_3_5():
    print("Generating Figure 3.5 (Intra-Annual Agro-Climatic Seasonality)...")
    
    daily_records = []
    daily_1988 = []
    
    for yr in years_available:
        fpath = os.path.join(WEATHER_DIR, f"county_daily_{yr}.parquet")
        df_yr = pd.read_parquet(fpath, columns=[
            "date", "air_temperature_mean", "air_temperature_maximum", "air_temperature_minimum",
            "total_precipitation", "et0_fao56", "vapor_pressure_deficit",
            "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3"
        ])
        df_yr["doy"] = df_yr["date"].dt.dayofyear
        df_yr = df_yr[(df_yr["doy"] >= 91) & (df_yr["doy"] <= 304)]
        
        # Panel-wide daily mean across 135 counties
        df_d = df_yr.groupby("doy").mean().reset_index()
        df_d["year"] = yr
        daily_records.append(df_d)
        
        if yr == 1988:
            daily_1988 = df_d.copy()
            
    df_all = pd.concat(daily_records, ignore_index=True)
    
    def p10(x):
        return np.percentile(x, 10)
    def p90(x):
        return np.percentile(x, 90)

    # Climatological stats by DOY
    doy_stats = df_all.groupby("doy").agg({
        "air_temperature_maximum": ["mean", p10, p90],
        "air_temperature_mean": ["mean"],
        "air_temperature_minimum": ["mean", p10, p90],
        "total_precipitation": ["mean"],
        "et0_fao56": ["mean"],
        "vapor_pressure_deficit": ["mean"],
        "volumetric_soil_water_layer_1": ["mean"],
        "volumetric_soil_water_layer_2": ["mean"],
        "volumetric_soil_water_layer_3": ["mean"]
    })
    doy_stats.columns = ['_'.join(c).strip() for c in doy_stats.columns.values]
    doy_stats = doy_stats.reset_index()
    
    ref_dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(doy_stats["doy"] - 1, unit="D")
    
    # Cumulative water balance
    cum_p = np.cumsum(doy_stats["total_precipitation_mean"])
    cum_et0 = np.cumsum(doy_stats["et0_fao56_mean"])
    
    # 1988 cumulative
    cum_p_88 = np.cumsum(daily_1988["total_precipitation"].values) if len(daily_1988) > 0 else None
    cum_et0_88 = np.cumsum(daily_1988["et0_fao56"].values) if len(daily_1988) > 0 else None
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    
    stages = [
        ("Sowing & Emergence", 105, 151, "#e5f5e0"),
        ("Vegetative Dev.", 151, 182, "#edf8e9"),
        ("Flowering (R1–R3)", 182, 213, "#fee6ce"),
        ("Pod Filling (R4–R6)", 213, 244, "#fdd0a2"),
        ("Maturity & Harvest", 244, 298, "#f0f0f0")
    ]
    
    for ax in axes:
        for name, start, end, col in stages:
            d_start = pd.to_datetime("2021-01-01") + pd.to_timedelta(start - 1, unit="D")
            d_end = pd.to_datetime("2021-01-01") + pd.to_timedelta(end - 1, unit="D")
            ax.axvspan(d_start, d_end, color=col, alpha=0.45, lw=0)
            
    # Stage labels at top of panel 0
    y_top = 37.0
    for name, start, end, _ in stages:
        d_mid = pd.to_datetime("2021-01-01") + pd.to_timedelta((start + end) / 2 - 1, unit="D")
        axes[0].text(d_mid, y_top, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#333333")
        
    # --- PANEL (a): Thermal Dynamics ---
    ax0 = axes[0]
    ax0.fill_between(ref_dates, doy_stats["air_temperature_maximum_p10"], doy_stats["air_temperature_maximum_p90"],
                     color="#d95f02", alpha=0.18, label=r"$T_{\text{max}}$ 10th–90th percentile envelope")
    ax0.fill_between(ref_dates, doy_stats["air_temperature_minimum_p10"], doy_stats["air_temperature_minimum_p90"],
                     color="#2b83ba", alpha=0.18, label=r"$T_{\text{min}}$ 10th–90th percentile envelope")
    
    ax0.plot(ref_dates, doy_stats["air_temperature_maximum_mean"], color="#d95f02", lw=2.2, label=r"Climatological Mean $T_{\text{max}}$")
    ax0.plot(ref_dates, doy_stats["air_temperature_mean_mean"], color="#4daf4a", lw=1.5, ls="--", label=r"Climatological Mean $T_{\text{mean}}$")
    ax0.plot(ref_dates, doy_stats["air_temperature_minimum_mean"], color="#2b83ba", lw=2.0, label=r"Climatological Mean $T_{\text{min}}$")
    
    # Overplot 1988 extreme heat
    if len(daily_1988) > 0:
        ax0.plot(ref_dates, daily_1988["air_temperature_maximum"].values, color="#990000", lw=1.8, ls=":",
                 label=r"1988 Severe Drought $T_{\text{max}}$ (Record Peak)")
        
    # Thresholds
    ax0.axhline(30, color="#d95f02", ls="-.", lw=1.0, alpha=0.7)
    ax0.text(ref_dates[2], 30.3, r"Stress Threshold: $T_{\text{max}} \geq 30^\circ\text{C}$", color="#d95f02", fontsize=8.2, fontweight="bold")
    ax0.axhline(35, color="#990000", ls="-.", lw=1.0, alpha=0.7)
    ax0.text(ref_dates[2], 35.3, r"Acute Damage Threshold: $T_{\text{max}} \geq 35^\circ\text{C}$", color="#990000", fontsize=8.2, fontweight="bold")
    
    ax0.set_ylabel(r"Temperature ($^\circ$C)")
    ax0.set_title(r"(a) Thermal Dynamics and Phenological Calendar Across the Growing Cycle (April 1 – October 31)", loc="left", fontweight="bold")
    ax0.set_ylim(2, 39)
    ax0.legend(loc="lower right", frameon=True, ncol=2, fontsize=8.5)
    
    # --- PANEL (b): Hydrological Balance & Water Deficit ---
    ax1 = axes[1]
    ax1.plot(ref_dates, cum_p, color="#08519c", lw=2.5, label="Cumulative Precipitation ($P$) [Normal Climatology]")
    ax1.plot(ref_dates, cum_et0, color="#d95f02", lw=2.5, label="Cumulative Reference Evapotranspiration ($ET_0$) [Normal]")
    
    # Fill deficit between ET0 and P
    ax1.fill_between(ref_dates, cum_p, cum_et0, where=(cum_et0 >= cum_p), color="#fee0d2", alpha=0.6,
                     label=r"Midsummer Atmospheric Water Deficit ($ET_0 > P$)")
    ax1.fill_between(ref_dates, cum_p, cum_et0, where=(cum_et0 < cum_p), color="#c6dbef", alpha=0.5,
                     label=r"Early Spring Moisture Recharge ($P > ET_0$)")
    
    # Overplot 1988
    if cum_p_88 is not None:
        ax1.plot(ref_dates, cum_p_88, color="#08519c", lw=1.8, ls=":", label="1988 Cumulative $P$ (Stalled Rainfall)")
        ax1.plot(ref_dates, cum_et0_88, color="#d95f02", lw=1.8, ls=":", label="1988 Cumulative $ET_0$ (Elevated Demand)")
        # Annotate 1988 deficit with clear positioning
        ax1.annotate("1988 Deficit: ~400 mm", xy=(ref_dates[135], cum_p_88[135]), xytext=(ref_dates[90], 520),
                     arrowprops=dict(arrowstyle="->", color="#990000", lw=1.5),
                     fontweight="bold", color="#990000", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#990000", alpha=0.85))
        
    ax1.set_ylabel("Cumulative Depth (mm)")
    ax1.set_title(r"(b) Hydrological Budget: Cumulative Atmospheric Water Deficit ($P$ vs. $ET_0$)", loc="left", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)
    
    # --- PANEL (c): Soil Moisture Depletion Across Horizons ---
    ax2 = axes[2]
    ax2.plot(ref_dates, doy_stats["volumetric_soil_water_layer_1_mean"], color="#74c476", lw=2.0,
             label=r"Layer 1 (0--7 cm, Topsoil): Rapid Evaporative Fluctuations")
    ax2.plot(ref_dates, doy_stats["volumetric_soil_water_layer_2_mean"], color="#31a354", lw=2.2,
             label=r"Layer 2 (7--28 cm, Upper Root-Zone): Vegetative Extraction")
    ax2.plot(ref_dates, doy_stats["volumetric_soil_water_layer_3_mean"], color="#08519c", lw=2.5,
             label=r"Layer 3 (28--100 cm, Deep Root-Zone): Critical Pod-Fill Moisture Buffer")
    
    if len(daily_1988) > 0:
        ax2.plot(ref_dates, daily_1988["volumetric_soil_water_layer_3"].values, color="#990000", lw=1.8, ls=":",
                 label="1988 Layer 3 (Deep Moisture Collapse)")
        
    ax2.set_ylabel(r"Soil Moisture ($\text{m}^3/\text{m}^3$)")
    ax2.set_xlabel("Calendar Date Across Growing Cycle")
    ax2.set_title(r"(c) Edaphic Moisture Depletion Across Rooting Horizons ($SM_1$, $SM_2$, $SM_3$)", loc="left", fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=8.5)
    ax2.set_ylim(0.12, 0.44)
    
    # Format X axis dates
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.set_xlim(ref_dates[0], ref_dates.iloc[-1])
    
    plt.tight_layout()
    pdf_out = os.path.join(FIGURES_DIR, "fig_weather_seasonality.pdf")
    png_out = os.path.join(FIGURES_DIR, "fig_weather_seasonality.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out)
    plt.close()
    print(f"Figure 3.5 (Seasonality) saved to {pdf_out}")


# ==============================================================================
# FIGURE 3.6: Multi-Decadal Stress Evolution & 10 Worst Yield Shocks Nexus
# ==============================================================================
def generate_figure_3_6():
    print("Generating Figure 3.6 (Multi-Decadal Stress Evolution & 10 Worst Yield Shocks)...")
    
    annual_metrics = []
    for yr in years_available:
        fpath = os.path.join(WEATHER_DIR, f"county_daily_{yr}.parquet")
        df_yr = pd.read_parquet(fpath, columns=[
            "county_fips", "date", "air_temperature_maximum", "total_precipitation",
            "vapor_pressure_deficit", "heat_day_30"
        ])
        df_yr["month"] = df_yr["date"].dt.month
        
        # Summer (Jul-Aug) VPD
        summer = df_yr[df_yr["month"].isin([7, 8])]
        vpd_summer = summer["vapor_pressure_deficit"].mean()
        
        # Annual heat stress days (>30°C) per county
        hd30_annual = df_yr.groupby("county_fips")["heat_day_30"].sum().mean()
        
        annual_metrics.append({
            "year": yr,
            "vpd_summer": vpd_summer,
            "hd30_annual": hd30_annual
        })
        
    df_clim = pd.DataFrame(annual_metrics).sort_values("year").reset_index(drop=True)
    df_merged = pd.merge(df_clim, pmean[["year", "yield_bu_per_acre", "trend", "anomaly_pct", "yoy_change_pct"]], on="year", how="left")
    
    # 2-Panel Synchronized Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={'height_ratios': [1.1, 1.2]})
    
    top10_worst = df_merged.sort_values("anomaly_pct").head(10).sort_values("year")
    top10_years = set(top10_worst["year"].tolist())
    
    # --- PANEL (a): Yield Anomaly (%) & 10 Worst Shock Years ---
    ax0 = axes[0]
    
    for _, row in df_merged.iterrows():
        yr = int(row["year"])
        val = row["anomaly_pct"]
        if yr in top10_years:
            ax0.bar(yr, val, width=0.85, color="#b2182b", alpha=0.9, edgecolor="#67001f", lw=1.2)
        else:
            c = "#2166ac" if val >= 0 else "#92c5de"
            ax0.bar(yr, val, width=0.85, color=c, alpha=0.7, edgecolor="none")
            
    ax0.axhline(0, color="#333333", lw=1.0)
    ax0.axhline(-10, color="#b2182b", ls=":", lw=1.2, alpha=0.7)
    
    # Annotate the 10 worst years with staggered positions to prevent overlap
    for _, row in top10_worst.iterrows():
        yr = int(row["year"])
        val = row["anomaly_pct"]
        # Stagger 1983 vs 1984
        if yr == 1983:
            ax0.text(yr - 0.4, val - 3.8, f"{yr}\n({val:.1f}%)", ha="center", va="top",
                     fontsize=7.8, fontweight="bold", color="#67001f")
        elif yr == 1984:
            ax0.text(yr + 0.5, val - 1.8, f"{yr}\n({val:.1f}%)", ha="center", va="top",
                     fontsize=7.8, fontweight="bold", color="#67001f")
        else:
            offset = -3.2 if val < -15 else -2.5
            ax0.text(yr, val + offset, f"{yr}\n({val:.1f}%)", ha="center", va="top",
                     fontsize=8.0, fontweight="bold", color="#67001f")
        
    ax0.set_ylabel(r"Yield Anomaly relative to Trend (%)")
    ax0.set_title(r"(a) Midwestern County Soybean Yield Detrended Anomalies (1950–Present) and the 10 Worst Shock Years",
                  loc="left", fontweight="bold")
    ax0.set_ylim(-34, 18)
    
    from matplotlib.patches import Patch
    legend_elements_a = [
        Patch(facecolor="#2166ac", alpha=0.7, label="Positive Yield Anomaly (> Trend)"),
        Patch(facecolor="#92c5de", alpha=0.7, label="Moderate Negative Anomaly (0% to -10%)"),
        Patch(facecolor="#b2182b", alpha=0.9, label="10 Worst Historical Shock Years (Catastrophic Loss)")
    ]
    ax0.legend(handles=legend_elements_a, loc="upper right", frameon=True, fontsize=9)
    
    # --- PANEL (b): Heat Stress Days (HD30) & Summer VPD ---
    ax1 = axes[1]
    
    # Bars for Heat Days
    bars = ax1.bar(df_merged["year"], df_merged["hd30_annual"], width=0.85, color="#fdbb84", alpha=0.65,
                   label=r"Annual Extreme Heat Days ($HD_{30}$, $T_{\text{max}} \geq 30^\circ$C)")
    
    for idx, row in df_merged.iterrows():
        yr = int(row["year"])
        if yr in top10_years:
            ax1.bar(yr, row["hd30_annual"], width=0.85, color="#e34a33", alpha=0.85, edgecolor="#b30000", lw=1.2)
            
    # 5-year rolling trend
    ax1.plot(df_merged["year"], df_merged["hd30_annual"].rolling(5, center=True).mean(),
             color="#b30000", lw=2.2, label=r"Heat Days (5-yr rolling mean)")
    
    # Secondary Y-axis for summer VPD
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_merged["year"], df_merged["vpd_summer"], color="#542788", lw=2.2, ls="--", marker="o", markersize=3.5,
                  label=r"Summer $VPD$ (Jul–Aug Mean, kPa)")
    
    # Vertical guideline bands linking worst 10 years across both panels
    for yr in top10_years:
        ax0.axvspan(yr - 0.45, yr + 0.45, color="#fee8c8", alpha=0.35, zorder=-1)
        ax1.axvspan(yr - 0.45, yr + 0.45, color="#fee8c8", alpha=0.35, zorder=-1)
        
    # Annotate specific atmospheric mechanisms on panel b
    shock_annotations = {
        1953: ("1953: Heatwave (51d)", (1953, 56)),
        1974: ("1974: Wet Sowing & Freeze", (1974, 22)),
        1983: ("1983: Pod-Fill Drought (51d)", (1983, 58)),
        1988: ("1988: Record Heat & Drought (56d)", (1988, 63)),
        1993: ("1993: Record Flood / Anoxia", (1993, 24)),
        2003: ("2003: Late-Season Deficit", (2003, 40))
    }
    for yr, (text, (x, y)) in shock_annotations.items():
        if yr in df_merged["year"].values:
            ax1.text(x, y, text, ha="center", fontsize=8.0, fontweight="bold", color="#67001f",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#e34a33", alpha=0.85, lw=0.8))
            
    ax1.set_ylabel(r"Days / Year ($T_{\text{max}} \geq 30^\circ$C)")
    ax1_twin.set_ylabel(r"Summer $VPD$ (Jul–Aug, kPa)")
    ax1.set_xlabel("Crop Year")
    ax1.set_title(r"(b) Atmospheric Heat and Evaporative Demand: Co-Occurrence of $HD_{30}$ Spikes and $VPD$ Surges",
                  loc="left", fontweight="bold")
    ax1.set_ylim(0, 75)
    ax1_twin.set_ylim(0.65, 1.35)
    ax1_twin.grid(False)
    
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax1_twin.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax1.set_xlim(min(df_merged["year"]) - 1, max(df_merged["year"]) + 1)
    
    plt.tight_layout()
    pdf_out = os.path.join(FIGURES_DIR, "fig_weather_climatology.pdf")
    png_out = os.path.join(FIGURES_DIR, "fig_weather_climatology.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out)
    plt.close()
    print(f"Figure 3.6 (Climatology & Yield Shocks) saved to {pdf_out}")


# ==============================================================================
# FIGURE 3.7 (or 3.6): Cumulative Intra-Seasonal Phenological Progression (2x2)
# ==============================================================================
def generate_figure_intra_seasonal_cumulative():
    print("Generating Cumulative Intra-Seasonal Trajectories (fig_intra_seasonal_extremes)...")
    
    stages = [
        ("Sowing (Apr--May)", pd.to_datetime("2021-04-01"), pd.to_datetime("2021-05-31"), "#f0f0f0"),
        ("Vegetative (Jun)", pd.to_datetime("2021-06-01"), pd.to_datetime("2021-06-30"), "#e5f5e0"),
        ("Flowering (Jul)", pd.to_datetime("2021-07-01"), pd.to_datetime("2021-07-31"), "#fee6ce"),
        ("Pod-Fill (Aug)", pd.to_datetime("2021-08-01"), pd.to_datetime("2021-08-31"), "#fdd0a2"),
        ("Maturity (Sep--Oct)", pd.to_datetime("2021-09-01"), pd.to_datetime("2021-10-31"), "#f0f0f0")
    ]
    
    target_years = [1988, 2003, 1974, 1983, 1993, 1952, 2021, 2016, 1994]
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
        
        # Depth-weighted root-zone soil moisture (Eq 226)
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), sharex=True)
    
    def add_pheno_clean(ax, show_labels=False):
        for name, start, end, col in stages:
            ax.axvspan(start, end, color=col, alpha=0.40, zorder=0)
        if show_labels:
            y_lim = ax.get_ylim()
            y_pos = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.94
            for name, start, end, _ in stages:
                mid = start + (end - start) / 2
                ax.text(mid, y_pos, name, ha="center", va="center", fontsize=8.0, fontweight="bold",
                        color="#444444", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
                
    styles_cum = {
        1988: {"label": "1988: Severe Heat & Drought (-25.6%)", "color": "#d73027", "lw": 2.2, "ls": "-"},
        2003: {"label": "2003: Late-Season Deficit (-21.6%)", "color": "#fc8d59", "lw": 1.8, "ls": "--"},
        1974: {"label": "1974: Wet Sowing & Freeze (-19.7%)", "color": "#762a83", "lw": 1.8, "ls": "-."},
        1983: {"label": "1983: Pod-Fill Thermal Drought (-11.4%)", "color": "#e08214", "lw": 1.6, "ls": ":"},
        1993: {"label": "1993: Great Midwest Flood (-10.5%)", "color": "#4575b4", "lw": 2.2, "ls": "-"},
        1952: {"label": "1952: Early Favorable Bumper (+11.9%)", "color": "#1a9850", "lw": 1.8, "ls": "-."},
        2021: {"label": "2021: Modern Favorable Bumper (+10.8%)", "color": "#006837", "lw": 2.0, "ls": "--"},
        2016: {"label": "2016: Modern Benchmark Bumper (+10.8%)", "color": "#41ab5d", "lw": 2.0, "ls": "-"},
        1994: {"label": "1994: Climatic Optimum Bumper (+10.7%)", "color": "#238443", "lw": 1.8, "ls": ":"}
    }
    
    # (a) Cumulative Heat Stress Days
    ax_a = axes[0, 0]
    ax_a.plot(ref_dates, clim_doy["cum_hd30_median"], color="black", lw=2.2, ls=":", label="Climatological Normal", zorder=3)
    for yr in target_years:
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
    for yr in target_years:
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
    for yr in target_years:
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
    for yr in target_years:
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
        
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5,
               frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.2)
    
    plt.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig_intra_seasonal_extremes.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig_intra_seasonal_extremes.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Figure 3.6/3.7 (Cumulative Extremes) saved to {pdf_path}")
    return daily_by_year, clim_doy, ref_dates, month_ticks, month_labels


# ==============================================================================
# FIGURE 3.8 (or 3.7): Non-Cumulative Dynamic States: Top 3 Best vs Top 3 Worst (3x1)
# ==============================================================================
def generate_figure_shocks_noncumulative(daily_by_year, clim_doy, ref_dates, month_ticks, month_labels):
    print("Generating Figure 3.7/3.8 (Non-Cumulative 3x1: Top 3 Best vs Top 3 Worst)...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    
    styles_shocks = {
        1988: {"label": "1988: Severe Heat & Drought (-25.6%)", "color": "#d73027", "lw": 2.2, "ls": "-"},
        2003: {"label": "2003: Late-Season Moisture Deficit (-21.6%)", "color": "#e08214", "lw": 2.0, "ls": "--"},
        1974: {"label": "1974: Wet Sowing Delay & Autumn Freeze (-19.7%)", "color": "#762a83", "lw": 2.0, "ls": "-."},
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
    pdf_path = os.path.join(FIGURES_DIR, "fig_weather_shocks_footprint.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig_weather_shocks_footprint.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Figure 3.7/3.8 (Shock Footprints) saved to {pdf_path}")


# ==============================================================================
# FIGURE 3.9 (or 3.8): Phenological Climate-Yield Sensitivity / Correlations
# ==============================================================================
def generate_figure_climate_yield_sensitivity():
    print("Generating Figure 3.8/3.9 (Monthly Climate-Yield Correlations)...")
    
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
        df_yr = df_yr[(df_yr["month"] >= 4) & (df_yr["month"] <= 10)].copy()
        df_yr["rootzone_sm"] = (0.07 * df_yr["volumetric_soil_water_layer_1"] +
                                0.21 * df_yr["volumetric_soil_water_layer_2"] +
                                0.72 * df_yr["volumetric_soil_water_layer_3"])
        
        df_mo = df_yr.groupby(["county_fips", "year", "month"]).agg({
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
        
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(months))
    w = 0.20
    
    rects1 = ax.bar(x - 1.5*w, corr_results["tmax"], w, label=r"Max Temperature ($T_{\text{max}}$)", color="#d95f02", alpha=0.9)
    rects2 = ax.bar(x - 0.5*w, corr_results["vpd"], w, label=r"Vapor Pressure Deficit ($VPD$)", color="#e7298a", alpha=0.85)
    rects3 = ax.bar(x + 0.5*w, corr_results["precip"], w, label="Precipitation ($P$)", color="#2b83ba", alpha=0.9)
    rects4 = ax.bar(x + 1.5*w, corr_results["sm"], w, label=r"Root-Zone Soil Water ($SM$)", color="#1b9e77", alpha=0.9)
    
    ax.axhline(0, color="black", lw=1.0, ls="-")
    ax.set_ylabel("Pearson Correlation ($r$) with Detrended Yield Anomaly")
    ax.set_title("Exploratory Climate-Yield Sensitivity: Monthly Bivariate Correlations across Phenological Cycle", loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(month_names, fontweight="bold")
    ax.set_ylim(-0.55, 0.55)
    ax.legend(loc="upper left", frameon=True, ncol=2)
    
    ax.text(0, -0.48, "Sowing\nExcess rain", ha="center", fontsize=8.5, fontstyle="italic", color="#555")
    ax.text(2, -0.48, "Vegetative\nCanopy closure", ha="center", fontsize=8.5, fontstyle="italic", color="#555")
    ax.text(3, -0.48, "Flowering (R1-R3)\nHeat penalty", ha="center", fontsize=8.5, fontstyle="italic", color="#b2182b", fontweight="bold")
    ax.text(4, 0.44, "Pod-Fill (R4-R6)\nMoisture critical", ha="center", fontsize=8.5, fontstyle="italic", color="#005a32", fontweight="bold")
    ax.text(6, -0.48, "Harvest\nDesiccation", ha="center", fontsize=8.5, fontstyle="italic", color="#555")
    
    plt.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig_climate_yield_sensitivity.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig_climate_yield_sensitivity.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()
    print(f"Figure 3.8/3.9 (Sensitivity) saved to {pdf_path}")


if __name__ == "__main__":
    generate_table_3_4()
    generate_figure_3_6()
    daily_by_year, clim_doy, ref_dates, month_ticks, month_labels = generate_figure_intra_seasonal_cumulative()
    generate_figure_shocks_noncumulative(daily_by_year, clim_doy, ref_dates, month_ticks, month_labels)
    generate_figure_climate_yield_sensitivity()
    print("All Chapter 3 weather artifacts generated successfully!")
