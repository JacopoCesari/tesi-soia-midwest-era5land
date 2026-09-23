"""
Test script to generate:
1. fig_weather_seasonality (Intra-Annual Agro-Climatic Cycle & Water Deficit)
2. fig_weather_climatology (Multi-Decadal Climate Stress & 10 Worst Yield Shocks Nexus)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

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
FIGURES_DIR = os.path.join(ROOT_DIR, "output", "thesis", "figures")

parquet_files = sorted(glob.glob(os.path.join(WEATHER_DIR, "county_daily_*.parquet")))
years_available = [int(os.path.basename(f).replace("county_daily_", "").replace(".parquet", "")) for f in parquet_files]
print(f"Available weather years: {min(years_available)} to {max(years_available)} ({len(years_available)} years)")

df_yield = pd.read_csv(TARGET_CSV)
pmean = df_yield.groupby("year")["yield_bu_per_acre"].mean().reset_index()
z = np.polyfit(pmean["year"], pmean["yield_bu_per_acre"], 1)
pmean["trend"] = np.polyval(z, pmean["year"])
pmean["anomaly"] = pmean["yield_bu_per_acre"] - pmean["trend"]
pmean["anomaly_pct"] = (pmean["anomaly"] / pmean["trend"]) * 100.0
pmean["yoy_change_pct"] = (pmean["yield_bu_per_acre"].diff() / pmean["yield_bu_per_acre"].shift(1)) * 100.0

# Identify top 10 worst shock years within available weather years
avail_yields = pmean[pmean["year"].isin(years_available)].copy()
worst_10_anom = avail_yields.sort_values("anomaly_pct").head(10)
worst_years = sorted(worst_10_anom["year"].tolist())
print(f"Top 10 worst shock years identified: {worst_years}")


# ==============================================================================
# FIGURE 1: Intra-Annual Seasonality & Phenological Water Deficit
# ==============================================================================
def make_seasonality_figure():
    print("Generating fig_weather_seasonality...")
    # Load all weather daily records (DOY 91 to 304: April 1 to October 31)
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
    
    # Stage boundaries (DOY)
    # Planting: Apr 15 (105) to May 31 (151)
    # Vegetative: Jun 1 (152) to Jun 30 (181)
    # Flowering R1-R3: Jul 1 (182) to Jul 31 (212)
    # Pod Fill R4-R6: Aug 1 (213) to Aug 31 (243)
    # Maturity R7-R8: Sep 1 (244) to Oct 20 (293)
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
    ax0.text(ref_dates[5], 30.3, r"Stress Threshold: $T_{\text{max}} \geq 30^\circ\text{C}$", color="#d95f02", fontsize=8.5, fontweight="bold")
    ax0.axhline(35, color="#990000", ls="-.", lw=1.0, alpha=0.7)
    ax0.text(ref_dates[5], 35.3, r"Acute Threshold: $T_{\text{max}} \geq 35^\circ\text{C}$", color="#990000", fontsize=8.5, fontweight="bold")
    
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
        # Annotate 1988 deficit
        ax1.annotate("1988 Deficit: ~400 mm", xy=(ref_dates[135], cum_p_88[135]), xytext=(ref_dates[105], 430),
                     arrowprops=dict(arrowstyle="->", color="#990000", lw=1.5),
                     fontweight="bold", color="#990000", fontsize=9)
        
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
    print(f"Seasonality figure saved to {pdf_out} and {png_out}")


# ==============================================================================
# FIGURE 2: Multi-Decadal Stress Evolution & 10 Worst Yield Shocks Nexus
# ==============================================================================
def make_climatology_figure():
    print("Generating fig_weather_climatology (revised 2-panel with 10 worst yield shocks)...")
    
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
    
    # Merge with yield anomalies
    df_merged = pd.merge(df_clim, pmean[["year", "yield_bu_per_acre", "trend", "anomaly_pct", "yoy_change_pct"]], on="year", how="left")
    
    # 2-Panel Synchronized Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={'height_ratios': [1.1, 1.2]})
    
    # Identify top 10 worst yield shock years in this available range
    top10_worst = df_merged.sort_values("anomaly_pct").head(10).sort_values("year")
    top10_years = set(top10_worst["year"].tolist())
    
    # --- PANEL (a): Yield Anomaly (%) & 10 Worst Shock Years ---
    ax0 = axes[0]
    
    # Plot normal years vs worst 10 years
    for _, row in df_merged.iterrows():
        yr = int(row["year"])
        val = row["anomaly_pct"]
        if yr in top10_years:
            ax0.bar(yr, val, width=0.85, color="#b2182b", alpha=0.9, edgecolor="#67001f", lw=1.2)
        else:
            c = "#2166ac" if val >= 0 else "#92c5de"
            ax0.bar(yr, val, width=0.85, color=c, alpha=0.7, edgecolor="none")
            
    ax0.axhline(0, color="#333333", lw=1.0)
    ax0.axhline(-10, color="#b2182b", ls=":", lw=1.2, alpha=0.7, label="Severe Deficit Threshold (-10%)")
    
    # Annotate the 10 worst years
    for _, row in top10_worst.iterrows():
        yr = int(row["year"])
        val = row["anomaly_pct"]
        offset = -3.2 if val < -15 else -2.5
        ax0.text(yr, val + offset, f"{yr}\n({val:.1f}%)", ha="center", va="top",
                 fontsize=8.0, fontweight="bold", color="#67001f")
        
    ax0.set_ylabel(r"Yield Anomaly relative to Trend (%)")
    ax0.set_title(r"(a) Midwestern County Soybean Yield Detrended Anomalies (1950–Present) and the 10 Worst Shock Years",
                  loc="left", fontweight="bold")
    ax0.set_ylim(-34, 18)
    
    # Custom legend for panel a
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
    
    # Highlight bars for the 10 worst shock years
    for idx, row in df_merged.iterrows():
        yr = int(row["year"])
        if yr in top10_years:
            ax1.bar(yr, row["hd30_annual"], width=0.85, color="#e34a33", alpha=0.85, edgecolor="#b30000", lw=1.2)
            
    # 5-year rolling trend of heat days
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
        2003: ("2003: Aphids + Moisture Deficit", (2003, 40))
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
    
    # Legends
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax1_twin.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax1.set_xlim(min(df_merged["year"]) - 1, max(df_merged["year"]) + 1)
    
    plt.tight_layout()
    pdf_out = os.path.join(FIGURES_DIR, "fig_weather_climatology.pdf")
    png_out = os.path.join(FIGURES_DIR, "fig_weather_climatology.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out)
    plt.close()
    print(f"Climatology figure saved to {pdf_out} and {png_out}")

if __name__ == "__main__":
    make_seasonality_figure()
    make_climatology_figure()
    print("Done generating new test figures!")
