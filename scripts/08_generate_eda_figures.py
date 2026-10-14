"""
08_generate_eda_figures.py

Generates exploratory data analysis figures for Chapter 3 (Data & Study Area):
1. Figure 3.1: Geographic Distribution of the 135 Study Counties (fig_study_area_map.pdf).
2. Figure 3.2: 75-Year Regional Soybean Yield Trajectory & Secular Trend (fig_yield_trajectory.pdf).
3. Figure 3.8: Monthly Agrometeorological Sensitivity Profile (fig_climate_yield_sensitivity.pdf).

Outputs:
- output/thesis/figures/fig_study_area_map.pdf / .png
- output/thesis/figures/fig_yield_trajectory.pdf / .png
- output/thesis/figures/fig_climate_yield_sensitivity.pdf / .png
"""

import glob
from pathlib import Path
import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
COUNTIES_CSV = ROOT / "data" / "auxiliary" / "counties.csv"
SHAPEFILE = ROOT / "data" / "raw" / "census_counties" / "cb_2020_us_county_500k.shp"
WEATHER_DIR = ROOT / "data" / "weather" / "daily_counties"
FIGURES_DIR = ROOT / "output" / "thesis" / "figures"


def generate_figure_3_1(df_counties):
    """Figure 3.1: Study Area Cartographic Map."""
    print("Generating Figure 3.1: Study Area Map...")
    gdf = gpd.read_file(SHAPEFILE)
    
    # 6 Corn Belt states: IL (17), IN (18), IA (19), MN (27), MO (29), OH (39)
    midwest_fips = ["17", "18", "19", "27", "29", "39"]
    gdf_midwest = gdf[gdf["STATEFP"].isin(midwest_fips)].copy()
    
    selected_fips = set(df_counties["county_fips"].astype(str).str.zfill(5))
    gdf_midwest["status"] = np.where(gdf_midwest["GEOID"].isin(selected_fips), "Selected", "Other")

    gdf_midwest_proj = gdf_midwest.to_crs(epsg=5070)
    gdf_states = gdf_midwest_proj.dissolve(by="STATE_NAME")

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_midwest_proj[gdf_midwest_proj["status"] == "Other"].plot(ax=ax, color="#f5f5f5", edgecolor="#d9d9d9", linewidth=0.4)
    gdf_midwest_proj[gdf_midwest_proj["status"] == "Selected"].plot(ax=ax, color="#02818a", edgecolor="#01464c", linewidth=0.8)
    gdf_states.plot(ax=ax, facecolor="none", edgecolor="#252525", linewidth=1.2)

    # State center labels
    state_coords = {
        "Illinois": (-89.2, 40.0), "Indiana": (-86.1, 39.8), "Iowa": (-93.5, 42.0),
        "Minnesota": (-94.5, 45.8), "Missouri": (-92.5, 38.3), "Ohio": (-82.7, 40.2)
    }
    pts = pd.DataFrame([{"state": k, "lon": v[0], "lat": v[1]} for k, v in state_coords.items()])
    pts_gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=4326).to_crs(epsg=5070)
    for _, row in pts_gdf.iterrows():
        ax.annotate(row["state"].upper(), xy=(row.geometry.x, row.geometry.y), fontsize=11, fontweight="bold", color="#252525", alpha=0.6, ha="center")

    legend_elements = [
        Patch(facecolor="#02818a", edgecolor="#01464c", label="Balanced Panel Counties (N=135)"),
        Patch(facecolor="#f5f5f5", edgecolor="#d9d9d9", label="Other Regional Counties"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax.set_title("Geographic Distribution of Balanced Study Counties (1951–2025)", fontsize=12.5, fontweight="bold", pad=12)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_study_area_map.pdf")
    fig.savefig(FIGURES_DIR / "fig_study_area_map.png")
    plt.close(fig)


def generate_figure_3_2(df_yield):
    """Figure 3.2: 75-Year Panel Yield Progression, Secular Trend, and Historical Shocks."""
    print("Generating Figure 3.2: Long-Term Yield Trajectory...")
    annual = df_yield.groupby("year")["yield_bu_per_acre"].agg(
        mean="mean", q05=lambda s: s.quantile(0.05), q95=lambda s: s.quantile(0.95)
    ).reset_index()

    slope, intercept, r_val, _, _ = stats.linregress(annual["year"], annual["mean"])
    annual["trend"] = slope * annual["year"] + intercept

    fig, ax = plt.subplots(figsize=(12, 6.2))

    # Cross-county trajectories and 90% dispersion envelope
    for _, grp in df_yield.groupby("county_fips"):
        ax.plot(grp["year"], grp["yield_bu_per_acre"], color="#64748b", alpha=0.15, linewidth=0.7, zorder=1)
    
    ax.fill_between(annual["year"], annual["q05"], annual["q95"], color="#bae6fd", alpha=0.45, label="90% Cross-County Dispersion (5th–95th Percentile)", zorder=2)
    ax.plot(annual["year"], annual["mean"], color="#0369a1", linewidth=2.4, label="Panel Mean Yield", zorder=4)
    ax.plot(annual["year"], annual["trend"], color="#ea580c", linestyle="--", linewidth=2.0, label=rf"Secular Linear Trend ($R^2 = {r_val**2:.3f}$)", zorder=3)

    # Historical benchmark shocks
    shocks = [
        (1988, 28.08, "1988 Severe Drought\n(-25.6%)", (-15, -45)),
        (2003, 35.26, "2003 Heat Stress\n(-21.6%)", (5, -50)),
        (1974, 24.84, "1974 Early Freeze\n(-19.7%)", (-15, -45)),
        (2012, 43.55, "2012 Flash Drought\n(-11.7%)", (10, -40)),
        (1994, 44.95, "1994 Bumper Crop\n(+10.7%)", (-15, 30)),
        (2016, 56.81, "2016 Bumper Crop\n(+10.8%)", (-25, 28)),
    ]
    for yr, val, txt, offset in shocks:
        col = "#dc2626" if "Bumper" not in txt else "#16a34a"
        ax.scatter([yr], [val], color=col, s=38, zorder=6)
        ax.annotate(txt, xy=(yr, val), xytext=offset, textcoords="offset points", fontsize=8.0, fontweight="bold",
                    color=col, ha="center", arrowprops=dict(arrowstyle="->", color="#475569", lw=0.7))

    ax.set_title("Midwestern Soybean Yield Trajectory, Technological Trend, and Climatic Shocks (1951–2025)", loc="left", fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Crop Year")
    ax.set_ylabel("Yield (bu/acre)")
    ax.set_xlim(1950, 2026)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_yield_trajectory.pdf")
    fig.savefig(FIGURES_DIR / "fig_yield_trajectory.png")
    plt.close(fig)


def generate_figure_climate_sensitivity(df_yield):
    """Figure 3.8: Pearson correlations between monthly weather anomalies and final yield."""
    print("Generating Figure 3.8: Monthly Climate-Yield Sensitivity...")
    files = sorted(glob.glob(str(WEATHER_DIR / "county_daily_*.parquet")))
    if not files:
        print("  Weather files not found. Skipping sensitivity figure.")
        return

    # Compute county detrended yield anomalies
    trends = {}
    for fips, grp in df_yield.groupby("county_fips"):
        slope, intercept, _, _, _ = stats.linregress(grp["year"], grp["yield_bu_per_acre"])
        trends[fips] = (slope, intercept)

    df_yield_anom = df_yield.copy()
    df_yield_anom["anom_pct"] = [
        ((r["yield_bu_per_acre"] - (trends[r["county_fips"]][0] * r["year"] + trends[r["county_fips"]][1])) /
         (trends[r["county_fips"]][0] * r["year"] + trends[r["county_fips"]][1])) * 100
        for _, r in df_yield_anom.iterrows()
    ]

    # Aggregate monthly weather across growing season (Apr to Oct)
    monthly_records = []
    for f in files:
        yr = int(Path(f).stem.replace("county_daily_", ""))
        if yr < 1951:
            continue
        df_yr = pd.read_parquet(f, columns=["county_fips", "date", "air_temperature_maximum", "total_precipitation", "vapor_pressure_deficit", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3"])
        df_yr["month"] = pd.to_datetime(df_yr["date"]).dt.month
        df_yr = df_yr[(df_yr["month"] >= 4) & (df_yr["month"] <= 10)].copy()
        df_yr["sm_root"] = 0.07 * df_yr["volumetric_soil_water_layer_1"] + 0.21 * df_yr["volumetric_soil_water_layer_2"] + 0.72 * df_yr["volumetric_soil_water_layer_3"]

        mo_agg = df_yr.groupby(["county_fips", "month"]).agg(
            tmax=("air_temperature_maximum", "mean"),
            precip=("total_precipitation", "sum"),
            vpd=("vapor_pressure_deficit", "mean"),
            sm=("sm_root", "mean")
        ).reset_index()
        mo_agg["year"] = yr
        monthly_records.append(mo_agg)

    df_mo = pd.concat(monthly_records, ignore_index=True)
    df_mo["county_fips"] = df_mo["county_fips"].astype(str).str.zfill(5)
    df_merged = df_mo.merge(df_yield_anom, on=["county_fips", "year"])

    # Compute weather anomalies relative to local county-month climatological means
    clim = df_merged.groupby(["county_fips", "month"])[["tmax", "precip", "vpd", "sm"]].transform("mean")
    for v in ["tmax", "precip", "vpd", "sm"]:
        df_merged[f"{v}_anom"] = df_merged[v] - clim[v]

    months = [4, 5, 6, 7, 8, 9, 10]
    month_names = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    corrs = {v: [stats.pearsonr(df_merged[df_merged["month"] == m][f"{v}_anom"], df_merged[df_merged["month"] == m]["anom_pct"])[0] for m in months] for v in ["tmax", "vpd", "precip", "sm"]}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(months))
    w = 0.20

    ax.bar(x - 1.5 * w, corrs["tmax"], w, label=r"Max Temperature ($T_{\max}$)", color="#d95f02", alpha=0.9)
    ax.bar(x - 0.5 * w, corrs["vpd"], w, label=r"Vapor Pressure Deficit ($VPD$)", color="#e7298a", alpha=0.85)
    ax.bar(x + 0.5 * w, corrs["precip"], w, label=r"Precipitation ($P$)", color="#2b83ba", alpha=0.9)
    ax.bar(x + 1.5 * w, corrs["sm"], w, label=r"Root-Zone Soil Moisture ($SM_{\mathrm{root}}$)", color="#1b9e77", alpha=0.9)

    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel("Pearson Correlation ($r$) with Yield Anomaly")
    ax.set_title("Monthly Agrometeorological Sensitivity across the Growing Season (1951–2025)", loc="left", fontweight="bold", fontsize=11.5)
    ax.set_xticks(x)
    ax.set_xticklabels(month_names, fontweight="bold")
    ax.set_ylim(-0.55, 0.55)
    ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=9.0, facecolor="white", edgecolor="#cbd5e1")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_climate_yield_sensitivity.pdf")
    fig.savefig(FIGURES_DIR / "fig_climate_yield_sensitivity.png")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df_counties = pd.read_csv(COUNTIES_CSV)
    df_yield = pd.read_csv(TARGET_CSV)
    df_yield["county_fips"] = df_yield["county_fips"].astype(str).str.zfill(5)

    generate_figure_3_1(df_counties)
    generate_figure_3_2(df_yield)
    generate_figure_climate_sensitivity(df_yield)
    print("Chapter 3 figures generated successfully.")


if __name__ == "__main__":
    main()
