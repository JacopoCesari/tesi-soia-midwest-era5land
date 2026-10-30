"""
08_generate_eda_figures.py

Generates all 8 official exploratory data analysis figures for Chapter 3 (Data & Study Area):
1. Figure 3.1: Study Area Cartographic Map (fig_study_area_map.pdf / .png)
2. Figure 3.2: 75-Year Regional Soybean Yield Trajectory & Secular Trend (fig_yield_trajectory.pdf / .png)
3. Figure 3.3: Spatial Heterogeneity and State-Level Decadal Distributions (fig_state_yield_distributions.pdf / .png)
4. Figure 3.4: ERA5-Land Spatial Discretization and Area-Weighted Aggregation (fig_era5_spatial_structure.pdf / .png)
5. Figure 3.5: Cumulative Intra-Seasonal Agro-Climatic Trajectories (fig_intra_seasonal_extremes.pdf / .png)
6. Figure 3.6: Dynamic Shock Footprint Profiles (fig_weather_shocks_footprint.pdf / .png)
7. Figure 3.7: Spatial Contrast of Historical Agro-Climatic Shock Archetypes (fig_spatial_shock_comparison.pdf / .png)
8. Figure 3.8: Monthly Agrometeorological Sensitivity across the Growing Season (fig_climate_yield_sensitivity.pdf / .png)

Outputs:
- output/thesis/figures/fig_study_area_map.pdf / .png
- output/thesis/figures/fig_yield_trajectory.pdf / .png
- output/thesis/figures/fig_state_yield_distributions.pdf / .png
- output/thesis/figures/fig_era5_spatial_structure.pdf / .png
- output/thesis/figures/fig_intra_seasonal_extremes.pdf / .png
- output/thesis/figures/fig_weather_shocks_footprint.pdf / .png
- output/thesis/figures/fig_spatial_shock_comparison.pdf / .png
- output/thesis/figures/fig_climate_yield_sensitivity.pdf / .png
"""

import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch, Patch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import Rbf
import shapely.geometry
from shapely.geometry import box
from shapely.ops import unary_union

# Styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

ROOT = Path(__file__).resolve().parent.parent
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
COUNTIES_CSV = ROOT / "data" / "auxiliary" / "counties.csv"
WEIGHTS_CSV = ROOT / "data" / "auxiliary" / "spatial_weights.csv"
SHAPEFILE = ROOT / "data" / "raw" / "census_counties" / "cb_2020_us_county_500k.shp"
WEATHER_DIR = ROOT / "data" / "weather" / "daily_counties"
FIGURES_DIR = ROOT / "output" / "thesis" / "figures"


# ==============================================================================
# FIGURE 3.1: Study Area Map
# ==============================================================================
def generate_figure_3_1(df_counties, gdf_all):
    print("Generating Figure 3.1: Study Area Map...")
    midwest_fips = ["17", "18", "19", "27", "29", "39"]
    gdf_midwest = gdf_all[gdf_all["STATEFP"].isin(midwest_fips)].copy()
    
    selected_fips = set(df_counties["county_fips"].astype(str).str.zfill(5))
    gdf_midwest["status"] = np.where(gdf_midwest["GEOID"].isin(selected_fips), "Selected", "Other")

    gdf_midwest_proj = gdf_midwest.to_crs(epsg=5070)
    gdf_states = gdf_midwest_proj.dissolve(by="STATE_NAME")

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_midwest_proj[gdf_midwest_proj["status"] == "Other"].plot(
        ax=ax, color="#f5f5f5", edgecolor="#d9d9d9", linewidth=0.4
    )
    gdf_midwest_proj[gdf_midwest_proj["status"] == "Selected"].plot(
        ax=ax, color="#02818a", edgecolor="#01464c", linewidth=0.8
    )
    gdf_states.plot(ax=ax, facecolor="none", edgecolor="#252525", linewidth=1.2)

    state_coords = {
        "Illinois": (-89.2, 40.0), "Indiana": (-86.1, 39.8), "Iowa": (-93.5, 42.0),
        "Minnesota": (-94.5, 45.8), "Missouri": (-92.5, 38.3), "Ohio": (-82.7, 40.2)
    }
    pts = pd.DataFrame([{"state": k, "lon": v[0], "lat": v[1]} for k, v in state_coords.items()])
    pts_gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=4326).to_crs(epsg=5070)
    for _, row in pts_gdf.iterrows():
        ax.annotate(row["state"].upper(), xy=(row.geometry.x, row.geometry.y), fontsize=11,
                    fontweight="bold", color="#252525", alpha=0.6, ha="center")

    legend_elements = [
        Patch(facecolor="#02818a", edgecolor="#01464c", label="Study Counties (N=135)"),
        Patch(facecolor="#f5f5f5", edgecolor="#d9d9d9", label="Other Regional Counties"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax.set_title("Geographic Distribution of Study Counties (1951–2025)", fontsize=12.5, fontweight="bold", pad=12)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_study_area_map.pdf")
    fig.savefig(FIGURES_DIR / "fig_study_area_map.png")
    plt.close(fig)


# ==============================================================================
# FIGURE 3.2: Yield Trajectory & Secular Trend
# ==============================================================================
def generate_figure_3_2(df_yield):
    print("Generating Figure 3.2: Long-Term Yield Trajectory (Dual Petrol)...")
    cache_1950 = FIGURES_DIR / "cache_1950_yield.csv"
    if cache_1950.exists():
        df_1950 = pd.read_csv(cache_1950, dtype={'county_fips': str})
        df_1950['county_fips'] = df_1950['county_fips'].str.zfill(5)
        df_full = pd.concat([df_1950, df_yield], ignore_index=True)
    else:
        df_full = df_yield.copy()

    annual = df_full.groupby("year")["yield_bu_per_acre"].agg(
        mean="mean", min="min", max="max",
        q05=lambda s: s.quantile(0.05), q95=lambda s: s.quantile(0.95)
    ).reset_index()

    slope, intercept, r_val, _, _ = stats.linregress(annual["year"], annual["mean"])
    annual["linear_trend"] = slope * annual["year"] + intercept

    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ref = annual["linear_trend"].values
    years = annual["year"].values

    COLOR_ABOVE = "#059669"
    COLOR_ABOVE_FILL = "#10b981"
    COLOR_BELOW = "#b45309"
    COLOR_BELOW_FILL = "#f97316"
    COLOR_COUNTY = "#94a3b8"
    COLOR_REF_LINE = "#164e63"
    COLOR_MEAN_LINE = "#475569"
    COLOR_ARROW = "#0f172a"

    # County trajectories
    for _, grp in df_full.groupby("county_fips"):
        ax.plot(grp["year"], grp["yield_bu_per_acre"], color=COLOR_COUNTY, alpha=0.15, linewidth=0.60, zorder=1)

    # Dual Envelopes
    u_rec, l_rec = annual["max"].values, annual["min"].values
    u_p90, l_p90 = annual["q95"].values, annual["q05"].values

    ax.fill_between(years, np.maximum(l_rec, ref), u_rec, where=(u_rec >= ref), interpolate=True, color=COLOR_ABOVE_FILL, alpha=0.18, zorder=2)
    ax.fill_between(years, l_rec, np.minimum(u_rec, ref), where=(l_rec <= ref), interpolate=True, color=COLOR_BELOW_FILL, alpha=0.18, zorder=2)
    ax.fill_between(years, np.maximum(l_p90, ref), u_p90, where=(u_p90 >= ref), interpolate=True, color=COLOR_ABOVE_FILL, alpha=0.35, label="Above Reference (Upper County Envelope)", zorder=2)
    ax.fill_between(years, l_p90, np.minimum(u_p90, ref), where=(l_p90 <= ref), interpolate=True, color=COLOR_BELOW_FILL, alpha=0.35, label="Below Reference (Lower County Envelope)", zorder=2)

    ax.plot(years, u_rec, color=COLOR_ABOVE, linewidth=0.9, linestyle=":", alpha=0.6, zorder=3)
    ax.plot(years, l_rec, color=COLOR_BELOW, linewidth=0.9, linestyle=":", alpha=0.6, zorder=3)
    ax.plot(years, u_p90, color=COLOR_ABOVE, linewidth=1.3, linestyle="-", alpha=0.9, zorder=3)
    ax.plot(years, l_p90, color=COLOR_BELOW, linewidth=1.3, linestyle="-", alpha=0.9, zorder=3)

    ref_label = f"Linear Trend (OLS: +{slope*10:.2f} bu/ac/decade)"
    ax.plot(years, ref, color=COLOR_REF_LINE, linestyle="--", linewidth=2.3, label=ref_label, zorder=5)
    ax.plot(years, annual["mean"], color=COLOR_MEAN_LINE, linewidth=2.0, linestyle="-", label="Panel Cross-Sectional Mean Yield", zorder=4)

    # Shocks
    shocks = [
        (1988, 28.08, "1988 Severe Drought\n(-25.6%)", (-15, -42)),
        (2003, 35.26, "2003 Heat & Aphids\n(-21.6%)", (5, -48)),
        (1974, 24.84, "1974 Early Freeze\n(-19.7%)", (-15, -42)),
        (2012, 43.55, "2012 Flash Drought\n(-11.7%)", (10, -38)),
        (1993, 35.92, "1993 Great Flood\n(-10.5%)", (15, -40)),
        (1994, 44.95, "1994 Bumper\n(+10.7%)", (-15, 28)),
        (2016, 56.81, "2016 Bumper\n(+10.8%)", (-30, 26)),
        (2021, 59.50, "2021 Bumper\n(+10.8%)", (-22, 24)),
    ]
    for yr, val, txt, offset in shocks:
        ax.scatter([yr], [val], color="#dc2626" if "Bumper" not in txt else "#16a34a", s=30, zorder=6)
        ax.annotate(txt, xy=(yr, val), xytext=offset, textcoords="offset points", fontsize=7.8, fontweight="bold",
                    color="#991b1b" if "Bumper" not in txt else "#15803d", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.6, shrinkA=3, shrinkB=3), zorder=7)

    # Dispersion arrows (1951 vs 2024)
    if 1951 in annual["year"].values and 2024 in annual["year"].values:
        e_row = annual[annual["year"] == 1951].iloc[0]
        r_row = annual[annual["year"] == 2024].iloc[0]
        e_top, e_bot = e_row["q95"], e_row["q05"]
        r_top, r_bot = r_row["q95"], r_row["q05"]
        e_delta, r_delta = e_top - e_bot, r_top - r_bot

        ax.annotate("", xy=(1951, e_top), xytext=(1951, e_bot),
                    arrowprops=dict(arrowstyle="<->,head_width=0.38,head_length=0.65", color=COLOR_ARROW, lw=2.2, shrinkA=1, shrinkB=1), zorder=8)
        cap_w = 0.55
        ax.plot([1951 - cap_w, 1951 + cap_w], [e_top, e_top], color=COLOR_ARROW, lw=2.0, zorder=8)
        ax.plot([1951 - cap_w, 1951 + cap_w], [e_bot, e_bot], color=COLOR_ARROW, lw=2.0, zorder=8)
        ax.annotate(f"Narrow county dispersion\n$(\\Delta = {e_delta:.1f}$ bu/ac)", xy=(1951, e_top), xytext=(14, 14),
                    textcoords="offset points", fontsize=8.5, fontweight="bold", color=COLOR_ARROW, ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor=COLOR_ARROW, alpha=0.96, lw=1.0),
                    arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=0.9, shrinkB=4, connectionstyle="arc3,rad=-0.10"), zorder=9)

        ax.annotate("", xy=(2024, r_top), xytext=(2024, r_bot),
                    arrowprops=dict(arrowstyle="<->,head_width=0.38,head_length=0.65", color=COLOR_ARROW, lw=2.2, shrinkA=1, shrinkB=1), zorder=8)
        ax.plot([2024 - cap_w, 2024 + cap_w], [r_top, r_top], color=COLOR_ARROW, lw=2.0, zorder=8)
        ax.plot([2024 - cap_w, 2024 + cap_w], [r_bot, r_bot], color=COLOR_ARROW, lw=2.0, zorder=8)
        pct_expand = ((r_delta - e_delta) / e_delta) * 100.0
        ax.annotate(f"Wide county dispersion\n$(\\Delta = {r_delta:.1f}$ bu/ac, $+{pct_expand:.0f}\\%)$", xy=(2024, r_top), xytext=(-14, 16),
                    textcoords="offset points", fontsize=8.5, fontweight="bold", color=COLOR_ARROW, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor=COLOR_ARROW, alpha=0.96, lw=1.0),
                    arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=0.9, shrinkB=4, connectionstyle="arc3,rad=0.10"), zorder=9)

    ax.set_title(
        "County Soybean Yield Trajectories & Dispersion Dynamics (1951–2025)\n"
        "Reference Line: Linear Trend | Envelope: Dual Envelope (Min–Max Records + 90% Dispersion Core)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Crop Year", fontsize=11, fontweight="bold")
    ax.set_ylabel("Soybean Yield (bu/acre)", fontsize=11, fontweight="bold")
    ax.set_xlim(1948, 2027)
    ax.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2025])
    ax.set_ylim(0, 96)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.95, fontsize=8.8)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.10)

    for out_name in ["fig_yield_trajectory_dual_petrol", "fig_yield_trajectory"]:
        fig.savefig(FIGURES_DIR / f"{out_name}.pdf")
        fig.savefig(FIGURES_DIR / f"{out_name}.png", dpi=300)
    plt.close(fig)


# ==============================================================================
# FIGURE 3.3: Spatial Heterogeneity & State-Level Decadal Yields
# ==============================================================================
def generate_figure_3_3(df_yield, df_counties):
    print("Generating Figure 3.3: State-Level Yield Distributions...")
    merged = df_yield.merge(df_counties[["county_fips", "state"]], on="county_fips", how="left")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6), gridspec_kw={'width_ratios': [1.2, 1]})
    
    state_order = merged.groupby("state")["yield_bu_per_acre"].median().sort_values(ascending=False).index.tolist()
    cmap = plt.cm.Blues_r
    palette = [cmap(0.15 + 0.70 * (i / max(1, len(state_order) - 1))) for i in range(len(state_order))]
    
    boxes_data = [merged[merged["state"] == st]["yield_bu_per_acre"].values for st in state_order]
    bp = ax1.boxplot(boxes_data, patch_artist=True, tick_labels=state_order,
                     flierprops=dict(marker='o', markersize=1.5, markerfacecolor='#555555', markeredgecolor='none', alpha=0.5),
                     medianprops=dict(color='#000000', linewidth=1.2),
                     whiskerprops=dict(linewidth=1.0),
                     capprops=dict(linewidth=1.0))
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.0)
        
    ax1.set_title("(a) Cross-Sectional Yield Distribution by State (1951–2025)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("State", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Soybean Yield (bu/acre)", fontsize=10, fontweight="bold")
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    merged["decade"] = (merged["year"] // 10) * 10
    state_dec = merged.groupby(["decade", "state"])["yield_bu_per_acre"].mean().reset_index()
    
    for idx, st in enumerate(state_order):
        sub = state_dec[state_dec["state"] == st]
        ax2.plot(sub["decade"], sub["yield_bu_per_acre"], marker="o", markersize=4, linewidth=1.8, 
                 label=st, color=palette[idx])
        
    ax2.set_title("(b) Decadal Mean Yield Trajectories by State", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Decade", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Soybean Yield (bu/acre)", fontsize=10, fontweight="bold")
    ax2.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
    ax2.set_xticklabels(["'50s", "'60s", "'70s", "'80s", "'90s", "'00s", "'10s", "'20s"])
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_state_yield_distributions.pdf")
    fig.savefig(FIGURES_DIR / "fig_state_yield_distributions.png")
    plt.close(fig)


# ==============================================================================
# FIGURE 3.4: ERA5-Land Grid Structure & Tazewell County Aggregation
# ==============================================================================
def generate_figure_3_4(gdf_all, df_weights, df_counties):
    print("Generating Figure 3.4: ERA5-Land Spatial Discretization...")
    selected_fips = set(df_counties['county_fips'].astype(str).str.zfill(5))

    gdf_il = gdf_all[gdf_all['STATEFP'] == '17'].copy()
    gdf_selected_il = gdf_il[gdf_il['GEOID'].isin(selected_fips)].copy()
    gdf_il_state = gdf_il.dissolve(by='STATEFP')

    surrounding_fips = ['19', '29', '21', '18', '55']
    gdf_surrounding = gdf_all[gdf_all['STATEFP'].isin(surrounding_fips)].copy()
    gdf_surr_states = gdf_surrounding.dissolve(by='STATEFP')

    tazewell_row = gdf_il[gdf_il['GEOID'] == '17179'].iloc[0]
    tazewell_geom = tazewell_row.geometry
    tb = tazewell_geom.bounds
    tw_w = df_weights[df_weights['county_fips'].astype(str).str.zfill(5) == '17179'].copy()

    fig, (ax_macro, ax_micro) = plt.subplots(
        1, 2, figsize=(14.0, 6.8),
        gridspec_kw={'width_ratios': [1.0, 1.28], 'wspace': 0.22}
    )

    gdf_surr_states.plot(ax=ax_macro, facecolor='#f7f7f7', edgecolor='#d9d9d9', linewidth=0.8, zorder=1)
    gdf_il.plot(ax=ax_macro, facecolor='#ffffff', edgecolor='#d0d0d0', linewidth=0.45, zorder=2)
    gdf_selected_il.plot(ax=ax_macro, facecolor='#a1dab4', edgecolor='#238b45', linewidth=0.7, zorder=3)
    gpd.GeoSeries([tazewell_geom]).plot(
        ax=ax_macro, facecolor='#e41a1c', edgecolor='#99000d', linewidth=1.8, alpha=0.8, zorder=5
    )
    gdf_il_state.plot(ax=ax_macro, facecolor='none', edgecolor='#252525', linewidth=1.4, zorder=6)

    ax_macro.text(-90.5, 43.1, 'WISCONSIN', fontsize=9, color='#888888', fontweight='bold', ha='center', zorder=4)
    ax_macro.text(-92.3, 41.5, 'IOWA', fontsize=9, color='#888888', fontweight='bold', ha='center', zorder=4)
    ax_macro.text(-91.8, 38.3, 'MISSOURI', fontsize=9, color='#888888', fontweight='bold', ha='center', zorder=4)
    ax_macro.text(-86.2, 40.2, 'INDIANA', fontsize=9, color='#888888', fontweight='bold', ha='center', zorder=4)
    ax_macro.text(-87.8, 37.3, 'KENTUCKY', fontsize=8.5, color='#888888', fontweight='bold', ha='center', zorder=4)

    taz_box_pad_x = 0.08
    taz_box_pad_y = 0.06
    macro_taz_rect = mpatches.Rectangle(
        (tb[0] - taz_box_pad_x, tb[1] - taz_box_pad_y),
        tb[2] - tb[0] + 2 * taz_box_pad_x,
        tb[3] - tb[1] + 2 * taz_box_pad_y,
        fill=False, edgecolor='#99000d', linewidth=1.8, linestyle='-', zorder=7
    )
    ax_macro.add_patch(macro_taz_rect)

    ax_macro.annotate(
        "Tazewell County, IL",
        xy=((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2),
        xytext=(-92.0, 40.1),
        fontsize=9.5, fontweight='bold', color='#99000d',
        arrowprops=dict(arrowstyle='->', color='#99000d', lw=1.3, connectionstyle='arc3,rad=-0.12'),
        bbox=dict(boxstyle='square,pad=0.35', facecolor='white', edgecolor='#99000d', alpha=0.95),
        zorder=8
    )

    p_sel = mpatches.Patch(facecolor='#a1dab4', edgecolor='#238b45', label='Balanced Study Counties')
    p_other = mpatches.Patch(facecolor='#ffffff', edgecolor='#d0d0d0', label='Other Counties')
    p_taz = mpatches.Patch(facecolor='#e41a1c', edgecolor='#99000d', label='Tazewell County')
    ax_macro.legend(
        handles=[p_sel, p_other, p_taz],
        loc='lower left', bbox_to_anchor=(0.02, 0.02),
        fontsize=9.0, framealpha=0.95, facecolor='white', edgecolor='#cccccc'
    )
    ax_macro.set_xlim(-92.8, -86.0)
    ax_macro.set_ylim(36.8, 43.5)
    ax_macro.set_xlabel(r'Longitude ($^\circ$W)', fontsize=9.5)
    ax_macro.set_ylabel(r'Latitude ($^\circ$N)', fontsize=9.5)
    ax_macro.set_title('(a) Regional Domain: Illinois State', fontsize=11, fontweight='bold', pad=8)
    ax_macro.set_aspect(1.0 / np.cos(np.radians(40.0)))

    # Micro view
    gpd.GeoSeries([tazewell_geom]).plot(
        ax=ax_micro, facecolor='#edf8fb', edgecolor='#01464c', linewidth=2.4, zorder=2
    )
    for lon in np.arange(-89.95, -89.15, 0.1):
        ax_micro.axvline(lon, color='#08519c', linestyle='--', linewidth=0.75, alpha=0.55, zorder=3)
    for lat in np.arange(40.25, 40.85, 0.1):
        ax_micro.axhline(lat, color='#08519c', linestyle='--', linewidth=0.75, alpha=0.55, zorder=3)

    for _, r in tw_w.iterrows():
        b = box(r['longitude'] - 0.05, r['latitude'] - 0.05, r['longitude'] + 0.05, r['latitude'] + 0.05)
        inter = tazewell_geom.intersection(b)
        w = r['weight']
        alpha_val = min(0.65, 0.22 + w * 7.5)
        gpd.GeoSeries([inter]).plot(
            ax=ax_micro, facecolor='#2b83ba', edgecolor='#08519c', linewidth=0.65, alpha=alpha_val, zorder=4
        )

    badge_text = (
        r"$\mathbf{Area-Weighted\ Aggregation\ (Eq.\ 3.1)}$" + "\n" +
        r"$\bar{X}_{c,t} = \sum_{g=1}^{29} w_{c,g} \, X_{g,t}, \quad \text{with } \sum w_{c,g} = 100\%$" + "\n" +
        r"Weights: $w_{c,g} = \frac{\mathrm{Area}(c \cap g)}{\mathrm{Area}(c)}$"
    )
    ax_micro.text(
        0.04, 0.96, badge_text,
        transform=ax_micro.transAxes, fontsize=9.2, verticalalignment='top',
        bbox=dict(boxstyle='square,pad=0.45', facecolor='white', edgecolor='#08519c', alpha=0.95),
        zorder=7
    )

    leg_bound = Line2D([0], [0], color='#01464c', linewidth=2.4, label='County Boundary (Tazewell, IL)')
    leg_grid = Line2D([0], [0], color='#08519c', linestyle='--', linewidth=1.2, label=r'ERA5-Land Grid ($0.1^\circ \approx 9$ km)')
    leg_cells = mpatches.Patch(facecolor='#2b83ba', edgecolor='#08519c', alpha=0.55, label='Intersecting Grid Cells (Area-Weighted)')

    ax_micro.legend(
        handles=[leg_bound, leg_grid, leg_cells],
        loc='lower right', bbox_to_anchor=(0.98, 0.03),
        fontsize=9.0, framealpha=0.95, facecolor='white', edgecolor='#cccccc'
    )
    ax_micro.set_xlim(-89.98, -89.20)
    ax_micro.set_ylim(40.22, 40.80)
    ax_micro.set_xlabel(r'Longitude ($^\circ$W)', fontsize=9.5)
    ax_micro.set_ylabel(r'Latitude ($^\circ$N)', fontsize=9.5)
    ax_micro.set_title(r'(b) ERA5-Land Grid Discretization: Tazewell County, IL', fontsize=11, fontweight='bold', pad=8)
    ax_micro.set_aspect(1.0 / np.cos(np.radians(40.5)))

    con_top = ConnectionPatch(
        xyA=(tb[2] + taz_box_pad_x, tb[3] + taz_box_pad_y), coordsA=ax_macro.transData,
        xyB=(-89.98, 40.72), coordsB=ax_micro.transData,
        color='#99000d', linestyle=':', linewidth=1.2, alpha=0.7
    )
    fig.add_artist(con_top)

    con_bottom = ConnectionPatch(
        xyA=(tb[2] + taz_box_pad_x, tb[1] - taz_box_pad_y), coordsA=ax_macro.transData,
        xyB=(-89.98, 40.26), coordsB=ax_micro.transData,
        color='#99000d', linestyle=':', linewidth=1.2, alpha=0.7
    )
    fig.add_artist(con_bottom)

    fig.savefig(FIGURES_DIR / "fig_era5_spatial_structure.pdf")
    fig.savefig(FIGURES_DIR / "fig_era5_spatial_structure.png")
    plt.close(fig)


# ==============================================================================
# Helper: Load Daily Weather Trajectories for Shocks & Climatology
# ==============================================================================
def load_daily_weather_trajectories():
    print("Loading daily weather across 1951–2025 for intra-seasonal trajectories...")
    files = sorted(glob.glob(str(WEATHER_DIR / "county_daily_*.parquet")))
    target_years = [2012, 1993, 2016, 1994]
    
    daily_by_year = {}
    climatology_list = []
    
    for fpath in files:
        yr = int(Path(fpath).stem.replace("county_daily_", ""))
        if yr < 1951:
            continue
        df_yr = pd.read_parquet(fpath, columns=[
            "date", "air_temperature_maximum", "total_precipitation", "et0_fao56", "p_minus_et0",
            "vapor_pressure_deficit", "heat_day_30",
            "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3"
        ])
        df_yr["date"] = pd.to_datetime(df_yr["date"])
        df_yr["doy"] = df_yr["date"].dt.dayofyear
        df_season = df_yr[(df_yr["doy"] >= 91) & (df_yr["doy"] <= 304)].copy()
        
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

    return daily_by_year, clim_doy, ref_dates, month_ticks, month_labels


# ==============================================================================
# FIGURE 3.5: Cumulative Intra-Seasonal Extremes
# ==============================================================================
def generate_figure_3_5(daily_by_year, clim_doy, ref_dates, month_ticks, month_labels):
    print("Generating Figure 3.5: Cumulative Intra-Seasonal Trajectories...")
    stages = [
        ("Sowing", pd.to_datetime("2021-04-01"), pd.to_datetime("2021-05-31"), "#f0f0f0"),
        ("Vegetative", pd.to_datetime("2021-06-01"), pd.to_datetime("2021-06-30"), "#e5f5e0"),
        ("Flowering", pd.to_datetime("2021-07-01"), pd.to_datetime("2021-07-31"), "#fee6ce"),
        ("Pod-Filling", pd.to_datetime("2021-08-01"), pd.to_datetime("2021-08-31"), "#fdd0a2"),
        ("Harvest", pd.to_datetime("2021-09-01"), pd.to_datetime("2021-10-31"), "#f0f0f0")
    ]
    styles_cum = {
        2012: {"label": "2012 Shock: Flash Drought (-11.7%)", "color": "#d73027", "lw": 2.2, "ls": "-"},
        1993: {"label": "1993 Shock: Midwest Flood (-10.5%)", "color": "#762a83", "lw": 2.0, "ls": "-"},
        2016: {"label": "2016 Bumper: Optimum Buffer (+10.8%)", "color": "#006837", "lw": 2.2, "ls": "-"},
        1994: {"label": "1994 Bumper: Ideal Balance (+10.7%)", "color": "#2171b5", "lw": 2.0, "ls": "-"}
    }
    target_ordered = [2012, 1993, 2016, 1994]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), sharex=True)

    def add_pheno(ax, show_labels=False):
        for name, start, end, col in stages:
            ax.axvspan(start, end, color=col, alpha=0.35, zorder=0)
        if show_labels:
            y_lim = ax.get_ylim()
            y_pos = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.93
            for name, start, end, _ in stages:
                mid = start + (end - start) / 2
                ax.text(mid, y_pos, name, ha="center", va="center", fontsize=8.0, fontweight="bold",
                        color="#334155", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#cbd5e1", alpha=0.90, lw=0.6))

    # (a) Cumulative Heat Stress Days
    ax_a = axes[0, 0]
    add_pheno(ax_a, show_labels=True)
    ax_a.fill_between(ref_dates, clim_doy["cum_hd30_p10"], clim_doy["cum_hd30_p90"],
                      color="#cbd5e1", alpha=0.55, label="Climatological Range (10th–90th %ile)", zorder=2)
    ax_a.plot(ref_dates, clim_doy["cum_hd30_median"], color="black", lw=2.0, ls="--", label="Climatological Median", zorder=3)
    for yr in target_ordered:
        s = styles_cum[yr]
        ax_a.plot(ref_dates, daily_by_year[yr]["cum_hd30"], color=s["color"], lw=s["lw"], ls=s["ls"], label=s["label"], zorder=4)
    ax_a.set_ylabel("Cumulative Days (Count)")
    ax_a.set_title(r"(a) Heat Stress Accrual (Cumulative Days with $T_{\mathrm{max}} \geq 30^\circ\mathrm{C}$)", loc="left", fontweight="bold")
    ax_a.set_ylim(-1, 55)

    # (b) Cumulative Precipitation
    ax_b = axes[0, 1]
    add_pheno(ax_b, show_labels=True)
    ax_b.fill_between(ref_dates, clim_doy["cum_p_p10"], clim_doy["cum_p_p90"],
                      color="#cbd5e1", alpha=0.55, zorder=2)
    ax_b.plot(ref_dates, clim_doy["cum_p_median"], color="black", lw=2.0, ls="--", zorder=3)
    for yr in target_ordered:
        s = styles_cum[yr]
        ax_b.plot(ref_dates, daily_by_year[yr]["cum_p"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax_b.set_ylabel("Cumulative Depth (mm)")
    ax_b.set_title(r"(b) Hydrological Supply (Cumulative Total Precipitation $P$)", loc="left", fontweight="bold")
    ax_b.set_ylim(0, 1150)

    # (c) Cumulative Reference Evapotranspiration
    ax_c = axes[1, 0]
    add_pheno(ax_c, show_labels=False)
    ax_c.fill_between(ref_dates, clim_doy["cum_et0_p10"], clim_doy["cum_et0_p90"],
                      color="#cbd5e1", alpha=0.55, zorder=2)
    ax_c.plot(ref_dates, clim_doy["cum_et0_median"], color="black", lw=2.0, ls="--", zorder=3)
    for yr in target_ordered:
        s = styles_cum[yr]
        ax_c.plot(ref_dates, daily_by_year[yr]["cum_et0"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax_c.set_ylabel("Cumulative Evaporative Demand (mm)")
    ax_c.set_title(r"(c) Atmospheric Thirst (Cumulative Penman-Monteith $ET_0$)", loc="left", fontweight="bold")
    ax_c.set_ylim(400, 1050)

    # (d) Cumulative Climatic Water Balance
    ax_d = axes[1, 1]
    add_pheno(ax_d, show_labels=False)
    ax_d.fill_between(ref_dates, clim_doy["cum_balance_p10"], clim_doy["cum_balance_p90"],
                      color="#cbd5e1", alpha=0.55, zorder=2)
    ax_d.plot(ref_dates, clim_doy["cum_balance_median"], color="black", lw=2.0, ls="--", zorder=3)
    for yr in target_ordered:
        s = styles_cum[yr]
        ax_d.plot(ref_dates, daily_by_year[yr]["cum_balance"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax_d.axhline(0, color="gray", lw=1.0, ls=":", zorder=1)
    ax_d.set_ylabel("Cumulative Water Deficit / Surplus (mm)")
    ax_d.set_title(r"(d) Cumulative Net Climatic Water Balance ($P - ET_0$)", loc="left", fontweight="bold")
    ax_d.set_ylim(-500, 450)

    for ax in axes.flat:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(ref_dates.iloc[0], ref_dates.iloc[-1])
        ax.grid(True, linestyle="--", alpha=0.4)

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    fig.legend(handles_a, labels_a, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3,
               frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_intra_seasonal_extremes.pdf")
    fig.savefig(FIGURES_DIR / "fig_intra_seasonal_extremes.png")
    plt.close(fig)


# ==============================================================================
# FIGURE 3.6: Dynamic Shock Footprint Profiles
# ==============================================================================
def generate_figure_3_6(daily_by_year, clim_doy, ref_dates, month_ticks, month_labels):
    print("Generating Figure 3.6: Non-Cumulative Dynamic Shock Profiles...")
    stages = [
        ("Sowing", pd.to_datetime("2021-04-01"), pd.to_datetime("2021-05-31"), "#f0f0f0"),
        ("Vegetative", pd.to_datetime("2021-06-01"), pd.to_datetime("2021-06-30"), "#e5f5e0"),
        ("Flowering", pd.to_datetime("2021-07-01"), pd.to_datetime("2021-07-31"), "#fee6ce"),
        ("Pod-Filling", pd.to_datetime("2021-08-01"), pd.to_datetime("2021-08-31"), "#fdd0a2"),
        ("Harvest", pd.to_datetime("2021-09-01"), pd.to_datetime("2021-10-31"), "#f0f0f0")
    ]
    styles_shocks = {
        2012: {"label": "2012 Shock: Flash Drought (-11.7%)", "color": "#d73027", "lw": 2.2, "ls": "-"},
        1993: {"label": "1993 Shock: Midwest Flood (-10.5%)", "color": "#762a83", "lw": 2.0, "ls": "-"},
        2016: {"label": "2016 Bumper: Optimum Buffer (+10.8%)", "color": "#006837", "lw": 2.2, "ls": "-"},
        1994: {"label": "1994 Bumper: Ideal Balance (+10.7%)", "color": "#2171b5", "lw": 2.0, "ls": "-"}
    }
    target_ordered = [2012, 1993, 2016, 1994]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)

    def add_pheno_subtle(ax):
        for name, start, end, col in stages:
            ax.axvspan(start, end, color=col, alpha=0.30, zorder=0)

    # (a) Atmospheric Evaporative Demand (VPD)
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
    ax0.text(ref_dates[4], 1.54, r"Elevated Evaporative Demand Threshold ($1.5\text{ kPa}$)", color="#b2182b", fontsize=8.5, fontweight="bold")
    ax0.set_ylabel(r"15-Day Rolling $VPD$ (kPa)")
    ax0.set_title(r"(a) Atmospheric Evaporative Demand (15-Day Rolling $VPD$)", loc="left", fontweight="bold")
    ax0.set_ylim(0.35, 2.15)
    ax0.grid(True, linestyle="--", alpha=0.4)

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
    ax1.text(pd.to_datetime("2021-08-20"), 0.385, r"Regional Field Capacity ($0.38\ \mathrm{m}^3/\mathrm{m}^3$)", color="#2b83ba", fontsize=8.5, fontweight="bold", ha="right")
    ax1.axhline(0.18, color="#d73027", ls=":", lw=1.1, alpha=0.85)
    ax1.text(pd.to_datetime("2021-05-15"), 0.185, r"Incipient Wilting Stress Threshold ($0.18\ \mathrm{m}^3/\mathrm{m}^3$)", color="#d73027", fontsize=8.5, fontweight="bold")
    ax1.set_ylabel(r"Soil Moisture ($\text{m}^3/\text{m}^3$)")
    ax1.set_title(r"(b) Integrated Root-Zone Soil Moisture Dynamics ($SM_{\text{root}}$, $0$–$100\text{ cm}$)", loc="left", fontweight="bold")
    ax1.set_ylim(0.16, 0.44)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # (c) Maximum Temperature
    ax2 = axes[2]
    add_pheno_subtle(ax2)
    ax2.fill_between(ref_dates, clim_doy["tmax_roll15_p10"], clim_doy["tmax_roll15_p90"],
                     color="#cbd5e1", alpha=0.50, label="Climatological Range (10th–90th %ile)", zorder=2)
    ax2.plot(ref_dates, clim_doy["tmax_roll15_median"], color="black", lw=2.0, ls="--", label="Climatological Median", zorder=3)
    for yr in target_ordered:
        s = styles_shocks[yr]
        ax2.plot(ref_dates, daily_by_year[yr]["tmax_roll15"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=4)
    ax2.axhline(30, color="#d73027", ls=":", lw=1.2, alpha=0.85)
    ax2.text(pd.to_datetime("2021-05-15"), 30.5, r"Reproductive Thermal Stress Ceiling ($30^\circ\text{C}$)", color="#d73027", fontsize=8.5, fontweight="bold")
    ax2.set_ylabel(r"15-Day Rolling $T_{\text{max}}$ ($^\circ$C)")
    ax2.set_title(r"(c) Thermal Regimes (15-Day Rolling Maximum Temperature)", loc="left", fontweight="bold")
    ax2.set_ylim(10, 36)
    ax2.grid(True, linestyle="--", alpha=0.4)

    for ax in axes:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(ref_dates.iloc[0], ref_dates.iloc[-1])

    legend_handles_shocks = [
        h_range_0, h_med_0,
        line_handles_0[2012], line_handles_0[1993],
        line_handles_0[2016], line_handles_0[1994]
    ]
    legend_labels_shocks = [h.get_label() for h in legend_handles_shocks]
    fig.legend(legend_handles_shocks, legend_labels_shocks, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3,
               frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_weather_shocks_footprint.pdf")
    fig.savefig(FIGURES_DIR / "fig_weather_shocks_footprint.png")
    plt.close(fig)


# ==============================================================================
# FIGURE 3.7: Spatial Shock Comparison (Archetypes)
# ==============================================================================
def generate_figure_3_7(gdf_all, df_counties):
    print("Generating Figure 3.7: Spatial Shock Comparison Archetypes...")
    selected_fips = set(df_counties['county_fips'].astype(str).str.zfill(5))

    midwest_fips = ['17', '18', '19', '27', '29', '39']
    context_fips = ['55', '26', '21', '51', '54', '42', '31', '20', '46', '38']

    gdf_mw = gdf_all[gdf_all['STATEFP'].isin(midwest_fips)].copy()
    gdf_context = gdf_all[gdf_all['STATEFP'].isin(midwest_fips + context_fips)].copy()
    gdf_selected = gdf_mw[gdf_mw['GEOID'].isin(selected_fips)].copy()

    gdf_states = gdf_mw.dissolve(by='STATEFP')
    gdf_context_states = gdf_context.dissolve(by='STATEFP')
    mw_boundary = unary_union(gdf_states.geometry)

    gdf_selected['centroid_x'] = gdf_selected.geometry.centroid.x
    gdf_selected['centroid_y'] = gdf_selected.geometry.centroid.y

    bounds = mw_boundary.bounds
    grid_x, grid_y = np.mgrid[bounds[0]:bounds[2]:300j, bounds[1]:bounds[3]:300j]
    points_in_mw = [shapely.geometry.Point(x, y).within(mw_boundary) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
    mask_mw = np.array(points_in_mw).reshape(grid_x.shape)

    state_coords = {'MN': (-94.6, 46.2), 'IA': (-93.6, 42.1), 'MO': (-92.6, 38.6),
                    'IL': (-89.2, 40.0), 'IN': (-86.2, 40.0), 'OH': (-82.7, 40.3)}

    def get_metrics(year):
        fpath = WEATHER_DIR / f"county_daily_{year}.parquet"
        df = pd.read_parquet(fpath)
        df['date'] = pd.to_datetime(df['date'])
        df_ja = df[df['date'].dt.month.isin([7, 8])].copy()
        agg = df_ja.groupby('county_fips').agg({
            'total_precipitation': 'sum',
            'air_temperature_maximum': 'mean'
        }).reset_index()
        agg['county_fips'] = agg['county_fips'].astype(str).str.zfill(5)
        return agg

    def interpolate_var(metrics, var_col):
        merged = gdf_selected.merge(metrics, left_on='GEOID', right_on='county_fips')
        pts = np.vstack([merged['centroid_x'].values, merged['centroid_y'].values]).T
        vals = merged[var_col].values
        rbf = Rbf(pts[:, 0], pts[:, 1], vals, function='linear', smooth=0.5)
        grid = rbf(grid_x, grid_y)
        grid[~mask_mw] = np.nan
        return grid

    m2012 = get_metrics(2012)
    m2016 = get_metrics(2016)

    grid_p_2012 = interpolate_var(m2012, 'total_precipitation')
    grid_tmax_2012 = interpolate_var(m2012, 'air_temperature_maximum')
    grid_p_2016 = interpolate_var(m2016, 'total_precipitation')
    grid_tmax_2016 = interpolate_var(m2016, 'air_temperature_maximum')

    norm_p = Normalize(vmin=60, vmax=320)
    norm_tmax = Normalize(vmin=26.0, vmax=35.0)
    cmap_p = plt.cm.YlGnBu
    cmap_tmax = plt.cm.inferno

    configs = [
        (0, 0, grid_p_2012, norm_p, cmap_p,
         r'(a) 2012 Flash Drought: July–August Rainfall ($P$)',
         r'Negative Downside Shock (-11.7%) | Severe Rainfall Collapse (<80 mm in IL & IA)',
         ([80, 100, 120, 150], '%d', '#334155')),
        (0, 1, grid_tmax_2012, norm_tmax, cmap_tmax,
         r'(b) 2012 Flash Drought: July–August Max Temp ($T_{\mathrm{max}}$)',
         r'Negative Downside Shock (-11.7%) | Extreme Heat Waves (>33°C in Central/South)',
         ([28, 30, 32, 34], '%.0f°C', '#ffffff')),
        (1, 0, grid_p_2016, norm_p, cmap_p,
         r'(c) 2016 Climatic Optimum: July–August Rainfall ($P$)',
         r'Positive Bumper Harvest (+10.8%) | Abundant Recharge Waves (>260 mm in Core Belt)',
         ([180, 220, 260, 300], '%d', '#334155')),
        (1, 1, grid_tmax_2016, norm_tmax, cmap_tmax,
         r'(d) 2016 Climatic Optimum: July–August Max Temp ($T_{\mathrm{max}}$)',
         r'Positive Bumper Harvest (+10.8%) | Consistently Mild Regimes (<29°C Region-Wide)',
         ([27, 28, 29, 30], '%.0f°C', '#ffffff'))
    ]

    fig = plt.figure(figsize=(14.0, 12.5), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.0, 1.0, 0.07], hspace=0.25, wspace=0.18,
                           top=0.96, bottom=0.06, left=0.05, right=0.95)

    for r, c, gdata, norm, cmap, title, subtitle, contour_spec in configs:
        ax = fig.add_subplot(gs[r, c])
        gdf_context_states.plot(ax=ax, facecolor='#fafafa', edgecolor='#e2e8f0', linewidth=0.5, zorder=1)
        gdf_mw.plot(ax=ax, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=0.35, zorder=2)
        ax.contourf(grid_x, grid_y, gdata, levels=35, cmap=cmap, norm=norm, zorder=3, alpha=0.90)

        if contour_spec:
            levels, fmt, colors = contour_spec
            cs = ax.contour(grid_x, grid_y, gdata, levels=levels, colors=colors, linewidths=0.45, alpha=0.55, zorder=4)
            ax.clabel(cs, inline=True, fmt=fmt, fontsize=7.5)

        gdf_selected.plot(ax=ax, facecolor='none', edgecolor='#1e293b', linewidth=0.60, zorder=5)
        gdf_states.plot(ax=ax, facecolor='none', edgecolor='#0f172a', linewidth=1.1, zorder=6)
        for st, (sx, sy) in state_coords.items():
            ax.text(sx, sy, st, fontsize=9.5, fontweight='bold', color='#0f172a', alpha=0.65, ha='center', zorder=7)

        ax.set_xlim(-97.3, -80.5)
        ax.set_ylim(36.0, 49.2)
        ax.set_aspect(1.0 / np.cos(np.radians(42.5)))
        ax.set_xticks([-96, -92, -88, -84])
        ax.set_xticklabels([r'$96^\circ\mathrm{W}$', r'$92^\circ\mathrm{W}$', r'$88^\circ\mathrm{W}$', r'$84^\circ\mathrm{W}$'], fontsize=8.5)
        ax.set_yticks([38, 42, 46])
        ax.set_yticklabels([r'$38^\circ\mathrm{N}$', r'$42^\circ\mathrm{N}$', r'$46^\circ\mathrm{N}$'], fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.40, color='#94a3b8')
        ax.set_title(title, fontsize=11.2, fontweight='bold', pad=7)
        ax.text(0.5, -0.065, subtitle, transform=ax.transAxes, ha='center', fontsize=9.0, fontstyle='italic', color='#334155')

    cbar_ax_left = fig.add_subplot(gs[2, 0])
    cb_l = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_p, norm=norm_p), cax=cbar_ax_left, orientation='horizontal')
    cb_l.set_label(r'July--August Total Precipitation: $P$ (mm)', fontsize=10.0, fontweight='bold', labelpad=5)
    cb_l.ax.tick_params(labelsize=8.5)

    cbar_ax_right = fig.add_subplot(gs[2, 1])
    cb_r = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_tmax, norm=norm_tmax), cax=cbar_ax_right, orientation='horizontal')
    cb_r.set_label(r'July--August Mean Daily Maximum Temperature: $T_{\mathrm{max}}$ ($^\circ\mathrm{C}$)', fontsize=10.0, fontweight='bold', labelpad=5)
    cb_r.ax.tick_params(labelsize=8.5)

    fig.savefig(FIGURES_DIR / "fig_spatial_shock_comparison.pdf")
    fig.savefig(FIGURES_DIR / "fig_spatial_shock_comparison.png")
    plt.close(fig)


# ==============================================================================
# FIGURE 3.8: Monthly Agrometeorological Sensitivity Profile
# ==============================================================================
def generate_figure_3_8(df_yield):
    print("Generating Figure 3.8: Monthly Climate-Yield Sensitivity...")
    files = sorted(glob.glob(str(WEATHER_DIR / "county_daily_*.parquet")))
    if not files:
        print("  Weather files not found. Skipping sensitivity figure.")
        return

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


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generating All Official Chapter 3 Figures (3.1 to 3.8) ===")
    
    df_counties = pd.read_csv(COUNTIES_CSV, dtype={'county_fips': str})
    df_counties['county_fips'] = df_counties['county_fips'].str.zfill(5)

    df_yield = pd.read_csv(TARGET_CSV)
    df_yield["county_fips"] = df_yield["county_fips"].astype(str).str.zfill(5)

    gdf_all = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
    df_weights = pd.read_csv(WEIGHTS_CSV, dtype={'county_fips': str})

    # Figures 3.1 to 3.4
    generate_figure_3_1(df_counties, gdf_all)
    generate_figure_3_2(df_yield)
    generate_figure_3_3(df_yield, df_counties)
    generate_figure_3_4(gdf_all, df_weights, df_counties)

    # Figures 3.5 and 3.6 (shared daily weather processing)
    daily_by_year, clim_doy, ref_dates, month_ticks, month_labels = load_daily_weather_trajectories()
    generate_figure_3_5(daily_by_year, clim_doy, ref_dates, month_ticks, month_labels)
    generate_figure_3_6(daily_by_year, clim_doy, ref_dates, month_ticks, month_labels)

    # Figures 3.7 and 3.8
    generate_figure_3_7(gdf_all, df_counties)
    generate_figure_3_8(df_yield)

    print("All Chapter 3 figures generated successfully into output/thesis/figures/!")


if __name__ == "__main__":
    main()
