"""
Generate three distinct candidate designs for Figure 3.9:
Spatial Contrast of Benchmark Agro-Climatic Shock Archetypes across the Midwestern Corn Belt

Candidates:
1. Option 1: 1988 (Severe Historic Drought, -25.6%) vs 1994 (Historic Bumper, +10.7%)
   - Variables: Precipitation (P, mm) vs Heat Days (HD30, days)
2. Option 2: 1993 (Pluvial Flood Shock, -10.5%) vs 2021 (Modern Bumper, +10.8%)
   - Variables: Net Water Balance (P - ET0, mm) vs Heat Days (HD30, days)
3. Option 3: 2012 (Midsummer Flash Drought, -11.7%) vs 2016 (Modern Bumper, +10.8%)
   - Variables: Precipitation (P, mm) vs Mean Daily Maximum Temperature (Tmax, °C)
"""

import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import Rbf
import shapely.geometry
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = r"c:\Users\JacopoCesari-Aretésr\Desktop\Tesi"
SHAPEFILE = os.path.join(ROOT_DIR, "work", "data", "external", "usda_nass_and_county_boundaries", "census_counties", "cb_2020_us_county_500k.shp")
COUNTIES_CSV = os.path.join(ROOT_DIR, "output", "data", "auxiliary", "counties.csv")
WEATHER_DIR = os.path.join(ROOT_DIR, "work", "data", "interim", "county_daily_weather", "production")
FIGURES_DIR = os.path.join(ROOT_DIR, "output", "thesis", "figures")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 9.5, 'axes.titlesize': 11,
    'figure.titlesize': 13, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

print("Loading boundaries and setting up spatial interpolation...")
gdf_all = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
df_counties = pd.read_csv(COUNTIES_CSV, dtype={'county_fips': str})
df_counties['county_fips'] = df_counties['county_fips'].str.zfill(5)
selected_fips = set(df_counties['county_fips'])

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
    fpath = os.path.join(WEATHER_DIR, f"county_daily_{year}.parquet")
    df = pd.read_parquet(fpath)
    df['date'] = pd.to_datetime(df['date'])
    df_ja = df[df['date'].dt.month.isin([7, 8])].copy()
    df_ja['sm_root'] = (0.07 * df_ja['volumetric_soil_water_layer_1'] +
                        0.21 * df_ja['volumetric_soil_water_layer_2'] +
                        0.72 * df_ja['volumetric_soil_water_layer_3'])
    agg = df_ja.groupby('county_fips').agg({
        'p_minus_et0': 'sum',
        'total_precipitation': 'sum',
        'heat_day_30': 'sum',
        'air_temperature_maximum': 'mean',
        'vapor_pressure_deficit': 'mean',
        'sm_root': 'mean'
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

def plot_2x2_comparison(configs, cbar_left_info, cbar_right_info, out_png, out_pdf):
    fig = plt.figure(figsize=(13.5, 12.5), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.0, 1.0, 0.08], hspace=0.22, wspace=0.10,
                           top=0.95, bottom=0.06, left=0.06, right=0.94)

    for r, c, gdata, norm, cmap, title, subtitle, contour_spec in configs:
        ax = fig.add_subplot(gs[r, c])
        gdf_context_states.plot(ax=ax, facecolor='#fafafa', edgecolor='#e2e8f0', linewidth=0.5, zorder=1)
        gdf_mw.plot(ax=ax, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=0.35, zorder=2)
        cf = ax.contourf(grid_x, grid_y, gdata, levels=35, cmap=cmap, norm=norm, zorder=3, alpha=0.90)

        # Contours
        if contour_spec:
            levels, fmt, colors = contour_spec
            cs = ax.contour(grid_x, grid_y, gdata, levels=levels, colors=colors, linewidths=0.4, alpha=0.5, zorder=4)
            ax.clabel(cs, inline=True, fmt=fmt, fontsize=6.5)

        gdf_selected.plot(ax=ax, facecolor='none', edgecolor='#1e293b', linewidth=0.55, zorder=5)
        gdf_states.plot(ax=ax, facecolor='none', edgecolor='#0f172a', linewidth=1.0, zorder=6)
        for st, (sx, sy) in state_coords.items():
            ax.text(sx, sy, st, fontsize=8.5, fontweight='bold', color='#0f172a', alpha=0.6, ha='center', zorder=7)

        ax.set_xlim(-97.3, -80.5)
        ax.set_ylim(36.0, 49.2)
        ax.set_aspect(1.0 / np.cos(np.radians(42.5)))
        ax.set_xticks([-96, -92, -88, -84])
        ax.set_xticklabels([r'$96^\circ\mathrm{W}$', r'$92^\circ\mathrm{W}$', r'$88^\circ\mathrm{W}$', r'$84^\circ\mathrm{W}$'], fontsize=7.5)
        ax.set_yticks([38, 42, 46])
        ax.set_yticklabels([r'$38^\circ\mathrm{N}$', r'$42^\circ\mathrm{N}$', r'$46^\circ\mathrm{N}$'], fontsize=7.5)
        ax.grid(True, linestyle=':', alpha=0.40, color='#94a3b8')
        ax.set_title(title, fontsize=9.8, fontweight='bold', pad=5)
        ax.text(0.5, -0.065, subtitle, transform=ax.transAxes, ha='center', fontsize=7.6, fontstyle='italic', color='#475569')

    # Colorbars
    cbar_ax_left = fig.add_subplot(gs[2, 0])
    cmap_l, norm_l, label_l = cbar_left_info
    cb_l = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_l, norm=norm_l), cax=cbar_ax_left, orientation='horizontal')
    cb_l.set_label(label_l, fontsize=8.5, labelpad=4)
    cb_l.ax.tick_params(labelsize=7.5)

    cbar_ax_right = fig.add_subplot(gs[2, 1])
    cmap_r, norm_r, label_r = cbar_right_info
    cb_r = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_r, norm=norm_r), cax=cbar_ax_right, orientation='horizontal')
    cb_r.set_label(label_r, fontsize=8.5, labelpad=4)
    cb_r.ax.tick_params(labelsize=7.5)

    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {out_png}")

# ==============================================================================
# OPTION 1: 1988 (Historic Drought: -25.6%) vs 1994 (Historic Bumper: +10.7%)
# Variables: Precipitation (P, mm) vs Heat Days (HD30, days)
# ==============================================================================
print("\n--- Generating Option 1 (1988 vs 1994: P vs HD30) ---")
m1988 = get_metrics(1988)
m1994 = get_metrics(1994)

grid_p_1988 = interpolate_var(m1988, 'total_precipitation')
grid_hd_1988 = interpolate_var(m1988, 'heat_day_30')
grid_p_1994 = interpolate_var(m1994, 'total_precipitation')
grid_hd_1994 = interpolate_var(m1994, 'heat_day_30')

norm_p = Normalize(vmin=80, vmax=320)
cmap_p = plt.cm.YlGnBu
norm_hd = Normalize(vmin=0, vmax=55)
cmap_hd = plt.cm.YlOrRd

configs_1 = [
    (0, 0, grid_p_1988, norm_p, cmap_p,
     r'(a) 1988 Historic Drought: July--August Rainfall ($P$)',
     r'Yield Anomaly: -25.6% | Severe West Deficit (<150 mm) vs Scattered East Showers',
     ([100, 150, 200, 250], '%d', '#334155')),
    (0, 1, grid_hd_1988, norm_hd, cmap_hd,
     r'(b) 1988 Historic Drought: Extreme Heat Stress ($HD_{30}$)',
     r'Yield Anomaly: -25.6% | Relentless Heat Dome (>35 Days in Core Corn Belt)',
     ([20, 30, 40, 50], '%d', '#475569')),
    (1, 0, grid_p_1994, norm_p, cmap_p,
     r'(c) 1994 Historic Bumper: July--August Rainfall ($P$)',
     r'Yield Anomaly: +10.7% | Plentiful, Well-Distributed Rain (>200 mm Everywhere)',
     ([150, 200, 250, 300], '%d', '#334155')),
    (1, 1, grid_hd_1994, norm_hd, cmap_hd,
     r'(d) 1994 Historic Bumper: Extreme Heat Stress ($HD_{30}$)',
     r'Yield Anomaly: +10.7% | Near-Complete Absence of Heat Stress (<10 Days in North/East)',
     ([5, 10, 15, 25], '%d', '#475569'))
]

cbar_l_1 = (cmap_p, norm_p, r'July--August Total Precipitation: $P$ (mm)  [Yellow = Deficit $\leftarrow\rightarrow$ Blue = Plentiful Rainfall]')
cbar_r_1 = (cmap_hd, norm_hd, r'July--August Extreme Heat Days: $HD_{30}$ ($T_{\mathrm{max}} \geq 30^\circ\mathrm{C}$, days)  [Yellow = Mild $\leftarrow\rightarrow$ Red = High Thermal Stress]')

plot_2x2_comparison(
    configs_1, cbar_l_1, cbar_r_1,
    os.path.join(FIGURES_DIR, "fig_spatial_shock_option1_1988_1994.png"),
    os.path.join(FIGURES_DIR, "fig_spatial_shock_option1_1988_1994.pdf")
)

# ==============================================================================
# OPTION 2: 1993 (Flood Shock: -10.5%) vs 2021 (Modern Bumper: +10.8%)
# Variables: Net Climatic Water Balance (P - ET0, mm) vs Heat Days (HD30, days)
# ==============================================================================
print("\n--- Generating Option 2 (1993 vs 2021: P - ET0 vs HD30) ---")
m1993 = get_metrics(1993)
m2021 = get_metrics(2021)

grid_wb_1993 = interpolate_var(m1993, 'p_minus_et0')
grid_hd_1993 = interpolate_var(m1993, 'heat_day_30')
grid_wb_2021 = interpolate_var(m2021, 'p_minus_et0')
grid_hd_2021 = interpolate_var(m2021, 'heat_day_30')

norm_wb = TwoSlopeNorm(vmin=-250, vcenter=0, vmax=250)
cmap_wb = plt.cm.RdYlBu

configs_2 = [
    (0, 0, grid_wb_1993, norm_wb, cmap_wb,
     r'(a) 1993 Flood Shock: Net Water Balance ($P - ET_0$)',
     r'Yield Anomaly: -10.5% | Severe Pluvial Surplus West (>+200 mm) vs Deficit Southeast',
     ([-150, -100, -50, 0, 50, 100, 150, 200], '%d', '#334155')),
    (0, 1, grid_hd_1993, norm_hd, cmap_hd,
     r'(b) 1993 Flood Shock: Extreme Heat Days ($HD_{30}$)',
     r'Yield Anomaly: -10.5% | Cold/Overcast North (<5 Days) vs Southern Heat (>50 Days)',
     ([10, 20, 30, 40, 50], '%d', '#475569')),
    (1, 0, grid_wb_2021, norm_wb, cmap_wb,
     r'(c) 2021 Modern Bumper: Net Water Balance ($P - ET_0$)',
     r'Yield Anomaly: +10.8% | Controlled, Favorable Hydrological Regime Across Midwest',
     ([-100, -50, 0, 50], '%d', '#334155')),
    (1, 1, grid_hd_2021, norm_hd, cmap_hd,
     r'(d) 2021 Modern Bumper: Extreme Heat Days ($HD_{30}$)',
     r'Yield Anomaly: +10.8% | Balanced, Moderate Heat Accumulation Without Spikes',
     ([15, 20, 25, 35], '%d', '#475569'))
]

cbar_l_2 = (cmap_wb, norm_wb, r'July--August Net Climatic Water Balance: $P - ET_0$ (mm)  [Red = Deficit $\leftarrow\rightarrow$ Blue = Severe Pluvial Surplus]')
cbar_r_2 = (cmap_hd, norm_hd, r'July--August Extreme Heat Days: $HD_{30}$ ($T_{\mathrm{max}} \geq 30^\circ\mathrm{C}$, days)  [Yellow = Mild $\leftarrow\rightarrow$ Red = High Thermal Stress]')

plot_2x2_comparison(
    configs_2, cbar_l_2, cbar_r_2,
    os.path.join(FIGURES_DIR, "fig_spatial_shock_option2_1993_2021.png"),
    os.path.join(FIGURES_DIR, "fig_spatial_shock_option2_1993_2021.pdf")
)

# ==============================================================================
# OPTION 3: 2012 (Flash Drought: -11.7%) vs 2016 (Modern Bumper: +10.8%)
# Variables: Precipitation (P, mm) vs Mean Daily Maximum Temperature (Tmax, °C)
# ==============================================================================
print("\n--- Generating Option 3 (2012 vs 2016: P vs Tmax) ---")
m2012 = get_metrics(2012)
m2016 = get_metrics(2016)

grid_p_2012 = interpolate_var(m2012, 'total_precipitation')
grid_tmax_2012 = interpolate_var(m2012, 'air_temperature_maximum')
grid_p_2016 = interpolate_var(m2016, 'total_precipitation')
grid_tmax_2016 = interpolate_var(m2016, 'air_temperature_maximum')

norm_p3 = Normalize(vmin=60, vmax=320)
norm_tmax = Normalize(vmin=26.0, vmax=35.0)
cmap_tmax = plt.cm.inferno

configs_3 = [
    (0, 0, grid_p_2012, norm_p3, cmap_p,
     r'(a) 2012 Flash Drought: July--August Rainfall ($P$)',
     r'Yield Anomaly: -11.7% | Acute Rainfall Collapse in Iowa & Illinois (<80 mm)',
     ([80, 100, 120, 150], '%d', '#334155')),
    (0, 1, grid_tmax_2012, norm_tmax, cmap_tmax,
     r'(b) 2012 Flash Drought: Mean Max Temperature ($T_{\mathrm{max}}$)',
     r'Yield Anomaly: -11.7% | Scorching Summer Temperatures (>33°C in Central/South)',
     ([28, 30, 32, 34], '%.0f°C', '#ffffff')),
    (1, 0, grid_p_2016, norm_p3, cmap_p,
     r'(c) 2016 Modern Bumper: July--August Rainfall ($P$)',
     r'Yield Anomaly: +10.8% | Abundant Recharge Waves in Core Corn Belt (>260 mm)',
     ([180, 220, 260, 300], '%d', '#334155')),
    (1, 1, grid_tmax_2016, norm_tmax, cmap_tmax,
     r'(d) 2016 Modern Bumper: Mean Max Temperature ($T_{\mathrm{max}}$)',
     r'Yield Anomaly: +10.8% | Consistently Mild Daytime Temperatures (<29°C Everywhere)',
     ([27, 28, 29, 30], '%.0f°C', '#ffffff'))
]

cbar_l_3 = (cmap_p, norm_p3, r'July--August Total Precipitation: $P$ (mm)  [Yellow = Acute Drought $\leftarrow\rightarrow$ Blue = Plentiful Rainfall]')
cbar_r_3 = (cmap_tmax, norm_tmax, r'July--August Mean Daily Maximum Temperature: $T_{\mathrm{max}}$ ($^\circ\mathrm{C}$)  [Dark/Purple = Cool $\leftarrow\rightarrow$ Yellow = Extreme Heat]')

plot_2x2_comparison(
    configs_3, cbar_l_3, cbar_r_3,
    os.path.join(FIGURES_DIR, "fig_spatial_shock_option3_2012_2016.png"),
    os.path.join(FIGURES_DIR, "fig_spatial_shock_option3_2012_2016.pdf")
)

print("\nAll 3 candidate figures generated successfully!")
