"""
Generate Figure 3.9:
"Spatial Contrast of Benchmark Agro-Climatic Shock Archetypes across the Midwestern Corn Belt"
Layout: 2x2 grid of maps
- Row 1: 2012 Midsummer Flash Drought (Severe Shock: -11.7% yield anomaly)
- Row 2: 2016 Modern Climatic Optimum (Bumper Harvest: +10.8% yield anomaly)
- Column 1: July--August Total Cumulative Precipitation (P, mm) [YlGnBu]
- Column 2: July--August Mean Daily Maximum Temperature (T_max, °C) [inferno]

Features:
- Continuous interpolated spatial field (Rbf thin-plate spline, 300x300 grid)
- 135 study county boundaries overlaid
- State borders and labels
- Coordinate graticules
- Two separate synchronized colorbars (one for Precipitation, one for Maximum Temperature)
"""

import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.interpolate import Rbf
import shapely.geometry
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
_script_path = Path(__file__).resolve()
ROOT_DIR = _script_path.parents[2] if (_script_path.parents[1].name == "output") else _script_path.parents[1]
OUTPUT_DIR_PATH = ROOT_DIR / "output" if (ROOT_DIR / "output").exists() else ROOT_DIR

SHAPEFILE = str(ROOT_DIR / "data" / "raw" / "census_counties" / "cb_2020_us_county_500k.shp")
COUNTIES_CSV = str(ROOT_DIR / "data" / "auxiliary" / "counties.csv")
WEATHER_DIR = str(ROOT_DIR / "data" / "weather" / "daily_counties")
OUTPUT_PNG = str(OUTPUT_DIR_PATH / "thesis" / "figures" / "fig_spatial_shock_comparison.png")
OUTPUT_PDF = str(OUTPUT_DIR_PATH / "thesis" / "figures" / "fig_spatial_shock_comparison.pdf")

# Publication typography
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 9.5,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

print("Loading boundaries and preparing spatial context...")
gdf_all = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
df_counties = pd.read_csv(COUNTIES_CSV, dtype={'county_fips': str})
df_counties['county_fips'] = df_counties['county_fips'].str.zfill(5)
selected_fips = set(df_counties['county_fips'])

# Midwest study states + neighboring states for regional context
midwest_fips = ['17', '18', '19', '27', '29', '39']  # IL, IN, IA, MN, MO, OH
context_fips = ['55', '26', '21', '51', '54', '42', '31', '20', '46', '38']

gdf_mw = gdf_all[gdf_all['STATEFP'].isin(midwest_fips)].copy()
gdf_context = gdf_all[gdf_all['STATEFP'].isin(midwest_fips + context_fips)].copy()
gdf_selected = gdf_mw[gdf_mw['GEOID'].isin(selected_fips)].copy()

gdf_states = gdf_mw.dissolve(by='STATEFP')
gdf_context_states = gdf_context.dissolve(by='STATEFP')
mw_boundary = unary_union(gdf_states.geometry)

# Centroids for interpolation
gdf_selected['centroid_x'] = gdf_selected.geometry.centroid.x
gdf_selected['centroid_y'] = gdf_selected.geometry.centroid.y

# Spatial interpolation grid (300 x 300)
bounds = mw_boundary.bounds  # minx, miny, maxx, maxy
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

print("Calculating July--August metrics for 2012 and 2016...")
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

print("Rendering 2x2 publication figure...")
fig = plt.figure(figsize=(14.0, 12.5), dpi=300)
gs = gridspec.GridSpec(3, 2, height_ratios=[1.0, 1.0, 0.07], hspace=0.25, wspace=0.18,
                       top=0.96, bottom=0.06, left=0.05, right=0.95)

for r, c, gdata, norm, cmap, title, subtitle, contour_spec in configs:
    ax = fig.add_subplot(gs[r, c])
    gdf_context_states.plot(ax=ax, facecolor='#fafafa', edgecolor='#e2e8f0', linewidth=0.5, zorder=1)
    gdf_mw.plot(ax=ax, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=0.35, zorder=2)
    cf = ax.contourf(grid_x, grid_y, gdata, levels=35, cmap=cmap, norm=norm, zorder=3, alpha=0.90)

    # Contours
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

# Synchronized bottom colorbars
cbar_ax_left = fig.add_subplot(gs[2, 0])
cb_l = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_p, norm=norm_p), cax=cbar_ax_left, orientation='horizontal')
cb_l.set_label(r'July--August Total Precipitation: $P$ (mm)', fontsize=10.0, fontweight='bold', labelpad=5)
cb_l.ax.tick_params(labelsize=8.5)

cbar_ax_right = fig.add_subplot(gs[2, 1])
cb_r = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_tmax, norm=norm_tmax), cax=cbar_ax_right, orientation='horizontal')
cb_r.set_label(r'July--August Mean Daily Maximum Temperature: $T_{\mathrm{max}}$ ($^\circ\mathrm{C}$)', fontsize=10.0, fontweight='bold', labelpad=5)
cb_r.ax.tick_params(labelsize=8.5)

print(f"Saving to {OUTPUT_PNG} and {OUTPUT_PDF}...")
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PDF, bbox_inches='tight')
plt.close()
print("Figure 3.9 generation complete!")
