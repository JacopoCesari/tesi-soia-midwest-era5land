"""
Generate Figure 3.9:
"Spatial Contrast of Benchmark Agro-Climatic Shock Archetypes across the Midwestern Corn Belt"
Layout: 2x2 grid of maps
- Row 1: 2012 Severe Drought & Heatwave (Adverse Shock: -11.7%)
- Row 2: 2021 Modern Climatic Optimum (Bumper Harvest: +10.8%)
- Column 1: July--August Cumulative Net Climatic Water Balance (P - ET0, mm)
- Column 2: July--August Mean Vapor Pressure Deficit (VPD, kPa)

Features:
- Continuous interpolated spatial field (Rbf thin-plate spline, 300x300 grid)
- 135 study county boundaries overlaid
- State borders and labels
- Coordinate graticules
- Two separate synchronized colorbars (one for Water Balance, one for VPD)
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
OUTPUT_PNG = os.path.join(ROOT_DIR, "output", "thesis", "figures", "fig_spatial_shock_comparison.png")
OUTPUT_PDF = os.path.join(ROOT_DIR, "output", "thesis", "figures", "fig_spatial_shock_comparison.pdf")

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

# Load metrics for the 2 benchmark years: 2012 (drought) and 2021 (bumper)
years = [2012, 2021]
metrics = {}
for y in years:
    fpath = os.path.join(WEATHER_DIR, f"county_daily_{y}.parquet")
    df = pd.read_parquet(fpath)
    df['date'] = pd.to_datetime(df['date'])
    df_ja = df[df['date'].dt.month.isin([7, 8])].copy()
    agg = df_ja.groupby('county_fips').agg({
        'p_minus_et0': 'sum',
        'vapor_pressure_deficit': 'mean',
        'total_precipitation': 'sum'
    }).reset_index()
    agg['county_fips'] = agg['county_fips'].astype(str).str.zfill(5)
    metrics[y] = agg

# Spatial interpolation grid (300 x 300)
bounds = mw_boundary.bounds  # minx, miny, maxx, maxy
grid_x, grid_y = np.mgrid[bounds[0]:bounds[2]:300j, bounds[1]:bounds[3]:300j]

points_in_mw = [shapely.geometry.Point(x, y).within(mw_boundary) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
mask_mw = np.array(points_in_mw).reshape(grid_x.shape)

interp_water = {}
interp_vpd = {}

for y in years:
    merged = gdf_selected.merge(metrics[y], left_on='GEOID', right_on='county_fips')
    pts = np.vstack([merged['centroid_x'].values, merged['centroid_y'].values]).T
    
    # 1. P - ET0
    vals_wb = merged['p_minus_et0'].values
    rbf_wb = Rbf(pts[:, 0], pts[:, 1], vals_wb, function='linear', smooth=0.5)
    grid_wb = rbf_wb(grid_x, grid_y)
    grid_wb[~mask_mw] = np.nan
    interp_water[y] = grid_wb
    
    # 2. VPD
    vals_vpd = merged['vapor_pressure_deficit'].values
    rbf_vpd = Rbf(pts[:, 0], pts[:, 1], vals_vpd, function='linear', smooth=0.5)
    grid_vpd = rbf_vpd(grid_x, grid_y)
    grid_vpd[~mask_mw] = np.nan
    interp_vpd[y] = grid_vpd

print("Interpolation completed!")

# ==============================================================================
# BUILD FIGURE: 2x2 Grid of Maps + 2 Horizontal Colorbars at Bottom
# ==============================================================================
fig = plt.figure(figsize=(13.5, 12.5), dpi=300)
gs = gridspec.GridSpec(3, 2, height_ratios=[1.0, 1.0, 0.08], hspace=0.22, wspace=0.10,
                       top=0.95, bottom=0.06, left=0.06, right=0.94)

# Color norms
norm_wb = TwoSlopeNorm(vmin=-250, vcenter=0, vmax=150)
cmap_wb = plt.cm.RdYlBu

norm_vpd = Normalize(vmin=0.6, vmax=1.8)
cmap_vpd = plt.cm.YlOrRd

map_configs = [
    # (row, col, yr, var_type, grid_data, norm, cmap, title, subtitle)
    (0, 0, 2012, "wb", interp_water[2012], norm_wb, cmap_wb,
     "(a) 2012 Shock: Net Water Balance ($P - ET_0$)", "Yield Anomaly: −11.7% | Acute Moisture Deficit"),
    (0, 1, 2012, "vpd", interp_vpd[2012], norm_vpd, cmap_vpd,
     "(b) 2012 Shock: Evaporative Demand ($VPD$)", "Yield Anomaly: −11.7% | Severe Atmospheric Vapor Deficit"),
    (1, 0, 2021, "wb", interp_water[2021], norm_wb, cmap_wb,
     "(c) 2021 Bumper: Net Water Balance ($P - ET_0$)", "Yield Anomaly: +10.8% | Balanced Moisture Replenishment"),
    (1, 1, 2021, "vpd", interp_vpd[2021], norm_vpd, cmap_vpd,
     "(d) 2021 Bumper: Evaporative Demand ($VPD$)", "Yield Anomaly: +10.8% | Benign Transpirational Demand")
]

state_coords = {"MN": (-94.6, 46.2), "IA": (-93.6, 42.1), "MO": (-92.6, 38.6),
                "IL": (-89.2, 40.0), "IN": (-86.2, 40.0), "OH": (-82.7, 40.3)}

for r, c, yr, vtype, gdata, norm, cmap, title, subtitle in map_configs:
    ax = fig.add_subplot(gs[r, c])
    
    # 1. Background context states
    gdf_context_states.plot(ax=ax, facecolor='#fafafa', edgecolor='#e2e8f0', linewidth=0.5, zorder=1)
    # Background Midwest counties
    gdf_mw.plot(ax=ax, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=0.35, zorder=2)
    
    # 2. Continuous field
    cf = ax.contourf(grid_x, grid_y, gdata, levels=35, cmap=cmap, norm=norm, zorder=3, alpha=0.90)
    
    # 3. Contour isolines
    if vtype == "wb":
        cs = ax.contour(grid_x, grid_y, gdata, levels=[-200, -150, -100, -50, 0, 50, 100],
                        colors='#334155', linewidths=0.4, linestyles='-', alpha=0.5, zorder=4)
        cs_zero = ax.contour(grid_x, grid_y, gdata, levels=[0],
                             colors='#0f172a', linewidths=0.9, linestyles='--', zorder=4)
        ax.clabel(cs, inline=True, fmt='%d', fontsize=6.5, colors='#1e293b')
    else:
        cs = ax.contour(grid_x, grid_y, gdata, levels=[0.8, 1.0, 1.2, 1.4, 1.6],
                        colors='#475569', linewidths=0.4, linestyles='-', alpha=0.5, zorder=4)
        cs_stress = ax.contour(grid_x, grid_y, gdata, levels=[1.5],
                               colors='#7f1d1d', linewidths=0.9, linestyles='--', zorder=4)
        ax.clabel(cs, inline=True, fmt='%.1f', fontsize=6.5, colors='#0f172a')
        
    # 4. Study county boundaries
    gdf_selected.plot(ax=ax, facecolor='none', edgecolor='#1e293b', linewidth=0.55, zorder=5)
    
    # 5. State borders
    gdf_states.plot(ax=ax, facecolor='none', edgecolor='#0f172a', linewidth=1.0, zorder=6)
    
    # State labels
    for st, (sx, sy) in state_coords.items():
        ax.text(sx, sy, st, fontsize=8.5, fontweight='bold', color='#0f172a', alpha=0.6, ha='center', zorder=7)
        
    ax.set_xlim(-97.3, -80.5)
    ax.set_ylim(36.0, 49.2)
    ax.set_aspect(1.0 / np.cos(np.radians(42.5)))
    
    # Graticules
    ax.set_xticks([-96, -92, -88, -84])
    ax.set_xticklabels([r'$96^\circ\mathrm{W}$', r'$92^\circ\mathrm{W}$', r'$88^\circ\mathrm{W}$', r'$84^\circ\mathrm{W}$'], fontsize=7.5)
    ax.set_yticks([38, 42, 46])
    ax.set_yticklabels([r'$38^\circ\mathrm{N}$', r'$42^\circ\mathrm{N}$', r'$46^\circ\mathrm{N}$'], fontsize=7.5)
    ax.grid(True, linestyle=':', alpha=0.40, color='#94a3b8', zorder=1)
    
    ax.set_title(title, fontsize=9.8, fontweight='bold', pad=5, color='#0f172a')
    ax.text(0.5, -0.065, subtitle, transform=ax.transAxes, ha='center', fontsize=7.6, fontstyle='italic', color='#475569')

# Bottom Colorbars
cbar_ax_wb = fig.add_subplot(gs[2, 0])
sm_wb = plt.cm.ScalarMappable(cmap=cmap_wb, norm=norm_wb)
sm_wb.set_array([])
cb_wb = fig.colorbar(sm_wb, cax=cbar_ax_wb, orientation='horizontal')
cb_wb.set_label(r'July--August Net Climatic Water Balance: $P - ET_0$ (mm)  [Red = Deficit $\leftarrow\rightarrow$ Blue = Surplus]',
                fontsize=8.5, fontweight='medium', labelpad=4)
cb_wb.ax.tick_params(labelsize=7.5)

cbar_ax_vpd = fig.add_subplot(gs[2, 1])
sm_vpd = plt.cm.ScalarMappable(cmap=cmap_vpd, norm=norm_vpd)
sm_vpd.set_array([])
cb_vpd = fig.colorbar(sm_vpd, cax=cbar_ax_vpd, orientation='horizontal')
cb_vpd.set_label(r'July--August Mean Vapor Pressure Deficit: $VPD$ (kPa)  [Yellow = Benign $\leftarrow\rightarrow$ Red = High Stress]',
                 fontsize=8.5, fontweight='medium', labelpad=4)
cb_vpd.ax.tick_params(labelsize=7.5)

os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PDF, bbox_inches='tight')
plt.close()
print(f"Saved 2x2 publication grade Figure 3.9 to:\n  {OUTPUT_PNG}\n  {OUTPUT_PDF}")
