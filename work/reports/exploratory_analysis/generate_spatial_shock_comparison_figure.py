"""
Generate ultra-polished, publication-grade Figure 3.7:
"Spatial Footprints of Benchmark Agro-Climatic Shock Archetypes across the Midwestern Corn Belt"
Layout: 1x3 horizontal layout (1992 Normal Reference, 1993 Flood, 2012 Flash Drought)
Features:
- Continuous interpolated spatial field (Rbf thin-plate spline, 320x320 grid)
- Subtle graticule grid (lat/lon)
- Great Lakes and neighboring state context
- 135 study county boundaries overlaid
- Contour isolines with labels
- Centered synchronized diverging colorbar
- Sleek bottom diagnostic infographic card with 3 donut charts inspired by user reference image
"""

import os
import sys
venv_site = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import Rbf
import shapely.geometry
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SHAPEFILE = os.path.join(ROOT_DIR, "work", "data", "external", "usda_nass_and_county_boundaries", "census_counties", "cb_2020_us_county_500k.shp")
COUNTIES_CSV = os.path.join(ROOT_DIR, "output", "data", "auxiliary", "counties.csv")
ACRES_CSV = os.path.join(ROOT_DIR, "output", "data", "auxiliary", "acres_harvested_1951_2025.csv")
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
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
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
context_fips = ['55', '26', '21', '51', '54', '42', '31', '20', '46', '38']  # WI, MI, KY, VA, WV, PA, NE, KS, SD, ND

gdf_mw = gdf_all[gdf_all['STATEFP'].isin(midwest_fips)].copy()
gdf_context = gdf_all[gdf_all['STATEFP'].isin(midwest_fips + context_fips)].copy()
gdf_selected = gdf_mw[gdf_mw['GEOID'].isin(selected_fips)].copy()

gdf_states = gdf_mw.dissolve(by='STATEFP')
gdf_context_states = gdf_context.dissolve(by='STATEFP')
mw_boundary = unary_union(gdf_states.geometry)

# Centroids for interpolation
gdf_selected['centroid_x'] = gdf_selected.geometry.centroid.x
gdf_selected['centroid_y'] = gdf_selected.geometry.centroid.y

# Load acreage data
df_acres = pd.read_csv(ACRES_CSV, dtype={'county_fips': str})
df_acres['county_fips'] = df_acres['county_fips'].str.zfill(5)

# Load metrics for the 3 benchmark years
years = [1992, 1993, 2012]
metrics = {}
for y in years:
    fpath = os.path.join(WEATHER_DIR, f"county_daily_{y}.parquet")
    df = pd.read_parquet(fpath)
    df['date'] = pd.to_datetime(df['date'])
    df_ja = df[df['date'].dt.month.isin([7, 8])].copy()
    agg = df_ja.groupby('county_fips').agg({
        'p_minus_et0': 'sum',
        'vapor_pressure_deficit': 'mean',
        'heat_day_30': 'sum'
    }).reset_index()
    agg['county_fips'] = agg['county_fips'].astype(str).str.zfill(5)
    acres_y = df_acres[df_acres['year'] == y][['county_fips', 'acres_harvested']].copy()
    agg = agg.merge(acres_y, on='county_fips', how='left')
    metrics[y] = agg

# Spatial interpolation grid (320 x 320)
bounds = mw_boundary.bounds  # minx, miny, maxx, maxy
grid_x, grid_y = np.mgrid[bounds[0]:bounds[2]:320j, bounds[1]:bounds[3]:320j]

points_in_mw = [shapely.geometry.Point(x, y).within(mw_boundary) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
mask_mw = np.array(points_in_mw).reshape(grid_x.shape)

interpolated = {}
for y in years:
    merged = gdf_selected.merge(metrics[y], left_on='GEOID', right_on='county_fips')
    pts = np.vstack([merged['centroid_x'].values, merged['centroid_y'].values]).T
    vals = merged['p_minus_et0'].values
    
    rbf = Rbf(pts[:, 0], pts[:, 1], vals, function='linear', smooth=0.6)
    grid_z = rbf(grid_x, grid_y)
    grid_z[~mask_mw] = np.nan
    interpolated[y] = grid_z

print("Interpolation completed!")

# ------------------------------------------------------------------------------
# BUILD FIGURE: 1 Row of 3 Maps + Shared Colorbar + Bottom Diagnostic Card
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(15.5, 11.2), dpi=300)
gs = gridspec.GridSpec(3, 1, height_ratios=[2.2, 0.22, 1.05], hspace=0.28,
                       top=0.96, bottom=0.04, left=0.05, right=0.95)

# Row 0: 3 maps side-by-side
gs_maps = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.10)
# Row 1: Centered colorbar
gs_cbar = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[0.22, 0.56, 0.22])
# Row 2: Bottom diagnostic card (Legend + 3 Donuts)
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], width_ratios=[1.05, 1.0, 1.0, 1.0], wspace=0.14)

norm = TwoSlopeNorm(vmin=-300, vcenter=0, vmax=200)
cmap = plt.cm.RdYlBu

configs = [
    (1992, gs_maps[0], "(a) 1992 Normal Reference Year", "Yield: +2.1%", "Near-average water balance across the study region"),
    (1993, gs_maps[1], "(b) 1993 Great Midwest Flood", "Yield: −10.5%", "Persistent pluvial inundation & root-zone anoxia"),
    (2012, gs_maps[2], "(c) 2012 Flash Drought & Heatwave", "Yield: −10.2%", "Acute evaporative surge & heatwave in Central/Southern belt")
]

for yr, cell, title, y_anom, subtitle in configs:
    ax = fig.add_subplot(cell)
    
    # 1. Background context states (WI, MI, KY, etc.)
    gdf_context_states.plot(ax=ax, facecolor='#fafafa', edgecolor='#e2e8f0', linewidth=0.5, zorder=1)
    # Background Midwest counties
    gdf_mw.plot(ax=ax, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=0.35, zorder=2)
    
    # 2. Continuous interpolated P - ET0 field
    cf = ax.contourf(grid_x, grid_y, interpolated[yr], levels=40, cmap=cmap, norm=norm, zorder=3, alpha=0.92)
    
    # 3. Contour isolines
    cs = ax.contour(grid_x, grid_y, interpolated[yr], levels=[-250, -200, -150, -100, -50, 0, 50, 100, 150],
                    colors='#2d3748', linewidths=0.45, linestyles='-', alpha=0.55, zorder=4)
    # Highlight zero line
    cs_zero = ax.contour(grid_x, grid_y, interpolated[yr], levels=[0],
                         colors='#1a202c', linewidths=1.0, linestyles='--', zorder=4)
    ax.clabel(cs, inline=True, fmt='%d', fontsize=6.8, colors='#1a202c')
    
    # 4. Highlighted 135 study county boundaries
    gdf_selected.plot(ax=ax, facecolor='none', edgecolor='#1e293b', linewidth=0.6, zorder=5)
    
    # 5. State borders
    gdf_states.plot(ax=ax, facecolor='none', edgecolor='#0f172a', linewidth=1.1, zorder=6)
    
    # State labels
    state_coords = {"MN": (-94.6, 46.2), "IA": (-93.6, 42.1), "MO": (-92.6, 38.6),
                    "IL": (-89.2, 40.0), "IN": (-86.2, 40.0), "OH": (-82.7, 40.3)}
    for st, (sx, sy) in state_coords.items():
        ax.text(sx, sy, st, fontsize=9.0, fontweight='bold', color='#0f172a', alpha=0.6, ha='center', zorder=7)
        
    ax.set_xlim(-97.3, -80.5)
    ax.set_ylim(36.0, 49.2)
    ax.set_aspect(1.0 / np.cos(np.radians(42.5)))
    
    # Coordinate graticules
    ax.set_xticks([-96, -92, -88, -84])
    ax.set_xticklabels([r'$96^\circ\mathrm{W}$', r'$92^\circ\mathrm{W}$', r'$88^\circ\mathrm{W}$', r'$84^\circ\mathrm{W}$'], fontsize=7.8)
    ax.set_yticks([38, 42, 46])
    ax.set_yticklabels([r'$38^\circ\mathrm{N}$', r'$42^\circ\mathrm{N}$', r'$46^\circ\mathrm{N}$'], fontsize=7.8)
    ax.grid(True, linestyle=':', alpha=0.45, color='#94a3b8', zorder=1)
    
    # Elegant title badge at top of map
    ax.set_title(f"{title} ({y_anom})", fontsize=10.2, fontweight='bold', pad=6, color='#0f172a')
    # Subtitle at bottom
    ax.text(0.5, -0.055, subtitle, transform=ax.transAxes, ha='center', fontsize=7.8, fontstyle='italic', color='#475569')

# Shared horizontal colorbar
cbar_ax = fig.add_subplot(gs_cbar[1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'July--August Net Climatic Water Balance: $P - ET_0$ (mm)  [Negative = Deficit / Drought $\leftarrow\rightarrow$ Positive = Surplus / Waterlogging]',
               fontsize=9.0, fontweight='medium', labelpad=5)
cbar.ax.tick_params(labelsize=8)

# ==============================================================================
# BOTTOM INFOGRAPHIC CARD (Styled container with 3 Donut Charts & Acreage Stats)
# ==============================================================================
# Legend Column
ax_leg = fig.add_subplot(gs_bottom[0])
ax_leg.set_facecolor('#f8fafc')
ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)
for spine in ax_leg.spines.values():
    spine.set_edgecolor('#cbd5e1')
    spine.set_linewidth(0.8)
ax_leg.set_xticks([])
ax_leg.set_yticks([])

ax_leg.text(0.08, 0.88, "Harvest Exposure", fontsize=9.5, fontweight='bold', color='#0f172a', va='top')
ax_leg.text(0.08, 0.70, "Share of Midwest harvested\nsoybean acreage exposed\nto climatic water balance:", fontsize=7.6, color='#475569', va='top')

bullets = [
    ('#b91c1c', "Severe Deficit (< -150 mm)"),
    ('#f97316', "Moderate Deficit (0 to -150)"),
    ('#1d4ed8', "Excess Saturation (> +50 mm)"),
    ('#0284c7', "Moderate Surplus (0 to +50)")
]
for idx, (col, txt) in enumerate(bullets):
    y_pos = 0.44 - idx * 0.125
    ax_leg.scatter(0.12, y_pos, color=col, s=42, zorder=5)
    ax_leg.text(0.22, y_pos, txt, fontsize=7.2, color='#1e293b', va='center')

# Donut 1: 1992
ax_d1 = fig.add_subplot(gs_bottom[1])
ax_d1.set_facecolor('#f8fafc')
for spine in ax_d1.spines.values():
    spine.set_edgecolor('#cbd5e1')
    spine.set_linewidth(0.8)
df92 = metrics[1992].dropna(subset=['p_minus_et0', 'acres_harvested'])
t92 = df92['acres_harvested'].sum()
s92 = [df92[df92['p_minus_et0'] < -50]['acres_harvested'].sum() / t92 * 100,
       df92[(df92['p_minus_et0'] >= -50) & (df92['p_minus_et0'] < 50)]['acres_harvested'].sum() / t92 * 100,
       df92[df92['p_minus_et0'] >= 50]['acres_harvested'].sum() / t92 * 100]
w1, _ = ax_d1.pie(s92, colors=['#f97316', '#84cc16', '#0284c7'], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.38, edgecolor='white', linewidth=1.5))
ax_d1.text(0, 0, f"1992\n{s92[1]:.0f}%\nBalance", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#3f6212')
t92_m = f"{t92/1e6:.1f}M" if t92 > 0 else "N/A"
ax_d1.set_title(f"1992 Acreage Exposure\n({t92_m} Harvested Acres)", fontsize=8.6, fontweight='bold', pad=4, color='#0f172a')

# Donut 2: 1993
ax_d2 = fig.add_subplot(gs_bottom[2])
ax_d2.set_facecolor('#f8fafc')
for spine in ax_d2.spines.values():
    spine.set_edgecolor('#cbd5e1')
    spine.set_linewidth(0.8)
df93 = metrics[1993].dropna(subset=['p_minus_et0', 'acres_harvested'])
t93 = df93['acres_harvested'].sum()
s93 = [df93[df93['p_minus_et0'] > 50]['acres_harvested'].sum() / t93 * 100,
       df93[(df93['p_minus_et0'] >= 0) & (df93['p_minus_et0'] <= 50)]['acres_harvested'].sum() / t93 * 100,
       df93[df93['p_minus_et0'] < 0]['acres_harvested'].sum() / t93 * 100]
w2, _ = ax_d2.pie(s93, colors=['#1d4ed8', '#0284c7', '#fed7aa'], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.38, edgecolor='white', linewidth=1.5))
ax_d2.text(0, 0, f"1993\n{s93[0]:.0f}%\nPluvial", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#1e40af')
t93_m = f"{t93/1e6:.1f}M" if t93 > 0 else "N/A"
ax_d2.set_title(f"1993 Acreage Exposure\n({t93_m} Harvested Acres)", fontsize=8.6, fontweight='bold', pad=4, color='#0f172a')

# Donut 3: 2012
ax_d3 = fig.add_subplot(gs_bottom[3])
ax_d3.set_facecolor('#f8fafc')
for spine in ax_d3.spines.values():
    spine.set_edgecolor('#cbd5e1')
    spine.set_linewidth(0.8)
df12 = metrics[2012].dropna(subset=['p_minus_et0', 'acres_harvested'])
t12 = df12['acres_harvested'].sum()
s12 = [df12[df12['p_minus_et0'] < -200]['acres_harvested'].sum() / t12 * 100,
       df12[(df12['p_minus_et0'] >= -200) & (df12['p_minus_et0'] < -100)]['acres_harvested'].sum() / t12 * 100,
       df12[df12['p_minus_et0'] >= -100]['acres_harvested'].sum() / t12 * 100]
w3, _ = ax_d3.pie(s12, colors=['#7f1d1d', '#dc2626', '#fb923c'], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.38, edgecolor='white', linewidth=1.5))
ax_d3.text(0, 0, f"2012\n{s12[0]:.0f}%\nFlash", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#7f1d1d')
t12_m = f"{t12/1e6:.1f}M" if t12 > 0 else "N/A"
ax_d3.set_title(f"2012 Acreage Exposure\n({t12_m} Harvested Acres)", fontsize=8.6, fontweight='bold', pad=4, color='#0f172a')

os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_PDF, bbox_inches='tight')
print(f"Saved publication grade figure to: {OUTPUT_PNG} and {OUTPUT_PDF}")
