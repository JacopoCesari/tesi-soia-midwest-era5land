"""
Generate Figure 3.5: Spatial Architecture and Grid Discretization of ERA5-Land Reanalysis.

Left: Regional Study Domain - Illinois State with 28 balanced study counties and Tazewell County highlighted.
Right: Micro-scale Discretization on Tazewell County, IL, shaped by the Illinois River, showing the
regular 0.1° x 0.1° ERA5-Land grid mesh and the area-weighted aggregation mechanism (Eq. 3.1).
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from shapely.geometry import box

# Set high-quality academic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

ROOT_DIR = r"c:\Users\JacopoCesari-Aretésr\Desktop\Tesi"
SHAPEFILE = os.path.join(ROOT_DIR, "work", "data", "external", "usda_nass_and_county_boundaries", "census_counties", "cb_2020_us_county_500k.shp")
WEIGHTS_CSV = os.path.join(ROOT_DIR, "output", "data", "auxiliary", "spatial_weights.csv")
COUNTIES_CSV = os.path.join(ROOT_DIR, "output", "data", "auxiliary", "counties.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "thesis", "figures")

# Load data
print("Loading spatial geometries and weights...")
gdf_all = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
df_weights = pd.read_csv(WEIGHTS_CSV, dtype={'county_fips': str})
df_counties = pd.read_csv(COUNTIES_CSV, dtype={'county_fips': str})

selected_fips = set(df_counties['county_fips'].str.zfill(5))

# Illinois state and counties
gdf_il = gdf_all[gdf_all['STATEFP'] == '17'].copy()
gdf_selected_il = gdf_il[gdf_il['GEOID'].isin(selected_fips)].copy()
gdf_il_state = gdf_il.dissolve(by='STATEFP')

# Surrounding states for geographic context
surrounding_fips = ['19', '29', '21', '18', '55']  # IA, MO, KY, IN, WI
gdf_surrounding = gdf_all[gdf_all['STATEFP'].isin(surrounding_fips)].copy()
gdf_surr_states = gdf_surrounding.dissolve(by='STATEFP')

# Tazewell County geometry and weights
tazewell_row = gdf_il[gdf_il['GEOID'] == '17179'].iloc[0]
tazewell_geom = tazewell_row.geometry
tb = tazewell_geom.bounds
tw_w = df_weights[df_weights['county_fips'] == '17179'].copy()

# Setup 1-row, 2-column figure (14.0 x 6.8 inches)
fig, (ax_macro, ax_micro) = plt.subplots(
    1, 2, figsize=(14.0, 6.8),
    gridspec_kw={'width_ratios': [1.0, 1.28], 'wspace': 0.22}
)

# ==============================================================================
# LEFT: State of Illinois Zoomed In (Macro View)
# ==============================================================================
# Surrounding context
gdf_surr_states.plot(ax=ax_macro, facecolor='#f7f7f7', edgecolor='#d9d9d9', linewidth=0.8, zorder=1)

# All Illinois counties (non-selected)
gdf_il.plot(ax=ax_macro, facecolor='#ffffff', edgecolor='#d0d0d0', linewidth=0.45, zorder=2)

# Selected 28 balanced study counties in Illinois
gdf_selected_il.plot(ax=ax_macro, facecolor='#a1dab4', edgecolor='#238b45', linewidth=0.7, zorder=3)

# Highlight Tazewell County in central Illinois
gpd.GeoSeries([tazewell_geom]).plot(
    ax=ax_macro, facecolor='#e41a1c', edgecolor='#99000d', linewidth=1.8, alpha=0.8, zorder=5
)

# Illinois state outline
gdf_il_state.plot(ax=ax_macro, facecolor='none', edgecolor='#252525', linewidth=1.4, zorder=6)

# Labels for neighboring states
ax_macro.text(-90.5, 43.1, 'WISCONSIN', fontsize=9, color='#888888', fontweight='bold', ha='center', zorder=4)
ax_macro.text(-92.3, 41.5, 'IOWA', fontsize=9, color='#888888', fontweight='bold', ha='center', zorder=4)
ax_macro.text(-91.8, 38.3, 'MISSOURI', fontsize=9, color='#888888', fontweight='bold', ha='center', zorder=4)
ax_macro.text(-86.2, 40.2, 'INDIANA', fontsize=9, color='#888888', fontweight='bold', ha='center', zorder=4)
ax_macro.text(-87.8, 37.3, 'KENTUCKY', fontsize=8.5, color='#888888', fontweight='bold', ha='center', zorder=4)

# Callout rectangle around Tazewell County
taz_box_pad_x = 0.08
taz_box_pad_y = 0.06
macro_taz_rect = mpatches.Rectangle(
    (tb[0] - taz_box_pad_x, tb[1] - taz_box_pad_y),
    tb[2] - tb[0] + 2 * taz_box_pad_x,
    tb[3] - tb[1] + 2 * taz_box_pad_y,
    fill=False, edgecolor='#99000d', linewidth=1.8, linestyle='-', zorder=7
)
ax_macro.add_patch(macro_taz_rect)

# Tazewell County annotation box
ax_macro.annotate(
    "Tazewell County, IL\n(Enlarged on right)",
    xy=((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2),
    xytext=(-92.0, 40.1),
    fontsize=8.8, fontweight='bold', color='#99000d',
    arrowprops=dict(arrowstyle='->', color='#99000d', lw=1.3, connectionstyle='arc3,rad=-0.12'),
    bbox=dict(boxstyle='square,pad=0.35', facecolor='white', edgecolor='#99000d', alpha=0.95),
    zorder=8
)

# Macro Legend
p_sel = mpatches.Patch(facecolor='#a1dab4', edgecolor='#238b45', label=r'Balanced Study Counties ($N=28$)')
p_other = mpatches.Patch(facecolor='#ffffff', edgecolor='#d0d0d0', label=r'Other Counties ($N=74$)')
p_taz = mpatches.Patch(facecolor='#e41a1c', edgecolor='#99000d', label='Tazewell County (Focus)')
ax_macro.legend(
    handles=[p_sel, p_other, p_taz],
    loc='lower left', bbox_to_anchor=(0.02, 0.02),
    fontsize=8.5, framealpha=0.95, facecolor='white', edgecolor='#cccccc'
)

ax_macro.set_xlim(-92.8, -86.0)
ax_macro.set_ylim(36.8, 43.5)
ax_macro.set_xlabel(r'Longitude ($^\circ$W)', fontsize=9.5)
ax_macro.set_ylabel(r'Latitude ($^\circ$N)', fontsize=9.5)
ax_macro.set_title(r'(a) Regional Domain: Illinois State ($N=28$ Study Counties)', fontsize=10.5, fontweight='bold', pad=8)
ax_macro.set_aspect(1.0 / np.cos(np.radians(40.0)))

# ==============================================================================
# RIGHT: Micro Grid Discretization (Fast, Simple, Geographically Rigorous)
# ==============================================================================
# 1. County polygon with soft, clean light fill
gpd.GeoSeries([tazewell_geom]).plot(
    ax=ax_micro, facecolor='#edf8fb', edgecolor='#01464c', linewidth=2.4, zorder=2
)

# 2. Regular 0.1° ERA5-Land grid mesh (cardinal meridians and parallels)
for lon in np.arange(-89.95, -89.15, 0.1):
    ax_micro.axvline(lon, color='#08519c', linestyle='--', linewidth=0.75, alpha=0.55, zorder=3)
for lat in np.arange(40.25, 40.85, 0.1):
    ax_micro.axhline(lat, color='#08519c', linestyle='--', linewidth=0.75, alpha=0.55, zorder=3)

# 3. Soft shading of the intersecting cell polygons (proportional to area weight)
for _, r in tw_w.iterrows():
    b = box(r['longitude'] - 0.05, r['latitude'] - 0.05, r['longitude'] + 0.05, r['latitude'] + 0.05)
    inter = tazewell_geom.intersection(b)
    w = r['weight']
    alpha_val = min(0.65, 0.22 + w * 7.5)
    gpd.GeoSeries([inter]).plot(
        ax=ax_micro, facecolor='#2b83ba', edgecolor='#08519c', linewidth=0.65, alpha=alpha_val, zorder=4
    )

# 4. Illinois River annotation (highlighting the natural curved boundary without overlapping badge)
ax_micro.annotate(
    "Illinois River\n(Natural boundary)",
    xy=(-89.78, 40.55),
    xytext=(-89.95, 40.62),
    fontsize=8.5, fontweight='bold', color='#016c59',
    arrowprops=dict(arrowstyle='->', color='#016c59', lw=1.2, connectionstyle='arc3,rad=-0.12'),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e5f5f9', edgecolor='#016c59', alpha=0.95),
    zorder=6
)

# 5. Fast, crystal-clear explanation badge at the top
badge_text = (
    r"$\mathbf{Area-Weighted\ Aggregation\ (Eq.\ 3.1)}$" + "\n" +
    r"$\bar{X}_{c,t} = \sum_{g=1}^{29} w_{c,g} \, X_{g,t}, \quad \text{with } \sum w_{c,g} = 100\%$" + "\n" +
    r"Weights: $w_{c,g} = \frac{\mathrm{Area}(c \cap g)}{\mathrm{Area}(c)}$ (exact geometric fraction)"
)
ax_micro.text(
    0.04, 0.96, badge_text,
    transform=ax_micro.transAxes, fontsize=8.8, verticalalignment='top',
    bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='#08519c', alpha=0.95),
    zorder=7
)

# 6. Clean, self-explanatory legend at bottom right
leg_bound = Line2D([0], [0], color='#01464c', linewidth=2.4, label='County Boundary (Tazewell, IL)')
leg_grid = Line2D([0], [0], color='#08519c', linestyle='--', linewidth=1.2, label=r'ERA5-Land Grid ($0.1^\circ \approx 9$ km)')
leg_cells = mpatches.Patch(facecolor='#2b83ba', edgecolor='#08519c', alpha=0.55, label=r'Intersecting Cells ($N=29$, weighted by area)')

ax_micro.legend(
    handles=[leg_bound, leg_grid, leg_cells],
    loc='lower right', bbox_to_anchor=(0.98, 0.03),
    fontsize=8.5, framealpha=0.95, facecolor='white', edgecolor='#cccccc'
)

ax_micro.set_xlim(-89.98, -89.20)
ax_micro.set_ylim(40.22, 40.80)
ax_micro.set_xlabel(r'Longitude ($^\circ$W)', fontsize=9.5)
ax_micro.set_ylabel(r'Latitude ($^\circ$N)', fontsize=9.5)
ax_micro.set_title(
    r'(b) ERA5-Land $0.1^\circ$ Grid Discretization (Tazewell County, IL)',
    fontsize=10.5, fontweight='bold', pad=8
)
ax_micro.set_aspect(1.0 / np.cos(np.radians(40.5)))

# ==============================================================================
# PROJECTION CONNECTION LINES
# ==============================================================================
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

# Save figure
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_png = os.path.join(OUTPUT_DIR, "fig_era5_spatial_structure.png")
output_pdf = os.path.join(OUTPUT_DIR, "fig_era5_spatial_structure.pdf")

fig.savefig(output_png, bbox_inches='tight', dpi=300)
fig.savefig(output_pdf, bbox_inches='tight', dpi=300)
print(f"Saved figure successfully to:\n{output_png}\n{output_pdf}")
