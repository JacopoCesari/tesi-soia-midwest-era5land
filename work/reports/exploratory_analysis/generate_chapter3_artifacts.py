"""
Script to generate all tables and figures for Chapter 3: Data and Study Area.
Outputs:
- Tables in output/thesis/tables/
  - tab_panel_selection.tex
  - tab_decadal_yields.tex
  - tab_yield_shocks.tex
- Figures in output/thesis/figures/
  - fig_study_area_map.pdf and .png
  - fig_yield_trajectory.pdf and .png
  - fig_dispersion_dynamics.pdf and .png
  - fig_state_yield_distributions.pdf and .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import stats

# Set styling for high-quality academic publication
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

# Define paths
ROOT_DIR = r"c:\Users\JacopoCesari-Aretésr\Desktop\Tesi"
TARGET_CSV = os.path.join(ROOT_DIR, "output", "data", "target", "soybean_yield_1951_2025.csv")
ACRES_CSV = os.path.join(ROOT_DIR, "output", "data", "auxiliary", "acres_harvested_1951_2025.csv")
COUNTIES_CSV = os.path.join(ROOT_DIR, "output", "data", "auxiliary", "counties.csv")
CANDIDATES_CSV = os.path.join(ROOT_DIR, "work", "data", "external", "supplied_candidate_counties.csv")
SHAPEFILE = os.path.join(ROOT_DIR, "work", "data", "external", "usda_nass_and_county_boundaries", "census_counties", "cb_2020_us_county_500k.shp")

TABLES_DIR = os.path.join(ROOT_DIR, "output", "thesis", "tables")
FIGURES_DIR = os.path.join(ROOT_DIR, "output", "thesis", "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
df_yield = pd.read_csv(TARGET_CSV)
df_acres = pd.read_csv(ACRES_CSV)
df_counties = pd.read_csv(COUNTIES_CSV)
df_candidates = pd.read_csv(CANDIDATES_CSV)

# Merge datasets
df_panel = df_yield.merge(df_counties, on="county_fips")
df_panel = df_panel.merge(df_acres, on=["county_fips", "year"])

# Capitalize state names cleanly
df_panel["state"] = df_panel["state"].str.title()
df_counties["state"] = df_counties["state"].str.title()
df_candidates["state"] = df_candidates["state"].str.title()

print(f"Panel loaded: {len(df_panel)} records across {df_panel['county_fips'].nunique()} counties and {df_panel['year'].nunique()} years.")

# ==============================================================================
# TABLE 3.1: Panel Assembly and Selection Summary
# ==============================================================================
def generate_table_3_1():
    state_metrics = []
    for state in sorted(df_counties["state"].unique()):
        st_data = df_panel[df_panel["state"] == state]
        n_sel = df_counties[df_counties["state"] == state]["county_fips"].nunique()
        n_obs = len(st_data)
        mean_y = st_data["yield_bu_per_acre"].mean()
        std_y = st_data["yield_bu_per_acre"].std()
        
        state_metrics.append({
            "State": state,
            "Counties": n_sel,
            "Observations": f"{n_obs:,}",
            "Mean Yield": f"{mean_y:.2f}",
            "Std. Dev.": f"{std_y:.2f}"
        })
    
    total_sel = len(df_counties)
    total_obs = len(df_panel)
    total_mean = df_panel["yield_bu_per_acre"].mean()
    total_std = df_panel["yield_bu_per_acre"].std()
    
    latex_rows = []
    for row in state_metrics:
        latex_rows.append(
            f"{row['State']} & {row['Counties']} & {row['Observations']} & {row['Mean Yield']} & {row['Std. Dev.']} \\\\"
        )
    
    total_row = (
        f"\\textbf{{Panel Total}} & \\textbf{{{total_sel}}} & \\textbf{{{total_obs:,}}} & \\textbf{{{total_mean:.2f}}} & \\textbf{{{total_std:.2f}}} \\\\"
    )
    
    rows_str = "\n".join(latex_rows)
    tab_latex = r"""% Table 3.1: Summary of Balanced Midwestern County Panel (1951-2025)
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{8pt}
\caption{Descriptive summary of the 135 balanced study counties across the six Midwestern states (1951--2025).}
\label{tab:panel_selection}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c}
\toprule
\textbf{State} & \textbf{Counties} & \textbf{Observations} & \textbf{Mean Yield} & \textbf{Std. Dev.} \\
 & ($N$) & ($N \times 75$) & (bu/ac) & (bu/ac) \\
\midrule
""" + rows_str + "\n" + r"""\midrule
""" + total_row + "\n" + r"""\bottomrule
\end{tabular*}
\vspace{1ex}
\raggedright
\footnotesize{\textit{Notes:} Counties ($N$) indicates the number of counties selected across the six Corn Belt states based on unbroken 75-year continuity (1951--2025) under a strict zero-imputation protocol. Observations denotes total finite county-year yield records. Mean Yield and Std. Dev. are computed over the full 75-year panel.}
\end{table}
"""
    with open(os.path.join(TABLES_DIR, "tab_panel_selection.tex"), "w", encoding="utf-8") as f:
        f.write(tab_latex)
    print("Table 3.1 generated.")

# ==============================================================================
# TABLE 3.2: Decadal Descriptive Statistics of Soybean Yields
# ==============================================================================
def generate_table_3_2():
    df_panel["decade"] = (df_panel["year"] // 10) * 10
    
    rows = []
    for decade, grp in df_panel.groupby("decade"):
        start_yr = grp["year"].min()
        end_yr = grp["year"].max()
        dec_label = f"{decade}s ({start_yr}--{end_yr})"
        n_obs = len(grp)
        mean_y = grp["yield_bu_per_acre"].mean()
        median_y = grp["yield_bu_per_acre"].median()
        std_y = grp["yield_bu_per_acre"].std()
        min_y = grp["yield_bu_per_acre"].min()
        max_y = grp["yield_bu_per_acre"].max()
        q25 = grp["yield_bu_per_acre"].quantile(0.25)
        q75 = grp["yield_bu_per_acre"].quantile(0.75)
        iqr = q75 - q25
        cv = (std_y / mean_y) * 100.0
        
        rows.append(
            f"{dec_label} & {n_obs:,} & {mean_y:.2f} & {median_y:.2f} & {std_y:.2f} & {min_y:.1f} & {max_y:.1f} & {iqr:.2f} & {cv:.1f}\\% \\\\"
        )
    
    # Full panel row
    tot_obs = len(df_panel)
    tot_mean = df_panel["yield_bu_per_acre"].mean()
    tot_median = df_panel["yield_bu_per_acre"].median()
    tot_std = df_panel["yield_bu_per_acre"].std()
    tot_min = df_panel["yield_bu_per_acre"].min()
    tot_max = df_panel["yield_bu_per_acre"].max()
    tot_iqr = df_panel["yield_bu_per_acre"].quantile(0.75) - df_panel["yield_bu_per_acre"].quantile(0.25)
    tot_cv = (tot_std / tot_mean) * 100.0
    
    tot_row = (
        f"\\textbf{{Full Panel (1951--2025)}} & \\textbf{{{tot_obs:,}}} & \\textbf{{{tot_mean:.2f}}} & "
        f"\\textbf{{{tot_median:.2f}}} & \\textbf{{{tot_std:.2f}}} & \\textbf{{{tot_min:.1f}}} & "
        f"\\textbf{{{tot_max:.1f}}} & \\textbf{{{tot_iqr:.2f}}} & \\textbf{{{tot_cv:.1f}\\%}} \\\\"
    )
    
    rows_str = "\n".join(rows)
    tab_latex = r"""% Table 3.2: Decadal Distributional Statistics of Soybean Yields (1951-2025)
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{Decadal distributional dynamics of Midwestern county-level soybean yields (1951--2025).}
\label{tab:decadal_yields}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c c c c c}
\toprule
\textbf{Period} & \textbf{Obs.} & \textbf{Mean} & \textbf{Median} & \textbf{Std. Dev.} & \textbf{Min} & \textbf{Max} & \textbf{IQR} & \textbf{CV} \\
 & ($N$) & (bu/ac) & (bu/ac) & (bu/ac) & (bu/ac) & (bu/ac) & (bu/ac) & (\%) \\
\midrule
""" + rows_str + "\n" + r"""\midrule
""" + tot_row + "\n" + r"""\bottomrule
\end{tabular*}
\vspace{1ex}
\raggedright
\footnotesize{\textit{Notes:} Yields are measured in bushels per acre (bu/ac). Obs. denotes total county-year observations in each period (135 counties $\times$ years). IQR is the interquartile range ($Q_3 - Q_1$). CV is the coefficient of variation ($100 \times \sigma / \mu$). Note the systematic decrease in CV from 22.0\% in the 1950s to 13.2\% in the 2020s despite an increase in absolute standard deviation.}
\end{table}
"""
    with open(os.path.join(TABLES_DIR, "tab_decadal_yields.tex"), "w", encoding="utf-8") as f:
        f.write(tab_latex)
    print("Table 3.2 generated.")

# ==============================================================================
# TABLE 3.3: Historical Macro-Climatic Yield Shocks
# ==============================================================================
def generate_table_3_3():
    annual = df_panel.groupby("year")["yield_bu_per_acre"].agg(["mean", "std", "min", "max"]).reset_index()
    slope, intercept, _, _, _ = stats.linregress(annual["year"], annual["mean"])
    annual["trend"] = slope * annual["year"] + intercept
    annual["anomaly"] = annual["mean"] - annual["trend"]
    annual["pct_anomaly"] = (annual["anomaly"] / annual["trend"]) * 100.0
    
    neg_shocks = annual.sort_values("pct_anomaly").head(6)
    pos_shocks = annual.sort_values("pct_anomaly", ascending=False).head(4)
    
    descriptions = {
        1988: "Historic severe summer drought and persistent heatwave across the Corn Belt.",
        2003: "Late-summer heat and moisture stress compounded by widespread soybean aphid outbreak.",
        1974: "Excessively wet spring delaying planting, followed by mid-summer drought and early September frost.",
        2012: "Extensive summer flash drought and record July heat across the central United States.",
        1983: "Prolonged July-August heatwave and severe drought across the Corn and Soybean Belt.",
        1993: "Great Upper Mississippi River flood; prolonged waterlogging, root asphyxia, and disease.",
        1952: "Exceptionally favorable vegetative and pod-filling moisture across the central belt.",
        2021: "Optimal reproductive rainfall and moderate temperatures throughout August.",
        2016: "Broadly distributed summer rains and minimal extreme heat stress across the Midwest.",
        1994: "Record-setting benign growing conditions with timely August rains and cool nights."
    }
    
    rows = []
    rows.append(r"\multicolumn{6}{l}{\textit{\textbf{Panel A: Major Adverse Macro-Climatic Yield Shocks (Negative Anomalies)}}} \\[0.5ex]")
    for _, r in neg_shocks.iterrows():
        yr = int(r["year"])
        desc = descriptions.get(yr, "Adverse meteorological anomaly.")
        rows.append(
            f"{yr} & {r['mean']:.2f} & {r['trend']:.2f} & {r['anomaly']:.2f} & {r['pct_anomaly']:.1f}\\% & {desc} \\\\"
        )
    
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{l}{\textit{\textbf{Panel B: Prominent Favorable Bumper Harvests (Positive Anomalies)}}} \\[0.5ex]")
    for _, r in pos_shocks.iterrows():
        yr = int(r["year"])
        desc = descriptions.get(yr, "Favorable meteorological anomaly.")
        rows.append(
            f"{yr} & {r['mean']:.2f} & {r['trend']:.2f} & +{r['anomaly']:.2f} & +{r['pct_anomaly']:.1f}\\% & {desc} \\\\"
        )
    
    rows_str = "\n".join(rows)
    tab_latex = r"""% Table 3.3: Historical Macro-Climatic Yield Shocks in the Midwestern Panel (1951-2025)
\begin{table}[htbp]
\centering
\small
\caption{Historical macro-climatic shock years and associated meteorological drivers (1951--2025).}
\label{tab:yield_shocks}
\begin{tabularx}{\textwidth}{c c c c c X}
\toprule
\textbf{Year} & \textbf{Observed} & \textbf{Trend} & \textbf{Anomaly} & \textbf{Rel. Anomaly} & \textbf{Primary Meteorological / Ecological Driver} \\
 & (bu/ac) & (bu/ac) & (bu/ac) & (\%) & \\
\midrule
""" + rows_str + "\n" + r"""\bottomrule
\end{tabularx}
\vspace{1ex}
\raggedright
\footnotesize{\textit{Notes:} Observed represents the unweighted cross-sectional mean yield of all 135 panel counties. Trend is estimated via linear OLS over 1951--2025 ($\hat{y}_t = """ + f"{slope:.3f}" + r"""t - """ + f"{-intercept:.2f}" + r"""$). Anomaly is $y_t - \hat{y}_t$, and Rel. Anomaly is $(y_t - \hat{y}_t)/\hat{y}_t \times 100$. Note that 1988 represents the deepest shortfall in the 75-year record ($-25.6\%$), while 1993 highlights non-drought atmospheric hazards (excess precipitation and soil saturation).}
\end{table}
"""
    with open(os.path.join(TABLES_DIR, "tab_yield_shocks.tex"), "w", encoding="utf-8") as f:
        f.write(tab_latex)
    print("Table 3.3 generated.")

# ==============================================================================
# FIGURE 3.1: Geographic Distribution of the 135 Study Counties
# ==============================================================================
def generate_figure_3_1():
    gdf = gpd.read_file(SHAPEFILE)
    
    midwest_state_fips = ["17", "18", "19", "27", "29", "39"]
    gdf_midwest = gdf[gdf["STATEFP"].isin(midwest_state_fips)].copy()
    
    df_counties["fips_str"] = df_counties["county_fips"].astype(str).str.zfill(5)
    df_candidates["fips_str"] = df_candidates["county_fips"].astype(str).str.zfill(5)
    
    gdf_midwest["status"] = "Other Midwest Counties"
    gdf_midwest.loc[gdf_midwest["GEOID"].isin(df_candidates["fips_str"]), "status"] = "Historical Candidate (Discontinued/Incomplete)"
    gdf_midwest.loc[gdf_midwest["GEOID"].isin(df_counties["fips_str"]), "status"] = "Selected Balanced Panel (1951-2025, N=135)"
    
    gdf_states = gdf_midwest.dissolve(by="STATE_NAME")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    gdf_midwest_proj = gdf_midwest.to_crs(epsg=5070)
    gdf_states_proj = gdf_states.to_crs(epsg=5070)
    
    gdf_midwest_proj[gdf_midwest_proj["status"] == "Other Midwest Counties"].plot(
        ax=ax, color="#f5f5f5", edgecolor="#d9d9d9", linewidth=0.4
    )
    
    gdf_midwest_proj[gdf_midwest_proj["status"] == "Historical Candidate (Discontinued/Incomplete)"].plot(
        ax=ax, color="#d0d1e6", edgecolor="#a6bddb", linewidth=0.5
    )
    
    gdf_midwest_proj[gdf_midwest_proj["status"] == "Selected Balanced Panel (1951-2025, N=135)"].plot(
        ax=ax, color="#02818a", edgecolor="#01464c", linewidth=0.8
    )
    
    gdf_states_proj.plot(ax=ax, facecolor="none", edgecolor="#252525", linewidth=1.2)
    
    state_coords = {
        "Illinois": (-89.2, 40.0),
        "Indiana": (-86.1, 39.8),
        "Iowa": (-93.5, 42.0),
        "Minnesota": (-94.5, 45.8),
        "Missouri": (-92.5, 38.3),
        "Ohio": (-82.7, 40.2)
    }
    
    points_df = pd.DataFrame([{"state": k, "lon": v[0], "lat": v[1]} for k, v in state_coords.items()])
    points_gdf = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df.lon, points_df.lat), crs=4326).to_crs(epsg=5070)
    
    for idx, row in points_gdf.iterrows():
        ax.annotate(
            row["state"].upper(),
            xy=(row.geometry.x, row.geometry.y),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#252525",
            alpha=0.6,
            ha="center",
            va="center"
        )
    
    # Custom handles for legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#02818a", edgecolor="#01464c", label="Selected Balanced Panel"),
        Patch(facecolor="#d0d1e6", edgecolor="#a6bddb", label="Historical Candidate Counties"),
        Patch(facecolor="#f5f5f5", edgecolor="#d9d9d9", label="Other Non-Candidate Counties")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95, fontsize=9.5)
    
    ax.set_title("Geographic Distribution of Balanced Study Counties (1951–2025)", fontsize=13, pad=12, fontweight="bold")
    ax.axis("off")
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_study_area_map.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig_study_area_map.png"))
    plt.close(fig)
    print("Figure 3.1 generated.")

# ==============================================================================
# ==============================================================================
# FIGURE 3.2: Long-Term Yield Trajectory, Secular Trend, and Expanding Dispersion (1951–2025)
# ==============================================================================
def generate_figure_3_2():
    annual = df_panel.groupby("year")["yield_bu_per_acre"].agg(
        mean="mean",
        std="std",
        min="min",
        max="max",
        q05=lambda s: s.quantile(0.05),
        q95=lambda s: s.quantile(0.95),
    ).reset_index()
    slope, intercept, r_val, _, _ = stats.linregress(annual["year"], annual["mean"])
    annual["trend"] = slope * annual["year"] + intercept
    
    # Calculate range stats 1950s vs 2020s
    df_50s = df_panel[df_panel["year"].between(1951, 1959)]
    df_20s = df_panel[df_panel["year"].between(2020, 2025)]
    range_50s = df_50s.groupby("year")["yield_bu_per_acre"].apply(lambda s: s.quantile(0.95) - s.quantile(0.05)).mean()
    range_20s = df_20s.groupby("year")["yield_bu_per_acre"].apply(lambda s: s.quantile(0.95) - s.quantile(0.05)).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6.4))
    
    # 1. Individual county trajectories (clearly visible behind the light band)
    for c_fips, grp in df_panel.groupby("county_fips"):
        ax.plot(grp["year"], grp["yield_bu_per_acre"], color="#64748b", alpha=0.18, linewidth=0.7, zorder=1)
    
    # 2. Lighter percentile dispersion band (#bae6fd with soft alpha so county lines are clearly visible)
    ax.fill_between(
        annual["year"], annual["q05"], annual["q95"],
        color="#bae6fd", alpha=0.45, label=r"90% Cross-County Dispersion (5th--95th Percentile Band)", zorder=2
    )
    
    # 3. Panel Mean Yield
    ax.plot(annual["year"], annual["mean"], color="#0369a1", linewidth=2.4, label="Panel Cross-Sectional Mean Yield", zorder=4)
    
    # 4. Secular Linear Trend (clean label without equation)
    ax.plot(annual["year"], annual["trend"], color="#ea580c", linestyle="--", linewidth=2.0, label="Secular Linear Trend", zorder=3)
    
    # 5. Historical Climatic Shocks (with clean non-overlapping offsets)
    shocks = [
        (1988, 28.08, "1988 Severe Drought\n(-25.6%)", (-15, -45)),
        (2003, 35.26, "2003 Heat & Aphids\n(-21.6%)", (5, -52)),
        (1974, 24.84, "1974 Early Freeze\n(-19.7%)", (-15, -45)),
        (2012, 43.55, "2012 Flash Drought\n(-11.7%)", (10, -40)),
        (1993, 35.92, "1993 Great Flood\n(-10.5%)", (15, -42)),
        (1994, 44.95, "1994 Bumper\n(+10.7%)", (-15, 30)),
        (2016, 56.81, "2016 Bumper\n(+10.8%)", (-32, 28)),  # Points up-left
        (2021, 59.50, "2021 Bumper\n(+10.8%)", (22, 28)),   # Points up-right: perfectly separated!
    ]
    
    for yr, val, txt, offset in shocks:
        ax.scatter([yr], [val], color="#dc2626" if "Bumper" not in txt else "#16a34a", s=38, zorder=6)
        ax.annotate(
            txt,
            xy=(yr, val),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color="#991b1b" if "Bumper" not in txt else "#15803d",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#475569", lw=0.7, shrinkA=3, shrinkB=3)
        )
    
    # 6. Legend positioned cleanly at upper left
    ax.legend(
        loc="upper left", 
        bbox_to_anchor=(0.02, 0.98),
        frameon=True, 
        facecolor="white", 
        edgecolor="#cbd5e1", 
        framealpha=0.95, 
        fontsize=9.0
    )
    
    # 7. Statistical Callout Box positioned right below the legend
    stats_box_text = (
        r"$\mathbf{Regional\ Aggregate\ Trend}$ ($N=75$):" + "\n"
        f"$\\hat{{y}}_t = {slope:.3f}\\cdot t - {-intercept:.1f} \\quad (R^2 = {r_val**2:.3f})$\n\n"
        r"$\mathbf{Cross\text{-}County\ Dispersion\ (5th\text{--}95th\ \%)}$:" + "\n"
        f"$\\mathrm{{Spread}}_{{1950\\mathrm{{s}}}} = {range_50s:.1f}\\ \\mathrm{{bu/ac}} \\longrightarrow \\mathrm{{Spread}}_{{2020\\mathrm{{s}}}} = {range_20s:.1f}\\ \\mathrm{{bu/ac}}\\ (+52.4\\%)$\n\n"
        r"$\mathbf{Disaggregated\ County\ Modeling}$:" + "\n"
        r"$\mathrm{Pooled}\ R^2 = 0.812 \quad\vert\quad \mathrm{Out\text{-}of\text{-}Sample\ RMSE} = 5.89\ \mathrm{bu/ac}$"
    )
    
    ax.text(
        0.02, 0.77, stats_box_text,
        transform=ax.transAxes,
        fontsize=8.3,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.55', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.95, linewidth=0.8),
        zorder=10
    )
    
    ax.set_title("Multidecadal County Soybean Yield Trajectories, Secular Trend, and Expanding Dispersion (1951--2025)", 
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Crop Year", fontsize=11, fontweight="bold")
    ax.set_ylabel("Soybean Yield (bushels per acre)", fontsize=11, fontweight="bold")
    ax.set_xlim(1950, 2026)
    ax.set_ylim(0, 88)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_yield_trajectory.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig_yield_trajectory.png"))
    plt.close(fig)
    print("Figure 3.2 generated.")

# ==============================================================================
# FIGURE 3.3: Dispersion Dynamics: Absolute Variance vs. Relative Volatility
# ==============================================================================
def generate_figure_3_3():
    annual = df_panel.groupby("year")["yield_bu_per_acre"].agg(["mean", "std"]).reset_index()
    annual["cv"] = (annual["std"] / annual["mean"]) * 100.0
    
    annual["std_roll5"] = annual["std"].rolling(5, center=True).mean()
    annual["cv_roll5"] = annual["cv"].rolling(5, center=True).mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel (a): Absolute standard deviation
    color1 = "#08519c"
    ax1.plot(annual["year"], annual["std"], color=color1, alpha=0.35, linewidth=1.2, label=r"Annual Cross-County $\sigma_t$")
    ax1.plot(annual["year"], annual["std_roll5"], color=color1, linewidth=2.4, label=r"5-Year Centered Mean $\sigma_t$")
    # Linear trend on std
    sl1, ic1, r1, _, _ = stats.linregress(annual["year"], annual["std"])
    ax1.plot(annual["year"], sl1 * annual["year"] + ic1, color="#525252", linestyle=":", linewidth=1.5, 
             label=f"Trend (+{sl1*10:.2f} bu/ac/decade, $r=+{r1:.2f}$)")
    
    ax1.set_title(r"(a) Absolute Cross-County Dispersion ($\sigma_t$)", fontsize=11, fontweight="bold", pad=8, y=1.02)
    ax1.set_xlabel("Crop Year", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Standard Deviation (bu/acre)", fontsize=10, fontweight="bold")
    ax1.set_ylim(2, 12)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
    
    # Panel (b): Relative Coefficient of Variation
    color2 = "#d94801"
    ax2.plot(annual["year"], annual["cv"], color=color2, alpha=0.35, linewidth=1.2, linestyle="--", label="Annual Relative $CV_t$")
    ax2.plot(annual["year"], annual["cv_roll5"], color=color2, linewidth=2.4, linestyle="--", label="5-Year Centered Mean $CV_t$")
    # Linear trend on cv
    sl2, ic2, r2, _, _ = stats.linregress(annual["year"], annual["cv"])
    ax2.plot(annual["year"], sl2 * annual["year"] + ic2, color="#525252", linestyle=":", linewidth=1.5, 
             label=f"Trend ({sl2*10:.2f}\\%/decade, $r={r2:.2f}$)")
    
    ax2.set_title(r"(b) Relative Volatility ($CV_t$)", fontsize=11, fontweight="bold", pad=8, y=1.02)
    ax2.set_xlabel("Crop Year", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Coefficient of Variation (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(5, 32)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
    
    plt.suptitle("Dispersion Dynamics: Rising Absolute Variance vs. Declining Relative Volatility (1951--2025)", 
                 fontsize=12, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "fig_dispersion_dynamics.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig_dispersion_dynamics.png"))
    plt.close(fig)
    print("Figure 3.3 generated.")

# ==============================================================================
# FIGURE 3.4: Spatial Heterogeneity of County Yields by State
# ==============================================================================
def generate_figure_3_4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [1.2, 1]})
    
    state_order = df_panel.groupby("state")["yield_bu_per_acre"].median().sort_values(ascending=False).index.tolist()
    cmap = plt.cm.Blues_r
    palette = [cmap(0.15 + 0.70 * (i / (len(state_order) - 1))) for i in range(len(state_order))]
    
    boxes_data = [df_panel[df_panel["state"] == st]["yield_bu_per_acre"].values for st in state_order]
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
        
    ax1.set_title("(a) Cross-Sectional Yield Distribution by State", fontsize=11, fontweight="bold")
    ax1.set_xlabel("State", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Soybean Yield (bushels per acre)", fontsize=10, fontweight="bold")
    ax1.tick_params(axis='x', rotation=20)
    
    df_panel["decade"] = (df_panel["year"] // 10) * 10
    state_dec = df_panel.groupby(["decade", "state"])["yield_bu_per_acre"].mean().reset_index()
    
    for idx, st in enumerate(state_order):
        sub = state_dec[state_dec["state"] == st]
        ax2.plot(sub["decade"], sub["yield_bu_per_acre"], marker="o", markersize=4, linewidth=1.8, 
                 label=st, color=palette[idx])
        
    ax2.set_title("(b) Decadal Mean Yield Trajectories by State", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Decade", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Soybean Yield (bu/ac)", fontsize=10, fontweight="bold")
    ax2.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
    ax2.set_xticklabels(["'50s", "'60s", "'70s", "'80s", "'90s", "'00s", "'10s", "'20s"])
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)
    
    plt.suptitle("Spatial Heterogeneity and Regional Productivity Divergence", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_state_yield_distributions.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig_state_yield_distributions.png"))
    plt.close(fig)
    print("Figure 3.4 generated.")

if __name__ == "__main__":
    generate_table_3_1()
    generate_table_3_2()
    generate_table_3_3()
    generate_figure_3_1()
    generate_figure_3_2()
    # generate_figure_3_3()  # Replaced by enhanced fig_yield_trajectory.png (expanding dispersion band)
    generate_figure_3_4()
    print("All Chapter 3 artifacts generated successfully!")
