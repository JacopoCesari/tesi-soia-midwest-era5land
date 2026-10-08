"""
Script to generate the thesis alternative figure:
Figure 3.2b: County Soybean Yield Trajectories & Dispersion Dynamics (1950–2025).

Styling:
- Reference line: Long-Term Linear Trend (OLS 1950–2025) in Deep Petrol Cyan (#164e63, dashed)
- Cross-sectional panel mean: Grigino Color Petrolio (#475569, solid line, no dots)
- Dual dispersion envelope:
  * Inner core: 90% cross-county dispersion (5th–95th percentile, solid borders)
  * Outer bounds: Full empirical records (min–max bounds, dotted lines)
  * Fills: Mint/emerald above reference line, terracotta/amber below reference line
- Historical benchmark contrast:
  * Double-headed arrows in crisp black (#0f172a) with dimension end-caps
  * Early baseline at 1951 (Delta = 14.4 bu/ac)
  * Recent expansion at 2024 (Delta = 28.2 bu/ac, +96%)
  * Inward-facing callout badges avoiding axis label and margin crowding
- Historical climatic shocks annotated with non-overlapping callouts.
- Complete 76-year span: 1950–2025 (10,260 observations across 135 balanced counties).

Output targets:
  output/thesis/figures/fig_yield_trajectory_dual_petrol.pdf
  output/thesis/figures/fig_yield_trajectory_dual_petrol.png
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Paths
_script_path = Path(__file__).resolve()
ROOT_DIR = _script_path.parents[2] if (_script_path.parents[1].name == "output") else _script_path.parents[1]
DATA_PATH = ROOT_DIR / "data" / "target" / "soybean_yield_1951_2025.csv"
FIGURES_DIR = ROOT_DIR / "output" / "thesis" / "figures"
CACHE_1950 = FIGURES_DIR / "cache_1950_yield.csv"
ARTIFACT_DIR = Path(r"C:\Users\JacopoCesari-Aretésr\.gemini\antigravity\brain\d4217d9d-cccd-437b-b9dc-275f466c315b")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11.5,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.2,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Load yield data including 1950 for complete 1950–2025 panel
if CACHE_1950.exists():
    df_1950 = pd.read_csv(CACHE_1950)
else:
    alt_cache = FIGURES_DIR / "alternatives" / "cache_1950_yield.csv"
    if alt_cache.exists():
        df_1950 = pd.read_csv(alt_cache)
        df_1950.to_csv(CACHE_1950, index=False)
    else:
        excel_path = ROOT_DIR / "data" / "raw" / "usda_nass" / "soybean_yield_479_counties_1950_2025.xlsx"
        df_target_counties = set(pd.read_csv(DATA_PATH)["county_fips"].unique())
        df_all = pd.read_excel(excel_path, sheet_name="all_years_survey")
        df_1950 = df_all[(df_all["year"] == 1950) & (df_all["county_fips"].isin(df_target_counties))][
            ["county_fips", "year", "yield_bu_per_acre"]
        ].sort_values("county_fips").reset_index(drop=True)
        df_1950.to_csv(CACHE_1950, index=False)

df_yield = pd.concat([df_1950, pd.read_csv(DATA_PATH)], ignore_index=True)

# Compute annual statistics across the 135 counties (1950–2025)
annual = df_yield.groupby("year")["yield_bu_per_acre"].agg(
    mean="mean",
    std="std",
    min="min",
    max="max",
    q05=lambda s: s.quantile(0.05),
    q95=lambda s: s.quantile(0.95),
    q25=lambda s: s.quantile(0.25),
    q75=lambda s: s.quantile(0.75),
).reset_index()

# Linear Trend (OLS over all 76 years: 1950–2025)
slope, intercept, r_val, _, _ = stats.linregress(annual["year"], annual["mean"])
annual["linear_trend"] = slope * annual["year"] + intercept

# Benchmark years for dispersion comparison
EARLY_YEAR = 1951
RECENT_YEAR = 2024

# Dual Petrol Palette
COLOR_ABOVE = "#059669"       # Emerald green outline
COLOR_ABOVE_FILL = "#10b981"  # Mint / soft emerald fill
COLOR_BELOW = "#b45309"       # Warm terracotta outline
COLOR_BELOW_FILL = "#f97316"  # Terracotta / amber fill
COLOR_COUNTY = "#94a3b8"      # Soft slate for spaghetti trajectories
COLOR_REF_LINE = "#164e63"    # Deep petrol cyan (dashed)
COLOR_MEAN_LINE = "#475569"   # Grigino color petrolio / slate grey (solid)
COLOR_ARROW = "#0f172a"       # Crisp black for arrows and badges

def generate_figure():
    fig, ax = plt.subplots(figsize=(13.2, 7.0))

    ref = annual["linear_trend"].values
    years = annual["year"].values
    
    # 1. County Trajectories (spaghetti lines)
    for _, grp in df_yield.groupby("county_fips"):
        ax.plot(grp["year"], grp["yield_bu_per_acre"], color=COLOR_COUNTY, alpha=0.15, linewidth=0.60, zorder=1)

    # 2. Dual Envelope (Min–Max Records + 90% Core)
    u_rec = annual["max"].values
    l_rec = annual["min"].values
    u_p90 = annual["q95"].values
    l_p90 = annual["q05"].values

    # Outer fills (soft records)
    ax.fill_between(
        years, np.maximum(l_rec, ref), u_rec, where=(u_rec >= ref),
        interpolate=True, color=COLOR_ABOVE_FILL, alpha=0.18, zorder=2
    )
    ax.fill_between(
        years, l_rec, np.minimum(u_rec, ref), where=(l_rec <= ref),
        interpolate=True, color=COLOR_BELOW_FILL, alpha=0.18, zorder=2
    )
    # Inner fills (core 90%)
    ax.fill_between(
        years, np.maximum(l_p90, ref), u_p90, where=(u_p90 >= ref),
        interpolate=True, color=COLOR_ABOVE_FILL, alpha=0.35,
        label="Above Reference (Upper County Envelope)", zorder=2
    )
    ax.fill_between(
        years, l_p90, np.minimum(u_p90, ref), where=(l_p90 <= ref),
        interpolate=True, color=COLOR_BELOW_FILL, alpha=0.35,
        label="Below Reference (Lower County Envelope)", zorder=2
    )
    # Outlines
    ax.plot(years, u_rec, color=COLOR_ABOVE, linewidth=0.9, linestyle=":", alpha=0.6, zorder=3)
    ax.plot(years, l_rec, color=COLOR_BELOW, linewidth=0.9, linestyle=":", alpha=0.6, zorder=3)
    ax.plot(years, u_p90, color=COLOR_ABOVE, linewidth=1.3, linestyle="-", alpha=0.9, zorder=3)
    ax.plot(years, l_p90, color=COLOR_BELOW, linewidth=1.3, linestyle="-", alpha=0.9, zorder=3)

    # 3. Reference Line (Linear Trend OLS)
    ref_label = f"Linear Trend (OLS: +{slope*10:.2f} bu/ac/decade)"
    ax.plot(years, ref, color=COLOR_REF_LINE, linestyle="--", linewidth=2.3, label=ref_label, zorder=5)

    # 4. Panel Cross-Sectional Mean Yield (solid line, no dots)
    ax.plot(years, annual["mean"], color=COLOR_MEAN_LINE, linewidth=2.0, linestyle="-", label="Panel Cross-Sectional Mean Yield", zorder=4)

    # 5. Historical Climatic Shocks
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
        ax.annotate(
            txt,
            xy=(yr, val),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.8,
            fontweight="bold",
            color="#991b1b" if "Bumper" not in txt else "#15803d",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.6, shrinkA=3, shrinkB=3),
            zorder=7
        )

    # 6. Double-Headed Vertical Arrows (<->) with Dimension End-Caps
    e_row = annual[annual["year"] == EARLY_YEAR].iloc[0]
    r_row = annual[annual["year"] == RECENT_YEAR].iloc[0]
    
    e_top, e_bot = e_row["q95"], e_row["q05"]
    r_top, r_bot = r_row["q95"], r_row["q05"]
    e_delta = e_top - e_bot
    r_delta = r_top - r_bot

    # Early Arrow at 1951
    ax.annotate(
        "",
        xy=(EARLY_YEAR, e_top),
        xytext=(EARLY_YEAR, e_bot),
        arrowprops=dict(
            arrowstyle="<->,head_width=0.38,head_length=0.65",
            color=COLOR_ARROW,
            lw=2.2,
            shrinkA=1,
            shrinkB=1
        ),
        zorder=8
    )
    cap_w = 0.55
    ax.plot([EARLY_YEAR - cap_w, EARLY_YEAR + cap_w], [e_top, e_top], color=COLOR_ARROW, lw=2.0, zorder=8)
    ax.plot([EARLY_YEAR - cap_w, EARLY_YEAR + cap_w], [e_bot, e_bot], color=COLOR_ARROW, lw=2.0, zorder=8)

    # Inward-facing callout badge for early arrow (placed to the right of arrow)
    ax.annotate(
        f"Narrow county dispersion\n$(\\Delta = {e_delta:.1f}$ bu/ac)",
        xy=(EARLY_YEAR, e_top),
        xytext=(14, 14),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=COLOR_ARROW,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor=COLOR_ARROW, alpha=0.96, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=0.9, shrinkB=4, connectionstyle="arc3,rad=-0.10"),
        zorder=9
    )

    # Recent Arrow at 2024
    ax.annotate(
        "",
        xy=(RECENT_YEAR, r_top),
        xytext=(RECENT_YEAR, r_bot),
        arrowprops=dict(
            arrowstyle="<->,head_width=0.38,head_length=0.65",
            color=COLOR_ARROW,
            lw=2.2,
            shrinkA=1,
            shrinkB=1
        ),
        zorder=8
    )
    ax.plot([RECENT_YEAR - cap_w, RECENT_YEAR + cap_w], [r_top, r_top], color=COLOR_ARROW, lw=2.0, zorder=8)
    ax.plot([RECENT_YEAR - cap_w, RECENT_YEAR + cap_w], [r_bot, r_bot], color=COLOR_ARROW, lw=2.0, zorder=8)

    # Inward-facing callout badge for recent arrow (placed to the left of arrow)
    pct_expand = ((r_delta - e_delta) / e_delta) * 100.0
    ax.annotate(
        f"Wide county dispersion\n$(\\Delta = {r_delta:.1f}$ bu/ac, $+{pct_expand:.0f}\\%)$",
        xy=(RECENT_YEAR, r_top),
        xytext=(-14, 16),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=COLOR_ARROW,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor=COLOR_ARROW, alpha=0.96, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=0.9, shrinkB=4, connectionstyle="arc3,rad=0.10"),
        zorder=9
    )

    # Title and Labels
    ax.set_title(
        f"County Soybean Yield Trajectories & Dispersion Dynamics (1950–2025)\n"
        f"Reference Line: Linear Trend | Envelope: Dual Envelope (Min–Max Records + 90% Dispersion Core)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Crop Year", fontsize=11, fontweight="bold")
    ax.set_ylabel("Soybean Yield (bushels per acre)", fontsize=11, fontweight="bold")
    ax.set_xlim(1948, 2027)
    ax.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2025])
    ax.set_ylim(0, 96)

    # Legend
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.95,
        fontsize=8.8
    )

    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.10)

    # Save to thesis figures folder
    out_pdf = FIGURES_DIR / "fig_yield_trajectory_dual_petrol.pdf"
    out_png = FIGURES_DIR / "fig_yield_trajectory_dual_petrol.png"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Generated: {out_png} and {out_pdf}")

    # Copy to artifact folder for immediate viewing
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_png, ARTIFACT_DIR / "fig_yield_trajectory_dual_petrol.png")

if __name__ == "__main__":
    generate_figure()
