"""
generate_unified_decile_comparison.py

Generates:
1. tables/tab_decile_distribution.tex
   - Two-panel table comparing Trend Expanding vs Trend Full-Sample:
     * Panel A: Intero Storico 1951-2025 (10,125 osservazioni)
     * Panel B: Periodo di Test 2011-2025 (2,025 osservazioni)
     Decile per decile (D1 ... D10 e Totale) con Range, Media e % eps < 0.
     NO "Fonte:" line.

2. figures/fig_decile_distribution_comparison.pdf & .png
   - 2x2 grid:
     * Left column: Intero Storico (10,125 dati) - Densita (top), Medie Decili (bottom)
     * Right column: Periodo di Test (2,025 dati) - Densita (top), Medie Decili (bottom)
     Consistent colors: Blue = Expanding Trend, Red = Full-Sample Trend.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = ROOT / "work" / "experiments" / "full_sample_detrending"
FIG_DIR = EXP_DIR / "figures"
TAB_DIR = EXP_DIR / "tables"

sys.path.insert(0, str(EXP_DIR))
from generate_all_outputs_and_report import load_baseline_predictions, load_full_detrend_predictions

TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"

plt.rcParams.update({
    "font.size": 9.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def compute_historical_anomalies(df_y):
    """Compute historical expanding trend (T_min=30, expanding from 1981) and full-sample trend."""
    records = []
    for fips, grp in df_y.groupby("county_fips"):
        grp = grp.sort_values("year").reset_index(drop=True)
        # Full sample fit
        p_full = np.polyfit(grp["year"], grp["yield_bu_per_acre"], 1)
        tr_full = p_full[0] * grp["year"] + p_full[1]
        
        # Expanding fit (T_min=30, base window 1951-1980)
        tr_exp = []
        for i, r in grp.iterrows():
            yr = int(r["year"])
            if i < 30:
                sub = grp.iloc[:30]
            else:
                sub = grp.iloc[:i]
            p_sub = np.polyfit(sub["year"], sub["yield_bu_per_acre"], 1)
            tr_exp.append(p_sub[0] * yr + p_sub[1])
            
        for yr, y_val, tf, te in zip(grp["year"], grp["yield_bu_per_acre"], tr_full, tr_exp):
            records.append({
                "county_fips": fips,
                "year": yr,
                "yield": y_val,
                "anom_full": y_val - tf,
                "anom_exp": y_val - te
            })
    return pd.DataFrame(records)


def get_decile_stats(s):
    labels = list(range(1, 11))
    q = pd.qcut(s, q=10, labels=labels)
    rows = []
    for d in labels:
        sub = s[q == d]
        rows.append({
            "decile": d,
            "min": sub.min(),
            "max": sub.max(),
            "mean": sub.mean(),
            "neg_pct": (sub < 0).mean() * 100
        })
    df_res = pd.DataFrame(rows)
    tot = {
        "decile": "Totale",
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "neg_pct": (s < 0).mean() * 100
    }
    return df_res, tot


def main():
    print("Loading datasets...")
    df_y = pd.read_csv(TARGET_CSV)
    df_y["county_fips"] = df_y["county_fips"].astype(str).str.zfill(5)
    
    # 1. Historical anomalies (10,125 obs)
    df_hist = compute_historical_anomalies(df_y)
    
    # 2. Test set anomalies (2,025 obs, 2011-2025)
    df_b = load_baseline_predictions()
    df_f = load_full_detrend_predictions()
    sub_b = df_b[(df_b["H"] == 1) & (df_b["model"] == "ElasticNet") & (df_b["spec"] == "Integrated")].copy()
    sub_f = df_f[(df_f["H"] == 1) & (df_f["model"] == "ElasticNet") & (df_f["spec"] == "Integrated")].copy()
    
    s_hist_exp = df_hist["anom_exp"]
    s_hist_full = df_hist["anom_full"]
    s_test_exp = sub_b["y_true_anomaly"]
    s_test_full = sub_f["y_true_anomaly"]
    
    # Calculate deciles
    d_hist_exp, tot_hist_exp = get_decile_stats(s_hist_exp)
    d_hist_full, tot_hist_full = get_decile_stats(s_hist_full)
    d_test_exp, tot_test_exp = get_decile_stats(s_test_exp)
    d_test_full, tot_test_full = get_decile_stats(s_test_full)
    
    # -------------------------------------------------------------------------
    # 1. WRITE TABLE (TWO PANELS, NO SOURCE)
    # -------------------------------------------------------------------------
    print("Writing tab_decile_distribution.tex...")
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{Distribuzione delle anomalie di resa ($\epsilon$, bu/ac) per decile: range e quota di osservazioni negative ($\epsilon < 0$) a confronto tra \textbf{Trend Expanding} e \textbf{Trend Full-Sample} sullo storico (Panel A, $10{,}125$ dati) e sul periodo di test (Panel B, $2{,}025$ dati).}",
        r"\label{tab:decile_distribution}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l cc cc @{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Trend Expanding (Train-Only / OOS)}} & \multicolumn{2}{c}{\textbf{Trend Full-Sample (In-Sample / Oracle)}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Decile} & \textbf{Range} & \textbf{\% } $\epsilon < 0$ & \textbf{Range} & \textbf{\% } $\epsilon < 0$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Panel A: Intero Storico 1951--2025 ($N = 10{,}125$ osservazioni)}} \\",
        r"\midrule",
    ]
    
    for i in range(10):
        re = d_hist_exp.iloc[i]
        rf = d_hist_full.iloc[i]
        d_lbl = f"$D_{{{i+1}}}$" + (" (Crollo)" if i == 0 else (" (Picco)" if i == 9 else ""))
        range_e = f"$[{re['min']:+.1f}, {re['max']:+.1f}]$"
        range_f = f"$[{rf['min']:+.1f}, {rf['max']:+.1f}]$"
        tex.append(f"{d_lbl} & {range_e} & {re['neg_pct']:.0f}\\% & {range_f} & {rf['neg_pct']:.0f}\\% \\\\")
        
    range_tot_he = f"$[{tot_hist_exp['min']:+.1f}, {tot_hist_exp['max']:+.1f}]$"
    range_tot_hf = f"$[{tot_hist_full['min']:+.1f}, {tot_hist_full['max']:+.1f}]$"
    tex.append(r"\midrule")
    tex.append(f"\\textbf{{Totale}} & {range_tot_he} & \\textbf{{{tot_hist_exp['neg_pct']:.1f}\\%}} & {range_tot_hf} & \\textbf{{{tot_hist_full['neg_pct']:.1f}\\%}} \\\\")
    
    tex.extend([
        r"\midrule[\heavyrulewidth]",
        r"\multicolumn{5}{l}{\textbf{Panel B: Periodo di Test Out-of-Sample 2011--2025 ($N = 2{,}025$ osservazioni)}} \\",
        r"\midrule",
    ])
    
    for i in range(10):
        re = d_test_exp.iloc[i]
        rf = d_test_full.iloc[i]
        d_lbl = f"$D_{{{i+1}}}$" + (" (Crollo)" if i == 0 else (" (Picco)" if i == 9 else ""))
        range_e = f"$[{re['min']:+.1f}, {re['max']:+.1f}]$"
        range_f = f"$[{rf['min']:+.1f}, {rf['max']:+.1f}]$"
        tex.append(f"{d_lbl} & {range_e} & {re['neg_pct']:.0f}\\% & {range_f} & {rf['neg_pct']:.0f}\\% \\\\")
        
    range_tot_te = f"$[{tot_test_exp['min']:+.1f}, {tot_test_exp['max']:+.1f}]$"
    range_tot_tf = f"$[{tot_test_full['min']:+.1f}, {tot_test_full['max']:+.1f}]$"
    tex.append(r"\midrule")
    tex.append(f"\\textbf{{Totale}} & {range_tot_te} & \\textbf{{{tot_test_exp['neg_pct']:.1f}\\%}} & {range_tot_tf} & \\textbf{{{tot_test_full['neg_pct']:.1f}\\%}} \\\\")
    
    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
    
    out_tex = TAB_DIR / "tab_decile_distribution.tex"
    out_tex.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {out_tex}")
    
    # -------------------------------------------------------------------------
    # 2. WRITE FIGURE (2x2 GRID, HIGH CLARITY)
    # -------------------------------------------------------------------------
    print("Generating 2x2 comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 4.6))
    
    color_exp = "#1f77b4"   # Steel blue
    color_full = "#d62728"  # Crimson red
    
    # (a) Historical Density (Top-Left)
    ax_a = axes[0, 0]
    grid_h = np.linspace(-26, 24, 400)
    kde_he = gaussian_kde(s_hist_exp)
    kde_hf = gaussian_kde(s_hist_full)
    ax_a.plot(grid_h, kde_he(grid_h), label=f"Expanding (Media {s_hist_exp.mean():+.2f})", color=color_exp, linestyle="--", linewidth=1.8)
    ax_a.plot(grid_h, kde_hf(grid_h), label=f"Full-Sample (Media {s_hist_full.mean():+.2f})", color=color_full, linestyle="-", linewidth=1.8)
    ax_a.axvline(s_hist_exp.mean(), color=color_exp, linestyle="--", alpha=0.6, linewidth=1.0)
    ax_a.axvline(s_hist_full.mean(), color=color_full, linestyle="-", alpha=0.6, linewidth=1.0)
    ax_a.axvline(0, color="gray", linestyle=":", alpha=0.7, linewidth=0.9)
    ax_a.set_title("(a) Densità Anomalia: Storico 1951--2025 ($N=10{,}125$)")
    ax_a.set_xlabel("Anomalia di Resa $\\epsilon$ (bu/ac)")
    ax_a.set_ylabel("Densità")
    ax_a.set_xlim(-27, 25)
    ax_a.legend(loc="upper left")
    
    # (b) Test Set Density (Top-Right)
    ax_b = axes[0, 1]
    grid_t = np.linspace(-26, 24, 400)
    kde_te = gaussian_kde(s_test_exp)
    kde_tf = gaussian_kde(s_test_full)
    ax_b.plot(grid_t, kde_te(grid_t), label=f"Baseline Expanding (Media {s_test_exp.mean():+.2f})", color=color_exp, linestyle="--", linewidth=1.8)
    ax_b.plot(grid_t, kde_tf(grid_t), label=f"Full-Sample (Media {s_test_full.mean():+.2f})", color=color_full, linestyle="-", linewidth=1.8)
    ax_b.axvline(s_test_exp.mean(), color=color_exp, linestyle="--", alpha=0.6, linewidth=1.0)
    ax_b.axvline(s_test_full.mean(), color=color_full, linestyle="-", alpha=0.6, linewidth=1.0)
    ax_b.axvline(0, color="gray", linestyle=":", alpha=0.7, linewidth=0.9)
    ax_b.set_title("(b) Densità Anomalia: Periodo Test 2011--2025 ($N=2{,}025$)")
    ax_b.set_xlabel("Anomalia di Resa $\\epsilon$ (bu/ac)")
    ax_b.set_ylabel("Densità")
    ax_b.set_xlim(-27, 25)
    ax_b.legend(loc="upper left")
    
    # (c) Historical Decile Means (Bottom-Left)
    ax_c = axes[1, 0]
    x_dec = np.arange(1, 11)
    w = 0.35
    ax_c.bar(x_dec - w/2, d_hist_exp["mean"], width=w, label="Expanding Trend", color=color_exp, alpha=0.85, edgecolor="none")
    ax_c.bar(x_dec + w/2, d_hist_full["mean"], width=w, label="Full-Sample Trend", color=color_full, alpha=0.85, edgecolor="none")
    ax_c.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=0.9)
    ax_c.set_xticks(x_dec)
    ax_c.set_xticklabels([f"$D_{{{i}}}$" for i in x_dec])
    ax_c.set_title("(c) Media Anomalia per Decile: Storico 1951--2025")
    ax_c.set_xlabel("Decile di Anomalia")
    ax_c.set_ylabel("Media Anomalia (bu/ac)")
    ax_c.set_ylim(-11.5, 12.5)
    ax_c.legend(loc="upper left")
    
    # (d) Test Set Decile Means (Bottom-Right)
    ax_d = axes[1, 1]
    ax_d.bar(x_dec - w/2, d_test_exp["mean"], width=w, label="Baseline Expanding", color=color_exp, alpha=0.85, edgecolor="none")
    ax_d.bar(x_dec + w/2, d_test_full["mean"], width=w, label="Full-Sample Trend", color=color_full, alpha=0.85, edgecolor="none")
    ax_d.axhline(0, color="gray", linestyle=":", alpha=0.7, linewidth=0.9)
    ax_d.set_xticks(x_dec)
    ax_d.set_xticklabels([f"$D_{{{i}}}$" for i in x_dec])
    ax_d.set_title("(d) Media Anomalia per Decile: Periodo Test 2011--2025")
    ax_d.set_xlabel("Decile di Anomalia")
    ax_d.set_ylabel("Media Anomalia (bu/ac)")
    ax_d.set_ylim(-11.5, 12.5)
    ax_d.legend(loc="upper left")
    
    plt.tight_layout()
    pdf_path = FIG_DIR / "fig_decile_distribution_comparison.pdf"
    png_path = FIG_DIR / "fig_decile_distribution_comparison.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
