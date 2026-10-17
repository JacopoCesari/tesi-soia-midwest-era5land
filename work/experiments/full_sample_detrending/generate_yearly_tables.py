"""
generate_yearly_tables.py

Generates clean LaTeX tables for the 15 test years ordered by yield level:
1. tab_yearly_ordered_yield_performance.tex (multi-horizon anomaly predictions)
2. tab_yearly_raw_yields.tex (harvest closure physical yield level predictions)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = ROOT / "work" / "experiments" / "full_sample_detrending"
sys.path.append(str(EXP_DIR))
from generate_all_outputs_and_report import load_baseline_predictions, load_full_detrend_predictions

df_b = load_baseline_predictions()
df_f = load_full_detrend_predictions()

ydf = pd.read_csv(ROOT / "data" / "target" / "soybean_yield_1951_2025.csv")
ydf["county_fips"] = ydf["county_fips"].astype(str).str.zfill(5)
mean_yields = ydf[ydf["year"] >= 2011].groupby("year")["yield_bu_per_acre"].mean().to_dict()
sorted_years = [y for y, _ in sorted(mean_yields.items(), key=lambda x: x[1])]

TAB_DIR = EXP_DIR / "tables"

# 1. Anomaly table across H=5, 3, 1*
tex_anom = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\singlespacing",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2.5pt}",
    r"\renewcommand{\arraystretch}{1.06}",
    r"\caption{Replication of Table~5.3 across the 15 out-of-sample test years (2011--2025) \textbf{ordered by realized regional yield level} (lowest to highest), evaluating regional pooled mean point forecasts ($\bar{\hat{\epsilon}}$, bu/ac) across all 135 study counties. Direct comparison between \textbf{Baseline Expanding Trend} and \textbf{Full-Sample In-Sample Trend} across milestone horizons ($H = 5, 3, 1^*$).}",
    r"\label{tab:yearly_ordered_yield_performance}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c cc cc cc cc cc cc @{}}",
    r"\toprule",
    r"& & \multicolumn{2}{c}{\textbf{Observed ($\bar{\epsilon}$)}} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{Random Forest}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} \\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
    r"\textbf{Year} & \textbf{Yield} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
    r"\midrule",
]

panel_titles = {
    5: r"Panel A: Early Vegetative Horizon ($H=5$, June 30)",
    3: r"Panel B: Pod-Filling Horizon ($H=3$, August 31)",
    1: r"Panel C: Harvest Closure Horizon ($H=1^*$, October 31 / Sept 30 MN)",
}

for H in [5, 3, 1]:
    tex_anom.append(f"\\multicolumn{{14}}{{l}}{{\\textbf{{{panel_titles[H]}}}}} \\\\")
    sb = df_b[(df_b["H"] == H) & (df_b["spec"] == "Integrated")]
    sf = df_f[(df_f["H"] == H) & (df_f["spec"] == "Integrated")]

    for y in sorted_years:
        y_obs = mean_yields[y]
        sub_b = sb[sb["year"] == y]
        sub_f = sf[sf["year"] == y]

        yt_b = sub_b[sub_b["model"] == "ElasticNet"]["y_true_anomaly"].mean()
        yt_f = sub_f[sub_f["model"] == "ElasticNet"]["y_true_anomaly"].mean()

        en_b = sub_b[sub_b["model"] == "ElasticNet"]["y_pred_anomaly"].mean()
        en_f = sub_f[sub_f["model"] == "ElasticNet"]["y_pred_anomaly"].mean()

        rf_b = sub_b[sub_b["model"] == "RandomForest"]["y_pred_anomaly"].mean()
        rf_f = sub_f[sub_f["model"] == "RandomForest"]["y_pred_anomaly"].mean()

        xgb_b = sub_b[sub_b["model"] == "XGBoost"]["y_pred_anomaly"].mean()
        xgb_f = sub_f[sub_f["model"] == "XGBoost"]["y_pred_anomaly"].mean()

        ls_b = sub_b[sub_b["model"] == "LSTM"]["y_pred_anomaly"].mean()
        ls_f = sub_f[sub_f["model"] == "LSTM"]["y_pred_anomaly"].mean()

        ens_b = sub_b[sub_b["model"] == "Ensemble"]["y_pred_anomaly"].mean()
        ens_f = sub_f[sub_f["model"] == "Ensemble"]["y_pred_anomaly"].mean()

        line = f"{y} & {y_obs:.1f} & {yt_b:+.2f} & {yt_f:+.2f} & {en_b:+.2f} & {en_f:+.2f} & {rf_b:+.2f} & {rf_f:+.2f} & {xgb_b:+.2f} & {xgb_f:+.2f} & {ls_b:+.2f} & {ls_f:+.2f} & {ens_b:+.2f} & {ens_f:+.2f} \\\\"
        tex_anom.append(line)
    tex_anom.append(r"\midrule")

tex_anom.pop()
tex_anom.extend([
    r"\bottomrule",
    r"\end{tabular*}",
    r"\vspace{0.2em}",
    r"{\scriptsize \textit{Source: Author's calculation on USDA NASS and ECMWF ERA5-Land data, 2011--2025 out-of-sample test evaluations. Yield: regional mean realized yield across 135 study counties (bu/ac). Base: baseline expanding-window detrending; Full: full-sample in-sample trend (1951--2025). Values report mean point forecasts pooled over all counties.}}",
    r"\end{table}",
])

out_anom = TAB_DIR / "tab_yearly_ordered_yield_performance.tex"
out_anom.write_text("\n".join(tex_anom), encoding="utf-8")
print(f"Generated {out_anom}")

# 2. Raw yield level table at H=1*
sb_h1 = df_b[(df_b["H"] == 1) & (df_b["spec"] == "Integrated")]
sf_h1 = df_f[(df_f["H"] == 1) & (df_f["spec"] == "Integrated")]

tex_raw = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\singlespacing",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3.0pt}",
    r"\renewcommand{\arraystretch}{1.08}",
    r"\caption{Regional mean out-of-sample physical yield level projections ($\bar{\hat{y}} = \bar{\tau} + \bar{\hat{\epsilon}}$, bu/ac) at harvest closure $H^*(s)$ across the 15 test years ordered by realized yield. Contrasting \textbf{Baseline} vs \textbf{Full-Sample Detrending}.}",
    r"\label{tab:yearly_raw_yields}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c cc cc cc cc cc @{}}",
    r"\toprule",
    r"& & \multicolumn{2}{c}{\textbf{Secular Trend ($\bar{\tau}$)}} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} \\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
    r"\textbf{Year} & \textbf{Observed} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
    r"\midrule",
]

for y in sorted_years:
    y_obs = mean_yields[y]
    sub_b = sb_h1[sb_h1["year"] == y]
    sub_f = sf_h1[sf_h1["year"] == y]

    yt_b = sub_b[sub_b["model"] == "ElasticNet"]["y_true_anomaly"].mean()
    yt_f = sub_f[sub_f["model"] == "ElasticNet"]["y_true_anomaly"].mean()
    tau_b = y_obs - yt_b
    tau_f = y_obs - yt_f

    en_b = tau_b + sub_b[sub_b["model"] == "ElasticNet"]["y_pred_anomaly"].mean()
    en_f = tau_f + sub_f[sub_f["model"] == "ElasticNet"]["y_pred_anomaly"].mean()

    xgb_b = tau_b + sub_b[sub_b["model"] == "XGBoost"]["y_pred_anomaly"].mean()
    xgb_f = tau_f + sub_f[sub_f["model"] == "XGBoost"]["y_pred_anomaly"].mean()

    ls_b = tau_b + sub_b[sub_b["model"] == "LSTM"]["y_pred_anomaly"].mean()
    ls_f = tau_f + sub_f[sub_f["model"] == "LSTM"]["y_pred_anomaly"].mean()

    ens_b = tau_b + sub_b[sub_b["model"] == "Ensemble"]["y_pred_anomaly"].mean()
    ens_f = tau_f + sub_f[sub_f["model"] == "Ensemble"]["y_pred_anomaly"].mean()

    line = f"{y} & {y_obs:.2f} & {tau_b:.2f} & {tau_f:.2f} & {en_b:.2f} & {en_f:.2f} & {xgb_b:.2f} & {xgb_f:.2f} & {ls_b:.2f} & {ls_f:.2f} & {ens_b:.2f} & {ens_f:.2f} \\\\"
    tex_raw.append(line)

tex_raw.extend([
    r"\bottomrule",
    r"\end{tabular*}",
    r"\vspace{0.2em}",
    r"{\scriptsize \textit{Source: Author's calculation on USDA NASS and ECMWF ERA5-Land data, 2011--2025 out-of-sample evaluations ($2{,}025$ observations). Base: baseline expanding detrending; Full: full-sample in-sample trend (1951--2025). Values report regional mean yield in bu/ac.}}",
    r"\end{table}",
])

out_raw = TAB_DIR / "tab_yearly_raw_yields.tex"
out_raw.write_text("\n".join(tex_raw), encoding="utf-8")
print(f"Generated {out_raw}")
