"""
update_tables_with_summary_metrics.py

Updates:
1. tables/tab_decile_error_performance_comparison.tex (Table 2)
   - Adds "Media Modelli" column (Base vs Full).
   - Adds summary rows at the bottom: Bias Medio, MAE, RMSE, Pearson r.
   - Bolds lowest error / best metric between Base and Full.
   - NO source footnote.

2. tables/tab_yearly_raw_yields.tex (Table 3)
   - Adds "Media Modelli" column (Base vs Full).
   - Adds summary rows at the bottom: Media, MAE, RMSE, Pearson r.
   - Bolds lowest error / closest prediction / best metric between Base and Full.
   - NO source footnote.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = ROOT / "work" / "experiments" / "full_sample_detrending"
TAB_DIR = EXP_DIR / "tables"

sys.path.insert(0, str(EXP_DIR))
from generate_all_outputs_and_report import load_baseline_predictions, load_full_detrend_predictions

# =============================================================================
# 1. GENERATE UPDATED TABLE 2 (DECILES)
# =============================================================================
def generate_table_2():
    print("Generating updated Table 2...")
    df_b = load_baseline_predictions()
    df_f = load_full_detrend_predictions()
    
    sb = df_b[(df_b["H"] == 1) & (df_b["spec"] == "Integrated")].copy()
    sf = df_f[(df_f["H"] == 1) & (df_f["spec"] == "Integrated")].copy()
    
    sb["decile"] = pd.qcut(sb["y_true_anomaly"], 10, labels=range(1, 11))
    sf["decile"] = pd.qcut(sf["y_true_anomaly"], 10, labels=range(1, 11))
    
    models = ["ElasticNet", "XGBoost", "LSTM", "Ensemble"]
    
    decile_rows = []
    for d in range(1, 11):
        sub_b = sb[sb["decile"] == d]
        sub_f = sf[sf["decile"] == d]
        
        yt_b = sub_b[sub_b["model"] == "ElasticNet"]["y_true_anomaly"].mean()
        yt_f = sub_f[sub_f["model"] == "ElasticNet"]["y_true_anomaly"].mean()
        
        # predictions
        en_b = sub_b[sub_b["model"] == "ElasticNet"]["y_pred_anomaly"].mean()
        en_f = sub_f[sub_f["model"] == "ElasticNet"]["y_pred_anomaly"].mean()
        xgb_b = sub_b[sub_b["model"] == "XGBoost"]["y_pred_anomaly"].mean()
        xgb_f = sub_f[sub_f["model"] == "XGBoost"]["y_pred_anomaly"].mean()
        ls_b = sub_b[sub_b["model"] == "LSTM"]["y_pred_anomaly"].mean()
        ls_f = sub_f[sub_f["model"] == "LSTM"]["y_pred_anomaly"].mean()
        ens_b = sub_b[sub_b["model"] == "Ensemble"]["y_pred_anomaly"].mean()
        ens_f = sub_f[sub_f["model"] == "Ensemble"]["y_pred_anomaly"].mean()
        
        ml_b = np.mean([en_b, xgb_b, ls_b, ens_b])
        ml_f = np.mean([en_f, xgb_f, ls_f, ens_f])
        
        # errors
        e_en_b, e_en_f = en_b - yt_b, en_f - yt_f
        e_xgb_b, e_xgb_f = xgb_b - yt_b, xgb_f - yt_f
        e_ls_b, e_ls_f = ls_b - yt_b, ls_f - yt_f
        e_ens_b, e_ens_f = ens_b - yt_b, ens_f - yt_f
        e_ml_b, e_ml_f = ml_b - yt_b, ml_f - yt_f
        
        decile_rows.append({
            "d": d, "yt_b": yt_b, "yt_f": yt_f,
            "e_en_b": e_en_b, "e_en_f": e_en_f,
            "e_xgb_b": e_xgb_b, "e_xgb_f": e_xgb_f,
            "e_ls_b": e_ls_b, "e_ls_f": e_ls_f,
            "e_ens_b": e_ens_b, "e_ens_f": e_ens_f,
            "e_ml_b": e_ml_b, "e_ml_f": e_ml_f,
            "pred_en_b": en_b, "pred_en_f": en_f,
            "pred_xgb_b": xgb_b, "pred_xgb_f": xgb_f,
            "pred_ls_b": ls_b, "pred_ls_f": ls_f,
            "pred_ens_b": ens_b, "pred_ens_f": ens_f,
            "pred_ml_b": ml_b, "pred_ml_f": ml_f,
        })
        
    df_d = pd.DataFrame(decile_rows)
    
    # Calculate summary metrics over 10 deciles
    metrics = {}
    col_pairs = [
        ("en", "e_en_b", "e_en_f", "pred_en_b", "pred_en_f"),
        ("xgb", "e_xgb_b", "e_xgb_f", "pred_xgb_b", "pred_xgb_f"),
        ("ls", "e_ls_b", "e_ls_f", "pred_ls_b", "pred_ls_f"),
        ("ens", "e_ens_b", "e_ens_f", "pred_ens_b", "pred_ens_f"),
        ("ml", "e_ml_b", "e_ml_f", "pred_ml_b", "pred_ml_f"),
    ]
    
    for tag, eb_col, ef_col, pb_col, pf_col in col_pairs:
        # Bias
        bias_b = df_d[eb_col].mean()
        bias_f = df_d[ef_col].mean()
        # MAE
        mae_b = np.abs(df_d[eb_col]).mean()
        mae_f = np.abs(df_d[ef_col]).mean()
        # RMSE
        rmse_b = np.sqrt((df_d[eb_col]**2).mean())
        rmse_f = np.sqrt((df_d[ef_col]**2).mean())
        # Pearson r
        r_b, _ = pearsonr(df_d[pb_col], df_d["yt_b"])
        r_f, _ = pearsonr(df_d[pf_col], df_d["yt_f"])
        
        metrics[tag] = {
            "bias_b": bias_b, "bias_f": bias_f,
            "mae_b": mae_b, "mae_f": mae_f,
            "rmse_b": rmse_b, "rmse_f": rmse_f,
            "r_b": r_b, "r_f": r_f,
        }
        
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Errore medio di previsione per decile ($\bar{e} = \bar{\hat{\epsilon}} - \bar{\epsilon}$, bu/ac) a chiusura raccolto $H = 1^*$. Confronto tra \textbf{Baseline (Expanding)} e \textbf{Full-Sample}. Nelle righe riassuntive: Bias Medio, MAE, RMSE e Pearson $r$ sui decili. In \textbf{grassetto} la specificazione più performante.}",
        r"\label{tab:comp_decile_error_performance_short}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l cc cc cc cc cc cc @{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Observed } $\bar{\epsilon}$} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} & \multicolumn{2}{c}{\textbf{Media Modelli}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
        r"\textbf{Decile} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
        r"\midrule",
    ]
    
    def fmt_pair(vb, vf):
        ab, af = abs(vb), abs(vf)
        if ab < af:
            return f"\\textbf{{{vb:+.2f}}}", f"{vf:+.2f}"
        elif af < ab:
            return f"{vb:+.2f}", f"\\textbf{{{vf:+.2f}}}"
        else:
            return f"{vb:+.2f}", f"{vf:+.2f}"
            
    for i, r in df_d.iterrows():
        d = int(r["d"])
        d_lbl = f"$D_{{{d}}}$" + (" (Crollo)" if d == 1 else (" (Picco)" if d == 10 else ""))
        
        en_b_s, en_f_s = fmt_pair(r["e_en_b"], r["e_en_f"])
        xgb_b_s, xgb_f_s = fmt_pair(r["e_xgb_b"], r["e_xgb_f"])
        ls_b_s, ls_f_s = fmt_pair(r["e_ls_b"], r["e_ls_f"])
        ens_b_s, ens_f_s = fmt_pair(r["e_ens_b"], r["e_ens_f"])
        ml_b_s, ml_f_s = fmt_pair(r["e_ml_b"], r["e_ml_f"])
        
        tex.append(f"{d_lbl} & {r['yt_b']:+.2f} & {r['yt_f']:+.2f} & {en_b_s} & {en_f_s} & {xgb_b_s} & {xgb_f_s} & {ls_b_s} & {ls_f_s} & {ens_b_s} & {ens_f_s} & {ml_b_s} & {ml_f_s} \\\\")
        
    tex.append(r"\midrule")
    
    # Summary rows
    # 1. Bias Medio
    b_strs = []
    for tag in ["en", "xgb", "ls", "ens", "ml"]:
        vb = metrics[tag]["bias_b"]
        vf = metrics[tag]["bias_f"]
        s_b, s_f = fmt_pair(vb, vf)
        b_strs.extend([s_b, s_f])
    tex.append(r"\textbf{Bias Medio} & --- & --- & " + " & ".join(b_strs) + r" \\")
    
    # 2. MAE
    mae_strs = []
    for tag in ["en", "xgb", "ls", "ens", "ml"]:
        vb = metrics[tag]["mae_b"]
        vf = metrics[tag]["mae_f"]
        if vb < vf:
            mae_strs.extend([f"\\textbf{{{vb:.2f}}}", f"{vf:.2f}"])
        elif vf < vb:
            mae_strs.extend([f"{vb:.2f}", f"\\textbf{{{vf:.2f}}}"])
        else:
            mae_strs.extend([f"{vb:.2f}", f"{vf:.2f}"])
    tex.append(r"\textbf{MAE} & --- & --- & " + " & ".join(mae_strs) + r" \\")
    
    # 3. RMSE
    rmse_strs = []
    for tag in ["en", "xgb", "ls", "ens", "ml"]:
        vb = metrics[tag]["rmse_b"]
        vf = metrics[tag]["rmse_f"]
        if vb < vf:
            rmse_strs.extend([f"\\textbf{{{vb:.2f}}}", f"{vf:.2f}"])
        elif vf < vb:
            rmse_strs.extend([f"{vb:.2f}", f"\\textbf{{{vf:.2f}}}"])
        else:
            rmse_strs.extend([f"{vb:.2f}", f"{vf:.2f}"])
    tex.append(r"\textbf{RMSE} & --- & --- & " + " & ".join(rmse_strs) + r" \\")
    
    # 4. Pearson r
    r_strs = []
    for tag in ["en", "xgb", "ls", "ens", "ml"]:
        vb = metrics[tag]["r_b"]
        vf = metrics[tag]["r_f"]
        if vb > vf:
            r_strs.extend([f"\\textbf{{{vb:.3f}}}", f"{vf:.3f}"])
        elif vf > vb:
            r_strs.extend([f"{vb:.3f}", f"\\textbf{{{vf:.3f}}}"])
        else:
            r_strs.extend([f"{vb:.3f}", f"{vf:.3f}"])
    tex.append(r"\textbf{Pearson } $r$ & --- & --- & " + " & ".join(r_strs) + r" \\")
    
    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
    
    out_tab2 = TAB_DIR / "tab_decile_error_performance_comparison.tex"
    out_tab2.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {out_tab2}")


# =============================================================================
# 2. GENERATE UPDATED TABLE 3 (YEARLY MACRO-LEVEL)
# =============================================================================
def generate_table_3():
    print("Generating updated Table 3...")
    df_b = load_baseline_predictions()
    df_f = load_full_detrend_predictions()
    
    ydf = pd.read_csv(ROOT / "data" / "target" / "soybean_yield_1951_2025.csv")
    ydf["county_fips"] = ydf["county_fips"].astype(str).str.zfill(5)
    mean_yields = ydf[ydf["year"] >= 2011].groupby("year")["yield_bu_per_acre"].mean().to_dict()
    sorted_years = [y for y, _ in sorted(mean_yields.items(), key=lambda x: x[1])]
    
    sb_h1 = df_b[(df_b["H"] == 1) & (df_b["spec"] == "Integrated")]
    sf_h1 = df_f[(df_f["H"] == 1) & (df_f["spec"] == "Integrated")]
    
    records = []
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
        
        ml_b = np.mean([en_b, xgb_b, ls_b, ens_b])
        ml_f = np.mean([en_f, xgb_f, ls_f, ens_f])
        
        records.append({
            "year": y, "y_obs": y_obs,
            "tau_b": tau_b, "tau_f": tau_f,
            "en_b": en_b, "en_f": en_f,
            "xgb_b": xgb_b, "xgb_f": xgb_f,
            "ls_b": ls_b, "ls_f": ls_f,
            "ens_b": ens_b, "ens_f": ens_f,
            "ml_b": ml_b, "ml_f": ml_f,
        })
        
    df_y = pd.DataFrame(records)
    
    # Calculate summary metrics over 15 years
    col_pairs = [
        ("tau", "tau_b", "tau_f"),
        ("en", "en_b", "en_f"),
        ("xgb", "xgb_b", "xgb_f"),
        ("ls", "ls_b", "ls_f"),
        ("ens", "ens_b", "ens_f"),
        ("ml", "ml_b", "ml_f"),
    ]
    
    metrics = {}
    obs = df_y["y_obs"]
    for tag, b_col, f_col in col_pairs:
        pb = df_y[b_col]
        pf = df_y[f_col]
        
        mean_b = pb.mean()
        mean_f = pf.mean()
        mae_b = np.mean(np.abs(pb - obs))
        mae_f = np.mean(np.abs(pf - obs))
        rmse_b = np.sqrt(np.mean((pb - obs)**2))
        rmse_f = np.sqrt(np.mean((pf - obs)**2))
        r_b, _ = pearsonr(pb, obs)
        r_f, _ = pearsonr(pf, obs)
        
        metrics[tag] = {
            "mean_b": mean_b, "mean_f": mean_f,
            "mae_b": mae_b, "mae_f": mae_f,
            "rmse_b": rmse_b, "rmse_f": rmse_f,
            "r_b": r_b, "r_f": r_f,
        }
        
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{Previsioni aggregate regionali di resa fisica ($\bar{\hat{y}} = \bar{\tau} + \bar{\hat{\epsilon}}$, bu/ac) a chiusura raccolto $H^*(s)$ sui 15 anni di test. Confronto tra \textbf{Baseline (Expanding)} e \textbf{Full-Sample}. Nelle righe finali: Media, MAE, RMSE e Pearson $r$ sui 15 anni. In \textbf{grassetto} la specificazione più vicina al reale.}",
        r"\label{tab:yearly_raw_yields}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c cc cc cc cc cc cc @{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Secular } $\bar{\tau}$} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} & \multicolumn{2}{c}{\textbf{Media Modelli}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
        r"\textbf{Anno} & \textbf{Reale} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
        r"\midrule",
    ]
    
    def fmt_pred(vb, vf, target):
        ab, af = abs(vb - target), abs(vf - target)
        if ab < af:
            return f"\\textbf{{{vb:.2f}}}", f"{vf:.2f}"
        elif af < ab:
            return f"{vb:.2f}", f"\\textbf{{{vf:.2f}}}"
        else:
            return f"{vb:.2f}", f"{vf:.2f}"
            
    for _, r in df_y.iterrows():
        y = int(r["year"])
        y_obs = r["y_obs"]
        
        tau_b_s, tau_f_s = fmt_pred(r["tau_b"], r["tau_f"], y_obs)
        en_b_s, en_f_s = fmt_pred(r["en_b"], r["en_f"], y_obs)
        xgb_b_s, xgb_f_s = fmt_pred(r["xgb_b"], r["xgb_f"], y_obs)
        ls_b_s, ls_f_s = fmt_pred(r["ls_b"], r["ls_f"], y_obs)
        ens_b_s, ens_f_s = fmt_pred(r["ens_b"], r["ens_f"], y_obs)
        ml_b_s, ml_f_s = fmt_pred(r["ml_b"], r["ml_f"], y_obs)
        
        tex.append(f"{y} & {y_obs:.2f} & {tau_b_s} & {tau_f_s} & {en_b_s} & {en_f_s} & {xgb_b_s} & {xgb_f_s} & {ls_b_s} & {ls_f_s} & {ens_b_s} & {ens_f_s} & {ml_b_s} & {ml_f_s} \\\\")
        
    tex.append(r"\midrule")
    
    # 1. Media
    mean_obs = obs.mean()
    m_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = metrics[tag]["mean_b"]
        vf = metrics[tag]["mean_f"]
        s_b, s_f = fmt_pred(vb, vf, mean_obs)
        m_strs.extend([s_b, s_f])
    tex.append(f"\\textbf{{Media}} & {mean_obs:.2f} & " + " & ".join(m_strs) + r" \\")
    
    # 2. MAE
    mae_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = metrics[tag]["mae_b"]
        vf = metrics[tag]["mae_f"]
        if vb < vf:
            mae_strs.extend([f"\\textbf{{{vb:.2f}}}", f"{vf:.2f}"])
        elif vf < vb:
            mae_strs.extend([f"{vb:.2f}", f"\\textbf{{{vf:.2f}}}"])
        else:
            mae_strs.extend([f"{vb:.2f}", f"{vf:.2f}"])
    tex.append(r"\textbf{MAE} & --- & " + " & ".join(mae_strs) + r" \\")
    
    # 3. RMSE
    rmse_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = metrics[tag]["rmse_b"]
        vf = metrics[tag]["rmse_f"]
        if vb < vf:
            rmse_strs.extend([f"\\textbf{{{vb:.2f}}}", f"{vf:.2f}"])
        elif vf < vb:
            rmse_strs.extend([f"{vb:.2f}", f"\\textbf{{{vf:.2f}}}"])
        else:
            rmse_strs.extend([f"{vb:.2f}", f"{vf:.2f}"])
    tex.append(r"\textbf{RMSE} & --- & " + " & ".join(rmse_strs) + r" \\")
    
    # 4. Pearson r
    r_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = metrics[tag]["r_b"]
        vf = metrics[tag]["r_f"]
        if vb > vf:
            r_strs.extend([f"\\textbf{{{vb:.3f}}}", f"{vf:.3f}"])
        elif vf > vb:
            r_strs.extend([f"{vb:.3f}", f"\\textbf{{{vf:.3f}}}"])
        else:
            r_strs.extend([f"{vb:.3f}", f"{vf:.3f}"])
    tex.append(r"\textbf{Pearson } $r$ & --- & " + " & ".join(r_strs) + r" \\")
    
    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
    
    out_tab3 = TAB_DIR / "tab_yearly_raw_yields.tex"
    out_tab3.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {out_tab3}")


if __name__ == "__main__":
    generate_table_2()
    generate_table_3()
