"""
generate_cubic_report_artifacts.py

Generates updated tables and figures for Report_detrending using:
- Baseline: Expanding Window Linear Detrending
- Full-Sample: County-Level Cubic Polynomial Detrending (best in-sample R2 = 0.831)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = ROOT / "work" / "experiments" / "full_sample_detrending"
TAB_DIR = EXP_DIR / "tables"
FIG_DIR = EXP_DIR / "figures"
DATA_DIR = EXP_DIR / "data"

sys.path.insert(0, str(EXP_DIR))
from generate_all_outputs_and_report import load_baseline_predictions

plt.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def generate_decile_table_3_horizons():
    print("Generating tab_decile_error_performance_comparison.tex (3 horizons)...")
    df_b = load_baseline_predictions()
    df_c = pd.read_parquet(DATA_DIR / "predictions_cubic_full_detrend.parquet")

    # Clean df_b
    df_b_clean = df_b[df_b["spec"] == "Integrated"].drop_duplicates(subset=["model", "H", "year", "county_fips"]).copy()

    horizons = [
        (5, r"Panel A: Orizzonte Vegetativo Precoce ($H=5$, 30 Giugno)"),
        (3, r"Panel B: Orizzonte Riempimento Baccelli ($H=3$, 31 Agosto)"),
        (1, r"Panel C: Chiusura Raccolto ($H=1^*$, 31 Ottobre / 30 Settembre MN)"),
    ]

    models = ["ElasticNet", "XGBoost", "LSTM", "Ensemble"]

    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\singlespacing",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\caption{Previsione media dell'anomalia per decile ($\bar{\hat{\epsilon}}$, bu/ac) vs anomalia osservata ($\bar{\epsilon}$) nei tre orizzonti chiave ($H = 5, 3, 1^*$). Confronto tra \textbf{Base} (trend expanding lineare) e \textbf{Full} (trend cubico oracolo 1951--2025, $R^2=0.831$). In grassetto la specificazione con previsione pi\`u vicina al rispettivo valore osservato.}",
        r"\label{tab:comp_decile_error_performance}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l cc cc cc cc cc cc @{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Osservato } $\bar{\epsilon}$} & \multicolumn{2}{c}{\textbf{ElasticNet } $\bar{\hat{\epsilon}}$} & \multicolumn{2}{c}{\textbf{XGBoost } $\bar{\hat{\epsilon}}$} & \multicolumn{2}{c}{\textbf{LSTM } $\bar{\hat{\epsilon}}$} & \multicolumn{2}{c}{\textbf{Ensemble } $\bar{\hat{\epsilon}}$} & \multicolumn{2}{c}{\textbf{Media Modelli } $\bar{\hat{\epsilon}}$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
        r"\textbf{Decile} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
        r"\midrule",
    ]

    def fmt_pair(pb, pf, yb, yf):
        eb, ef = abs(pb - yb), abs(pf - yf)
        if eb < ef:
            return f"\\textbf{{{pb:+.2f}}}", f"{pf:+.2f}"
        elif ef < eb:
            return f"{pb:+.2f}", f"\\textbf{{{pf:+.2f}}}"
        else:
            return f"{pb:+.2f}", f"{pf:+.2f}"

    for H, title in horizons:
        tex.append(f"\\multicolumn{{13}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        tex.append(r"\midrule")

        sb = df_b_clean[df_b_clean["H"] == H].copy()
        sc = df_c[df_c["H"] == H].copy()

        # Decile assignments based on actual anomaly
        ref_b = sb[sb["model"] == "ElasticNet"][["county_fips", "year", "y_true_anomaly"]].copy()
        ref_b["decile"] = pd.qcut(ref_b["y_true_anomaly"], 10, labels=range(1, 11))
        sb = pd.merge(sb, ref_b[["county_fips", "year", "decile"]], on=["county_fips", "year"])

        ref_c = sc[sc["model"] == "ElasticNet"][["county_fips", "year", "y_true_anomaly"]].copy()
        ref_c["decile"] = pd.qcut(ref_c["y_true_anomaly"], 10, labels=range(1, 11))
        sc = pd.merge(sc, ref_c[["county_fips", "year", "decile"]], on=["county_fips", "year"])

        rows_h = []
        for d in range(1, 11):
            sub_b = sb[sb["decile"] == d]
            sub_c = sc[sc["decile"] == d]

            yt_b = sub_b["y_true_anomaly"].mean()
            yt_c = sub_c["y_true_anomaly"].mean()

            en_b = sub_b[sub_b["model"] == "ElasticNet"]["y_pred_anomaly"].mean()
            en_c = sub_c[sub_c["model"] == "ElasticNet"]["y_pred_anomaly"].mean()

            xgb_b = sub_b[sub_b["model"] == "XGBoost"]["y_pred_anomaly"].mean()
            xgb_c = sub_c[sub_c["model"] == "XGBoost"]["y_pred_anomaly"].mean()

            ls_b = sub_b[sub_b["model"] == "LSTM"]["y_pred_anomaly"].mean()
            ls_c = sub_c[sub_c["model"] == "LSTM"]["y_pred_anomaly"].mean()

            ens_b = sub_b[sub_b["model"] == "Ensemble"]["y_pred_anomaly"].mean()
            ens_c = sub_c[sub_c["model"] == "Ensemble"]["y_pred_anomaly"].mean()

            ml_b = np.mean([en_b, xgb_b, ls_b, ens_b])
            ml_c = np.mean([en_c, xgb_c, ls_c, ens_c])

            rows_h.append({
                "d": d, "yt_b": yt_b, "yt_c": yt_c,
                "en_b": en_b, "en_c": en_c,
                "xgb_b": xgb_b, "xgb_c": xgb_c,
                "ls_b": ls_b, "ls_c": ls_c,
                "ens_b": ens_b, "ens_c": ens_c,
                "ml_b": ml_b, "ml_c": ml_c,
            })

            d_lbl = f"$D_{{{d}}}$" + (" (Crollo)" if d == 1 else (" (Picco)" if d == 10 else ""))
            en_s_b, en_s_c = fmt_pair(en_b, en_c, yt_b, yt_c)
            xgb_s_b, xgb_s_c = fmt_pair(xgb_b, xgb_c, yt_b, yt_c)
            ls_s_b, ls_s_c = fmt_pair(ls_b, ls_c, yt_b, yt_c)
            ens_s_b, ens_s_c = fmt_pair(ens_b, ens_c, yt_b, yt_c)
            ml_s_b, ml_s_c = fmt_pair(ml_b, ml_c, yt_b, yt_c)

            tex.append(f"{d_lbl} & {yt_b:+.2f} & {yt_c:+.2f} & {en_s_b} & {en_s_c} & {xgb_s_b} & {xgb_s_c} & {ls_s_b} & {ls_s_c} & {ens_s_b} & {ens_s_c} & {ml_s_b} & {ml_s_c} \\\\")

        # Summary for Panel C
        if H == 1:
            tex.append(r"\midrule")
            df_h = pd.DataFrame(rows_h)
            
            # Bias
            b_strs = []
            for m in ["en", "xgb", "ls", "ens", "ml"]:
                bias_b = np.mean(df_h[f"{m}_b"] - df_h["yt_b"])
                bias_c = np.mean(df_h[f"{m}_c"] - df_h["yt_c"])
                if abs(bias_b) < abs(bias_c):
                    b_strs.extend([f"\\textbf{{{bias_b:+.2f}}}", f"{bias_c:+.2f}"])
                elif abs(bias_c) < abs(bias_b):
                    b_strs.extend([f"{bias_b:+.2f}", f"\\textbf{{{bias_c:+.2f}}}"])
                else:
                    b_strs.extend([f"{bias_b:+.2f}", f"{bias_c:+.2f}"])
            tex.append(r"\textbf{Bias Medio ($H=1^*$)} & --- & --- & " + " & ".join(b_strs) + r" \\")

            # MAE
            mae_strs = []
            for m in ["en", "xgb", "ls", "ens", "ml"]:
                mae_b = np.mean(np.abs(df_h[f"{m}_b"] - df_h["yt_b"]))
                mae_c = np.mean(np.abs(df_h[f"{m}_c"] - df_h["yt_c"]))
                if mae_b < mae_c:
                    mae_strs.extend([f"\\textbf{{{mae_b:.2f}}}", f"{mae_c:.2f}"])
                elif mae_c < mae_b:
                    mae_strs.extend([f"{mae_b:.2f}", f"\\textbf{{{mae_c:.2f}}}"])
                else:
                    mae_strs.extend([f"{mae_b:.2f}", f"{mae_c:.2f}"])
            tex.append(r"\textbf{MAE ($H=1^*$)} & --- & --- & " + " & ".join(mae_strs) + r" \\")

            # RMSE
            rmse_strs = []
            for m in ["en", "xgb", "ls", "ens", "ml"]:
                rmse_b = np.sqrt(np.mean((df_h[f"{m}_b"] - df_h["yt_b"])**2))
                rmse_c = np.sqrt(np.mean((df_h[f"{m}_c"] - df_h["yt_c"])**2))
                if rmse_b < rmse_c:
                    rmse_strs.extend([f"\\textbf{{{rmse_b:.2f}}}", f"{rmse_c:.2f}"])
                elif rmse_c < rmse_b:
                    rmse_strs.extend([f"{rmse_b:.2f}", f"\\textbf{{{rmse_c:.2f}}}"])
                else:
                    rmse_strs.extend([f"{rmse_b:.2f}", f"{rmse_c:.2f}"])
            tex.append(r"\textbf{RMSE ($H=1^*$)} & --- & --- & " + " & ".join(rmse_strs) + r" \\")

            # Pearson r
            r_strs = []
            for m in ["en", "xgb", "ls", "ens", "ml"]:
                rb, _ = pearsonr(df_h[f"{m}_b"], df_h["yt_b"])
                rc, _ = pearsonr(df_h[f"{m}_c"], df_h["yt_c"])
                if rb > rc:
                    r_strs.extend([f"\\textbf{{{rb:.3f}}}", f"{rc:.3f}"])
                elif rc > rb:
                    r_strs.extend([f"{rb:.3f}", f"\\textbf{{{rc:.3f}}}"])
                else:
                    r_strs.extend([f"{rb:.3f}", f"{rc:.3f}"])
            tex.append(r"\textbf{Pearson $r$ ($H=1^*$)} & --- & --- & " + " & ".join(r_strs) + r" \\")
        else:
            tex.append(r"\midrule")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])

    out_path = TAB_DIR / "tab_decile_error_performance_comparison.tex"
    out_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {out_path}")


def generate_yearly_raw_yields_table():
    print("Generating tab_yearly_raw_yields.tex (Cubic Full Detrending)...")
    df_b = load_baseline_predictions()
    df_c = pd.read_parquet(DATA_DIR / "predictions_cubic_full_detrend.parquet")

    ydf = pd.read_csv(ROOT / "data" / "target" / "soybean_yield_1951_2025.csv")
    ydf["county_fips"] = ydf["county_fips"].astype(str).str.zfill(5)
    mean_yields = ydf[ydf["year"] >= 2011].groupby("year")["yield_bu_per_acre"].mean().to_dict()

    sb_h1 = df_b[(df_b["H"] == 1) & (df_b["spec"] == "Integrated")].drop_duplicates(subset=["model", "year", "county_fips"])
    sc_h1 = df_c[df_c["H"] == 1].drop_duplicates(subset=["model", "year", "county_fips"])

    # Pre-calculate cubic anomalies to sort by deviation from cubic trend
    year_diffs = {}
    for y, y_obs in mean_yields.items():
        sub_c = sc_h1[sc_h1["year"] == y]
        tau_c = sub_c[sub_c["model"] == "ElasticNet"]["trend"].mean()
        year_diffs[y] = (y_obs - tau_c, tau_c)

    # Sort years by discostamento dal trend cubico (lowest/most severe shortfall to highest)
    sorted_years = [y for y, _ in sorted(year_diffs.items(), key=lambda x: x[1][0])]

    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{Previsioni regionali aggregate di resa ($\bar{\hat{y}}$, bu/ac) a chiusura raccolto ($H = 1^*$) sui 15 anni di test ordinati per discostamento dal trend cubico. Tra parentesi l'anomalia cubica reale ($\bar{\epsilon}_{\text{cub}}$). In grassetto il valore pi\`u vicino al dato reale.}",
        r"\label{tab:yearly_raw_yields}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c cc cc cc cc cc cc @{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Secular } $\bar{\tau}$} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} & \multicolumn{2}{c}{\textbf{Media Modelli}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
        r"\textbf{Anno} & \textbf{Reale ($\bar{\epsilon}_{\text{cub}}$)} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
        r"\midrule",
    ]

    records = []
    for y in sorted_years:
        y_obs = mean_yields[y]
        diff_c, tau_c = year_diffs[y]

        sub_b = sb_h1[sb_h1["year"] == y]
        sub_c = sc_h1[sc_h1["year"] == y]

        # Baseline secular trend
        yt_b = sub_b[sub_b["model"] == "ElasticNet"]["y_true_anomaly"].mean()
        tau_b = y_obs - yt_b

        # Model physical predictions
        en_b = tau_b + sub_b[sub_b["model"] == "ElasticNet"]["y_pred_anomaly"].mean()
        en_c = sub_c[sub_c["model"] == "ElasticNet"]["raw_yield_pred"].mean()

        xgb_b = tau_b + sub_b[sub_b["model"] == "XGBoost"]["y_pred_anomaly"].mean()
        xgb_c = sub_c[sub_c["model"] == "XGBoost"]["raw_yield_pred"].mean()

        ls_b = tau_b + sub_b[sub_b["model"] == "LSTM"]["y_pred_anomaly"].mean()
        ls_c = sub_c[sub_c["model"] == "LSTM"]["raw_yield_pred"].mean()

        ens_b = tau_b + sub_b[sub_b["model"] == "Ensemble"]["y_pred_anomaly"].mean()
        ens_c = sub_c[sub_c["model"] == "Ensemble"]["raw_yield_pred"].mean()

        ml_b = np.mean([en_b, xgb_b, ls_b, ens_b])
        ml_c = np.mean([en_c, xgb_c, ls_c, ens_c])

        records.append({
            "year": y, "y_obs": y_obs, "diff_c": diff_c,
            "tau_b": tau_b, "tau_c": tau_c,
            "en_b": en_b, "en_c": en_c,
            "xgb_b": xgb_b, "xgb_c": xgb_c,
            "ls_b": ls_b, "ls_c": ls_c,
            "ens_b": ens_b, "ens_c": ens_c,
            "ml_b": ml_b, "ml_c": ml_c,
        })

        def pair_fmt(vb, vc):
            eb = abs(vb - y_obs)
            ec = abs(vc - y_obs)
            if eb < ec:
                return f"\\textbf{{{vb:.2f}}}", f"{vc:.2f}"
            elif ec < eb:
                return f"{vb:.2f}", f"\\textbf{{{vc:.2f}}}"
            else:
                return f"{vb:.2f}", f"{vc:.2f}"

        t_b, t_c = pair_fmt(tau_b, tau_c)
        e_b, e_c = pair_fmt(en_b, en_c)
        x_b, x_c = pair_fmt(xgb_b, xgb_c)
        l_b, l_c = pair_fmt(ls_b, ls_c)
        ens_b_s, ens_c_s = pair_fmt(ens_b, ens_c)
        ml_b_s, ml_c_s = pair_fmt(ml_b, ml_c)

        y_label_str = f"{y_obs:.2f} {{\\tiny ({diff_c:+.2f})}}"
        tex.append(f"{y} & {y_label_str} & {t_b} & {t_c} & {e_b} & {e_c} & {x_b} & {x_c} & {l_b} & {l_c} & {ens_b_s} & {ens_c_s} & {ml_b_s} & {ml_c_s} \\\\")

    tex.append(r"\midrule")
    df_rec = pd.DataFrame(records)

    # 1. Bias Medio
    bias_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = np.mean(df_rec[f"{tag}_b"] - df_rec["y_obs"])
        vc = np.mean(df_rec[f"{tag}_c"] - df_rec["y_obs"])
        if abs(vb) < abs(vc):
            bias_strs.extend([f"\\textbf{{{vb:+.2f}}}", f"{vc:+.2f}"])
        elif abs(vc) < abs(vb):
            bias_strs.extend([f"{vb:+.2f}", f"\\textbf{{{vc:+.2f}}}"])
        else:
            bias_strs.extend([f"{vb:+.2f}", f"{vc:+.2f}"])
    tex.append(r"\textbf{Bias Medio} & --- & " + " & ".join(bias_strs) + r" \\")

    # 2. MAE
    mae_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = np.mean(np.abs(df_rec[f"{tag}_b"] - df_rec["y_obs"]))
        vc = np.mean(np.abs(df_rec[f"{tag}_c"] - df_rec["y_obs"]))
        if vb < vc:
            mae_strs.extend([f"\\textbf{{{vb:.2f}}}", f"{vc:.2f}"])
        elif vc < vb:
            mae_strs.extend([f"{vb:.2f}", f"\\textbf{{{vc:.2f}}}"])
        else:
            mae_strs.extend([f"{vb:.2f}", f"{vc:.2f}"])
    tex.append(r"\textbf{MAE} & --- & " + " & ".join(mae_strs) + r" \\")

    # 3. RMSE
    rmse_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        vb = np.sqrt(np.mean((df_rec[f"{tag}_b"] - df_rec["y_obs"])**2))
        vc = np.sqrt(np.mean((df_rec[f"{tag}_c"] - df_rec["y_obs"])**2))
        if vb < vc:
            rmse_strs.extend([f"\\textbf{{{vb:.2f}}}", f"{vc:.2f}"])
        elif vc < vb:
            rmse_strs.extend([f"{vb:.2f}", f"\\textbf{{{vc:.2f}}}"])
        else:
            rmse_strs.extend([f"{vb:.2f}", f"{vc:.2f}"])
    tex.append(r"\textbf{RMSE} & --- & " + " & ".join(rmse_strs) + r" \\")

    # 4. Pearson r
    r_strs = []
    for tag in ["tau", "en", "xgb", "ls", "ens", "ml"]:
        rb, _ = pearsonr(df_rec[f"{tag}_b"], df_rec["y_obs"])
        rc, _ = pearsonr(df_rec[f"{tag}_c"], df_rec["y_obs"])
        if rb > rc:
            r_strs.extend([f"\\textbf{{{rb:.3f}}}", f"{rc:.3f}"])
        elif rc > rb:
            r_strs.extend([f"{rb:.3f}", f"\\textbf{{{rc:.3f}}}"])
        else:
            r_strs.extend([f"{rb:.3f}", f"{rc:.3f}"])
    tex.append(r"\textbf{Pearson } $r$ & --- & " + " & ".join(r_strs) + r" \\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])

    out_path = TAB_DIR / "tab_yearly_raw_yields.tex"
    out_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {out_path}")


def update_figures_cubic():
    print("Updating fig_decile_calibration_and_residuals_comparison.pdf with Cubic Full Detrending...")
    df_b = load_baseline_predictions()
    df_c = pd.read_parquet(DATA_DIR / "predictions_cubic_full_detrend.parquet")

    sb = df_b[(df_b['H'] == 1) & (df_b['spec'] == 'Integrated')].drop_duplicates(subset=["model", "year", "county_fips"]).copy()
    sc = df_c[df_c['H'] == 1].drop_duplicates(subset=["model", "year", "county_fips"]).copy()

    # Deciles
    ref_b = sb[sb['model'] == 'ElasticNet'][['year', 'county_fips', 'y_true_anomaly']].copy()
    ref_b['decile'] = pd.qcut(ref_b['y_true_anomaly'], 10, labels=range(1, 11))
    sb = pd.merge(sb, ref_b[['year', 'county_fips', 'decile']], on=['year', 'county_fips'])

    ref_c = sc[sc['model'] == 'ElasticNet'][['year', 'county_fips', 'y_true_anomaly']].copy()
    ref_c['decile'] = pd.qcut(ref_c['y_true_anomaly'], 10, labels=range(1, 11))
    sc = pd.merge(sc, ref_c[['year', 'county_fips', 'decile']], on=['year', 'county_fips'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.4))

    deciles = np.arange(1, 11)

    models_plot = [
        ("ElasticNet", "#1f77b4", "ElasticNet"),
        ("XGBoost", "#d62728", "XGBoost"),
        ("LSTM", "#8e44ad", "LSTM (+Lag)"),
        ("Ensemble", "#16a085", "Ensemble"),
    ]

    for m_key, col, lbl in models_plot:
        # Base
        obs_b = [sb[(sb['decile'] == d) & (sb['model'] == m_key)]['y_true_anomaly'].mean() for d in deciles]
        pred_b = [sb[(sb['decile'] == d) & (sb['model'] == m_key)]['y_pred_anomaly'].mean() for d in deciles]
        ax1.plot(obs_b, pred_b, linestyle="--", marker="s", markersize=4.0, color=col, alpha=0.75, label=f"{lbl} (Base)")

        # Full Cubic
        obs_c = [sc[(sc['decile'] == d) & (sc['model'] == m_key)]['y_true_anomaly'].mean() for d in deciles]
        pred_c = [sc[(sc['decile'] == d) & (sc['model'] == m_key)]['y_pred_anomaly'].mean() for d in deciles]
        ax1.plot(obs_c, pred_c, linestyle="-", marker="o", markersize=4.5, color=col, label=f"{lbl} (Full Cubico)")

    ax1.plot([-12, 12], [-12, 12], color="black", linestyle=":", linewidth=1.2, label="Calibrazione Perfetta")
    ax1.set_xlabel("Anomalia Osservata nel Decile (bu/ac)")
    ax1.set_ylabel("Anomalia Prevista dal Modello (bu/ac)")
    ax1.set_title(r"(a) Curva di Calibrazione per Decili ($D_1 \dots D_{10}$)")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=7.2, ncol=2)
    ax1.set_xlim(-12, 13)
    ax1.set_ylim(-6.5, 7.5)

    # Panel (b): Decile Error (Pred - Obs) across D1-D10
    for m_key, col, lbl in [("ElasticNet", "#1f77b4", "ElasticNet"), ("LSTM", "#8e44ad", "LSTM (+Lag)"), ("Ensemble", "#16a085", "Ensemble")]:
        errs_b = [sb[(sb['decile'] == d) & (sb['model'] == m_key)]['y_pred_anomaly'].mean() - sb[(sb['decile'] == d) & (sb['model'] == m_key)]['y_true_anomaly'].mean() for d in deciles]
        errs_c = [sc[(sc['decile'] == d) & (sc['model'] == m_key)]['y_pred_anomaly'].mean() - sc[(sc['decile'] == d) & (sc['model'] == m_key)]['y_true_anomaly'].mean() for d in deciles]

        ax2.plot(deciles, errs_b, linestyle="--", marker="s", markersize=4.0, color=col, alpha=0.75, label=f"{lbl} (Base)")
        ax2.plot(deciles, errs_c, linestyle="-", marker="o", markersize=4.5, color=col, label=f"{lbl} (Full Cubico)")

    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.set_xlabel("Decile di Resa ($D_1 = $ Crollo Massimo, $D_{10} = $ Picco Massimo)")
    ax2.set_ylabel(r"Errore Medio $\bar{\hat{\epsilon}} - \bar{\epsilon}$ (bu/ac)")
    ax2.set_title(r"(b) Errore di Previsione Medio con Segno ($\bar{e}$)")
    ax2.set_xticks(deciles)
    ax2.set_xticklabels([f"$D_{{{d}}}$" for d in deciles])
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=7.5, ncol=2)
    ax2.set_ylim(-10.5, 9.5)

    plt.tight_layout()
    pdf_out = FIG_DIR / "fig_decile_calibration_and_residuals_comparison.pdf"
    png_out = FIG_DIR / "fig_decile_calibration_and_residuals_comparison.png"
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {pdf_out} and {png_out}")


if __name__ == "__main__":
    generate_decile_table_3_horizons()
    generate_yearly_raw_yields_table()
    update_figures_cubic()
