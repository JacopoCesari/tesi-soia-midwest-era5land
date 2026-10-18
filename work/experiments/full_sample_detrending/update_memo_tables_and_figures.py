"""
update_memo_tables_and_figures.py

Updates:
1. tables/tab_historical_vs_test_distribution.tex (Table 1A: Distribution across all 10,125 observations vs 2,025 test observations)
2. tables/tab_decile_distribution.tex (Table 1B: Decile bounds, counts, and negative shares)
3. tables/tab_decile_error_performance_comparison.tex (Table 2: Mean signed error pred - obs with bold for lowest absolute error between Base and Full)
4. tables/tab_yearly_raw_yields.tex (Table 3: Raw physical yields with bold for lowest absolute error between Base and Full for each model)
5. figures/fig_decile_calibration_and_residuals_comparison.pdf & .png:
   - Panel (a): Reliability curves (Observed vs Predicted anomaly) comparing Base vs Full.
   - Panel (b): Decile Prediction Error (Pred - Obs) across D1-D10 comparing Base vs Full.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = ROOT / "work" / "experiments" / "full_sample_detrending"
FIG_DIR = EXP_DIR / "figures"
TAB_DIR = EXP_DIR / "tables"

sys.path.insert(0, str(EXP_DIR))
from generate_all_outputs_and_report import load_baseline_predictions, load_full_detrend_predictions

TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def update_table_1a():
    print("Generating tab_historical_vs_test_distribution.tex...")
    df_y = pd.read_csv(TARGET_CSV)
    df_y["county_fips"] = df_y["county_fips"].astype(str).str.zfill(5)
    
    # Full-sample trend for all 10,125 observations
    full_res = []
    for c, g in df_y.groupby("county_fips"):
        g = g.sort_values("year")
        y = g["yield_bu_per_acre"].values
        t = g["year"].values - 1951
        p = np.polyfit(t, y, 1)
        trend = np.polyval(p, t)
        anom = y - trend
        for yr, y_val, tr, an in zip(g["year"], y, trend, anom):
            full_res.append({
                "county_fips": c, "year": yr, "yield": y_val, "trend_full": tr, "anom_full": an
            })
    df_full_all = pd.DataFrame(full_res)

    # Expanding trend for test set
    df_b = load_baseline_predictions()
    sub_b = df_b[(df_b["H"] == 1) & (df_b["model"] == "ElasticNet") & (df_b["spec"] == "Integrated")]
    sub_b["county_fips"] = sub_b["county_fips"].astype(str).str.zfill(5)
    exp_test_anoms = sub_b.set_index(["county_fips", "year"])["y_true_anomaly"].to_dict()
    df_full_all["anom_base_test"] = df_full_all.set_index(["county_fips", "year"]).index.map(exp_test_anoms)

    rows = []
    
    # 1. Historical 1951-2025 (10,125 obs)
    y_hist = df_full_all["yield"]
    a_full_hist = df_full_all["anom_full"]
    
    # 2. Test period 2011-2025 (2,025 obs)
    test_mask = df_full_all["year"] >= 2011
    y_test = df_full_all[test_mask]["yield"]
    a_full_test = df_full_all[test_mask]["anom_full"]
    a_base_test = df_full_all[test_mask]["anom_base_test"].dropna()

    series_dict = [
        ("Resa Grezza ($y$)", "1951--2025 (Totale)", 10125, y_hist),
        ("Anomalia Full-Sample ($\\epsilon_{\\text{full}}$)", "1951--2025 (Totale)", 10125, a_full_hist),
        ("Resa Grezza ($y$)", "2011--2025 (Test)", 2025, y_test),
        ("Anomalia Baseline Expanding ($\\epsilon_{\\text{base}}$)", "2011--2025 (Test)", 2025, a_base_test),
        ("Anomalia Full-Sample ($\\epsilon_{\\text{full}}$)", "2011--2025 (Test)", 2025, a_full_test),
    ]

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\caption{Distribuzione statistica delle rese fisiche ($y$) e delle anomalie detrendizzate ($\epsilon$) sull'intero storico (1951--2025, $10{,}125$ osservazioni) a confronto con la finestra out-of-sample di test (2011--2025, $2{,}025$ osservazioni). Valori in bu/ac.}",
        r"\label{tab:historical_vs_test_distribution}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l c c c c c c c c c c @{}}",
        r"\toprule",
        r"\textbf{Variabile} & \textbf{Periodo} & \textbf{N} & \textbf{Media} & \textbf{Dev.Std} & \textbf{Min} & \textbf{P10} & \textbf{P50 (Med)} & \textbf{P90} & \textbf{Max} & \textbf{Skewness} \\",
        r"\midrule",
    ]

    for var_lbl, per_lbl, n_obs, s in series_dict:
        m = s.mean()
        sd = s.std()
        mi = s.min()
        p10 = s.quantile(0.10)
        p50 = s.median()
        p90 = s.quantile(0.90)
        ma = s.max()
        sk = s.skew()
        tex.append(f"{var_lbl:<36} & {per_lbl} & {n_obs} & {m:+.2f} & {sd:.2f} & {mi:+.2f} & {p10:+.2f} & {p50:+.2f} & {p90:+.2f} & {ma:+.2f} & {sk:+.2f} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{0.2em}",
        r"{\scriptsize \textit{Fonte: Elaborazione dell'autore su dati USDA NASS (135 contee del Midwest). Si evidenzia la marcata asimmetria a sinistra (skewness negativa) strutturale delle rese agrarie.}}",
        r"\end{table}",
    ])

    out_tex = TAB_DIR / "tab_historical_vs_test_distribution.tex"
    out_tex.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {out_tex}")


def update_table_2_errors():
    print("Generating tab_decile_error_performance_comparison.tex with signed errors and bolding...")
    df_b = load_baseline_predictions()
    df_f = load_full_detrend_predictions()

    # H=1, Integrated
    sb = df_b[(df_b['H'] == 1) & (df_b['spec'] == 'Integrated')].copy()
    sf = df_f[(df_f['H'] == 1) & (df_f['spec'] == 'Integrated')].copy()

    # Deciles based on ElasticNet y_true
    b_h1 = sb[sb['model'] == 'ElasticNet'][['year', 'county_fips', 'y_true_anomaly']].drop_duplicates()
    b_h1['decile'] = pd.qcut(b_h1['y_true_anomaly'], 10, labels=range(1, 11))
    dec_map_b = b_h1.set_index(['year', 'county_fips'])['decile'].to_dict()
    sb['decile'] = sb.set_index(['year', 'county_fips']).index.map(dec_map_b)

    f_h1 = sf[sf['model'] == 'ElasticNet'][['year', 'county_fips', 'y_true_anomaly']].drop_duplicates()
    f_h1['decile'] = pd.qcut(f_h1['y_true_anomaly'], 10, labels=range(1, 11))
    dec_map_f = f_h1.set_index(['year', 'county_fips'])['decile'].to_dict()
    sf['decile'] = sf.set_index(['year', 'county_fips']).index.map(dec_map_f)

    # Add Ensemble
    def add_ens(df):
        p_enet = df[df['model'] == 'ElasticNet'].set_index(['year', 'county_fips', 'decile'])['y_pred_anomaly']
        p_lstm = df[df['model'] == 'LSTM'].set_index(['year', 'county_fips', 'decile'])['y_pred_anomaly']
        p_ens = 0.5 * p_enet + 0.5 * p_lstm
        ens_df = p_ens.reset_index().rename(columns={'y_pred_anomaly': 'y_pred_anomaly'})
        ens_df['model'] = 'Ensemble'
        y_true = df[df['model'] == 'ElasticNet'].set_index(['year', 'county_fips'])['y_true_anomaly'].to_dict()
        ens_df['y_true_anomaly'] = ens_df.set_index(['year', 'county_fips']).index.map(y_true)
        return pd.concat([df, ens_df], ignore_index=True)

    sb = add_ens(sb)
    sf = add_ens(sf)

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\caption{Errore medio di previsione per decile ($\bar{e} = \bar{\hat{\epsilon}} - \bar{\epsilon}$, bu/ac) a chiusura raccolto ($H = 1^*$, 31 Ottobre). Confronto diretto tra \textbf{Baseline (Expanding)} e \textbf{Full-Sample Trend (1951--2025)}. In \textbf{grassetto} il risultato con errore assoluto minore (più vicino all'osservato) per ogni modello.}",
        r"\label{tab:comp_decile_error_performance_short}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l cc cc cc cc cc @{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Observed } $\bar{\epsilon}$} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"\textbf{Decile} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
        r"\midrule",
    ]

    for d in range(1, 11):
        sbd = sb[sb['decile'] == d]
        sfd = sf[sf['decile'] == d]

        ob_b = sbd[sbd['model'] == 'ElasticNet']['y_true_anomaly'].mean()
        ob_f = sfd[sfd['model'] == 'ElasticNet']['y_true_anomaly'].mean()

        def fmt_pair(m_key):
            eb = sbd[sbd['model'] == m_key]['y_pred_anomaly'].mean() - ob_b
            ef = sfd[sfd['model'] == m_key]['y_pred_anomaly'].mean() - ob_f
            if abs(eb) < abs(ef):
                return f"\\textbf{{{eb:+.2f}}}", f"{ef:+.2f}"
            elif abs(ef) < abs(eb):
                return f"{eb:+.2f}", f"\\textbf{{{ef:+.2f}}}"
            else:
                return f"\\textbf{{{eb:+.2f}}}", f"\\textbf{{{ef:+.2f}}}"

        e_b, e_f = fmt_pair('ElasticNet')
        x_b, x_f = fmt_pair('XGBoost')
        l_b, l_f = fmt_pair('LSTM')
        ens_b, ens_f = fmt_pair('Ensemble')

        d_lbl = f"$D_{{{d}}}$" + (" (Shortfall)" if d == 1 else (" (Peak)" if d == 10 else ""))
        tex.append(f"{d_lbl} & {ob_b:+.2f} & {ob_f:+.2f} & {e_b} & {e_f} & {x_b} & {x_f} & {l_b} & {l_f} & {ens_b} & {ens_f} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{0.2em}",
        r"{\scriptsize \textit{Fonte: Elaborazione dell'autore su USDA NASS ed ECMWF ERA5-Land (2011--2025, 2.025 osservazioni). I valori mostrano l'errore contabile con segno ($\bar{\hat{\epsilon}} - \bar{\epsilon}$). Il grassetto indica la specificazione con minore deviazione dall'osservato reale.}}",
        r"\end{table}",
    ])

    out_tex = TAB_DIR / "tab_decile_error_performance_comparison.tex"
    out_tex.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {out_tex}")


def update_table_3_macro():
    print("Generating tab_yearly_raw_yields.tex with bolding on closer physical prediction...")
    df_b = load_baseline_predictions()
    df_f = load_full_detrend_predictions()

    ydf = pd.read_csv(TARGET_CSV)
    ydf["county_fips"] = ydf["county_fips"].astype(str).str.zfill(5)
    mean_yields = ydf[ydf["year"] >= 2011].groupby("year")["yield_bu_per_acre"].mean().to_dict()
    sorted_years = [y for y, _ in sorted(mean_yields.items(), key=lambda x: x[1])]

    sb_h1 = df_b[(df_b["H"] == 1) & (df_b["spec"] == "Integrated")]
    sf_h1 = df_f[(df_f["H"] == 1) & (df_f["spec"] == "Integrated")]

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Previsioni aggregate regionali di resa fisica ($\bar{\hat{y}} = \bar{\tau} + \bar{\hat{\epsilon}}$, bu/ac) a chiusura raccolto $H^*(s)$ sui 15 anni di test ordinati per livello di resa reale. Confronto tra \textbf{Baseline (Expanding)} e \textbf{Full-Sample}. In \textbf{grassetto} la specificazione più vicina al valore osservato per ciascun modello.}",
        r"\label{tab:yearly_raw_yields}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c cc cc cc cc cc @{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Trend Secular } $\bar{\tau}$} & \multicolumn{2}{c}{\textbf{ElasticNet}} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{LSTM (+Lag)}} & \multicolumn{2}{c}{\textbf{Ensemble}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
        r"\textbf{Anno} & \textbf{Reale} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} & \textbf{Base} & \textbf{Full} \\",
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

        ens_b = 0.5 * en_b + 0.5 * ls_b
        ens_f = 0.5 * en_f + 0.5 * ls_f

        def pair_fmt(vb, vf):
            eb = abs(vb - y_obs)
            ef = abs(vf - y_obs)
            if eb < ef:
                return f"\\textbf{{{vb:.2f}}}", f"{vf:.2f}"
            elif ef < eb:
                return f"{vb:.2f}", f"\\textbf{{{vf:.2f}}}"
            else:
                return f"\\textbf{{{vb:.2f}}}", f"\\textbf{{{vf:.2f}}}"

        t_b, t_f = pair_fmt(tau_b, tau_f)
        e_b, e_f = pair_fmt(en_b, en_f)
        x_b, x_f = pair_fmt(xgb_b, xgb_f)
        l_b, l_f = pair_fmt(ls_b, ls_f)
        ens_b_str, ens_f_str = pair_fmt(ens_b, ens_f)

        tex.append(f"{y} & {y_obs:.2f} & {t_b} & {t_f} & {e_b} & {e_f} & {x_b} & {x_f} & {l_b} & {l_f} & {ens_b_str} & {ens_f_str} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\vspace{0.2em}",
        r"{\scriptsize \textit{Fonte: Elaborazione dell'autore su dati USDA NASS ed ECMWF ERA5-Land (pooling su 135 contee per ciascun anno). Il grassetto evidenzia quale dei due approcci di detrending minimizza l'errore assoluto rispetto al dato aggregato reale.}}",
        r"\end{table}",
    ])

    out_tex = TAB_DIR / "tab_yearly_raw_yields.tex"
    out_tex.write_text("\n".join(tex), encoding="utf-8")
    print(f"Saved {out_tex}")


def update_figure_2_comparison():
    print("Generating updated fig_decile_calibration_and_residuals_comparison.pdf & .png...")
    df_b = load_baseline_predictions()
    df_f = load_full_detrend_predictions()

    sb = df_b[(df_b['H'] == 1) & (df_b['spec'] == 'Integrated')].copy()
    sf = df_f[(df_f['H'] == 1) & (df_f['spec'] == 'Integrated')].copy()

    # Deciles
    b_h1 = sb[sb['model'] == 'ElasticNet'][['year', 'county_fips', 'y_true_anomaly']].drop_duplicates()
    b_h1['decile'] = pd.qcut(b_h1['y_true_anomaly'], 10, labels=range(1, 11))
    sb['decile'] = sb.set_index(['year', 'county_fips']).index.map(b_h1.set_index(['year', 'county_fips'])['decile'])

    f_h1 = sf[sf['model'] == 'ElasticNet'][['year', 'county_fips', 'y_true_anomaly']].drop_duplicates()
    f_h1['decile'] = pd.qcut(f_h1['y_true_anomaly'], 10, labels=range(1, 11))
    sf['decile'] = sf.set_index(['year', 'county_fips']).index.map(f_h1.set_index(['year', 'county_fips'])['decile'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.0))

    deciles = np.arange(1, 11)

    # Panel (a): Reliability Diagram (Observed vs Predicted Anomaly)
    models = [
        ("ElasticNet", "#1f77b4", "ElasticNet"),
        ("XGBoost", "#d62728", "XGBoost"),
        ("LSTM", "#8e44ad", "LSTM (+Lag)"),
    ]

    for m_key, col, lbl in models:
        # Base
        obs_b = [sb[(sb['decile'] == d) & (sb['model'] == m_key)]['y_true_anomaly'].mean() for d in deciles]
        pred_b = [sb[(sb['decile'] == d) & (sb['model'] == m_key)]['y_pred_anomaly'].mean() for d in deciles]
        ax1.plot(obs_b, pred_b, linestyle="--", marker="s", markersize=4.5, color=col, alpha=0.75, label=f"{lbl} (Base Expanding)")

        # Full
        obs_f = [sf[(sf['decile'] == d) & (sf['model'] == m_key)]['y_true_anomaly'].mean() for d in deciles]
        pred_f = [sf[(sf['decile'] == d) & (sf['model'] == m_key)]['y_pred_anomaly'].mean() for d in deciles]
        ax1.plot(obs_f, pred_f, linestyle="-", marker="o", markersize=5.0, color=col, label=f"{lbl} (Full-Sample)")

    # 45-degree line (perfect calibration)
    ax1.plot([-12, 12], [-12, 12], color="black", linestyle=":", linewidth=1.2, label="Calibrazione Perfetta (45°)")
    ax1.set_xlabel("Anomalia Osservata nel Decile (bu/ac)")
    ax1.set_ylabel("Anomalia Prevista dal Modello (bu/ac)")
    ax1.set_title(r"(a) Curva di Calibrazione per Decili ($D_1 \dots D_{10}$)")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=8.0)
    ax1.set_xlim(-11, 12)
    ax1.set_ylim(-6, 7)

    # Panel (b): Decile Error (Pred - Obs) across D1-D10
    width = 0.38
    for m_key, col, lbl in [("ElasticNet", "#1f77b4", "ElasticNet"), ("LSTM", "#8e44ad", "LSTM (+Lag)")]:
        errs_b = [sb[(sb['decile'] == d) & (sb['model'] == m_key)]['y_pred_anomaly'].mean() - sb[(sb['decile'] == d) & (sb['model'] == m_key)]['y_true_anomaly'].mean() for d in deciles]
        errs_f = [sf[(sf['decile'] == d) & (sf['model'] == m_key)]['y_pred_anomaly'].mean() - sf[(sf['decile'] == d) & (sf['model'] == m_key)]['y_true_anomaly'].mean() for d in deciles]
        
        offset = -width/2 if m_key == "ElasticNet" else width/2
        ax2.plot(deciles, errs_b, linestyle="--", marker="s", markersize=4.5, color=col, alpha=0.75, label=f"{lbl} Error (Base)")
        ax2.plot(deciles, errs_f, linestyle="-", marker="o", markersize=5.0, color=col, label=f"{lbl} Error (Full)")

    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.set_xlabel("Decile di Resa ($D_1 = $ Crollo Massimo, $D_{10} = $ Picco Massimo)")
    ax2.set_ylabel("Errore Medio di Previsione $\\bar{\\hat{\\epsilon}} - \\bar{\\epsilon}$ (bu/ac)")
    ax2.set_title(r"(b) Errore di Previsione Medio con Segno ($\bar{e}$)")
    ax2.set_xticks(deciles)
    ax2.set_xticklabels([f"$D_{{{d}}}$" for d in deciles])
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=8.5)
    ax2.set_ylim(-9.5, 8.5)

    plt.tight_layout()
    pdf_out = FIG_DIR / "fig_decile_calibration_and_residuals_comparison.pdf"
    png_out = FIG_DIR / "fig_decile_calibration_and_residuals_comparison.png"
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {pdf_out} and {png_out}")


if __name__ == "__main__":
    update_table_1a()
    update_table_2_errors()
    update_table_3_macro()
    update_figure_2_comparison()
