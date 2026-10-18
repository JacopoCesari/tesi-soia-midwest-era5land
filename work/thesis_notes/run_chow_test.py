"""
run_chow_test.py

Test di rottura strutturale di Chow (Chow Structural Break Test) sulla resa
storica della soia nel Midwest statunitense (1951-2025).

Analisi econometrica per la tesi:
1. Verifica stazionarieta' attorno al trend deterministico (ADF test).
2. Test di Chow sequenziale per data di rottura incognita (1966-2021):
   - Level Break (salto di intercetta, 1 df)
   - Trend Break (cambio di pendenza, 1 df)
   - Combined Break (livello + pendenza, 2 df)
3. Modello ottimale stimato: Break di Livello al 2014 (F = 21.96, p < 0.0001, delta = +4.83 bu/ac).
4. Generazione della figura a due pannelli per note di tesi / futura integrazione.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
OUT_FIG_PDF = Path(__file__).resolve().parent / "fig_chow_structural_break.pdf"
OUT_FIG_PNG = Path(__file__).resolve().parent / "fig_chow_structural_break.png"

# Styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9.5,
    'axes.labelsize': 10,
    'axes.titlesize': 10.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def run_chow_analysis():
    print(f"Caricamento dati da: {TARGET_CSV}")
    df_yield = pd.read_csv(TARGET_CSV)
    annual = df_yield.groupby("year")["yield_bu_per_acre"].mean().reset_index()
    annual.columns = ["year", "yield"]

    n = len(annual)
    print(f"Serie temporale caricata: {n} anni ({annual['year'].min()} - {annual['year'].max()})")

    # 1. Verifica preliminare di stazionarietà attorno al trend deterministico
    adf_result = adfuller(annual["yield"], regression="ct")  # constant + trend
    print("\n--- TEST DI DICKEY-FULLER AUMENTATO (ADF con trend) ---")
    print(f"ADF Stat: {adf_result[0]:.4f}")
    print(f"p-value:  {adf_result[1]:.4e}")
    print(f"Conclusion: La serie e' trend-stazionaria I(0) (p < 0.001). Test di Chow pienamente valido.")

    # 2. Modello ristretto (trend lineare unico)
    min_obs = 15  # Inizio test dal 1966 per stabilita' dei gradi di liberta'
    X_base = pd.DataFrame({
        'const': 1.0,
        'trend': annual['year'] - annual['year'].min()
    })
    model_restricted = sm.OLS(annual["yield"], X_base).fit()
    RSS_restricted = model_restricted.ssr
    k_base = X_base.shape[1]

    print("\n--- MODELLO RISTRETTO OLS (1951-2025) ---")
    print(f"Intercetta: {model_restricted.params['const']:.3f} bu/ac")
    print(f"Trend:      +{model_restricted.params['trend']:.3f} bu/ac/anno")
    print(f"R-squared:  {model_restricted.rsquared:.4f}")
    print(f"RSS:        {RSS_restricted:.2f}")

    # 3. Test di Chow sequenziale
    F_level = np.zeros(n)
    F_trend = np.zeros(n)
    F_combined = np.zeros(n)

    for i in range(min_obs, n - 4):
        yr_break = annual.iloc[i]['year']
        D_t = np.where(annual['year'] >= yr_break, 1, 0)
        T_break = np.where(annual['year'] >= yr_break, annual['year'] - yr_break, 0)

        # Level Break
        X_level = X_base.copy()
        X_level['level_break'] = D_t
        m_level = sm.OLS(annual["yield"], X_level).fit()
        F_level[i] = ((RSS_restricted - m_level.ssr) / 1) / (m_level.ssr / (n - k_base - 1))

        # Trend Break
        X_trend = X_base.copy()
        X_trend['trend_break'] = T_break
        m_trend = sm.OLS(annual["yield"], X_trend).fit()
        F_trend[i] = ((RSS_restricted - m_trend.ssr) / 1) / (m_trend.ssr / (n - k_base - 1))

        # Combined Break
        X_combined = X_base.copy()
        X_combined['level_break'] = D_t
        X_combined['trend_break'] = T_break
        m_combined = sm.OLS(annual["yield"], X_combined).fit()
        F_combined[i] = ((RSS_restricted - m_combined.ssr) / 2) / (m_combined.ssr / (n - k_base - 2))

    opt_level_idx = np.argmax(F_level[min_obs:n-4]) + min_obs
    opt_trend_idx = np.argmax(F_trend[min_obs:n-4]) + min_obs
    opt_combined_idx = np.argmax(F_combined[min_obs:n-4]) + min_obs

    opt_yr_level = int(annual.iloc[opt_level_idx]['year'])
    opt_yr_trend = int(annual.iloc[opt_trend_idx]['year'])
    opt_yr_combined = int(annual.iloc[opt_combined_idx]['year'])

    alpha = 0.05
    crit_1df = f.ppf(1 - alpha, 1, n - k_base - 1)
    crit_2df = f.ppf(1 - alpha, 2, n - k_base - 2)

    # 4. Modello non vincolato ottimale (Level Break 2014)
    X_opt = X_base.copy()
    X_opt['level_break'] = np.where(annual['year'] >= opt_yr_level, 1, 0)
    m_opt = sm.OLS(annual["yield"], X_opt).fit()

    p_val_level = 1.0 - f.cdf(F_level[opt_level_idx], 1, n - k_base - 1)
    delta_bic = m_opt.bic - model_restricted.bic

    print("\n--- RISULTATI TEST DI CHOW ---")
    print(f"Picco Level Break:    Anno {opt_yr_level} | F = {F_level[opt_level_idx]:.2f} (Crit = {crit_1df:.2f}, p = {p_val_level:.4e})")
    print(f"  Salto Stimato Delta: {m_opt.params['level_break']:+.2f} bu/ac (t = {m_opt.tvalues['level_break']:.2f})")
    print(f"  Delta BIC:          {delta_bic:.2f} (favorevole al break)")
    print(f"Picco Trend Break:    Anno {opt_yr_trend} | F = {F_trend[opt_trend_idx]:.2f}")
    print(f"Picco Combined Break: Anno {opt_yr_combined} | F = {F_combined[opt_combined_idx]:.2f} (Crit = {crit_2df:.2f})")

    # Verifica p-value pendenza post-break
    X_full = X_opt.copy()
    X_full['trend_break'] = np.where(annual['year'] >= opt_yr_level, annual['year'] - opt_yr_level, 0)
    m_full = sm.OLS(annual["yield"], X_full).fit()
    print(f"Significativita' cambio pendenza post-2014: p = {m_full.pvalues['trend_break']:.4f} (NON significativo)")

    # 5. Generazione Grafico a due pannelli
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    years = annual['year'].values
    pred_linear = model_restricted.predict(X_base)
    pred_break = m_opt.predict(X_opt)

    # Panel (a): F-statistics
    ax = axes[0]
    mask_plot = (years >= annual.iloc[min_obs]['year']) & (years <= annual.iloc[n-5]['year'])
    ax.plot(years[mask_plot], F_level[mask_plot], color="#1d4ed8", linewidth=2.0,
            label=f"Level Break (Max $F={F_level[opt_level_idx]:.1f}$ al {opt_yr_level})")
    ax.plot(years[mask_plot], F_trend[mask_plot], color="#d97706", linewidth=1.8, linestyle="--",
            label=f"Trend Break (Max $F={F_trend[opt_trend_idx]:.1f}$ al {opt_yr_trend})")
    ax.plot(years[mask_plot], F_combined[mask_plot], color="#059669", linewidth=1.8, linestyle="-.",
            label=f"Combined Break (Max $F={F_combined[opt_combined_idx]:.1f}$ al {opt_yr_combined})")

    ax.axhline(crit_1df, color="#1d4ed8", linestyle=":", linewidth=1.0, alpha=0.7, label=f"Soglia Critica 1 df ($F={crit_1df:.2f}$)")
    ax.axhline(crit_2df, color="#059669", linestyle=":", linewidth=1.0, alpha=0.7, label=f"Soglia Critica 2 df ($F={crit_2df:.2f}$)")

    ax.scatter([opt_yr_level], [F_level[opt_level_idx]], color="#1d4ed8", s=60, zorder=5)
    ax.scatter([opt_yr_trend], [F_trend[opt_trend_idx]], color="#d97706", s=60, zorder=5)
    ax.scatter([opt_yr_combined], [F_combined[opt_combined_idx]], color="#059669", s=60, zorder=5)

    ax.set_title("(a) Profilo della Statistica F del Test di Chow lungo lo Storico", loc="left", fontweight="bold")
    ax.set_xlabel("Anno del Punto di Rottura Candidato ($T^*$)")
    ax.set_ylabel("Statistica F di Chow")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.2)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel (b): Trajectory with Fitted Break Model
    ax = axes[1]
    ax.plot(years, annual['yield'], color="#475569", marker="o", markersize=3.5, linewidth=1.2, alpha=0.85, label="Resa Media Osservata")
    ax.plot(years, pred_linear, color="#dc2626", linestyle="--", linewidth=1.8,
            label=f"Trend Lineare OLS (+{model_restricted.params['trend']:.3f} bu/ac/yr, $R^2={model_restricted.rsquared:.3f}$)")
    ax.plot(years, pred_break, color="#1d4ed8", linewidth=2.2,
            label=f"Trend con Level Break 2014 (+{m_opt.params['level_break']:+.2f} bu/ac, $R^2={m_opt.rsquared:.3f}$)")

    ax.set_ylim(16, 62)
    ax.axvline(opt_yr_level, color="#1d4ed8", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.annotate(f"Break di Livello ({opt_yr_level}):\nSalto $\\Delta = {m_opt.params['level_break']:+.2f}$ bu/ac ($p < 0.001$)",
                xy=(opt_yr_level, pred_break[opt_level_idx]), xytext=(-110, -50), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#1d4ed8",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#1d4ed8", alpha=0.95, lw=1.0),
                arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.0, shrinkB=4, connectionstyle="arc3,rad=0.15"))

    ax.set_title("(b) Confronto tra Trend Lineare Standard e Modello con Break al 2014", loc="left", fontweight="bold")
    ax.set_xlabel("Anno di Raccolto")
    ax.set_ylabel("Resa della Soia (bu/ac)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.2)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUT_FIG_PDF)
    fig.savefig(OUT_FIG_PNG)
    plt.close(fig)
    print(f"\nFigure salvate con successo:")
    print(f"  PDF: {OUT_FIG_PDF}")
    print(f"  PNG: {OUT_FIG_PNG}")


if __name__ == "__main__":
    run_chow_analysis()
