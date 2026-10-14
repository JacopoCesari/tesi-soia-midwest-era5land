"""Script per la generazione della Figura 5.5: Grouped PFI Macro-Domini a 3 modelli.

Calcola il Grouped Permutation Feature Importance (GPFI) out-of-sample per i modelli
a Horizon H=1 (ElasticNet, XGBoost e LSTM surrogato) permutando congiuntamente tutte
le variabili appartenenti a ciascuno dei 5 macro-domini fisici agronomici.
Genera tre pannelli orizzontali affiancati speculari alla Figura 5.4.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# 1. Configurazione percorsi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "output/data/processed/model_datasets/features_H01.parquet"
TARGET_PATH = ROOT / "data/target/soybean_yield_1951_2025.csv"
HPARAMS_PATH = ROOT / "output/data/processed/hyperparameters_optimal_integrated.json"
LSTM_PREDS_PATH = ROOT / "work/reports/experiments/lstm/preds_10d_H01_linear_lag.parquet"
OUTPUT_FIG_PATH = ROOT / "output/thesis/figures/fig55_pfi_domain_bars.png"

# ---------------------------------------------------------------------------
# 2. Caricamento dati e detrending per contea
# ---------------------------------------------------------------------------
feat_df = pd.read_parquet(FEATURES_PATH)
target_df = pd.read_csv(TARGET_PATH)

if "crop_year" in feat_df.columns and "year" not in feat_df.columns:
    feat_df = feat_df.rename(columns={"crop_year": "year"})

feat_df["county_fips"] = feat_df["county_fips"].astype(str).str.zfill(5)
target_df["county_fips"] = target_df["county_fips"].astype(str).str.zfill(5)

df = target_df.merge(feat_df, on=["county_fips", "year"])

# Split train (1951-2010) e test (2011-2025)
train_df = df[df["year"] <= 2010].copy()
test_df = df[df["year"] >= 2011].copy()

# Trend lineare per contea stimato solo su train
trend_dict = {}
for fips, grp in train_df.groupby("county_fips"):
    trend_dict[fips] = np.polyfit(grp["year"], grp["yield_bu_per_acre"], 1)

for part in (train_df, test_df):
    part["trend"] = part.apply(
        lambda r: trend_dict[r["county_fips"]][0] * r["year"] + trend_dict[r["county_fips"]][1],
        axis=1
    )
    part["anomaly"] = part["yield_bu_per_acre"] - part["trend"]

# Lag dell'anomalia dell'anno precedente
df_all = df.copy()
df_all["trend"] = df_all.apply(
    lambda r: trend_dict[r["county_fips"]][0] * r["year"] + trend_dict[r["county_fips"]][1],
    axis=1
)
df_all["anomaly"] = df_all["yield_bu_per_acre"] - df_all["trend"]
df_all = df_all.sort_values(["county_fips", "year"])
df_all["lag_anomaly"] = df_all.groupby("county_fips")["anomaly"].shift(1)

train_df["lag_anomaly"] = df_all.loc[train_df.index, "lag_anomaly"].fillna(0.0)
test_df["lag_anomaly"] = df_all.loc[test_df.index, "lag_anomaly"].fillna(0.0)

# Matrici per il fit
feature_cols = [c for c in feat_df.columns if c not in ("county_fips", "year")] + ["lag_anomaly"]
X_train = train_df[feature_cols].values
y_train = train_df["anomaly"].values
X_test = test_df[feature_cols].values
y_test = test_df["anomaly"].values

# Standardizzazione per ElasticNet
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

# SS_tot standard: sum((y - mean(y))^2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

# ---------------------------------------------------------------------------
# 3. Addestramento modelli con iperparametri ottimali (Horizon 1)
# ---------------------------------------------------------------------------
with open(HPARAMS_PATH, "r", encoding="utf-8") as f:
    hparams_h1 = json.load(f)["1"]

enet_params = hparams_h1["ElasticNet"]
xgb_params = hparams_h1["XGBoost"]

# ElasticNet
enet = ElasticNet(**enet_params, max_iter=2000, tol=1e-3, random_state=42)
enet.fit(X_train_sc, y_train)
r2_base_enet = 1.0 - np.sum((y_test - enet.predict(X_test_sc)) ** 2) / ss_tot

# XGBoost
xgb = XGBRegressor(**xgb_params, n_estimators=300, random_state=42)
xgb.fit(X_train, y_train)
r2_base_xgb = 1.0 - np.sum((y_test - xgb.predict(X_test)) ** 2) / ss_tot

# LSTM via modello surrogato (XGBoost addestrato sulle predizioni LSTM)
lstm_df = pd.read_parquet(LSTM_PREDS_PATH)
test_merged = test_df[["county_fips", "year"]].merge(lstm_df, on=["county_fips", "year"], how="left")
y_lstm = test_merged["pred_anomaly"].values

surrogate = XGBRegressor(**xgb_params, n_estimators=300, random_state=42)
surrogate.fit(X_test, y_lstm)
r2_base_lstm = 1.0 - np.sum((y_test - surrogate.predict(X_test)) ** 2) / ss_tot

print(f"R2 Out-of-Sample Base: ENet={r2_base_enet:.4f}, XGB={r2_base_xgb:.4f}, LSTM={r2_base_lstm:.4f}")

# ---------------------------------------------------------------------------
# 4. Definizione dei 5 macro-domini fisici (identici a Figura 5.4)
# ---------------------------------------------------------------------------
canonical_domains = [
    {
        "domain": "Atmospheric Demand & Heat",
        "color": "#e66101",
        "vars": ["VPD", "ET0", "HD30", "HD35", "EDD30"]
    },
    {
        "domain": "Hydrological Supply",
        "color": "#1f78b4",
        "vars": ["P", "P_minus_ET0"]
    },
    {
        "domain": "Soil Moisture Dynamics",
        "color": "#8c510a",
        "vars": ["SM1", "SM2", "SM3", "SM_root"]
    },
    {
        "domain": "Thermal & Radiation Regimes",
        "color": "#e7298a",
        "vars": ["T_mean", "T_max", "T_min", "GDD", "T_dew", "SSRD"]
    },
    {
        "domain": "Memory & Spatial Baseline",
        "color": "#7570b3",
        "vars": ["lag_anomaly", "spatial"]
    }
]

# Sequenza visiva concordata: raggruppa le due classi dell'area termica/atmosferica (Arancione e Rosa)
DISPLAY_DOMAIN_ORDER = [
    "Atmospheric Demand & Heat",
    "Thermal & Radiation Regimes",
    "Hydrological Supply",
    "Soil Moisture Dynamics",
    "Memory & Spatial Baseline"
]

# ---------------------------------------------------------------------------
# 5. Calcolo Grouped Permutation Feature Importance (GPFI per Macro-Dominio)
# ---------------------------------------------------------------------------
np.random.seed(42)
N_REPS = 10

domain_results = []
for d_info in canonical_domains:
    col_idx = []
    for key in d_info["vars"]:
        if key == "lag_anomaly":
            col_idx.append(feature_cols.index("lag_anomaly"))
        elif key == "spatial":
            col_idx.extend([i for i, c in enumerate(feature_cols) if c in ("lat_norm", "lon_norm")])
        elif key == "P":
            col_idx.extend([i for i, c in enumerate(feature_cols) if c.startswith("P_m") and not c.startswith("P_minus")])
        else:
            col_idx.extend([i for i, c in enumerate(feature_cols) if c.startswith(f"{key}_m")])

    drops_enet, drops_xgb, drops_lstm = [], [], []
    for _ in range(N_REPS):
        perm = np.random.permutation(len(X_test))

        # ElasticNet: permutazione congiunta di tutte le feature del dominio
        X_p_sc = X_test_sc.copy()
        X_p_sc[:, col_idx] = X_p_sc[perm][:, col_idx]
        r2_p_enet = 1.0 - np.sum((y_test - enet.predict(X_p_sc)) ** 2) / ss_tot
        drops_enet.append(max(0.0, r2_base_enet - r2_p_enet))

        # XGBoost
        X_p = X_test.copy()
        X_p[:, col_idx] = X_p[perm][:, col_idx]
        r2_p_xgb = 1.0 - np.sum((y_test - xgb.predict(X_p)) ** 2) / ss_tot
        drops_xgb.append(max(0.0, r2_base_xgb - r2_p_xgb))

        # LSTM (surrogato)
        r2_p_lstm = 1.0 - np.sum((y_test - surrogate.predict(X_p)) ** 2) / ss_tot
        drops_lstm.append(max(0.0, r2_base_lstm - r2_p_lstm))

    mean_enet = float(np.mean(drops_enet))
    mean_xgb = float(np.mean(drops_xgb))
    mean_lstm = float(np.mean(drops_lstm))

    domain_results.append({
        "domain": d_info["domain"],
        "color": d_info["color"],
        "gpfi_enet": mean_enet,
        "gpfi_xgb": mean_xgb,
        "gpfi_lstm": mean_lstm,
    })
    print(f"  {d_info['domain']:30s} | ENet: {mean_enet:.3f} | XGB: {mean_xgb:.3f} | LSTM: {mean_lstm:.3f}")

df_domains = pd.DataFrame(domain_results)

# Riordino per sequenza espositiva di macro-area
df_domains["domain_cat"] = pd.Categorical(df_domains["domain"], categories=DISPLAY_DOMAIN_ORDER, ordered=True)
df_domains = df_domains.sort_values("domain_cat", kind="stable").reset_index(drop=True).drop(columns=["domain_cat"])

# ---------------------------------------------------------------------------
# 6. Plot Horizontal 100% Stacked Bars (Armonizzato con Figura 5.4)
# ---------------------------------------------------------------------------
models = ["(a) ElasticNet (+Lag)", "(b) XGBoost (+Lag)", "(c) LSTM (Surrogate)"]
model_cols = ["gpfi_enet", "gpfi_xgb", "gpfi_lstm"]

# Calcolo quote percentuali relative (100% Stacked)
matrix_vals = np.array([df_domains[c].values for c in model_cols])  # shape: (3, 5)
totals = matrix_vals.sum(axis=1, keepdims=True)
shares_matrix = (matrix_vals / totals) * 100.0  # shape: (3, 5)

print("\n=== GPFI 5 MACRO DOMAINS H=1 ===")
for i, d_name in enumerate(df_domains["domain"]):
    enet_share, xgb_share, lstm_share = shares_matrix[0, i], shares_matrix[1, i], shares_matrix[2, i]
    enet_d, xgb_d, lstm_d = matrix_vals[0, i], matrix_vals[1, i], matrix_vals[2, i]
    print(f"{d_name:<28} | ENet={enet_share:5.1f}% (dR2={enet_d:.3f}) | XGB={xgb_share:5.1f}% (dR2={xgb_d:.3f}) | LSTM={lstm_share:5.1f}% (dR2={lstm_d:.3f})")
print("================================\n")

fig, ax = plt.subplots(figsize=(11.5, 4.8))
y_pos = np.arange(len(models))[::-1]
height = 0.48
lefts = np.zeros(len(models))

domain_names = df_domains["domain"].tolist()
domain_colors = df_domains["color"].tolist()

for i in range(len(domain_names)):
    s = shares_matrix[:, i]
    bars = ax.barh(
        y_pos,
        s,
        left=lefts,
        color=domain_colors[i],
        alpha=0.92,
        height=height,
        edgecolor="white",
        linewidth=1.5
    )
    for j, b in enumerate(bars):
        pct = s[j]
        if pct >= 4.5:
            ax.text(
                lefts[j] + pct / 2,
                b.get_y() + b.get_height() / 2,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=10.0,
                color="white",
                fontweight="bold"
            )
    lefts += s

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=11.5, fontweight="bold")
ax.set_xlabel("Macro-Domain Relative Information Share (%)", fontsize=11.0)
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

# Griglia verticale leggera solo su asse X
ax.grid(axis="x", color="#e8e8e8", linestyle=":", linewidth=0.7)
ax.yaxis.grid(False)

# Rimuovi bordi superiore e destro
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["bottom"].set_color("#888888")

# Legenda minimale orizzontale a 5 voci in cima, identica a Figura 5.4
legend_handles = [
    Patch(facecolor=c, edgecolor="none", label=d)
    for d, c in zip(domain_names, domain_colors)
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.93),
    ncol=5,
    frameon=False,
    fontsize=9.2
)

plt.suptitle(
    "Figure 5.5: Macro-Domain Information Share across Architectures (100% Stacked, H=2)",
    fontsize=12.0,
    fontweight="bold",
    y=0.985
)
plt.tight_layout(rect=[0, 0, 1, 0.88])

# Salvataggio figure
OUTPUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FIG_PATH, dpi=300, bbox_inches="tight")
pdf_path = OUTPUT_FIG_PATH.with_suffix(".pdf")
plt.savefig(pdf_path, bbox_inches="tight")

# Salvataggio compatibilità con nome precedente
legacy_pdf = OUTPUT_FIG_PATH.parent / "fig_shap_horizon_stack.pdf"
legacy_png = OUTPUT_FIG_PATH.parent / "fig_shap_horizon_stack.png"
plt.savefig(legacy_pdf, bbox_inches="tight")
plt.savefig(legacy_png, dpi=300, bbox_inches="tight")

plt.close(fig)

print(f"Figure salvate con successo in:\n  - {OUTPUT_FIG_PATH}\n  - {pdf_path}\n  - {legacy_pdf}")
