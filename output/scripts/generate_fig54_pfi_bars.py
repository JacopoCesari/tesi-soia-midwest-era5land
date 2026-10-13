"""Script per la generazione della Figura 5.4: Barplot a tre modelli (PFI).

Calcola la Grouped Permutation Feature Importance (PFI) out-of-sample per i modelli
a Horizon H=2 (XGBoost, ElasticNet e LSTM surrogato) e genera tre grafici a barre
orizzontali affiancati, raggruppati e colorati per macro-dominio fisico agronomico.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# 1. Configurazione percorsi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "output/data/processed/model_datasets/features_H02.parquet"
TARGET_PATH = ROOT / "data/target/soybean_yield_1951_2025.csv"
HPARAMS_PATH = ROOT / "output/data/processed/hyperparameters_optimal_integrated.json"
LSTM_PREDS_PATH = ROOT / "work/reports/experiments/lstm/preds_10d_H02_linear_lag.parquet"
OUTPUT_FIG_PATH = ROOT / "output/thesis/figures/fig54_pfi_faceted_bars.png"

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
# 3. Addestramento modelli con iperparametri ottimali (Horizon 2)
# ---------------------------------------------------------------------------
with open(HPARAMS_PATH, "r", encoding="utf-8") as f:
    hparams_h2 = json.load(f)["2"]

enet_params = hparams_h2["ElasticNet"]
xgb_params = hparams_h2["XGBoost"]

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
# 4. Definizione variabili e macro-domini fisici
# ---------------------------------------------------------------------------
domains = [
    {
        "domain": "Atmospheric Demand & Heat",
        "color": "#e66101",  # Arancio
        "vars": [
            ("VPD", "Vapor Pressure Deficit (VPD)"),
            ("ET0", "Reference Evapotranspiration (ET0)"),
            ("HD30", "Moderate Heat Days (≥30°C)"),
            ("HD35", "Extreme Heat Days (≥35°C)"),
            ("EDD30", "Extreme Degree Days (>30°C)"),
        ]
    },
    {
        "domain": "Hydrological Supply",
        "color": "#1f78b4",  # Blu
        "vars": [
            ("P", "Cumulative Precipitation (P)"),
            ("P_minus_ET0", "Climatic Water Balance (P - ET0)"),
        ]
    },
    {
        "domain": "Soil Moisture Dynamics",
        "color": "#8c510a",  # Marrone
        "vars": [
            ("SM1", "Topsoil Moisture (0-7 cm)"),
            ("SM2", "Mid Subsoil Moisture (7-28 cm)"),
            ("SM3", "Deep Subsoil Moisture (28-100 cm)"),
            ("SM_root", "Root-Zone Soil Moisture"),
        ]
    },
    {
        "domain": "Thermal & Radiation Regimes",
        "color": "#e7298a",  # Magenta
        "vars": [
            ("T_mean", "Mean Temperature (T_mean)"),
            ("T_max", "Max Temperature (T_max)"),
            ("T_min", "Min Temperature (T_min)"),
            ("GDD", "Growing Degree Days (GDD)"),
            ("T_dew", "Dewpoint Temperature (T_dew)"),
            ("SSRD", "Surface Solar Radiation (SSRD)"),
        ]
    },
    {
        "domain": "Memory & Spatial Baseline",
        "color": "#7570b3",  # Viola
        "vars": [
            ("lag_anomaly", r"Lagged Yield Memory ($\epsilon_{t-1}$)"),
            ("spatial", "Spatial Coordinates (Lat / Lon)"),
        ]
    }
]

# ---------------------------------------------------------------------------
# 5. Calcolo Grouped Permutation Feature Importance (PFI)
# ---------------------------------------------------------------------------
np.random.seed(42)
N_REPS = 5

all_rows = []
for d_info in domains:
    for key, label in d_info["vars"]:
        if key == "lag_anomaly":
            col_idx = [feature_cols.index("lag_anomaly")]
        elif key == "spatial":
            col_idx = [i for i, c in enumerate(feature_cols) if c in ("lat_norm", "lon_norm")]
        elif key == "P":
            col_idx = [i for i, c in enumerate(feature_cols) if c.startswith("P_m") and not c.startswith("P_minus")]
        else:
            col_idx = [i for i, c in enumerate(feature_cols) if c.startswith(f"{key}_m")]

        drops_enet, drops_xgb, drops_lstm = [], [], []
        for _ in range(N_REPS):
            perm = np.random.permutation(len(X_test))

            # ElasticNet
            X_p_sc = X_test_sc.copy()
            for ci in col_idx:
                X_p_sc[:, ci] = X_p_sc[perm, ci]
            r2_p_enet = 1.0 - np.sum((y_test - enet.predict(X_p_sc)) ** 2) / ss_tot
            drops_enet.append(max(0.0, r2_base_enet - r2_p_enet))

            # XGBoost
            X_p = X_test.copy()
            for ci in col_idx:
                X_p[:, ci] = X_p[perm, ci]
            r2_p_xgb = 1.0 - np.sum((y_test - xgb.predict(X_p)) ** 2) / ss_tot
            drops_xgb.append(max(0.0, r2_base_xgb - r2_p_xgb))

            # LSTM (surrogato)
            r2_p_lstm = 1.0 - np.sum((y_test - surrogate.predict(X_p)) ** 2) / ss_tot
            drops_lstm.append(max(0.0, r2_base_lstm - r2_p_lstm))

        all_rows.append({
            "label": label,
            "domain": d_info["domain"],
            "color": d_info["color"],
            "pfi_xgb": np.mean(drops_xgb),
            "pfi_enet": np.mean(drops_enet),
            "pfi_lstm": np.mean(drops_lstm),
        })

results_df = pd.DataFrame(all_rows)

# Calcolo quota percentuale relativa (Relative Information Share %)
for m in ["xgb", "enet", "lstm"]:
    tot = results_df[f"pfi_{m}"].sum()
    results_df[f"pct_{m}"] = (results_df[f"pfi_{m}"] / tot) * 100.0 if tot > 0 else 0.0

# ---------------------------------------------------------------------------
# 6. Plot Faceted Bars a 3 modelli (Semplice, lineare e pulito)
# ---------------------------------------------------------------------------
# 6. Plot Faceted Bars a 3 modelli (Minimalista, pulito ed essenziale)
# ---------------------------------------------------------------------------
df_plot = results_df.iloc[::-1].reset_index(drop=True)
y_pos = np.arange(len(df_plot))
bar_h = 0.65

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 10.0), sharey=True)

# Separatori sottili tra domini fisici
curr_dom = None
for i in range(len(df_plot)):
    d_name = df_plot.iloc[i]["domain"]
    if curr_dom is not None and d_name != curr_dom:
        for ax in (ax1, ax2, ax3):
            ax.axhline(i - 0.5, color="#d0d0d0", linewidth=0.8)
    curr_dom = d_name

# Configurazione pannelli: (modello, asse, titolo) - Progressione dal modello lineare al complesso
panels = [
    ("pct_enet", ax1, "(a) ElasticNet (+Lag)"),
    ("pct_xgb", ax2, "(b) XGBoost (+Lag)"),
    ("pct_lstm", ax3, "(c) LSTM (Surrogate)")
]

for col_name, ax, title in panels:
    bars = ax.barh(
        y_pos,
        df_plot[col_name],
        height=bar_h,
        color=df_plot["color"],
        edgecolor="none",
        alpha=0.90
    )
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Information Share (%)", fontsize=10.0)
    ax.set_xlim(0, 31)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    # Griglia verticale leggera solo sull'asse X
    ax.grid(axis="x", color="#e8e8e8", linestyle=":", linewidth=0.7)
    ax.yaxis.grid(False)

    # Rimuovi bordi superiore e destro
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    # Percentuale pulita a lato della barra
    for b, val in zip(bars, df_plot[col_name]):
        if val >= 0.2:
            ax.text(
                b.get_width() + 0.45,
                b.get_y() + b.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=8.5,
                color="#333333"
            )

# Etichette asse Y sul pannello di sinistra
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_plot["label"], fontsize=9.2)
ax1.tick_params(axis="y", length=0)

# Legenda minimale senza cornice
legend_handles = [
    Patch(facecolor=d["color"], edgecolor="none", label=d["domain"])
    for d in domains
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=5,
    frameon=False,
    fontsize=9.0
)

plt.suptitle("Figure 5.4: Relative Information Share (%) across Models and Physical Domains", fontsize=12.0, fontweight="bold", y=0.985)
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Salvataggio figura (PNG ad alta risoluzione e PDF vettoriale per la tesi)
OUTPUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FIG_PATH, dpi=300, bbox_inches="tight")
pdf_path = OUTPUT_FIG_PATH.with_suffix(".pdf")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Figura salvata con successo in:\n  - {OUTPUT_FIG_PATH}\n  - {pdf_path}")
