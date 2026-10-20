"""
train_cubic_full_detrend.py

Fits county-specific Cubic Polynomial Detrending on the entire 1951-2025 sample:
    y_{c,t} = alpha_c + beta_{1,c} * t + beta_{2,c} * t^2 + beta_{3,c} * t^3

Then trains tabular models (ElasticNet, RandomForest, XGBoost) and LSTM (Integrated +Lag)
across expanding test years (2011-2025) at key horizons (H=1, 3, 5).
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = ROOT / "work" / "experiments" / "full_sample_detrending"
FEAT_DIR = ROOT / "output" / "data" / "processed" / "model_datasets"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
HPARAMS_IN_JSON = ROOT / "output" / "data" / "processed" / "hyperparameters_optimal_integrated.json"
SEQ_FILE = ROOT / "output" / "data" / "processed" / "lstm_sequences" / "seq_10d_full_v3.npz"
OUT_PARQUET = EXP_DIR / "data" / "predictions_cubic_full_detrend.parquet"

TEST_START = 2011
TEST_END = 2025
EVAL_HORIZONS = [5, 3, 1]

CAMPAIGN_MONTH_DAYS = {1: 30, 2: 31, 3: 31, 4: 28, 5: 31, 6: 30, 7: 31, 8: 30, 9: 31, 10: 31, 11: 30, 12: 31}


def get_seq_length(H):
    if H == 12:
        return 0
    elapsed_months = 13 - H
    return sum(CAMPAIGN_MONTH_DAYS[m] // 10 for m in range(1, elapsed_months + 1))


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, attn_size=32):
        super().__init__()
        self.w_a = nn.Linear(hidden_size, attn_size)
        self.v_a = nn.Linear(attn_size, 1, bias=False)

    def forward(self, hidden_states):
        u = torch.tanh(self.w_a(hidden_states))
        scores = self.v_a(u).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=19, hidden_size=128, num_layers=2, dropout=0.4, include_lag=True):
        super().__init__()
        self.include_lag = include_lag
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = TemporalAttention(hidden_size)
        in_head = hidden_size + (1 if include_lag else 0)
        self.head = nn.Linear(in_head, 1)

    def forward(self, x, lag=None):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        context, _ = self.attention(out)
        features = torch.cat([context, lag], dim=-1) if self.include_lag and lag is not None else context
        return self.head(features).squeeze(-1)


def compute_cubic_full_trends_and_anomalies(yield_df):
    df = yield_df.copy()
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    
    trends = {}
    for fips, grp in df.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = (grp["year"].values - 1951).astype(float)
        X = np.column_stack([np.ones_like(t_vals), t_vals, t_vals**2, t_vals**3])
        beta = np.linalg.lstsq(X, y_vals, rcond=None)[0]
        trends[str(fips)] = beta

    trend_list = []
    for _, r in df.iterrows():
        t = float(r["year"] - 1951)
        beta = trends[str(r["county_fips"])]
        trend_val = beta[0] + beta[1] * t + beta[2] * (t**2) + beta[3] * (t**3)
        trend_list.append(trend_val)
        
    df["trend_cubic"] = trend_list
    df["anomaly_cubic"] = df["yield_bu_per_acre"] - df["trend_cubic"]
    
    # Lagged anomaly
    lookup = df.set_index(["county_fips", "year"])["anomaly_cubic"].to_dict()
    df["anomaly_lag1"] = [lookup.get((str(r["county_fips"]), int(r["year"]) - 1), 0.0) for _, r in df.iterrows()]
    
    return df, lookup


def train_tabular_models(yield_df):
    print("Training Tabular Models on Cubic Detrending...")
    records = []
    
    for H in EVAL_HORIZONS:
        feat_path = FEAT_DIR / f"features_H{H:02d}.parquet"
        feat_df = pd.read_parquet(feat_path)
        feat_df["county_fips"] = feat_df["county_fips"].astype(str).str.zfill(5)
        feat_df = feat_df.rename(columns={"crop_year": "year"})
        
        panel = yield_df.merge(feat_df, on=["county_fips", "year"]).sort_values(["year", "county_fips"]).reset_index(drop=True)
        
        exclude_cols = ["year", "county_fips", "state_alpha", "yield_bu_per_acre", "trend_cubic", "anomaly_cubic", "anomaly_lag1"]
        feat_cols = [c for c in panel.columns if c not in exclude_cols] + ["anomaly_lag1"]
        
        for y_test in range(TEST_START, TEST_END + 1):
            train = panel[panel["year"] < y_test]
            test = panel[panel["year"] == y_test]
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train[feat_cols])
            X_te = scaler.transform(test[feat_cols])
            y_tr = train["anomaly_cubic"].values
            
            # 1. Naive (predict 0 anomaly)
            for _, r in test.iterrows():
                records.append({
                    "model": "Naive", "spec": "Naive", "H": H, "year": y_test,
                    "county_fips": r["county_fips"],
                    "y_true_anomaly": r["anomaly_cubic"], "y_pred_anomaly": 0.0,
                    "trend": r["trend_cubic"], "raw_yield_true": r["yield_bu_per_acre"],
                    "raw_yield_pred": r["trend_cubic"]
                })
                
            # 2. ElasticNet
            en = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=2000, random_state=42)
            en.fit(X_tr, y_tr)
            p_en = en.predict(X_te)
            for idx, (_, r) in enumerate(test.iterrows()):
                records.append({
                    "model": "ElasticNet", "spec": "Integrated", "H": H, "year": y_test,
                    "county_fips": r["county_fips"],
                    "y_true_anomaly": r["anomaly_cubic"], "y_pred_anomaly": float(p_en[idx]),
                    "trend": r["trend_cubic"], "raw_yield_true": r["yield_bu_per_acre"],
                    "raw_yield_pred": float(r["trend_cubic"] + p_en[idx])
                })
                
            # 3. RandomForest
            rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", min_samples_leaf=5, max_depth=14, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            p_rf = rf.predict(X_te)
            for idx, (_, r) in enumerate(test.iterrows()):
                records.append({
                    "model": "RandomForest", "spec": "Integrated", "H": H, "year": y_test,
                    "county_fips": r["county_fips"],
                    "y_true_anomaly": r["anomaly_cubic"], "y_pred_anomaly": float(p_rf[idx]),
                    "trend": r["trend_cubic"], "raw_yield_true": r["yield_bu_per_acre"],
                    "raw_yield_pred": float(r["trend_cubic"] + p_rf[idx])
                })
                
            # 4. XGBoost
            xgb = XGBRegressor(n_estimators=150, learning_rate=0.06, max_depth=5, colsample_bytree=0.7, subsample=0.8, reg_lambda=1.0, reg_alpha=0.0, min_child_weight=3, random_state=42, verbosity=0, n_jobs=-1)
            xgb.fit(X_tr, y_tr)
            p_xgb = xgb.predict(X_te)
            for idx, (_, r) in enumerate(test.iterrows()):
                records.append({
                    "model": "XGBoost", "spec": "Integrated", "H": H, "year": y_test,
                    "county_fips": r["county_fips"],
                    "y_true_anomaly": r["anomaly_cubic"], "y_pred_anomaly": float(p_xgb[idx]),
                    "trend": r["trend_cubic"], "raw_yield_true": r["yield_bu_per_acre"],
                    "raw_yield_pred": float(r["trend_cubic"] + p_xgb[idx])
                })
                
    return pd.DataFrame(records)


def train_lstm_models(yield_df, lag_lookup):
    print("Training LSTM Models on Cubic Detrending...")
    seq_data = np.load(SEQ_FILE, allow_pickle=True)
    X_full = seq_data["X_seq"].astype(np.float32)
    seq_counties = [str(c).zfill(5) for c in seq_data["counties"]]
    seq_years = seq_data["years"].astype(int)

    key_to_idx = {}
    for i in range(len(seq_years)):
        key_to_idx[(str(seq_counties[i]), int(seq_years[i]))] = i

    records = []
    
    for H in EVAL_HORIZONS:
        seq_len = get_seq_length(H)
        X_H = X_full[:, :seq_len, :]

        torch.manual_seed(42)
        torch.set_num_threads(8)
        model = LSTMForecaster(input_size=19, hidden_size=128, include_lag=True)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        crit = nn.MSELoss()

        for y_idx, y_test in enumerate(range(TEST_START, TEST_END + 1)):
            train_sub = yield_df[yield_df["year"] < y_test]
            test_sub = yield_df[yield_df["year"] == y_test]

            tr_idx, tr_y, tr_l = [], [], []
            for _, r in train_sub.iterrows():
                f, yr = r["county_fips"], r["year"]
                if (f, yr) in key_to_idx:
                    tr_idx.append(key_to_idx[(f, yr)])
                    tr_y.append(r["anomaly_cubic"])
                    tr_l.append(r["anomaly_lag1"])

            te_idx, te_y, te_l, te_f, te_tr = [], [], [], [], []
            for _, r in test_sub.iterrows():
                f, yr = r["county_fips"], r["year"]
                if (f, yr) in key_to_idx:
                    te_idx.append(key_to_idx[(f, yr)])
                    te_y.append(r["anomaly_cubic"])
                    te_l.append(r["anomaly_lag1"])
                    te_f.append(f)
                    te_tr.append(r["trend_cubic"])

            X_tr = torch.tensor(X_H[tr_idx], dtype=torch.float32)
            y_tr = torch.tensor(tr_y, dtype=torch.float32)
            l_tr = torch.tensor(tr_l, dtype=torch.float32).unsqueeze(1)

            X_te = torch.tensor(X_H[te_idx], dtype=torch.float32)
            l_te = torch.tensor(te_l, dtype=torch.float32).unsqueeze(1)

            epochs = 15 if y_idx == 0 else 2
            ds = TensorDataset(X_tr, y_tr, l_tr)
            loader = DataLoader(ds, batch_size=128, shuffle=True)

            model.train()
            for _ in range(epochs):
                for batch in loader:
                    opt.zero_grad()
                    xb, yb, lb = batch
                    pred = model(xb, lb)
                    loss = crit(pred, yb)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                te_pred = model(X_te, l_te).cpu().numpy()

            for i in range(len(te_idx)):
                records.append({
                    "model": "LSTM", "spec": "Integrated", "H": H, "year": y_test,
                    "county_fips": te_f[i],
                    "y_true_anomaly": te_y[i], "y_pred_anomaly": float(te_pred[i]),
                    "trend": te_tr[i], "raw_yield_true": float(te_tr[i] + te_y[i]),
                    "raw_yield_pred": float(te_tr[i] + te_pred[i])
                })

    return pd.DataFrame(records)


def main():
    t_start = time.time()
    print("Computing Cubic Full-Sample Detrending...")
    yield_raw = pd.read_csv(TARGET_CSV)
    yield_df, lag_lookup = compute_cubic_full_trends_and_anomalies(yield_raw)

    df_tab = train_tabular_models(yield_df)
    df_lstm = train_lstm_models(yield_df, lag_lookup)

    df_all = pd.concat([df_tab, df_lstm], ignore_index=True)

    # Build Ensemble (ElasticNet + LSTM average)
    en = df_all[df_all["model"] == "ElasticNet"].copy()
    ls = df_all[df_all["model"] == "LSTM"].copy()
    mrg = en.merge(ls, on=["H", "year", "county_fips", "trend", "raw_yield_true", "y_true_anomaly"], suffixes=("_en", "_ls"))
    mrg["model"] = "Ensemble"
    mrg["spec"] = "Integrated"
    mrg["y_pred_anomaly"] = (mrg["y_pred_anomaly_en"] + mrg["y_pred_anomaly_ls"]) / 2.0
    mrg["raw_yield_pred"] = (mrg["raw_yield_pred_en"] + mrg["raw_yield_pred_ls"]) / 2.0

    ens_df = mrg[["model", "spec", "H", "year", "county_fips", "y_true_anomaly", "y_pred_anomaly", "trend", "raw_yield_true", "raw_yield_pred"]].copy()

    final_df = pd.concat([df_all, ens_df], ignore_index=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved all Cubic Detrending predictions to {OUT_PARQUET} ({len(final_df)} rows) in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
