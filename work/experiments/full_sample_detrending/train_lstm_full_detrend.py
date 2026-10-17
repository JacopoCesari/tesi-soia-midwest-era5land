"""
train_lstm_full_detrend.py

Trains sequential LSTM models (Weather-Only and Integrated +Lag)
under Full-Sample (In-Sample 1951-2025) OLS Detrending across horizons H=11..1
and 15 expanding test years (2011-2025).
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SEQ_FILE = ROOT / "output" / "data" / "processed" / "lstm_sequences" / "seq_10d_full_v3.npz"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
OUT_PARQUET = ROOT / "work" / "experiments" / "full_sample_detrending" / "data" / "predictions_lstm_full_detrend.parquet"

INPUT_SIZE = 19
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.4
BATCH_SIZE = 128
LR = 3e-4
TEST_START = 2011
TEST_END = 2025

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
    def __init__(self, input_size=19, hidden_size=128, num_layers=2, dropout=0.4, include_lag=False):
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


def compute_full_sample_trends_and_anomalies(yield_df):
    df = yield_df.copy()
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    
    trends = {}
    for fips, grp in df.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        b = np.sum((t_vals - t_vals.mean()) * (y_vals - y_vals.mean())) / np.sum((t_vals - t_vals.mean()) ** 2)
        a = y_vals.mean() - b * t_vals.mean()
        trends[str(fips)] = (a, b)
        
    df["trend_full"] = [trends[str(r["county_fips"])][0] + trends[str(r["county_fips"])][1] * r["year"] for _, r in df.iterrows()]
    df["anomaly_full"] = df["yield_bu_per_acre"] - df["trend_full"]
    
    lookup = df.set_index(["county_fips", "year"])["anomaly_full"].to_dict()
    df["anomaly_lag1"] = [lookup.get((str(r["county_fips"]), int(r["year"]) - 1), 0.0) for _, r in df.iterrows()]
    
    return df, lookup


def train_lstm_horizon(H, X_full, yield_df, lag_lookup, key_to_idx):
    t0 = time.time()
    seq_len = get_seq_length(H)
    X_H = X_full[:, :seq_len, :]

    records = []

    for include_lag in [False, True]:
        torch.manual_seed(42)
        torch.set_num_threads(8)
        model = LSTMForecaster(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, include_lag=include_lag)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        crit = nn.MSELoss()

        spec_name = "Integrated" if include_lag else "Weather-Only"

        for y_idx, y_test in enumerate(range(TEST_START, TEST_END + 1)):
            train_sub = yield_df[yield_df["year"] < y_test]
            test_sub = yield_df[yield_df["year"] == y_test]

            tr_idx, tr_y, tr_l = [], [], []
            for _, r in train_sub.iterrows():
                f, yr = r["county_fips"], r["year"]
                if (f, yr) in key_to_idx:
                    tr_idx.append(key_to_idx[(f, yr)])
                    tr_y.append(r["anomaly_full"])
                    tr_l.append(r["anomaly_lag1"])

            te_idx, te_y, te_l, te_f, te_tr = [], [], [], [], []
            for _, r in test_sub.iterrows():
                f, yr = r["county_fips"], r["year"]
                if (f, yr) in key_to_idx:
                    te_idx.append(key_to_idx[(f, yr)])
                    te_y.append(r["anomaly_full"])
                    te_l.append(r["anomaly_lag1"])
                    te_f.append(f)
                    te_tr.append(r["trend_full"])

            X_tr = torch.tensor(X_H[tr_idx], dtype=torch.float32)
            y_tr = torch.tensor(tr_y, dtype=torch.float32)
            l_tr = torch.tensor(tr_l, dtype=torch.float32).unsqueeze(1)

            X_te = torch.tensor(X_H[te_idx], dtype=torch.float32)
            l_te = torch.tensor(te_l, dtype=torch.float32).unsqueeze(1)

            epochs = 15 if y_idx == 0 else 2
            ds = TensorDataset(X_tr, y_tr, l_tr) if include_lag else TensorDataset(X_tr, y_tr)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

            model.train()
            for _ in range(epochs):
                for batch in loader:
                    opt.zero_grad()
                    if include_lag:
                        xb, yb, lb = batch
                        pred = model(xb, lb)
                    else:
                        xb, yb = batch
                        pred = model(xb)
                    loss = crit(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            model.eval()
            with torch.no_grad():
                pred_te = model(X_te, l_te if include_lag else None).numpy()

            for f, yt, tr, yp in zip(te_f, te_y, te_tr, pred_te):
                records.append({
                    "model": "LSTM",
                    "spec": spec_name,
                    "H": H,
                    "year": y_test,
                    "county_fips": f,
                    "y_true_anomaly": float(yt),
                    "y_pred_anomaly": float(yp),
                    "trend": float(tr),
                })

    elapsed = time.time() - t0
    sub_df = pd.DataFrame(records)
    r2_in = 1.0 - np.sum((sub_df[sub_df["spec"] == "Integrated"]["y_true_anomaly"] - sub_df[sub_df["spec"] == "Integrated"]["y_pred_anomaly"]) ** 2) / np.sum(sub_df[sub_df["spec"] == "Integrated"]["y_true_anomaly"] ** 2)
    print(f"LSTM Horizon H={H:02d} completed in {elapsed:.1f}s (Integrated R2: {r2_in:+.3f}).")
    return records


def run_all():
    print("Loading sequence data and target yields...")
    seq_data = np.load(SEQ_FILE, allow_pickle=True)
    X_full = seq_data["X_seq"].astype(np.float32)
    seq_counties = [str(c).zfill(5) for c in seq_data["counties"]]
    seq_years = seq_data["years"].astype(int)
    key_to_idx = {(c, y): i for i, (c, y) in enumerate(zip(seq_counties, seq_years))}

    yield_df_raw = pd.read_csv(TARGET_CSV)
    yield_df, lag_lookup = compute_full_sample_trends_and_anomalies(yield_df_raw)

    all_records = []
    horizons = list(range(11, 0, -1))

    t_start = time.time()
    for H in horizons:
        recs = train_lstm_horizon(H, X_full, yield_df, lag_lookup, key_to_idx)
        all_records.extend(recs)

    df_out = pd.DataFrame(all_records)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nAll LSTM predictions saved to {OUT_PARQUET} ({len(df_out)} rows in {time.time() - t_start:.1f}s).")


if __name__ == "__main__":
    run_all()
