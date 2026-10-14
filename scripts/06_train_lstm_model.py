"""
06_train_lstm_model.py

Recurrent neural network forecasting pipeline (LSTM with Bahdanau Temporal Attention).

Methodological architecture:
1. Multi-variate 10-day time sequences (19 features: 17 agrometeorological indicators + lat_norm + lon_norm).
2. Backbone: 2-layer LSTM (hidden_size=128, dropout=0.4) + LayerNorm.
3. Attention: Bahdanau temporal attention pooling to extract time-salient weather extremes.
4. Output head: Linear projection head (or PReLU) trained with AdamW on train-only detrended anomalies.
5. Blind expanding-window evaluation across test years (2011-2025).

Outputs:
- output/data/processed/predictions_lstm.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
SEQ_FILE = ROOT / "output" / "data" / "processed" / "lstm_sequences" / "seq_10d_full_v3.npz"
TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
OUT_PREDS_PARQUET = ROOT / "output" / "data" / "processed" / "predictions_lstm.parquet"

# Model hyperparameters
INPUT_SIZE = 19
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.4
BATCH_SIZE = 32
LR = 3e-4
MAX_EPOCHS = 25
TEST_START = 2011
TEST_END = 2025

# Campaign calendar: 10-day windows per campaign month (Nov Y-1 to Oct Y)
CAMPAIGN_MONTH_DAYS = {1: 30, 2: 31, 3: 31, 4: 28, 5: 31, 6: 30, 7: 31, 8: 30, 9: 31, 10: 31, 11: 30, 12: 31}


def get_seq_length(H):
    """Number of elapsed 10-day windows available up to forecast horizon H."""
    if H == 12:
        return 0
    elapsed_months = 13 - H
    return sum(CAMPAIGN_MONTH_DAYS[m] // 10 for m in range(1, elapsed_months + 1))


class TemporalAttention(nn.Module):
    """Bahdanau additive temporal attention over recurrent hidden states."""
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
    """2-layer LSTM with temporal attention and linear projection."""
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


def fit_county_trends(yield_df, train_years):
    """Fit train-only linear trend per county."""
    sub = yield_df[yield_df["year"].isin(train_years)]
    trends = {}
    for fips, grp in sub.groupby("county_fips"):
        y_vals = grp["yield_bu_per_acre"].values
        t_vals = grp["year"].values.astype(float)
        b = np.sum((t_vals - t_vals.mean()) * (y_vals - y_vals.mean())) / np.sum((t_vals - t_vals.mean()) ** 2)
        a = y_vals.mean() - b * t_vals.mean()
        trends[str(fips)] = (a, b)
    return trends


def main():
    print("Loading 10-day agrometeorological sequences and yield data...")
    if not SEQ_FILE.exists():
        raise FileNotFoundError(f"Sequence cache {SEQ_FILE} not found. Run sequence generator first.")

    seq_data = np.load(SEQ_FILE, allow_pickle=True)
    X_full = seq_data["X_seq"].astype(np.float32)
    seq_counties = [str(c).zfill(5) for c in seq_data["counties"]]
    seq_years = seq_data["years"].astype(int)

    yield_df = pd.read_csv(TARGET_CSV)
    yield_df["county_fips"] = yield_df["county_fips"].astype(str).str.zfill(5)

    # Map (county_fips, year) to matrix row index
    key_to_idx = {(c, y): i for i, (c, y) in enumerate(zip(seq_counties, seq_years))}

    results = []

    # Evaluate across all 11 active horizons
    for H in range(11, 0, -1):
        seq_len = get_seq_length(H)
        X_H = X_full[:, :seq_len, :]
        print(f"\n--- Training LSTM for Horizon H={H:02d} (Sequence length: {seq_len} ten-day steps) ---")

        for test_year in range(TEST_START, TEST_END + 1):
            train_years = [y for y in sorted(yield_df["year"].unique()) if y < test_year]

            trends = fit_county_trends(yield_df, train_years)

            # Compute train and test anomalies
            train_sub = yield_df[yield_df["year"].isin(train_years)]
            test_sub = yield_df[yield_df["year"] == test_year]

            # Build train tensors
            tr_indices, tr_anom, tr_lag = [], [], []
            for _, r in train_sub.iterrows():
                fips, yr = str(r["county_fips"]), int(r["year"])
                tr = trends[fips][0] + trends[fips][1] * yr
                anom = r["yield_bu_per_acre"] - tr
                if (fips, yr) in key_to_idx:
                    tr_indices.append(key_to_idx[(fips, yr)])
                    tr_anom.append(anom)
                    # Lagged anomaly
                    lag_anom = 0.0
                    if yr > min(train_years):
                        prior_tr = trends[fips][0] + trends[fips][1] * (yr - 1)
                        prior_row = train_sub[(train_sub["county_fips"] == fips) & (train_sub["year"] == yr - 1)]
                        if not prior_row.empty:
                            lag_anom = prior_row["yield_bu_per_acre"].values[0] - prior_tr
                    tr_lag.append(lag_anom)

            te_indices, te_anom, te_lag, te_fips = [], [], [], []
            for _, r in test_sub.iterrows():
                fips, yr = str(r["county_fips"]), int(r["year"])
                tr = trends[fips][0] + trends[fips][1] * yr
                anom = r["yield_bu_per_acre"] - tr
                if (fips, yr) in key_to_idx:
                    te_indices.append(key_to_idx[(fips, yr)])
                    te_anom.append(anom)
                    te_fips.append(fips)
                    # Lagged anomaly from year Y-1
                    prior_tr = trends[fips][0] + trends[fips][1] * (yr - 1)
                    prior_row = yield_df[(yield_df["county_fips"] == fips) & (yield_df["year"] == yr - 1)]
                    lag_anom = prior_row["yield_bu_per_acre"].values[0] - prior_tr if not prior_row.empty else 0.0
                    te_lag.append(lag_anom)

            X_tr = torch.tensor(X_H[tr_indices], dtype=torch.float32)
            y_tr = torch.tensor(tr_anom, dtype=torch.float32)
            lag_tr = torch.tensor(tr_lag, dtype=torch.float32).unsqueeze(1)

            X_te = torch.tensor(X_H[te_indices], dtype=torch.float32)
            lag_te = torch.tensor(te_lag, dtype=torch.float32).unsqueeze(1)

            # Fit 1: Weather-Only LSTM
            torch.manual_seed(42)
            model_wo = LSTMForecaster(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, include_lag=False)
            opt_wo = torch.optim.AdamW(model_wo.parameters(), lr=LR, weight_decay=1e-4)
            criterion = nn.MSELoss()

            ds_wo = TensorDataset(X_tr, y_tr)
            loader_wo = DataLoader(ds_wo, batch_size=BATCH_SIZE, shuffle=True)

            model_wo.train()
            for _ in range(MAX_EPOCHS):
                for xb, yb in loader_wo:
                    opt_wo.zero_grad()
                    pred = model_wo(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model_wo.parameters(), 1.0)
                    opt_wo.step()

            model_wo.eval()
            with torch.no_grad():
                preds_wo = model_wo(X_te).numpy()

            # Fit 2: LSTM + Lagged Anomaly
            torch.manual_seed(42)
            model_lag = LSTMForecaster(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, include_lag=True)
            opt_lag = torch.optim.AdamW(model_lag.parameters(), lr=LR, weight_decay=1e-4)

            ds_lag = TensorDataset(X_tr, y_tr, lag_tr)
            loader_lag = DataLoader(ds_lag, batch_size=BATCH_SIZE, shuffle=True)

            model_lag.train()
            for _ in range(MAX_EPOCHS):
                for xb, yb, lb in loader_lag:
                    opt_lag.zero_grad()
                    pred = model_lag(xb, lb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model_lag.parameters(), 1.0)
                    opt_lag.step()

            model_lag.eval()
            with torch.no_grad():
                preds_lag = model_lag(X_te, lag_te).numpy()

            for fips, true_val, p_wo, p_lg in zip(te_fips, te_anom, preds_wo, preds_lag):
                results.append({
                    "year": test_year,
                    "county_fips": fips,
                    "actual_anomaly": float(true_val),
                    "pred_anomaly": float(p_wo),
                    "horizon": H,
                    "head_variant": "linear",
                    "cell_type": "lstm",
                    "include_lag": False,
                    "frequency": 10,
                })
                results.append({
                    "year": test_year,
                    "county_fips": fips,
                    "actual_anomaly": float(true_val),
                    "pred_anomaly": float(p_lg),
                    "horizon": H,
                    "head_variant": "linear",
                    "cell_type": "lstm",
                    "include_lag": True,
                    "frequency": 10,
                })

        print(f"  Completed H={H:02d} blind evaluation.")

    OUT_PREDS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_parquet(OUT_PREDS_PARQUET, index=False)
    print(f"\nLSTM predictions saved to {OUT_PREDS_PARQUET} ({len(df_out)} rows).")


if __name__ == "__main__":
    main()
