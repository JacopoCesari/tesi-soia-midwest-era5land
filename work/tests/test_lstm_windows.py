"""Coverage and cutoff contracts for recurrent weather windows."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

spec = importlib.util.spec_from_file_location("lstm", Path(__file__).resolve().parents[2] / "scripts/06_train_lstm_model.py")
lstm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lstm)


class WindowTests(unittest.TestCase):
    def test_each_date_occurs_once_at_every_horizon_and_frequency(self):
        for year in [2019, 2020]:
            for horizon in range(1, 12):
                for width in [5, 10, 30]:
                    windows = lstm.campaign_windows(year, horizon, width)
                    dates = [date for window in windows for date in window]
                    cutoff = pd.Timestamp(year, 11, 1) - pd.offsets.MonthEnd(horizon)
                    expected = list(pd.date_range(f"{year-1}-11-01", cutoff))
                    self.assertEqual(dates, expected)
                    self.assertTrue(all(len(w) == width for w in windows[:-1]))
                    self.assertTrue(1 <= len(windows[-1]) <= width)

    def test_blocks_cross_month_boundaries(self):
        windows = lstm.campaign_windows(2020, 1, 10)
        self.assertTrue(any(len(set(w.month)) > 1 for w in windows))
        self.assertEqual(sum(map(len, windows)), 366)
        self.assertEqual(len(windows[-1]), 6)

    def test_cutoffs_match_named_months(self):
        self.assertEqual(lstm.campaign_windows(2020, 11, 10)[-1][-1], pd.Timestamp("2019-12-31"))
        self.assertEqual(lstm.campaign_windows(2020, 5, 10)[-1][-1], pd.Timestamp("2020-06-30"))

    def test_padding_is_excluded_from_training_scaling(self):
        values = np.full((2, 3, 19), 2, dtype=np.float32)
        values[0, :2] = 4
        values[0, 2] = 1000
        scaled = lstm.scale_sequences(values, np.array([2, 3]), np.array([True, True]))
        valid = np.array([[True, True, False], [True, True, True]])
        np.testing.assert_allclose(scaled[valid].mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(scaled[valid].std(axis=0), 1, atol=1e-6)
        np.testing.assert_array_equal(scaled[0, 2], 0)

    def test_february_and_month_end_are_retained(self):
        dates = [d for w in lstm.campaign_windows(2020, 1, 30) for d in w]
        self.assertIn(pd.Timestamp("2020-02-29"), dates)
        self.assertIn(pd.Timestamp("2020-01-31"), dates)

    def test_last_window_is_aggregated_only_through_cutoff(self):
        daily = np.ones((1, 61, 17), dtype=np.float32)
        daily[:, 60, :] = 1000
        values = lstm.aggregate_daily_windows(daily[:, :60], 10)
        self.assertEqual(values.shape, (1, 6, 17))
        np.testing.assert_equal(values[:, :, :8], 10)
        np.testing.assert_equal(values[:, :, 8:], 1)

    def test_partial_window_has_actual_sum_and_mean(self):
        daily = np.full((2, 35, 17), 2, dtype=np.float32)
        values = lstm.aggregate_daily_windows(daily, 10)
        np.testing.assert_equal(values[:, -1, :8], 10)
        np.testing.assert_equal(values[:, -1, 8:], 2)

    def test_missing_daily_values_are_rejected(self):
        daily = np.ones((1, 35, 17)); daily[0, 31, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            lstm.aggregate_daily_windows(daily, 10)

    def test_padding_does_not_change_forecast_or_receive_attention(self):
        torch.manual_seed(1)
        model = lstm.RecurrentForecaster(hidden_size=8, num_layers=1, dropout=0, head_type="linear")
        model.eval()
        x = torch.randn(1, 4, 19)
        with torch.no_grad():
            expected, _ = model(x, lengths=torch.tensor([4]))
            padded = torch.cat([x, torch.zeros(1, 3, 19)], dim=1)
            actual, attention = model(padded, lengths=torch.tensor([4]))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(attention[:, 4:], torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
