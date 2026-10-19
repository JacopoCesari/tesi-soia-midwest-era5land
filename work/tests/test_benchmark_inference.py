"""Offline checks for annual paired forecast inference."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmarks", ROOT / "scripts/07_evaluate_benchmarks.py")
benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmarks)


class InferenceTests(unittest.TestCase):
    def test_dm_matches_intercept_regression_with_newey_west_covariance(self):
        loss_difference = np.array([4, 3, 5, 7, 8, 6, 9, 8, 6, 4, 3, 2, 5, 7, 9], dtype=float)
        fitted = sm.OLS(loss_difference, np.ones((15, 1))).fit()
        expected = loss_difference.mean() / np.sqrt(cov_hac(fitted, nlags=None)[0, 0])
        statistic, p_value = benchmarks.diebold_mariano_test(loss_difference + 10, np.full(15, 10.0))
        self.assertAlmostEqual(statistic, expected, places=12)
        self.assertAlmostEqual(p_value, stats.norm.sf(expected), places=12)

    def test_alternative_is_superior_second_model(self):
        difference = np.arange(1.0, 16.0)
        statistic, p_value = benchmarks.diebold_mariano_test(np.full(15, 30.0), 30 - difference)
        self.assertGreater(statistic, 0)
        self.assertLess(p_value, 0.5)
        reverse_stat, reverse_p = benchmarks.diebold_mariano_test(30 - difference, np.full(15, 30.0))
        self.assertAlmostEqual(reverse_stat, -statistic)
        self.assertAlmostEqual(reverse_p, 1 - p_value)

    def test_non_finite_losses_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            benchmarks.diebold_mariano_test([1, 2, np.nan], [1, 2, 3])

    def test_different_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same"):
            benchmarks.diebold_mariano_test([1, 2], [1, 2, 3])

    def test_identical_losses_do_not_receive_a_fabricated_p_value(self):
        with self.assertRaisesRegex(ValueError, "variance"):
            benchmarks.diebold_mariano_test(np.ones(15), np.ones(15))

    def records(self):
        rows = []
        for year in range(2011, 2026):
            for county in ["17001", "27001"]:
                for model, prediction in [("Naive", 0.0), ("Candidate", 0.5)]:
                    rows.append({"year": year, "county_fips": county, "H": 1,
                                 "model": model, "y_true_anomaly": float(year - 2010),
                                 "y_pred_anomaly": prediction})
        return pd.DataFrame(rows)

    def test_counties_are_averaged_before_inference(self):
        result = benchmarks.annual_loss_differentials(self.records(), "Candidate", 1)
        self.assertEqual(len(result), 15)
        np.testing.assert_allclose(result.to_numpy(), np.arange(1, 16) - 0.25)

    def test_unpaired_county_is_rejected(self):
        records = self.records().drop(index=1)
        with self.assertRaisesRegex(ValueError, "paired"):
            benchmarks.annual_loss_differentials(records, "Candidate", 1)

    def test_different_targets_are_rejected(self):
        records = self.records()
        records.loc[records.model.eq("Candidate"), "y_true_anomaly"] += 1
        with self.assertRaisesRegex(ValueError, "target"):
            benchmarks.annual_loss_differentials(records, "Candidate", 1)

    def test_duplicate_predictions_are_rejected(self):
        records = self.records()
        records = pd.concat([records, records.iloc[[1]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            benchmarks.annual_loss_differentials(records, "Candidate", 1)

    def test_missing_prediction_is_not_averaged_away(self):
        records = self.records()
        records.loc[1, "y_pred_anomaly"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            benchmarks.annual_loss_differentials(records, "Candidate", 1)

    def test_minnesota_uses_its_paired_september_prediction(self):
        october = self.records()
        september = october.copy()
        september["H"] = 2
        september["y_pred_anomaly"] = 0.25
        result = benchmarks.apply_minnesota_phenological_rule(pd.concat([october, september]))
        selected = result[result.H.eq(1) & result.model.eq("Candidate")]
        self.assertTrue(selected[selected.county_fips.eq("27001")].y_pred_anomaly.eq(0.25).all())
        self.assertTrue(selected[selected.county_fips.eq("17001")].y_pred_anomaly.eq(0.5).all())

    def test_missing_september_forecast_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "paired H=2"):
            benchmarks.apply_minnesota_phenological_rule(self.records())


if __name__ == "__main__":
    unittest.main()
