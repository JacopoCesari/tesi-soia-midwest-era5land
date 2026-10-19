"""Presentation regressions: preserve refreshed evidence in the established layout."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExhibitFormattingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = load_script("07_evaluate_benchmarks")
        cls.figures = load_script("09_generate_results_figures")
        cls.predictions = cls.figures.load_data()

    def test_progression_uses_original_panels_with_current_scores(self):
        captured = []
        with patch("matplotlib.figure.Figure.savefig", lambda fig, *args, **kwargs: captured.append(fig)):
            self.figures.plot_figure_5_1(self.predictions)
        fig = captured[0]
        np.testing.assert_allclose(fig.get_size_inches(), [15, 5.4])
        self.assertEqual(len(fig.axes), 2)
        self.assertIn("RMSE", fig.axes[1].get_ylabel())
        for model in ["LSTM (+Lag)", "Ensemble (ENet + LSTM)", "ElasticNet (+Lag)"]:
            line = next(line for line in fig.axes[0].lines if line.get_label() == model)
            error_line = next(line for line in fig.axes[1].lines if line.get_label() == model)
            expected_r2, expected_rmse = [], []
            for horizon in self.figures.HORIZONS:
                frame = self.predictions[self.predictions.model.eq(model) & self.predictions.H.eq(horizon)]
                error = frame.y_pred_anomaly-frame.y_true_anomaly
                expected_r2.append(1-(error**2).sum()/(frame.y_true_anomaly**2).sum())
                expected_rmse.append(np.sqrt((error**2).mean()))
            np.testing.assert_allclose(line.get_ydata(), expected_r2)
            np.testing.assert_allclose(error_line.get_ydata(), expected_rmse)
        plt.close("all")

    def test_tables_keep_current_values_in_original_panels_and_year_order(self):
        diagnostics = ROOT / "output/data/processed/diagnostics"
        metrics = pd.read_csv(diagnostics / "forecast_metrics.csv")
        inference = pd.read_csv(diagnostics / "forecast_inference.csv")
        regional = pd.read_csv(diagnostics / "regional_performance.csv")
        with tempfile.TemporaryDirectory(dir=ROOT / "work/scratch") as directory:
            with patch.object(self.evaluation, "TABLE_DIR", Path(directory)):
                self.evaluation.write_evidence_tables(self.predictions, metrics, inference, regional)
            self.assertFalse((Path(directory) / "tab_forecast_inference.tex").exists())
            deciles = (Path(directory) / "tab_decile_error_performance.tex").read_text()
            annual = (Path(directory) / "tab_yearly_macro_yields.tex").read_text()
            self.assertEqual(deciles.count(r"\multicolumn{8}"), 3)
            self.assertIn("Range (bu/ac)", deciles)
            self.assertIn("-2.65", deciles)
            self.assertIn("2.185", annual)
            self.assertLess(annual.index("2012 &"), annual.index("2011 &"))
            horizons = (Path(directory) / "tab_horizon_performance.tex").read_text()
            shocks = (Path(directory) / "tab_tail_risk_performance.tex").read_text()
            self.assertIn(r"\multicolumn{4}", horizons)
            self.assertIn(r"\multicolumn{5}", horizons)
            self.assertEqual(shocks.count(r"\multicolumn{7}"), 2)
            for path in Path(directory).glob("*.tex"):
                self.assertNotIn(r"\color{red}", path.read_text())

    def test_thesis_sources_have_no_revision_red(self):
        for path in (ROOT / "output/thesis").rglob("*.tex"):
            self.assertNotIn(r"\color{red}", path.read_text(encoding="utf-8"), str(path))
            self.assertNotIn("tab:forecast_inference", path.read_text(encoding="utf-8"), str(path))
            self.assertNotIn("sec:res_significance", path.read_text(encoding="utf-8"), str(path))


if __name__ == "__main__":
    unittest.main()
