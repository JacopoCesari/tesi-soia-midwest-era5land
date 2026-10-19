"""Predictor-group boundaries for the regenerated attribution exhibits."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("figures", Path(__file__).resolve().parents[2] / "scripts/09_generate_results_figures.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


class AttributionGroups(unittest.TestCase):
    def test_precipitation_does_not_include_climatic_balance(self):
        columns = ["P_m01", "P_minus_ET0_m01", "P_m12", "P_minus_ET0_m12"]
        self.assertEqual(figures.attribution_columns(columns, ["P"]), [0, 2])
        self.assertEqual(figures.attribution_columns(columns, ["P_minus_ET0"]), [1, 3])

    def test_coordinates_form_one_joint_group(self):
        columns = ["lat_norm", "lon_norm", "lag_anomaly", "SM_root_m01"]
        self.assertEqual(figures.attribution_columns(columns, ["spatial"]), [0, 1])
        self.assertEqual(figures.attribution_columns(columns, ["lag_anomaly"]), [2])

    def test_absent_groups_stop_the_calculation(self):
        with self.assertRaisesRegex(ValueError, "No attribution columns"):
            figures.attribution_columns(["HD30_m01"], ["HD35"])


if __name__ == "__main__":
    unittest.main()
