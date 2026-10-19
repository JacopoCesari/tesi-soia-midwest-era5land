"""Calendar boundaries for future tabular feature regeneration."""
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

spec = importlib.util.spec_from_file_location("monthly", Path(__file__).resolve().parents[2] / "scripts/03_build_feature_matrices.py")
monthly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monthly)


class MonthlyCalendar(unittest.TestCase):
    def test_baseline_has_no_future_weather_and_active_horizons_retain_every_day(self):
        target = pd.DataFrame({"county_fips": ["17001"], "year": [2020]})
        coords = pd.DataFrame({"county_fips": ["17001"], "lat_norm": [.5], "lon_norm": [.5]})
        saved = {}

        def daily(path):
            year = int(path.stem.rsplit("_", 1)[1])
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
            frame = pd.DataFrame({name: 1.0 for name in [*monthly.FLUX_MAP, *monthly.STATE_MAP]}, index=range(len(dates)))
            frame["date"], frame["county_fips"] = dates, "17001"
            return frame

        def capture(frame, path, **kwargs):
            saved[path.name] = frame.copy()

        with patch.object(monthly.pd, "read_csv", return_value=target.copy()), \
             patch.object(monthly, "compute_normalized_centroids", return_value=coords), \
             patch.object(monthly.pd, "read_parquet", side_effect=daily), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "mkdir"), \
             patch.object(pd.DataFrame, "to_parquet", capture), redirect_stdout(StringIO()):
            monthly.main()
        self.assertEqual(list(saved["features_H12.parquet"].columns), ["county_fips", "crop_year", "lat_norm", "lon_norm"])
        first = saved["features_H11.parquet"]
        self.assertEqual(first.P_m01.iloc[0], 30)
        self.assertEqual(first.P_m02.iloc[0], 31)
        full = saved["features_H01.parquet"]
        self.assertEqual(full[[f"P_m{m:02d}" for m in range(1, 13)]].sum(axis=1).iloc[0], 366)


if __name__ == "__main__":
    unittest.main()
