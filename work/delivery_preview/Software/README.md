# Replication Pipeline: Agrometeorological Soybean Yield Forecasting (1951–2025)

This directory contains the standalone Python replication suite supporting the econometric and machine learning empirical analyses.

The scripts are organized sequentially (`01` through `09`) to allow full replication of the panel assembly, feature engineering, statistical detrending models, predictive pipelines, and exhibits.

---

## 1. Execution Pipeline and Script Catalog

All scripts are self-contained and resolve paths relative to the repository root. Commands can be run directly with standard Python:

```bash
# Example: execute any script from the repository root
python scripts/01_build_balanced_panel.py
```

| Script | Purpose / Methodology | Primary Artifacts Generated |
| :--- | :--- | :--- |
| `01_build_balanced_panel.py` | Ingests raw USDA NASS survey yields and acreage (1950–2025); selects unbroken 135-county balanced panel (10,125 observations) without imputation. | `data/target/soybean_yield_1951_2025.csv`<br>`data/auxiliary/counties.csv`<br>`data/auxiliary/acres_harvested_1951_2025.csv` |
| `02_compute_spatial_weights.py` | Intersects 0.1-degree ERA5-Land regular grid cells with Census county polygons in EPSG:5070 equal-area projection to compute planar area weights. | `data/auxiliary/spatial_weights.csv` |
| `03_build_feature_matrices.py` | Aggregates daily county agrometeorological series into monthly sums (fluxes) and means (states) across 12 crop campaign horizons ($H=12 \dots 1$). | `output/data/processed/model_datasets/features_H01.parquet` to `features_H12.parquet` |
| `04_compare_detrending.py` | Econometric comparison of candidate detrending models (Linear OLS, Quadratic, Cubic) and sequential Chow structural break tests (1951--2025). | `tab_detrending_comparison.tex` (Table 3.4)<br>`fig_detrending_diagnostics.pdf` (Figure 3.4)<br>`fig_chow_structural_break.pdf` (Figure A.1) |
| `05_train_tabular_models.py` | Chronological expanding-window out-of-sample evaluation (2011–2025) for Naive, ElasticNet, Random Forest, and XGBoost with train-only OLS detrending and scaling. | `output/data/processed/predictions.parquet`<br>`output/data/processed/evaluation_summary.csv` |
| `06_train_lstm_model.py` | Deep learning pipeline: trains 2-layer LSTM with Bahdanau temporal attention on 10-day agrometeorological sequences across horizons $H=11 \dots 1$. Supports multi-seed evaluation and deep seed ensembling (`--seeds`). | `output/data/processed/predictions_lstm.parquet`<br>`output/data/processed/diagnostics/lstm_multiseed_evaluation.csv` |
| `07_evaluate_benchmarks.py` | Computes multi-horizon $R^2_{\text{OOS}}$ matrix, Diebold-Mariano inferential significance tests, and extreme weather shock diagnostics (2012 drought vs 2016 bumper). | Regenerates the performance, regional, shock, decile and reconstructed-yield tables. |
| `08_generate_eda_figures.py` | Produces exploratory data figures: study area cartographic map, 75-year secular yield trajectory, and monthly climate-yield sensitivity correlations. | `fig_study_area_map.pdf` (Figure 3.1)<br>`fig_yield_trajectory.pdf` (Figure 3.2)<br>`fig_climate_yield_sensitivity.pdf` (Figure 3.8) |
| `09_generate_results_figures.py` | Produces empirical results figures (Figures 5.1--5.7, including Macro-Regional Error Trajectory) and Appendix A robustness exhibits: decile distributions, calibration/residuals, lead-time comparison, and summary tables. | Figures 5.1--5.7<br>Figures A.2--A.4<br>Tables A.1--A.3 |

---

## 2. Environment and Software Dependencies

The scripts rely strictly on standard scientific and statistical Python packages:

- `python >= 3.10`
- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `xgboost`
- `torch` (PyTorch)
- `geopandas` & `shapely`
- `matplotlib`
- `pyarrow` (for Parquet I/O)

To install all dependencies:
```bash
pip install pandas numpy scipy statsmodels scikit-learn xgboost torch geopandas matplotlib pyarrow
```

---

## 3. Data Flow Architecture

```
data/raw/usda_nass/              data/raw/census_counties/
        │                                  │
        ▼                                  ▼
[01_build_balanced_panel.py]    [02_compute_spatial_weights.py]
        │                                  │
        └─────────────────┬────────────────┘
                          ▼
             data/weather/daily_counties/
                          │
                          ▼
            [03_build_feature_matrices.py]
                          │
                          ▼
               [04_compare_detrending.py] ───► Table 3.4 & Fig 3.4
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
[05_train_tabular_models.py]   [06_train_lstm_model.py]
            │                           │
            └─────────────┬─────────────┘
                          ▼
             [07_evaluate_benchmarks.py] ───► Table 5.1 & Table 5.3
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
[08_generate_eda_figures.py]   [09_generate_results_figures.py]
            │                           │
            ▼                           ▼
       EDA Figures               Results Figures
```

## Verified forecast evidence (2026-10-18)

`07_evaluate_benchmarks.py` reads the explicit weather-only and integrated tabular
exports and the consolidated LSTM predictions. It selects the PReLU weather-only
head and linear integrated head, aligns county-years, and forms the ensemble from
integrated ElasticNet and linear LSTM. H=1 uses September forecasts for Minnesota.
No model is fitted and no original prediction file is modified.

Outputs in `output/data/processed/diagnostics/`:
- `forecast_metrics.csv`: county-level accuracy by model and horizon.
- `forecast_inference.csv`: one-sided annual DM tests against the trend and matched
  weather-only comparators; Bartlett HAC, two annual lags, covariance correction.
  The dedicated Chapter 5 inference subsection and table are deferred. The subsection text is retained in the project procedural notes in `work/thesis_notes/operational_todo.md`.
- `regional_performance.csv`: separate anomaly and reconstructed-level correlations,
  with both calendar-H1 and harvest-closure scopes explicitly identified.
- `forecast_residual_diagnostics.csv`: AR(1) and Moran diagnostics on the same panel.
- `test_refresh_comparison.csv` records changes from the previously published county
  metrics when a refreshed set of test forecasts is evaluated.

The script regenerates the horizon, inference, regional, shock, decile and
reconstructed-yield tables under `output/thesis/tables/`. Chapter 5 includes these
files directly. Run from `output/`:

```powershell
..\.venv\Scripts\python.exe ../scripts/07_evaluate_benchmarks.py
```

The forecast-based functions in `09_generate_results_figures.py` use the same
loader and closure rule. To refresh only the four forecast exhibits, without fitting
feature-attribution models, run from `output/`:

```powershell
..\.venv\Scripts\python.exe -c "import runpy; m=runpy.run_path('../scripts/09_generate_results_figures.py'); p=m['load_data'](); [m[n](p) for n in ['plot_figure_5_1','plot_figure_5_2','plot_figure_5_3','plot_figure_5_6']]"
```

## Complete recurrent windows and frozen-selection test refresh

`06_train_lstm_model.py` constructs horizon-specific consecutive windows directly
from existing daily county Parquet files. It retains leap days, month-end dates and
incomplete last windows, and masks trailing batch padding. No weather download,
monthly matrix rebuild or validation search is part of this command. The existing
head-specific selections in `checkpoint_lstm.json` are read-only. Six independent
CPU workers fit the same 2011--2025 expanding folds, with the existing three-year
inner early-stopping split, fixed seeds, optimizer and architecture. Progress is
saved after each complete county-year fold. Existing monthly tabular forecasts are
unchanged. Run from `output/`:

```powershell
..\.venv\Scripts\python.exe ../scripts/06_train_lstm_model.py
```

New input caches have the `continuous_daily_v1` identifier; legacy monthly-reset caches
are not reused. `work/reports/experiments/lstm/test_continuous_manifest.json` binds
the resumable test predictions to the frozen selection checksum. The published
`predictions_lstm.parquet` is replaced only when all 660 complete test folds exist.
Run script 07 and the forecast-based functions of script 09 after successful export,
then refresh Chapter 5--6 narrative. The retained development validation entry point
also uses the canonical complete-window loader/engine and a distinct checkpoint;
it is not executed by the test-only command.

The attribution functions in script 09 refresh the LSTM surrogate from the consolidated
forecasts. They fit separate fixed-2010 diagnostic tabular models at H=1 and H=2,
using the corresponding cutoff for each state. They translate recurrent forecasts
to that diagnostic trend space, and report surrogate fidelity separately. Attribution
figures describe these diagnostic fits, not additional expanding-window accuracy tests.
Weather groups match exact monthly indicator prefixes, keeping precipitation separate
from climatic water balance. No missing surrogate is replaced by a fabricated score.

After successful publication, script 06 records the consolidated prediction checksum
in the run manifest and removes the duplicate working prediction file. A subsequent
invocation verifies that checksum and skips the completed folds. The validation
checkpoint is never rewritten. Script 07 also refreshes the existing calendar-horizon
LSTM summary CSV/Parquet and matched-head lag comparison, separately from the H*
thesis metrics. Undefined naive directional accuracy is left empty.

For the current manuscript, use the verified forecast, inference, regional,
residual and refreshed attribution reports listed above. Other retained diagnostic
exports belong to earlier analyses and do not define the current thesis results.
The stored Wilcoxon report remains distinct from the annual HAC DM evidence.
