# Master's Thesis Replication Scripts: Agrometeorological Soybean Yield Forecasting (1951–2025)

This directory contains the standalone Python replication suite supporting the econometric and machine learning empirical analyses in the master's thesis.

The scripts are organized sequentially (`01` through `09`) to allow full replication of the panel assembly, feature engineering, statistical detrending models, predictive pipelines, and thesis exhibits.

---

## 1. Execution Pipeline and Script Catalog

All scripts are self-contained and resolve paths relative to the repository root. Commands can be run directly with standard Python:

```bash
# Example: execute any script from the repository root
python thesis_scripts/01_build_balanced_panel.py
```

| Script | Purpose / Methodology | Primary Artifacts Generated |
| :--- | :--- | :--- |
| `01_build_balanced_panel.py` | Ingests raw USDA NASS survey yields and acreage (1950–2025); selects unbroken 135-county balanced panel (10,125 observations) without imputation. | `data/target/soybean_yield_1951_2025.csv`<br>`data/auxiliary/counties.csv`<br>`data/auxiliary/acres_harvested_1951_2025.csv` |
| `02_compute_spatial_weights.py` | Intersects 0.1-degree ERA5-Land regular grid cells with Census county polygons in EPSG:5070 equal-area projection to compute planar area weights. | `data/auxiliary/spatial_weights.csv` |
| `03_build_feature_matrices.py` | Aggregates daily county agrometeorological series into monthly sums (fluxes) and means (states) across 12 crop campaign horizons ($H=12 \dots 1$). | `output/data/processed/model_datasets/features_H01.parquet` to `features_H12.parquet` |
| `04_compare_detrending.py` | Econometric comparison of candidate detrending models (Linear OLS, Quadratic, Cubic, Natural Spline, HP filter) via in-sample criteria and 45-fold expanding-window OOS RMSE. | `output/thesis/tables/tab_detrending_comparison.tex` (Table 3.4)<br>`output/thesis/figures/fig_detrending_diagnostics.pdf` (Figure 3.4) |
| `05_train_tabular_models.py` | Chronological expanding-window out-of-sample evaluation (2011–2025) for Naive, ElasticNet, Random Forest, and XGBoost with train-only OLS detrending and scaling. | `output/data/processed/predictions.parquet`<br>`output/data/processed/evaluation_summary.csv` |
| `06_train_lstm_model.py` | Deep learning pipeline: trains 2-layer LSTM with Bahdanau temporal attention on 10-day agrometeorological sequences across horizons $H=11 \dots 1$. | `output/data/processed/predictions_lstm.parquet` |
| `07_evaluate_benchmarks.py` | Computes multi-horizon $R^2_{\text{OOS}}$ matrix, Diebold-Mariano inferential significance tests, and extreme weather shock diagnostics (2012 drought vs 2016 bumper). | Replicates **Table 5.1** and **Table 5.3** in thesis text. |
| `08_generate_chapter3_figures.py` | Produces exploratory data figures: study area cartographic map, 75-year secular yield trajectory, and monthly climate-yield sensitivity correlations. | `fig_study_area_map.pdf` (Figure 3.1)<br>`fig_yield_trajectory.pdf` (Figure 3.2)<br>`fig_climate_yield_sensitivity.pdf` (Figure 3.8) |
| `09_generate_chapter5_figures.py` | Produces empirical results figures: multi-horizon $R^2_{\text{OOS}}$ progression curves, extreme shock bias profiles, and county $R^2$ spatial choropleth map. | `fig_r2_horizon_progression.pdf` (Figure 5.1)<br>`fig_shock_bias_profiles.pdf` (Figure 5.2)<br>`fig_county_r2_map.pdf` (Figure 5.5) |

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
[08_generate_chapter3_figures.py] [09_generate_chapter5_figures.py]
            │                           │
            ▼                           ▼
     Chapter 3 Figures           Chapter 5 Figures
```
