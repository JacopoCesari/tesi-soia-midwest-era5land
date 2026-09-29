# Research Scripts Catalog and Replication Guide

This directory contains the complete suite of Python research scripts supporting the master's thesis. All scripts are standalone, deterministic, and organized by functional domain to allow direct replication of data pipelines, econometric estimations, and all thesis tables and figures.

Commands should be executed from the `output/` project root directory:
```powershell
python scripts/<script_name>.py
```

---

## 1. Master Script Navigation Matrix

| Script Name | Thesis Chapter / Section | Primary Artifacts Generated | Description / Methodological Role |
| :--- | :--- | :--- | :--- |
| [`prepare_thesis_data.py`](file:///scripts/prepare_thesis_data.py) | **Chap 3 (Sec 3.1)** | `data/target/soybean_yield_1951_2025.csv`<br>`data/auxiliary/counties.csv`<br>`data/auxiliary/acres_harvested_1951_2025.csv` | Ingests USDA NASS SURVEY exports and builds the unbroken 135-county balanced panel (1951--2025). |
| [`compute_spatial_weights.py`](file:///scripts/compute_spatial_weights.py) | **Chap 3 (Sec 3.3)** | `data/auxiliary/spatial_weights.csv` | Computes exact planar polygon intersection weights between ERA5-Land grid cells and county geometries in EPSG:5070. |
| [`download_era5_land.py`](file:///scripts/download_era5_land.py) | **Chap 3 (Sec 3.3)** | Retrospective weather store | Acquires hourly ECMWF ERA5-Land reanalysis via Analysis-Ready Cloud-Optimized (ARCO) Zarr HTTP slicing. |
| [`build_county_daily_weather.py`](file:///scripts/build_county_daily_weather.py) | **Chap 3 (Sec 3.3)** | Daily agrometeorological series | Aggregates hourly weather to 17 daily indicators (FAO-56 $ET_0$, $VPD$, $GDD$, $HD_{30}$, moisture). |
| [`validate_downloaded_data.py`](file:///scripts/validate_downloaded_data.py) | **Chap 3 (Sec 3.3)** | QC validation reports | Audits daily weather series for temporal completeness, missingness, and strict physical range bounds. |
| [`analyze_price_yield_shocks.py`](file:///scripts/analyze_price_yield_shocks.py) | **Chap 1 (Sec 1.1)**<br>**Chap 3 (Sec 3.2)** | Econometric elasticity estimates & benchmark price response | Estimates price elasticity to yield shocks (1960--2025) and quantifies market impacts for 1988, 2012, 1994. |
| [`generate_figures_yield_eda.py`](file:///scripts/generate_figures_yield_eda.py) | **Chap 3 (Sec 3.1 & 3.2)** | **Tables 3.1, 3.2, 3.3**<br>**Figures 3.1, 3.2, 3.3** | Produces panel selection criteria, spatial study area map, secular yield trend, decadal statistics, and historical shock table. |
| [`compare_detrending_methods.py`](file:///scripts/compare_detrending_methods.py) | **Chap 3 (Sec 3.2)** | **Table 3.4**<br>`fig_detrending_diagnostics.pdf` | Econometric comparison of linear OLS, quadratic, cubic, natural spline, and Hodrick-Prescott filters across 45 expanding folds. |
| [`generate_figure_grid_structure.py`](file:///scripts/generate_figure_grid_structure.py) | **Chap 3 (Sec 3.3)** | **Figure 3.4** | Visualizes ERA5-Land regular grid mesh ($0.1^\circ \times 0.1^\circ$) intersecting irregular county boundaries (Tazewell County, IL). |
| [`generate_figures_weather_trajectories.py`](file:///scripts/generate_figures_weather_trajectories.py) | **Chap 3 (Sec 3.3 & 3.4)** | **Table 3.5**<br>**Figures 3.5, 3.6, 3.8** | Compiles agrometeorological variable definitions, cumulative phenological trajectories, dynamic state profiles, and monthly climate-yield sensitivity. |
| [`generate_figure_spatial_shocks.py`](file:///scripts/generate_figure_spatial_shocks.py) | **Chap 3 (Sec 3.4)** | **Figure 3.7** | Generates 4-panel spatial interpolation maps contrasting the 2012 flash drought against the 2016 bumper harvest. |

---

## 2. Functional Script Categories

### A. Data Ingestion, Geospatial Processing & Quality Control
- **`prepare_thesis_data.py`**: Reads raw USDA NASS Excel exports, extracts candidate counties, and constructs the primary balanced 75-year panel ($10{,}125$ county-year records) without data imputation.
- **`compute_spatial_weights.py`**: Intersects the regular $0.1^\circ$ ERA5-Land grid with U.S. Census Bureau county shapefiles using equal-area planar projection (EPSG:5070 Conus Albers).
- **`download_era5_land.py`**: Directly queries ECMWF ARCO Zarr data stores on Google Cloud / AWS, fetching 37 raw atmospheric, soil, and radiative variables.
- **`build_county_daily_weather.py`**: Transforms hourly reanalysis into 17 standardized daily agronomic indicators, including vapor pressure deficit ($VPD$), reference evapotranspiration ($ET_0$), and heat degree days ($HD_{30}$).
- **`validate_downloaded_data.py`**: Applies strict physical bounds (e.g., non-negative precipitation, temperature boundaries, solar radiation) to ensure zero corruption.

### B. Economic Motivation & Price Response Analysis
- **`analyze_price_yield_shocks.py`**: 
  - Fits OLS regressions between detrended county yield anomalies ($\epsilon_t$) and national farm-gate prices (USDA NASS / farmdoc, University of Illinois).
  - Quantifies that a $10\%$ yield shortfall increases annual farm prices by an average of $7.5\%$ ($p = 0.014$) and triggers intra-seasonal summer price rallies of $+8.5\%$ ($p = 0.0002$).
  - Reproduces benchmark price responses: the 1988 drought ($+26.2\%$ annual price, $+60\%$ CBOT prompt futures), 2012 flash drought ($+15.2\%$ annual price, all-time record $\$17.89\text{/bu}$ CBOT), and 1994 bumper crop ($-18.4\%$ post-harvest cash collapse).

### C. Exploratory Data Analysis & Thesis Artifact Generation (Chapter 3)
- **`generate_figures_yield_eda.py`**:
  - `tab_panel_selection.tex` (Table 3.1): Multi-stage attrition and panel filtering criteria.
  - `fig_study_area_map.pdf` (Figure 3.1): Cartographic representation of the 135 study counties.
  - `fig_yield_trajectory.pdf` (Figure 3.2): 75-year regional mean yield progression, linear secular trend ($R^2 = 0.81$), and anomaly bars.
  - `tab_decadal_yields.tex` (Table 3.2): Decadal descriptive statistics and productivity tripling metrics.
  - `fig_state_yield_distributions.pdf` (Figure 3.3): State-level violin distributions and regional trend growth rates.
  - `tab_yield_shocks.tex` (Table 3.3): Catalog of historical macro-climatic shock campaigns ($|\epsilon_t / \hat{y}_t| \ge 10\%$).
- **`compare_detrending_methods.py`**:
  - `tab_detrending_comparison.tex` (Table 3.4): Comparative goodness-of-fit (AIC, BIC, RSE) and 45-fold expanding out-of-sample RMSE (1981--2025).
- **`generate_figure_grid_structure.py`**:
  - `fig_era5_spatial_structure.pdf` (Figure 3.4): Micro-scale illustration of area-weighted grid discretization along the Illinois River.
- **`generate_figures_weather_trajectories.py`**:
  - `tab_weather_variables.tex` (Table 3.5): Comprehensive technical dictionary of all 17 daily surface and derived agrometeorological indicators.
  - `fig_intra_seasonal_extremes.pdf` (Figure 3.5): Cumulative growing-season thermal and hydrological budgets for shortfall vs. bumper campaigns.
  - `fig_weather_shocks_footprint.pdf` (Figure 3.6): 15-day rolling dynamic state trajectories ($VPD$, root-zone soil moisture, $T_{\text{max}}$).
  - `fig_climate_yield_sensitivity.pdf` (Figure 3.8): Pearson correlation profiles between monthly weather anomalies and final yield.
- **`generate_figure_spatial_shocks.py`**:
  - `fig_spatial_shock_comparison.pdf` (Figure 3.7): Spatial contrast of midsummer drought (2012) versus agro-climatic optimum (2016).

---

## 3. Environment & Dependencies

To execute these scripts, ensure the research environment is active (`Python 3.12`):
```powershell
# From the thesis repository root
.venv\Scripts\activate

# Move into output/ and execute any target script
cd output
python scripts/analyze_price_yield_shocks.py
```
All scripts rely strictly on standard scientific packages declared in `pyproject.toml` (`pandas`, `numpy`, `scipy`, `statsmodels`, `geopandas`, `matplotlib`, `shapely`).
