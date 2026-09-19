# Deliberate Chapter Boundaries and Content Mapping
*Permanent reference for thesis development and supervisor delivery*

---

## 1. Chapter 3: Data and Study Area (Current Chapter)
* **Scope:** The empirical inputs and the raw study environment.
* **Included:**
  - Regional agronomic and soil context of the Midwestern Corn and Soybean Belt (Mollisols/Alfisols, phenological calendar from April sowing to October harvest).
  - USDA NASS Survey dataset assembly and variables (yield in bu/acre, harvested acres).
  - The "Why" and "How" of panel selection: prioritizing a 75-year balanced panel (1951–2025, 135 counties, 10,125 observations) without synthetic imputation, accepting a deliberate trade-off between multidecadal temporal depth/extreme year coverage and broader geographic generalization.
  - Handling of the 1950 crop year: exclusion of 1950 yield due to the October 1949 ERA5-Land boundary; preservation of 1950 weather for antecedent lookbacks of the 1951 campaign.
  - Auxiliary data: harvested acreage (documentation only; never a feature, target, or weight) and spatial intersection geometry (3,344 grid-county intersection polygons).
  - ECMWF ERA5-Land reanalysis specifications (0.1° grid, ~9 km, CHESSEL land model, daily aggregations, thermal/moisture/hydrological/soil variables).
  - Exploratory Data Analysis (EDA) of yields: secular trend (+0.484 bu/ac/year, R²=0.81), regional productivity heterogeneity across states, dispersion dynamics (rising absolute standard deviation $\sigma_t$ vs declining relative coefficient of variation $CV_t$), and macro-climatic historical shock years.

---

## 2. Chapter 4: Methodology (Next Chapter)
* **Scope:** Mathematical formulations, preprocessing, experimental design, and model architectures.
* **Deliberately deferred to Chapter 4:**
  - Mathematical formalization of deterministic detrending (linear, quadratic, cubic spline, LOESS) fitted strictly train-only to eliminate look-ahead leakage.
  - Mathematical formulation of categorical yield regimes (e.g., 20th percentile severe shortfall threshold, median, bumper) estimated train-only.
  - Definition of the twelve progressive monthly forecast horizons ($H=12, \dots, 1$) and monthly feature cut-offs anchored to October harvest.
  - Multi-granularity temporal feature aggregation (5-day, 10-day, 30-day summaries) and derived agro-climatic indices ($GDD, EDD, VPD, ET_0$, soil moisture anomalies).
  - Chronological expanding-window validation protocol (Hyndman framework) without random splits.
  - Candidate model taxonomies and algorithms: historical reference baselines (Trend, Climatology, Past-Yield), weather-only ablation benchmarks, and integrated models across regularized linear models (Ridge/Lasso), tree ensembles (Random Forest, XGBoost), and recurrent neural networks (LSTM).
  - Formal evaluation metrics: continuous metrics (RMSE, MAE, out-of-sample $R^2$), classification metrics (ROC-AUC, Brier score), and Diebold-Mariano statistical significance testing.

---

## 3. Chapter 5: Empirical Results
* **Scope:** Out-of-sample performance, comparative trajectories, and empirical discoveries.
* **Deliberately deferred to Chapter 5:**
  - Iterative model progression tracking from simple historical baselines to complex neural networks.
  - Out-of-sample predictive accuracy across the twelve lead times ($H=12 \dots 1$).
  - Lead-time emergence mapping: identifying the exact horizon at which observed weather delivers statistically significant skill beyond historical trend and climatology.
  - Multi-granularity impact: empirical comparison between 5-day, 10-day, and 30-day weather features.
  - Dedicated extreme shock year stress test: out-of-sample performance specifically evaluated on historical macro-anomaly years (1988, 2012, 1974, 2003, 1993, and bumper years 1994, 2016, 2021) to test tail-risk prediction.
  - Feature attribution and interpretability (SHAP values, permutation feature importance) across phenological growth stages.
  - Spatial heterogeneity of predictive performance across counties and states.

---

## 4. Chapter 6: Discussion
* **Scope:** Critical synthesis, managerial value, operational realities, and boundary conditions.
* **Deliberately deferred to Chapter 6:**
  - Critical examination of potential survivorship bias inherent in the 135 balanced surviving counties.
  - Reanalysis latency versus real-time operational deployment (5-day latency of ERA5T vs 2-3 month finalization of ERA5-Land).
  - Practical supply-chain implications for grain merchandisers, crushers, and hedging operations.
  - Limits of weather-only predictability and boundary conditions of the empirical design.
