# Model Progression and Experiment Tracking Protocol

**Status:** Active  
**Author Directive Date:** 2026-09-18  
**Scope:** Captures performance and metadata for all experimental model iterations, feature transformations, ablations, and architectural variants.

---

## 1. Rationale and Objective

In scientific reporting, presenting only the final "best" model specification obscures the empirical trajectory of how predictive gains were achieved. To allow the thesis (specifically Chapter 5: Empirical Results and Chapter 6: Discussion) to report the **complete journey of model improvement**, every experimental iteration must be systematically logged and persisted.

This protocol ensures that:
1. No model performance result is lost or overwritten, even if an iteration yields inferior or neutral skill.
2. The thesis can document the incremental contribution of each design element (e.g., adding VPD, shifting from 30-day to 10-day aggregations, moving from linear Ridge to XGBoost or LSTM).
3. Ablation studies can be reconstructed with exact quantitative backing (e.g., impact of dropping $Y_{t-1}$, changing EDD thresholds, varying sequence length).

---

## 2. Iteration Taxonomy

Every tested configuration falls into one of four progression ladders:

### Ladder A: Feature Engineering & Transformation Progression
Tracks how predictive skill evolves as meteorological representations become richer:
* **F0 (Raw Baseline):** Coarse monthly means of basic ERA5-Land variables ($T_{\text{mean}}$, $P_{\text{tot}}$, $R_{\text{solar}}$).
* **F1 (Thermal Accumulation):** F0 + Growing Degree Days (GDD base $10^\circ\text{C}$, cap $30^\circ\text{C}$).
* **F2 (Acute Thermal Stress):** F1 + Extreme Degree Days (EDD $>30^\circ\text{C}$) and threshold day counts.
* **F3 (Atmospheric Demand):** F2 + Vapor Pressure Deficit (VPD).
* **F4 (Surface Water Balance):** F3 + Cumulative Climatic Water Balance ($P - \text{ET}_0$ via FAO-56).
* **F5 (High-Frequency Granularity):** F4 calculated across 10-day and 5-day intervals rather than 30-day monthly blocks.

### Ladder B: Model Family Complexity Progression
Tracks how predictive skill evolves as model capacity increases:
* **M0 (Reference Baselines):**
  * M0.1: County Climatology (historical mean).
  * M0.2: County Technological Trend (linear, quadratic, natural cubic spline, rolling average).
  * M0.3: Lagged Yield Persistence ($Y_{t-1}$, subject to publication lag).
  * M0.4: Monthly Baseline inspired by Chen & Zhang (trend + climatology + $Y_{t-1}$ + monthly weather).
* **M1 (Linear Econometric Models):**
  * M1.1: Ordinary Least Squares (OLS) with train-only regularization.
  * M1.2: Ridge regression ($L_2$ penalty).
  * M1.3: Lasso regression ($L_1$ sparsity).
  * M1.4: Generalized Additive Models (GAM) / Spline components.
* **M2 (Tree-Based Ensembles):**
  * M2.1: Default Random Forest.
  * M2.2: Tuned Random Forest (depth, leaf size, feature subsampling).
  * M2.3: Gradient Boosted Trees (XGBoost / LightGBM) with train-only early stopping.
* **M3 (Sequential Deep Learning):**
  * M3.1: Multi-Layer Perceptron (MLP) on flattened tabular features.
  * M3.2: 1D Convolutional Neural Network (CNN) across temporal windows.
  * M3.3: Recurrent Long Short-Term Memory (LSTM) on daily/5-day sequences.
  * M3.4: Bidirectional / Multi-layer LSTM variants.

### Ladder C: Ablation & Sensitivity Experiments
* **Ablation 1 (Past-Yield Removal):** Removing $Y_{t-1}$ to isolate pure weather-driven skill (evaluating the Chen & Zhang baseline drop).
* **Ablation 2 (Weather-Only vs Integrated):** Category B (weather only) vs Category C (baseline + weather).
* **Ablation 3 (Thermal Damage Threshold):** EDD at $30^\circ\text{C}$ vs $32^\circ\text{C}$ vs $35^\circ\text{C}$.
* **Ablation 4 (Detrending Method Sensitivity):** Evaluating skill stability under linear vs quadratic vs cubic spline detrending.

---

## 3. Standardized Experiment Logging Schema

Every run executed during the modeling phase must generate a metadata record saved in `work/reports/experiments/<run_id>/metadata.json` and a metrics summary in `work/reports/experiments/<run_id>/metrics_summary.csv`.

### Metadata Fields
```json
{
  "run_id": "EXP_20261001_RF_F3_10D",
  "timestamp": "2026-10-01T14:30:00Z",
  "git_commit": "abcdef1234567890",
  "model_family": "RandomForest",
  "model_variant": "M2.2_tuned",
  "feature_ladder": "F3",
  "temporal_granularity": "10d",
  "hyperparameters": {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5
  },
  "validation_protocol": "expanding_window",
  "initial_training_years": 40,
  "test_years_span": [1991, 2025],
  "lead_times_evaluated": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
}
```

### Required Performance Metrics (per lead time $H=12 \dots 1$):
* **OOS RMSE:** Root Mean Squared Error on detrended anomaly.
* **OOS MAE:** Mean Absolute Error on detrended anomaly.
* **OOS $R^2$:** Out-of-sample coefficient of determination.
* **Relative Skill vs Climatology:** $1 - \text{RMSE} / \text{RMSE}_{\text{climatology}}$.
* **Relative Skill vs Trend:** $1 - \text{RMSE} / \text{RMSE}_{\text{trend}}$.
* **Relative Skill vs Lagged Yield ($Y_{t-1}$):** $1 - \text{RMSE} / \text{RMSE}_{Y-1}$.
* **Balanced Accuracy & Macro-$F_1$:** For categorical regime prediction (tertiles/quantiles).
* **ROC-AUC:** For extreme shortfall events ($<20$th percentile).
* **Worst-Year Performance:** Specific error during major historical drought years (1988, 2012).

---

## 4. Master Progression Register

A global register `work/reports/experiments/master_progression_register.csv` will aggregate all runs across time. This file enables generating:
1. **Model Progression Curves:** Plots showing RMSE reduction as features and models evolve from F0/M0 to F5/M3 across lead times $H=12 \dots 1$.
2. **Comprehensive Thesis Tables:** Direct LaTeX export of ablation tables for Chapter 5, showing exact incremental gains at each developmental milestone.
