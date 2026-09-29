# Author Working Notes & Pending Tasks

> **Note on Project Rules:**  
> All permanent writing protocols, author voice instructions, zero-LLM marker constraints, and literature management rules are maintained exclusively in [`AGENTS.md`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/AGENTS.md).  
> This file tracks **strictly temporary working notes, open decisions, and pending tasks/re-runs** to be addressed in subsequent chapters or upon data updates.

---

## 1. Deferred Methodological Decisions (To Resolve in Chapter 5)

### A. Target Selection: Pure Continuous Anomaly Regression (Classification Excluded)
*   **Decision (2026-09-28):** Categorical regime classification is formally excluded from the modeling pipeline. The research design focuses 100% on continuous detrended yield anomaly regression ($\epsilon_{c,Y}$).
*   **Rationale:** Eliminates arbitrary discretization thresholds, avoids splitting validation metrics across multiple loss paradigms, and ensures direct comparability with core literature benchmarks (Chen & Zhang 2026, Khaki et al. 2020).
*   **Tail-Risk Evaluation:** Rather than training separate discrete classifiers, the ability to anticipate extreme climate shocks is evaluated directly by stress-testing the continuous regression predictions against historical shock cohorts (Table 3.4 cohorts: 1988, 2012, 1993, 2016).
*   **Validation Interval (Consolidated 2026-09-28):** Three-phase temporal partition: Initial baseline training spans 1951–1984 ($N=34$ years, $4,590$ observations). Hyperparameter validation spans 1985–1995 ($N_{\text{val}}=11$ years, $1,485$ observations), achieving an equitable balance across climatic regimes with 6 positive anomaly campaigns (1985, 1986, 1987, 1990, 1992, 1994) and 5 negative anomaly campaigns (1988 drought, 1989, 1991, 1993 flood, 1995). Out-of-sample expanding-window testing spans 1996–2025 ($N_{\text{test}}=30$ crop years, $4,050$ blind evaluations per lead time; 17 positive, 13 negative), evaluating modern shock years (2003, 2012 drought, 2004, 2016 bumper). Primary loss function is $\text{RMSE}$ on detrended anomaly.

### B. Final Candidate Model Portfolio and Data Representation Paradigms
*   **Decision (2026-09-28):** The candidate model portfolio is streamlined into four complementary families, avoiding redundant variants:
    1.  *Reference Baselines:* Pure Naive Trend ($\hat{\epsilon}_{c,Y}=0$, $R^2_{\text{OOS}}=0$ at Month 1 with 0 weather) and Autoregressive Lagged Yield ($y_{t-1}$, lagged to $Y-2$ for $H \ge 8$).
    2.  *Linear Regularized:* ElasticNet (unifying $L_1$ sparsity and $L_2$ collinearity grouping into a single convex objective).
    3.  *Margin/Kernel Non-Linear:* Support Vector Regression (SVR) with RBF kernel and $\epsilon$-insensitive loss (constructs smooth continuous response surfaces matching plant physiology and ignores minor weather noise).
    4.  *Tree Ensembles:* Random Forest (bagging, variance reduction) and XGBoost (gradient boosting with shrinkage and leaf regularization).
    5.  *Sequential Deep Learning:* LSTM (Long Short-Term Memory) recurrent neural network (processes sequence dynamics; 1D-CNN and GRU referenced as alternative sequence architectures).
*   **Data Representation Paradigms:**
    *   *Horizon-Specific Direct Forecasting:* At each monthly origin $H \in \{12 \dots 1\}$, an independent model instance $\mathcal{M}_H$ is calibrated on the information set $\mathcal{I}_H$ available up to that cutoff.
    *   *Continuous Spatial Coordinates ($P_{\text{geo}}=2$):* County centroid coordinates ($\text{Lat}_c, \text{Lon}_c$) are normalized and included as continuous spatial covariates. Administrative state dummies are explicitly excluded to prevent artificial boundary discontinuities across contiguous bioclimatic zones.
    *   *2D Tabular Flattening (ElasticNet, SVR, RF, XGBoost):* Features are concatenated into monthly aggregated blocks ($17 \times m$ features) plus the 2 spatial coordinates ($P_H = 2 + 17 \times m$), avoiding feature explosion from ultra-high frequency tabular windows.
    *   *3D Sequence Tensor (LSTM):* Native $[N, T, 17]$ sequence representation, where hidden state $h_T$ summarizes the growing season progression into a scalar prediction.
*   **Hyperparameter Tuning Strategy (Single-Pass Grid Search):**
    *   To prevent look-ahead bias and human data-snooping leakage, hyperparameter calibration uses an automated **Single-Pass Grid Search** defined a priori and executed strictly inside inner expanding-window validation folds.
    *   *Grid specifications:* ElasticNet (25 combinations: $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$, $\alpha \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$); SVR (27 combinations: $C \in \{0.5, 2, 10\}$, $\epsilon \in \{0.1, 0.5, 1.0\}$, $\gamma \in \{0.1, 1, 5\} \times \gamma_{\text{scale}}$); Random Forest (27 combinations: trees=300, `max_features` $\in \{\text{'sqrt'}, 0.33, 0.5\}$, `min_samples_leaf` $\in \{5, 15, 30\}$, `max_depth` $\in \{8, 12, \text{None}\}$); XGBoost (16 combinations: $\eta \in \{0.03, 0.08\}$, depth $\in \{3, 5\}$, colsample $\in \{0.6, 0.8\}$, $\lambda \in \{1, 10\}$).

### C. Table 3.4 (Historical Yield Shocks and Macro-Climatic Benchmarks)
*   **Current State:** Positioned in Chapter 3 (`tab_yield_shocks.tex`) listing the major historical shortfall ($\le -10\%$) and bumper ($\ge +10\%$) cohorts.
*   **Pending Action:** Directly referenced in Chapter 4 for out-of-sample tail-risk stress testing and evaluated in Chapter 5.

### C. Horizon Progression, Pure Naive Trend Baseline, and Error-Only Anomaly Modeling
*   **Crop Campaign Definition (12 Months):** For target harvest year $Y$, the campaign spans exactly 12 calendar months from **November 1 of Year $Y-1$ to October 31 of Year $Y$**.
*   **Sequential Monthly Horizons & Weather Accumulation:**
    *   Forecasts are generated month-by-month through the campaign, with each horizon evaluated at the end of the corresponding campaign month.
    *   **Month 1 (End of November $Y-1$, $H=12$ lead months): 0 weather data ingested.**
        *   Even though Month 1 (November) has elapsed, the model is deliberately supplied with **zero meteorological data**.
        *   The prediction relies strictly on the historical secular trend ($\hat{y}_{c,Y} = \tau_{c,Y}$, or predicted anomaly $\hat{\epsilon}_{c,Y} = 0$).
        *   This serves as the pure, unaugmented **naive baseline** for the new campaign.
    *   **Month 2 (End of December $Y-1$, $H=11$ lead months): 2 months of cumulative weather (Nov + Dec $Y-1$).**
        *   The model receives 2 months of observed weather (Months 1 and 2).
        *   *Empirical test:* Evaluates whether having 2 months of distant antecedent autumn/early-winter weather delivers genuine out-of-sample predictive skill over the 0-data naive baseline ($R^2_{\text{OOS}} > 0$), or whether it merely introduces noise and overfitting ($R^2_{\text{OOS}} < 0$, $\text{RMSE} > \text{RMSE}_{\text{naive}}$).
    *   **Month 3 (End of January $Y$, $H=10$):** 3 months (Nov–Jan).
    *   **Month 4 (End of February $Y$, $H=9$):** 4 months (Nov–Feb).
    *   **Month 5 (End of March $Y$, $H=8$):** 5 months (Nov–Mar).
    *   **Month 6 (End of April $Y$, $H=7$):** 6 months (Nov–Apr, overwinter hydrological recharge + pre-sowing soil profile).
    *   **Month 7 (End of May $Y$, $H=6$):** 7 months (Nov–May, sowing window).
    *   **Month 8 (End of June $Y$, $H=5$):** 8 months (Nov–Jun, vegetative emergence & canopy development).
    *   **Month 9 (End of July $Y$, $H=4$):** 9 months (Nov–Jul, critical flowering & early pod set).
    *   **Month 10 (End of August $Y$, $H=3$):** 10 months (Nov–Aug, peak reproductive pod-filling).
    *   **Month 11 (End of September $Y$, $H=2$):** 11 months (Nov–Sep, physiological maturity).
    *   **Month 12 (End of October $Y$, $H=1$ / Harvest):** 12 months (Nov–Oct, complete 12-month campaign observed weather).
*   **Modeling Strictly on Detrended Anomaly Error:**
    *   Observed yield is decomposed as $y_{c,Y} = \tau_{c,Y} + \epsilon_{c,Y}$, where $\tau_{c,Y}$ is the deterministic trend estimated strictly train-only.
    *   All statistical, econometric, and ML models are trained **strictly to predict the anomaly $\epsilon_{c,Y}$**.
    *   Out-of-sample performance metrics (RMSE, MAE, $R^2_{\text{OOS}}$) are computed **directly on the anomaly error** $\hat{\epsilon}_{c,Y} - \epsilon_{c,Y}$ (or equivalently relative to $\tau_{c,Y}$).
    *   Because the secular trend accounts for over 80% of raw yield variance, evaluating metrics on raw yield would artificially inflate performance even with zero weather data.
    *   Evaluating directly on $\epsilon$ ensures:
        *   At **Month 1 (Naive, 0 data)**: $\hat{\epsilon} = 0 \implies \text{RMSE}_{\text{naive}} = \sigma(\epsilon)$ and $R^2_{\text{OOS}} = 0.0$.
        *   At **Month 2 (2 months data)**: $R^2_{\text{OOS}} < 0$ cleanly reveals whether early distant data causes out-of-sample degradation due to overfitting, while $R^2_{\text{OOS}} > 0$ establishes genuine early signal.
        *   Tracking the monthly progression from Month 1 to Month 12 maps the exact lead-time threshold where meteorological information begins yielding statistically significant predictive value beyond the naive baseline.

---

## 2. Weather Dataset (1950–2025) & Figure Synchronization [COMPLETED]

*   **Final State:** All 76 years of daily county-aggregated ERA5-Land parquets (1950–2025) have been extracted, validated through quality control (QC: `PASS` across all 76 annual files), and stored in `work/data/interim/county_daily_weather/production/`.
*   **Artifacts Synchronized:**
    *   *Figure 3.6 (Climatology):* Recomputed over the entire 75-year target period (1951–2025) with continuous 32-variable agrometeorological series.
    *   *Figures 3.7 & 3.8 (Intra-Seasonal Trajectories & State Profiles):* Verified across full historical extremes.
    *   *Figure 3.9 (Spatial Shock Comparison):* Finalized with Option 3 (2012 Flash Drought vs. 2016 Bumper, Rainfall vs. $T_{\text{max}}$ with *inferno* palette) and synchronized with Chapter 3 narrative.
    *   *Figure 3.10 (Phenological Sensitivity):* Recomputed across the complete 75-year panel.
    *   *Table 3.5 & Table 3.6:* Formally updated with 32 agrometeorological indicators and formal source attribution.

---

## 3. Pending Literature Deployments for Chapter 4 (Methodology)

The primary bibliography is strictly closed to the 13 papers in `Literature.zip`. Ten papers are cited in Chapters 1–3. The remaining three papers will be introduced naturally in Chapter 4:
1.  **`Torsoni et al. (2023)`**: To be cited in Section 4.4 when formalizing Random Forest and Gradient Boosting hyperparameter grids for soybean yield.
2.  **`Shook et al. (2021)`**: To be cited in Section 4.5 when detailing the recurrent deep learning architecture (LSTM) sequence processing.
3.  **`Xie, Huang & Meng (2025)`**: To be cited in Section 4.2 when formalizing the sub-monthly (5-day and 10-day) rolling feature engineering window.

---

## 4. In-Depth Figure-by-Figure Audit (Chapters 1–3 Figures)

*   **Objective:** Conduct an exhaustive, high-resolution visual and methodological audit of every individual graphic (`output/thesis/figures/fig_*.pdf`).
*   **Audit Checklist for Each Figure:**
    1.  *Visual Clarity & Legibility:* Check font sizing, line weights, color palettes (colorblind-friendly/distinguishable in monochrome print), and absence of overlapping text/elements.
    2.  *Axes, Labels & Units:* Ensure every axis has explicit domain units (e.g., $\text{bu}/\text{acre}$, $\text{mm}$, $^{\circ}\text{C}$, $\text{hPa}$, $\text{MJ}/\text{m}^2$) and descriptive, mathematically consistent labels.
    3.  *Legends & Colorbars:* Verify all legends are self-explanatory, unambiguous, and do not occlude data series.
    4.  *Self-Contained Captions & Formal Source Attributions:* Ensure captions fully explain what is plotted (thresholds, sample years, aggregations) and contain the formal source line (`\textit{Source: ...}`) complying with the university thesis guide.
    5.  *Text-Figure Synthesis:* Confirm that the narrative commentary in the chapter text accurately synthesizes the empirical patterns without mechanically repeating numbers visible on the axes.
    6.  *Inventory to Audit:*
        *   `fig_study_area_map.pdf` (Figure 3.1)
        *   `fig_yield_trajectory.pdf` (Figure 3.2)
        *   `fig_state_yield_distributions.pdf` (Figure 3.3)
        *   `fig_dispersion_dynamics.pdf` (Figure 3.4)
        *   `fig_era5_spatial_structure.pdf` (Figure 3.5)
        *   `fig_weather_climatology.pdf` (Figure 3.6)
        *   `fig_intra_seasonal_extremes.pdf` (Figure 3.7)
        *   `fig_weather_shocks_footprint.pdf` (Figure 3.8)
        *   `fig_spatial_shock_comparison.pdf` (Figure 3.9)
        *   `fig_climate_yield_sensitivity.pdf` (Figure 3.10)

---

## 5. Post-Chapter 4 Execution Roadmap

With the formalization of Chapter 4 (`Methodology`), the sequential pipeline transitioning from methodological design to machine learning model execution and empirical results generation is documented in detail in:
*   [`work/thesis_notes/modeling_execution_roadmap.md`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/work/thesis_notes/modeling_execution_roadmap.md)

Key milestones:
1.  **Gate 0**: Verification of Chapter 4 character budget (16k–19k chars via `count_thesis_characters.py`) and LaTeX zero-error compilation.
2.  **Step 1**: Machine learning environment configuration (`scikit-learn`, `xgboost`, `torch`).
3.  **Step 2**: Feature matrix assembly across 12 monthly horizons from ERA5-Land daily parquets (`build_feature_matrices.py`).
4.  **Step 3**: Train-only hermetic OLS detrending and feature standardization module.
5.  **Step 4**: Single-Pass Grid Search across the 1985–1995 balanced validation partition (11 years, 1,485 obs).
6.  **Step 5**: Expanding-window out-of-sample evaluation across 1996–2025 (30 years, 4,050 evaluations) and export of performance artifacts.
7.  **Step 6**: Drafting Chapter 5 (`05_empirical_results.tex`).
