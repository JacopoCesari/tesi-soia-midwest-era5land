# Action Plan: Chapter 4 (Methodology)

> **Document Status:** Active Working Roadmap  
> **Target Delivery File:** `output/thesis/chapters/04_methodology.tex`  
> **Master Reference:** `output/thesis/main.tex`, `output/docs/decisions.md`, `work/thesis_notes/operational_todo_and_action_plans.md`

---

## 1. Executive Summary & Core Framing

Chapter 4 establishes the formal mathematical, econometric, and algorithmic foundation of the thesis. Following the empirical description of the 75-year panel and climate dynamics in Chapter 3, Chapter 4 formalizes the predictive machinery without discussing empirical results.

### Core Scientific Decisions
1. **Target Formulation:** Pure continuous detrended yield anomaly regression ($\epsilon_{c,Y}$). Discrete regime classification is formally excluded, eliminating arbitrary discretization thresholds and multi-loss fragmentation.
2. **Tail-Risk Handling:** Historical tail-risk sensitivity is evaluated directly through an out-of-sample stress test on macro-climatic shock cohorts (e.g., 1988, 2012, 1993, 2016) using continuous anomaly predictions.
3. **Horizon-Specific Modeling:** Forecasters possess a dynamically expanding information set $\mathcal{I}_H$ as the season advances. Rather than a static model with masked features, an independent model instance $\mathcal{M}_H$ is calibrated for each monthly lead time $H \in \{12, \dots, 1\}$.
4. **Non-Redundancy Guarantee:**
   * *Yield detrending & EDA:* Fully detailed in Chapter 3 (Section 3.2.1); Chapter 4 strictly references the additive decomposition $y_{c,Y} = \tau_{c,Y} + \epsilon_{c,Y}$ and enforces its train-only estimation.
   * *Meteorological variables:* The 17 agrometeorological indicators are cataloged in Chapter 3 (Tables 3.5 & 3.6); Chapter 4 refers to them as the available input pool $K=17$.

---

## 2. Model Portfolio & Data Representation Paradigms

### A. Candidate Model Portfolio (Complexity Ladder)
The model suite is organized into five complementary families to test whether model complexity provides genuine out-of-sample skill over parsimonious baselines:

1. **Reference Baselines:**
   * *Naive Secular Trend Benchmark:* Predicts $\hat{\epsilon}_{c,Y} = 0$ (i.e., $\hat{y} = \tau_{c,Y}$), serving as the unaugmented zero-weather baseline at Month 1 ($H=12$, $R^2_{\text{OOS}} = 0$).
   * *Autoregressive Persistence ($y_{t-1}$):* Captures historical yield memory, constrained by a $Y-2$ lag for origins $H \ge 8$ to respect USDA NASS publication schedules.
2. **Linear Regularized:**
   * *ElasticNet:* Combines $L_1$ and $L_2$ penalties into a single convex loss, enabling both feature selection and robust handling of collinear weather variables.
3. **Kernel & Margin-Based Non-Linear:**
   * *Support Vector Regression (SVR):* Implements an $\epsilon$-insensitive loss function (ignoring background weather noise) and an RBF kernel to generate smooth continuous response surfaces reflecting crop physiology.
4. **Tree Ensembles (Non-Parametric Partitioning):**
   * *Random Forest:* Bagging ensemble of deep trees reducing estimation variance.
   * *XGBoost:* Gradient boosted decision tree ensemble optimizing regularized split loss with shrinkage.
5. **Sequential Deep Learning:**
   * *Long Short-Term Memory (LSTM):* Recurrent architecture processing time series without flattening; 1D-CNN and GRU referenced as natural architectural counterparts.

### B. Input Structuring: Tabular Flattening vs. Sequence Tensors
* **Continuous Spatial Coordinates ($P_{\text{geo}} = 2$):**
  * County centroid coordinates ($\text{Lat}_c, \text{Lon}_c$) are normalized and included as continuous spatial covariates.
  * *Explicit Exclusion of State Dummies:* Administrative state borders are deliberately excluded. Physical contiguity and bioclimatic transitions cross political state lines continuously (e.g., eastern Iowa and western Illinois share identical microclimates and soils, whereas northern and southern Illinois differ substantially). Continuous coordinates allow non-linear models (trees, SVR) to learn genuine spatial proximity and photoperiod gradients without artificial administrative boundaries.
* **2D Tabular Flattening (ElasticNet, SVR, Random Forest, XGBoost):**
  * Weather history up to origin $H$ is structured into monthly aggregated blocks ($17 \times m$ features) concatenated with the 2 spatial coordinates ($P_H = 2 + 17 \times m$).
  * *Methodological justification:* Monthly blocks preserve seasonal timing while preventing parameter explosion and collinearity collapse associated with ultra-high frequency tabular concatenation.
* **3D Sequence Tensor (LSTM):**
  * Data are structured as a 3D sequence tensor $[N_{\text{samples}}, T_{\text{timesteps}}, 17]$.
  * The hidden state $h_T$ at the cutoff step summarizes temporal trajectory dynamics into a scalar anomaly prediction $\hat{\epsilon}_{c,Y}$.

### C. Single-Pass Hyperparameter Grid Specifications (Inner Expanding-Window)
To guarantee strict immunity from data snooping, hyperparameter calibration follows a **Single-Pass Grid Search** defined a priori and executed strictly inside inner training folds:
* *ElasticNet:* `alpha` $\in \{10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0\}$, `l1_ratio` $\in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ (25 combinations).
* *SVR (RBF):* $C \in \{0.5, 2.0, 10.0\}$, $\epsilon \in \{0.1, 0.5, 1.0\}$, $\gamma \in \{0.1 \times \gamma_{\text{scale}}, 1 \times \gamma_{\text{scale}}, 5 \times \gamma_{\text{scale}}\}$ (27 combinations).
* *Random Forest:* `n_estimators` $= 300$, `max_features` $\in \{\text{'sqrt'}, 0.33, 0.5\}$, `min_samples_leaf` $\in \{5, 15, 30\}$, `max_depth` $\in \{8, 12, \text{None}\}$ (27 combinations).
* *XGBoost:* `learning_rate` $\in \{0.03, 0.08\}$, `max_depth` $\in \{3, 5\}$, `colsample_bytree` $\in \{0.6, 0.8\}$, `reg_lambda` $\in \{1.0, 10.0\}$ (16 combinations).

---

## 3. Detailed Structure of Chapter 4 (`04_methodology.tex`)

### Section 4.1 — Problem Formulation and Multi-Horizon Setup
* Formal definition of the prediction target: additive decomposition $y_{c,Y} = \tau_{c,Y} + \epsilon_{c,Y}$ with cross-reference to Chapter 3 for trend parameterization.
* Operational definition of the 12-month campaign (November 1 of $Y-1$ to October 31 of $Y$).
* Step-by-step monthly horizon timeline ($H=12 \dots 1$):
  * *Month 1 ($H=12$):* 0 weather features $\implies$ pure naive baseline ($\hat{\epsilon}=0$, $R^2_{\text{OOS}}=0$).
  * *Month 2 ($H=11$):* 2 months weather (Nov+Dec) $\implies$ empirical test of antecedent weather value vs. overfitting noise.
  * *Months 3..12 ($H=10 \dots 1$):* Progressive cumulative monthly weather until harvest.
* Formalization of the Direct Horizon-Specific Modeling paradigm ($\mathcal{M}_H: \mathcal{X}_H \to \hat{\epsilon}$).
* Operational lag constraints: USDA NASS release timing requiring $Y-2$ historical yields at $H \ge 8$.

### Section 4.2 — Input Representation and Temporal Aggregation Strategies
* Definition of the input feature space ($K=17$ agrometeorological indicators, citing Chapter 3).
* 2D tabular representation: monthly block concatenation and dimensional control ($P_H = 17 \times m$).
* 3D recurrent tensor representation: temporal sequence mapping for deep learning.
* Methodological synthesis on temporal aggregation: justifying the trade-offs between coarse monthly summaries, decadal blocks, and high-frequency pentads.

### Section 4.3 — Candidate Forecasting Models and Complexity Ladder
* Mathematical formalization of reference baselines (Naive Trend and Lagged Yield).
* ElasticNet: formulation of the dual-penalty loss function and parameter mixing ($\alpha, \lambda$).
* Support Vector Regression (SVR): dual formulation, $\epsilon$-tube definition, and RBF kernel projection.
* Tree Ensembles: Random Forest bagging mechanics and XGBoost objective with leaf regularization ($\gamma, \lambda$).
* Sequential Deep Learning: LSTM cell equations (input, forget, output gates, cell state $c_t$) and dense output projection.

### Section 4.4 — Chronological Expanding-Window Validation Protocol
* Description of the three-phase temporal partition:
  * *Phase 1 (Base Training):* 1951–1984 ($N=34$ years, 4,590 observations) establishing climatology and initial weights.
  * *Phase 2 (Validation):* 1985–1995 ($N_{\text{val}}=11$ years, 1,485 observations) balanced between 6 positive anomaly campaigns (1985, 1986, 1987, 1990, 1992, 1994) and 5 negative anomaly campaigns (1988 drought, 1989, 1991, 1993 flood, 1995).
  * *Phase 3 (Out-of-Sample Test):* 1996–2025 ($N_{\text{test}}=30$ years, 4,050 evaluations) expanding year-by-year from 1996 through 2025.
* Prevention of spatial and temporal leakage: testing all 135 counties simultaneously per crop year and rejecting random k-fold splits (citing Sweet et al. 2023).
* Strict train-only hermetic seal: detrending coefficients, standardization scalers, and hyperparameter tuning performed exclusively within outer training folds.

### Section 4.5 — Performance Evaluation and Tail-Risk Stress Testing
* Out-of-sample anomaly evaluation metrics across 1996–2025 ($M_{\text{test}} = 4{,}050$):
  $$\text{RMSE}_{\text{OOS}} = \sqrt{\frac{1}{M_{\text{test}}} \sum_{Y=1996}^{2025} \sum_{c=1}^C (\hat{\epsilon}_{c,Y} - \epsilon_{c,Y})^2}, \quad \text{MAE}_{\text{OOS}} = \frac{1}{M_{\text{test}}} \sum_{Y=1996}^{2025} \sum_{c=1}^C |\hat{\epsilon}_{c,Y} - \epsilon_{c,Y}|$$
  $$R^2_{\text{OOS}} = 1 - \frac{\sum_{Y=1996}^{2025} \sum_{c=1}^C (\epsilon_{c,Y} - \hat{\epsilon}_{c,Y})^2}{\sum_{Y=1996}^{2025} \sum_{c=1}^C \epsilon_{c,Y}^2}$$
* Baseline-relative skill score tracking incremental meteorological value over naive trend extrapolation.
* Tail-risk stress test protocol: evaluating continuous anomaly predictions specifically against historical shock benchmark cohorts (Table 3.4) to determine early shock sensitivity.

---

## 4. Sequential Action Items for Implementation

- [ ] **Step 1: File Setup & Main Linkage**
  - Create `output/thesis/chapters/04_methodology.tex`.
  - Uncomment `\include{chapters/04_methodology}` in `output/thesis/main.tex`.
- [ ] **Step 2: Draft Sections 4.1 & 4.2**
  - Formalize target decomposition, 12-month horizon sequence, direct horizon-specific paradigm, and tabular vs. sequence data structuring.
- [ ] **Step 3: Draft Section 4.3**
  - Write concise, mathematically rigorous formulations for all candidate models (Naive, Lagged Yield, ElasticNet, SVR, RF, XGBoost, LSTM).
- [ ] **Step 4: Draft Sections 4.4 & 4.5**
  - Detail expanding-window anti-leakage protocol, metric formulations on anomaly error, and tail-shock stress-testing rules.
- [ ] **Step 5: Compilation and Quality Assurance**
  - Compile the complete manuscript (`pdflatex` / `latexmk`).
  - Verify zero LaTeX errors, correct cross-referencing to Chapter 3 tables/figures, and Vademecum adherence.
