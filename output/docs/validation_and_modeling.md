# Validation and planned modeling

## Implemented temporal splitter

`evaluation/expanding_window.py` yields positional `train_indices`, `test_indices`,
`train_years` and `test_year`. It accepts county-year records and keeps every complete
crop year indivisible. It rejects duplicate keys, missing or noninteger years,
calendar gaps, changing county membership and insufficient history.

`initial_training_years` is unset in `configs/modeling.yaml`; the splitter rejects
an unset value. The author has not chosen the first validation year and explicitly
rejects a fixed 60/40 assumption. Given an approved initial length N, the first
training interval would be 1951 through 1950+N, testing 1951+N, then expanding
annually through 2025, subject to horizon-specific information availability.

All 135 counties from a crop year remain together. Random county-year splits and
ordinary row-based k-fold cross-validation are not accepted alternatives.

## Train-only processing and availability

Detrending, scaling, feature selection, tuning and any later categorical thresholds
must be fitted inside training only. Inner tuning must use earlier complete years
inside the outer training window. Each monthly horizon gets a separate feature set
and model instance using weather no later than its forecast origin.

The fold utility groups target years; it does not assert when their yields were
published. For early origins in Y-1, including yield from Y-1 in training may violate
actual label availability. The author must settle release timing and any resulting
horizon-specific training lag before fitting models. No silent lag or change to the
provisional crop-year folds is implemented here. Historical ERA5 availability also
needs an explicit operational interpretation.

## Planned baselines and candidate models

A fundamental distinction is drawn between **reference baselines** (measuring variance explained before adding current-season weather) and **candidate predictive models** (whose predictive skill is under test).

### Reference Baselines
1. **County Climatology:** Historical mean yield or anomaly estimated strictly from past training observations.
2. **County Technological Trend:** Historical trend line (linear, polynomial, natural cubic spline, or rolling average) fitted strictly train-only.
3. **Previous-Year Yield ($Y_{t-1}$):** Autoregressive persistence baseline, subject to official publication lags.
4. **Regularized Linear Model or GAM:** Interpretable benchmark (Ridge/Lasso) to establish whether non-linear complexity is necessary.
5. **Monthly Benchmark inspired by Chen & Zhang (2026):** Trend, climatology, lagged yield, and coarse monthly weather summaries available up to the cutoff.

### Candidate Predictive Models
- **Penalized Linear Models:** Ridge and Lasso regressions on high-resolution weather features.
- **Tree Ensembles:** Random Forest and Gradient Boosted Decision Trees (XGBoost / LightGBM).
- **Sequential Deep Learning:** Recurrent neural network (LSTM) processing high-frequency (daily, 5-day, 10-day) weather sequences.

### Model Taxonomy for Experiments
1. **Category A — Historical Baselines:** Climatology, Trend, Past-Yield Persistence ($Y_{t-1}$).
2. **Category B — Weather-Only Ablation:** Models trained exclusively on observed weather features without historical yield/trend inputs.
3. **Category C — Integrated Models:** Historical baseline + progressively observed weather features.

No model is implemented or fitted in this phase. Do not assume deep learning is superior.


## Evaluation specification and open decisions before experiments

The following methodological choices must be formally decided before running model training:

1. **Primary Regression Metric:** Selection of the primary loss function among RMSE,
   MAE, and out-of-sample $R^2$, along with the formal baseline-relative skill score:
   $$\text{Skill} = 1 - \frac{\text{Loss}_{\text{model}}}{\text{Loss}_{\text{baseline}}}$$
2. **Primary Classification Metric:** Selection of the primary metric for discrete
   regimes (e.g., Macro-F1, Balanced Accuracy, or Multiclass Brier Score).
3. **Pooled vs Year-Averaged Reporting:** Whether overall performance is computed
   pooled over all county-year test pairs, or averaged across annual test batches
   (which gives equal weight to each crop year regardless of year-specific variance).
4. **Statistical Definition of "Earliest Stable Skill":** The formal decision rule
   defining the emergence of skill: horizon $H^*$ is the earliest stable origin if
   the relative skill over the baseline is strictly positive and statistically significant
   at $H^*$, and remains strictly positive and significant at all subsequent origins
   $H < H^*$.
5. **Multiplicity Control across 12 Horizons:** Statistical adjustment for multiple
   comparisons across the twelve monthly cutoffs (e.g., False Discovery Rate control via
   Benjamini-Hochberg or family-wise error rate correction).
6. **Initial Training Length and Validation Interval:** Setting $N$, the number of initial
   training years (e.g., 1951 through $1950+N$), and the resulting span of the out-of-sample
   testing period ($1951+N$ through 2025).
7. **Label Availability at $Y-1$ Horizons ($H=12 \dots 8$):** Final USDA NASS county yield
   estimates for year $Y-1$ are officially released in January/February of year $Y$.
   Because origins $H=12$ through $H=8$ occur in autumn/winter of $Y-1$, the training
   set and antecedent yield features at these cutoffs cannot include year $Y-1$ yield
   without inducing label leakage. The training cutoff must be lagged to $Y-2$ for these
   early origins.
8. **Train-Only Detrending Selection Criterion:** The protocol for choosing among
   linear, polynomial, spline, and rolling-average detrending models using exclusively
   training data (e.g., inner cross-validation or information criteria like AIC/BIC),
   preventing test set leakage.
9. **Uncertainty Quantification:** Formalizing the block-bootstrap protocol across
   complete crop years (preserving the spatial dependence structure of all 135 counties)
   to construct confidence intervals and test hypotheses H1–H5.

## Model progression, ablation tracking, and iterative reporting

Per author directive (2026-09-18), the evaluation system must systematically record and persist out-of-sample performance metrics for **every tested model iteration, variant, and ablation**. This ensures that the thesis narrative in Chapter 5 (Results) and Chapter 6 (Discussion) can rigorously reconstruct the complete developmental journey and progression of improvements, rather than merely presenting the terminal best-performing specification.

### Systematic Tracking Dimensions
1. **Feature Engineering Progression:**
   - Raw weather aggregates (mean/min/max temperature, total precipitation, solar flux).
   - Non-linear derived indices: Growing Degree Days (GDD), Extreme Degree Days (EDD above 30°C and 35°C), Vapor Pressure Deficit (VPD), and surface water balance ($P - \text{ET}_0$ via FAO-56).
   - Temporal aggregation granularity: direct progression from coarse 30-day monthly means to 10-day and 5-day high-frequency windows.
2. **Model Complexity Ladder:**
   - Baseline tier: County climatology $\to$ linear trend $\to$ polynomial trend $\to$ spline $\to$ lagged yield persistence ($Y_{t-1}$).
   - Linear tier: Ordinary Least Squares $\to$ Ridge $\to$ Lasso $\to$ ElasticNet $\to$ GAM.
   - Tree ensemble tier: Default Random Forest $\to$ tuned Random Forest $\to$ XGBoost $\to$ LightGBM.
   - Sequential deep learning tier: Simple multi-layer perceptron (MLP) $\to$ 1D CNN $\to$ basic LSTM $\to$ bidirectional or multi-layer LSTM.
3. **Ablation and Sensitivity Variants:**
   - Impact of excluding lagged past yield ($Y_{t-1}$) across all model families (directly replicating and extending Chen & Zhang's finding).
   - Weather-only ablation versus integrated (weather + baseline) models.
   - Sensitivity to thermal stress threshold definitions (e.g., EDD $>30^\circ\text{C}$ vs $>35^\circ\text{C}$).
4. **Structured Run Artifacts:**
   - Every experimental run must persist a structured metadata manifest (JSON/YAML) detailing: git commit hash, configuration file, input feature catalog, model architecture/hyperparameters, random seed, training time, and out-of-sample error metrics per lead time ($H=12 \dots 1$) across folds.
   - Summary performance tables and progressive comparison curves must be archived in `work/reports/experiments/` to support direct LaTeX export into the thesis results chapter.


