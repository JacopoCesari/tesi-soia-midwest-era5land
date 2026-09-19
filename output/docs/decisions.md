# Research decision log

Dates below record documentation of the author's current request on 2026-09-15,
not fabricated historical approval dates. Earlier selection provenance is unavailable.

| Recorded | Status | Decision | Rationale / evidence | Affected stages |
|---|---|---|---|---|
| 2026-09-15 | consolidated | Weather-only observed information through each origin | Author's stated research question | Protocol, features, models |
| 2026-09-15 | consolidated | 1951–2025 primary panel, 135 counties, 75 years, 10,125 observations | Author prioritizes usable temporal coverage and excludes 1950 yield because October 1949 weather is unavailable | Target, auxiliary data and configuration |
| 2026-09-15 | consolidated | Twelve monthly relative horizons, 12 through 1 | Author's replacement of earlier seasonal cutoffs | Horizon utility, features config |
| 2026-09-15 | consolidated / day open | October campaign reference; exact harvest day unset | Author specifies October Y-1 to October Y campaign | All forecast origins |
| 2026-09-15 | consolidated | Complete-year expanding-window validation | Prevent county-year temporal leakage | Splitter, future tuning |
| 2026-09-15 | open | Initial training length and first validation year | Author clarified that there is no fixed 60/40 split; previous 40-year default removed | Modeling config and splitter |
| 2026-09-15 | consolidated | All preprocessing and thresholds training-only | Anti-leakage requirement | Future features, detrending, tuning |
| 2026-09-15 | consolidated | Support RMSE, MAE and out-of-sample R²; no primary metric | Author has not selected a primary loss | Evaluation design |
| 2026-09-15 | consolidated | No acreage predictor, target or weighting | Acreage is limited to sample construction | NASS audit, models and evaluation |
| 2026-09-15 | open | Recover earlier acreage threshold and reference period | Author reports low-acreage exclusion; original rule and script not recovered | Sample provenance |
| 2026-09-15 | consolidated | Keep 1950 weather; exclude only 1950 yield | First retained campaign is October 1950–October 1951; feature lookback details remain open | Weather period and feature datasets |
| 2026-09-15 | open | Label publication timing and horizon-specific training availability | Early Y-1 origins may precede Y-1 yield publication | Future model fitting |
| 2026-09-15 | open | ERA5 observation/release availability interpretation | Retrospective daily dates alone do not prove real-time availability | Forecast claims |
| 2026-09-15 | open | Metric denominators, annual aggregation and bootstrap settings | Multiple metrics supported; implementation deferred | Evaluation |
| 2026-09-15 | open | Evaporative-demand sign and aggregation window details | Preserve raw ERA5 signs and avoid inventing derived behavior | Future feature engineering |
| 2026-09-15 | consolidated | Two content sections: output/ and work/ | Clean thesis project separate from all original data and internal operations; no duplicate code tree | Layout, manifest path map, delivery export |
| 2026-09-17 | consolidated | Thesis manuscript in English and LaTeX format | Conforms to international academic standards; formal departmental guidelines to be verified before final submission | output/thesis/ |
| 2026-09-17 | open | Past yield and previous-campaign weather lookback for early horizons | Because long lead times (H=12 down to H=8) precede the current crop year, models necessarily rely on past observed yield data; the exact number of antecedent weather months from the previous campaign to include remains an open decision | Feature engineering & model fitting |
| 2026-09-17 | consolidated | Evaluation hierarchy: accuracy, earliness, stability | Stated research priority: forecast must first achieve accuracy over baselines, then establish lead-time earliness, and finally demonstrate temporal/spatial stability | Protocol, evaluation design |
| 2026-09-17 | consolidated | Modeling target is detrended continuous yield anomaly | Observed raw yield is preserved as ground truth; model predicts detrended anomaly (fitted train-only) to isolate weather sensitivity; absolute yield level reconstructed post-hoc | Target, features, models |
| 2026-09-17 | consolidated | Regression and classification evaluated in parallel | Continuous anomaly regression paired with discrete regime classification to test whether regimes can be recognized earlier than point estimates | Modeling, evaluation |
| 2026-09-17 | consolidated | Detrending comparison: linear, polynomial, spline, rolling average | Sensitivity analysis across deterministic trend models fitted train-only; Fourier series treated as exploratory alternative, not primary baseline | Preprocessing, baselines |
| 2026-09-17 | consolidated | Classification thresholds and extreme quantiles fitted train-only | Strict anti-leakage requirement for regime boundary estimation | Splitter, classification targets |
| 2026-09-17 | consolidated | Scientific validity of null / negative results | Failure to detect stable incremental skill or finding that skill emerges only near harvest is a valid empirical result defining predictability limits | Research design, thesis conclusions |
| 2026-09-17 | consolidated | Explicit research hypotheses H1–H5 | Formulate testable hypotheses on incremental weather skill, lead-time dependence, 5/10/30-day temporal aggregation, model complexity, and early regime recognition | Research design, empirical evaluation |
| 2026-09-18 | consolidated | Systematic logging of all model progressions, ablations, and iterations | Author explicitly requested preserving performance metrics for every tested iteration (progressive feature additions, model variants from simple to complex, deep learning architectural stages, ablations) to enable reporting the complete iterative modeling journey in the thesis | Experiment tracking, modeling, thesis reporting |
| 2026-09-19 | consolidated | Deliberate chapter boundary separation (Chapters 3--6) | Ensure clear separation between data/EDA (Chap 3), methodology/detrending/split (Chap 4), out-of-sample results (Chap 5), and survivorship/operational discussions (Chap 6) | Manuscript architecture, work/thesis_notes/chapter_boundaries.md |
| 2026-09-19 | consolidated | Panel selection justified as within-county temporal trade-off | Literature (Schlenker & Roberts 2009; Khaki et al. 2020; Xie et al. 2025; Yin et al. 2026; Sweet et al. 2023) supports prioritizing 75-year within-county temporal depth and extreme events over broader geographic generalization; rejects naive spatial homogeneity assumption | Chapter 3, Sample provenance |
| 2026-09-19 | consolidated | R²=0.81 trend rationale and extreme shock years benchmark | High explanatory power of secular trend motivates residual anomaly forecasting; unexpected shock years (1988, 2012, 1974, 2003, 1993 and bumper years) cataloged as core testbed for Chapter 5 model stress-testing | Chapter 3 EDA, Chapter 5 evaluation, work/thesis_notes/extreme_years_benchmark.md |


## Reorganization implementation choices

Existing ERA5 algorithms were extracted into modules. Strict calendar/schema checks,
failures that stop invalid output, latest-record checksum semantics and isolated
preflight directories address verified implementation gaps. The area-intersection
formula, evaporation permutation and magnitude-guarded unit conversions are retained.
New output columns use descriptive English names; old artifacts remain unchanged.
These engineering corrections do not resolve the open scientific decisions above.
