# Author Working Notes & Methodological Considerations
*Archived from Chapters 1 and 2 during supervisor review*

---

## 1. Forecast Cutoff Parameterization
- **Original provisional text (Chap 1):** *"The campaign is anchored to an October harvest reference, while the exact cutoff day remains an open methodological parameter."*
- **Operational decision:** Fixed at the final calendar day of each preceding month ($H=12, \dots, 1$) prior to the October harvest window.
- **Sensitivity note:** If evaluating administrative data publication latency (e.g., USDA NASS reports or ERA5-Land monthly updates), sensitivity tests can shift the cutoff from end-of-month (e.g. Day 30/31) to mid-month (Day 15) or test ERA5-Land real-time lag (approx. 5 days for ERA5T vs finalized ERA5-Land).

---

## 2. Modeling Target Priority (Regression vs. Classification)
- **Status (Chap 1):** Continuous regression is established as the primary empirical backbone, while categorical classification (shortfall, normal, bumper regimes) is retained as an exploratory/complementary framework.
- **Operational decision (Author Review, Sept 2026):** Keep both in Chapter 1 for now. A final evaluation will be made once full empirical runs are analyzed: if classification provides distinct operational signal, retain it as an ablation/secondary analysis; if redundant, drop or streamline before final thesis delivery.

---

## 3. Methodological Positioning & Expanding Window
- **Original provisional text (Chap 2):** *"Section 2.9.4: Provisional Character of the Methodological Positioning... remain provisional. While the literature establishes clear theoretical motivations for each design choice, the definitive verification of these propositions depends entirely on the execution of the expanding-window experiments and statistical significance tests formalized in Chapter 4."*
- **Operational decision:** Removed from literature review chapter. The empirical validation results are reported objectively in Chapter 5, while Chapter 2 stands as a definitive, critical state-of-the-art review and hypothesis formulation.

---

## 4. Methodological Contributions Consolidation
- **Chapter 1 Streamlining (Author Review, Sept 2026):** Consolidated into 2 core research contributions:
  1. Systematic 12-Month Lead-Time Mapping Without Forecast Inputs ($H=12, \dots, 1$).
  2. Identification of Optimal Model-Aggregation Configurations (winning setup across 5-day, 10-day, and 30-day summaries and model families).
- **Foundational Modeling Hygiene moved strictly to Methodology (Chap 4):**
  - Baseline decoupling (detrending and historical inertia isolation) is recognized as foundational data-science hygiene, not an independent headline contribution.
  - Expanding-window chronological validation (Hyndman).
  - Strict train-only fitting (detrending, scaling, quantile thresholds).
  - Standard error metrics (RMSE, MAE, ROC-AUC, Brier score).

---

## 5. Table 3.4 (Historical Yield Shocks and Macro-Climatic Benchmarks) Evaluation Status
- **Status (Author Review, Sept 2026):** Marked for evaluation. To be revisited upon completion of empirical experiments in Chapter 5.
- **Rationale:** The table compiles historical benchmark shock years (severe shortfall years $\le -10\%$ and bumper harvest years $\ge +10\%$) along with their meteorological drivers. The author noted skepticism regarding whether this specific tabular format is necessary in Chapter 3 or should be restructured, simplified, or moved/integrated directly into Chapter 5 where out-of-sample stress-testing of candidate model predictions across historical shock cohorts actually takes place.
- **Operational decision:** Retain provisionally in Chapter 3 with clean, unambiguous column headers and grouped cohort narrative; make a final decision on whether to keep, streamline, or relocate to Chapter 5 once full out-of-sample empirical results are available.

---

## 6. Chapter 3 Structural Principles & Panel Selection Logic
- **Two-stage county screening logic:** Filter 1 (agronomic significance and historical production stability, avoiding spatial scale-mismatch noise) + Filter 2 (complete 75-year continuity 1951–2025 under strict zero imputation).
- **Temporal depth vs. geographic expansion trade-off:** Prioritizing multidecadal depth (75 consecutive years capturing macro-climatic historical extremes) over sheer county count; 135 counties provide an even, uniform geographic distribution across the Corn and Soybean Belt.
- **1950 target yield boundary exclusion:** ERA5-Land begins Jan 1, 1950, so the Oct 31, 1949 lookback for $H=12$ is unavailable. 1950 weather is preserved to construct antecedent features for the 1951 campaign.
- **Auxiliary role of harvested acreage:** Documented strictly for initial screening and historical acreage representation; never used as a predictor, target, or sample weight.
- **Grouped cohort analysis for extremes:** Meteorological extremes and yield anomalies are analyzed through archetypal multi-event cohorts across the 75-year record (severe drought/heat clusters, pluvial/anoxia regimes, planting delay/frost truncation, and bumper harvest clusters) rather than isolated single-year anecdotes.
- **Demarcation between Chapter 3 and Chapter 4:** Chapter 3 covers empirical data provenance, basic spatial/temporal aggregation, and exploratory analysis. All predictive feature engineering (12 lead times $H=12, \dots, 1$, multi-granularity 5/10/30-day aggregations, lag windows), machine learning architectures, and expanding-window validation protocols belong strictly to Chapter 4.


