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
