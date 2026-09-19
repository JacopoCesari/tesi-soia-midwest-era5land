# Author Working Notes & Methodological Considerations
*Archived from Chapters 1 and 2 during supervisor review*

---

## 1. Forecast Cutoff Parameterization
- **Original provisional text (Chap 1):** *"The campaign is anchored to an October harvest reference, while the exact cutoff day remains an open methodological parameter."*
- **Operational decision:** Fixed at the final calendar day of each preceding month ($H=12, \dots, 1$) prior to the October harvest window.
- **Sensitivity note:** If evaluating administrative data publication latency (e.g., USDA NASS reports or ERA5-Land monthly updates), sensitivity tests can shift the cutoff from end-of-month (e.g. Day 30/31) to mid-month (Day 15) or test ERA5-Land real-time lag (approx. 5 days for ERA5T vs finalized ERA5-Land).

---

## 2. Modeling Target Priority (Regression vs. Classification)
- **Original provisional text (Chap 1):** *"Continuous regression and categorical classification are developed in parallel; the designation of a single primary modeling target will be settled after preliminary empirical experiments."*
- **Operational decision:** Both targets are formally preserved in the research design as dual complementary pillars:
  - **Primary Quantitative Target:** Continuous detrended yield anomaly (bushels/acre).
  - **Decision-Support Target:** Categorical regime classification (e.g. negative shortfall events below the 20th percentile / lower quartile) for supply chain risk alerts.

---

## 3. Methodological Positioning & Expanding Window
- **Original provisional text (Chap 2):** *"Section 2.9.4: Provisional Character of the Methodological Positioning... remain provisional. While the literature establishes clear theoretical motivations for each design choice, the definitive verification of these propositions depends entirely on the execution of the expanding-window experiments and statistical significance tests formalized in Chapter 4."*
- **Operational decision:** Removed from literature review chapter. The empirical validation results are reported objectively in Chapter 5, while Chapter 2 stands as a definitive, critical state-of-the-art review and hypothesis formulation.

---

## 4. Methodological Contributions Consolidation
- **Original list of 11 contributions (Chap 1):** Consolidated into 3 core academic contributions:
  1. Systematic Lead-Time Horizon Mapping ($H=12, \dots, 1$).
  2. Rigorous Decoupling of Historical Inertia and Weather Skill.
  3. Multi-Granularity Temporal and Architectural Comparison.
- **Good practices moved to Methodology (Chap 4):**
  - Expanding-window chronological validation (Hyndman).
  - Strict train-only fitting (detrending, scaling, quantile thresholds).
  - Standard error metrics (RMSE, MAE, ROC-AUC, Brier score).
