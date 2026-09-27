# Author Working Notes & Pending Tasks

> **Note on Project Rules:**  
> All permanent writing protocols, author voice instructions, zero-LLM marker constraints, and literature management rules are maintained exclusively in [`AGENTS.md`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/AGENTS.md).  
> This file tracks **strictly temporary working notes, open decisions, and pending tasks/re-runs** to be addressed in subsequent chapters or upon data updates.

---

## 1. Deferred Methodological Decisions (To Resolve in Chapter 5)

### A. Target Priority: Continuous Regression vs. Regime Classification
*   **Current State:** Both continuous yield anomalies and 3-class categorical regimes (shortfall, normal, bumper) are introduced in Chapter 1 and formalized in Chapter 4.
*   **Pending Action:** Once out-of-sample empirical runs are completed in Chapter 5, assess whether regime classification provides distinct operational signal or earlier lead-time skill. If redundant, drop or streamline to a secondary ablation before final thesis delivery.

### B. Table 3.4 (Historical Yield Shocks and Macro-Climatic Benchmarks)
*   **Current State:** Positioned provisionally in Chapter 3 (`tab_yield_shocks.tex`) listing the major historical shortfall ($\le -10\%$) and bumper ($\ge +10\%$) cohorts.
*   **Pending Action:** Evaluate in Chapter 5 whether to retain this table in Chapter 3 or relocate/merge it directly into Chapter 5, where candidate models are stress-tested against these exact historical shock cohorts.

---

## 2. Pending Script & Figure Re-Runs (Upon 1965–1990 ERA5-Land Extraction)

*   **Context:** Daily county-aggregated ERA5-Land parquets currently cover 50 years (1950–1964 and 1991–2025). The intermediate block (1965–1990) is pending completion of the CDS extraction pipeline.
*   **Current Temporary Benchmarks in Chapter 3:**
    *   Figures 3.7 & 3.8 (cumulative & dynamic intra-seasonal curves) plot post-1990 benchmark years: Shocks = 2003 (-21.6%), 2012 (-11.7%), 1993 (-10.5%); Bumpers = 2021 (+10.8%), 2016 (+10.8%), 1994 (+10.7%).
    *   Figure 3.9 (spatial 2x2 contrast) maps 2012 (drought) vs. 2021 (bumper).
*   **Re-Run Checklist When 1965–1990 Data is Available:**
    1.  *Figure 3.9 (Spatial Shock Comparison):* Re-run `work/reports/exploratory_analysis/generate_spatial_shock_comparison_figure.py` to spotlight **1988** (the absolute historical drought benchmark at -25.6%) against **2021** (bumper, +10.8%).
    2.  *Figures 3.7 & 3.8 (Intra-Seasonal Trajectories):* Re-run `work/reports/exploratory_analysis/generate_chapter3_weather_artifacts.py` to incorporate **1988** (compound thermal-drought desiccation) and **1974** (spring planting delay followed by early Arctic freeze truncation).
    3.  *Text Synchronization:* Update Section 3.4.2 in `output/thesis/chapters/03_data_and_study_area.tex` to reference 1988 and 1974 directly.

---

## 3. Pending Literature Deployments for Chapter 4 (Methodology)

The primary bibliography is strictly closed to the 13 papers in `Literature.zip`. Ten papers are cited in Chapters 1–3. The remaining three papers will be introduced naturally in Chapter 4:
1.  **`Torsoni et al. (2023)`**: To be cited in Section 4.4 when formalizing Random Forest and Gradient Boosting hyperparameter grids for soybean yield.
2.  **`Shook et al. (2021)`**: To be cited in Section 4.5 when detailing the recurrent deep learning architecture (LSTM) sequence processing.
3.  **`Xie, Huang & Meng (2025)`**: To be cited in Section 4.2 when formalizing the sub-monthly (5-day and 10-day) rolling feature engineering window.
