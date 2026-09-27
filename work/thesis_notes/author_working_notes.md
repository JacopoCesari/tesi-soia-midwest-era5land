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
