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
## 7. Data Science Identity Boundary & Evidence-Based Commentary (Author Rule, Sept 2026)
- **Role and voice:** The narrative voice is strictly that of applied data scientists and economists, NOT agronomists, plant biologists, or synoptic meteorologists.
- **Evidence-based claims:** All interpretations and comments regarding climatic effects or historical shocks must be strictly grounded in:
  1. Direct evidence from our own data, tables, and figures (e.g. observed values of $VPD$, precipitation, soil moisture, temperatures).
  2. Direct citations from the approved reference papers in the repository catalog (`output/docs/references.md`).
- **No external speculation or unwarranted domain digressions:** Avoid unverified domain conjectures (e.g. specific biological diseases, atmospheric jet stream/blocking dynamics, entomological outbreaks) that are not directly captured in our datasets or in our approved literature.
- **Strict citation perimeter:** Do not search for or add papers by external agronomists, biologists, or meteorologists outside the curated repository catalog.

---

## 8. Approved Spatial Multi-Panel Visualization Pattern (Faceted Comparative Maps)
- **Author Directive (Author Review, Sept 2026):** Implement high-impact, multi-panel comparative spatial maps ("Small Multiples" / faceted choropleths) accompanied by bottom diagnostic summaries (donut charts, regional share bars, or state-level distributions), inspired by global agricultural trade and acreage distribution layouts.
- **Approved Comparative Archetypes:**
  1. **Archetype A: Temporal Progression of the Same Variable (`stessa variabile in diversi periodi/anni/mesi`)**:
     - *Cross-Year Benchmark Comparison:* Stacked or faceted Midwest maps comparing spatial shock footprints across archetypal historical shock years (e.g. 1988 severe drought vs. 2012 flash drought vs. 1993 Midwestern pluvial vs. 1994/2016 record bumper harvests) using identical, synchronized colorbars.
     - *Intra-Seasonal Phenological Progression:* Month-by-month spatial propagation of soil-atmospheric stress through critical soybean phenological windows (June vegetative $H=4 \rightarrow$ July flowering $H=3 \rightarrow$ August pod fill $H=2 \rightarrow$ September maturation $H=1$).
  2. **Archetype B: Multi-Variable Compound Stress Footprint (`diverse variabili nello stesso periodo`)**:
     - Synchronized multi-panel spatial mapping of compound atmospheric-hydrological drivers during the critical July–August seed-filling window:
       1. Thermal stress: Extreme heat days ($HD_{30}$, days with $T_{\text{max}} \ge 30^\circ\text{C}$).
       2. Atmospheric evaporative demand: Vapor pressure deficit ($VPD$, kPa).
       3. Climatic water balance: Cumulative net water deficit ($P - ET_0$, mm).
       4. Hydrological memory: Root-zone volumetric soil moisture ($SM_{\text{root}}$, $0$--$100\text{ cm}$).
  3. **Bottom Diagnostic / Infographic Layer:**
     - Coordinated sub-cards or donut/bar distributions underneath the maps quantifying regional shares (e.g., percentage of study acreage under severe stress regimes, or state-by-state contrast) to give immediate, high-level executive readability.
- **Target Chapters:** Chapter 3 (Section 3.4 Climatology & Stress Footprint) and Chapter 5 (Out-of-sample shock ablation and historical cohort stress-testing).

---

## 9. Overarching Editorial Strategy: "Parlare con le Immagini" (Visual Information Density)
- **Institutional Constraint vs. Opportunity:**
  - Strict university text limit: Max 100,000 characters (spaces included) $\approx$ 35 pages of prose.
  - Zero limit on figures, tables, formulas, and visual diagnostics (all strictly excluded from the character count).
- **Core Directive:** Prioritize rich, self-contained visual artifacts (multi-panel maps, diagnostic infographics, structural badges, workflow flowcharts) to carry the substantive empirical and methodological depth, allowing the prose to remain lean, dense, and non-proliferative.
- **Operational Rules for Writing & Plotting:**
  1. *Self-Contained Figures:* Every figure must feature descriptive sub-titles, explicit keys, clean mathematical callout badges, and unambiguous visual hierarchies so it can be understood in seconds without reading paragraphs of prose.
  2. *Lean Surrounding Narrative:* Text does not describe visual details mechanically (e.g., "the blue bar is at 49.7%"). Instead, prose focuses exclusively on high-level economic/agronomic interpretation, supply-chain implications, and methodological synthesis.
  3. *Maximum Content per Character:* Whenever a complex mechanism arises (spatial aggregation, expanding-window validation, compound stress indexation, lead-time horizon progression), design an elegant figure rather than writing multiple explanatory paragraphs.

---

## 10. Chapter 3 Revision Roadmap: Visual-First Overhaul (+Images, Essential Tables, Less Prose)
- **Author Directive (Author Review, Sept 2026):** Revisit and restructure the entire Chapter 3 (`03_data_and_study_area.tex`) following the "parlare con le immagini" philosophy:
  1. **Visual Primacy:** Prioritize clean, self-contained figures and multi-panel infographics over descriptive textual narrative.
  2. **Selective, Essential Tables:** Include tables strictly when necessary and highly diagnostic. Avoid table redundancy or bloat.
  3. **Textual Compression:** Drastically compress textual explanations; replace prose descriptions of empirical facts with direct figure/table cross-references and high-level analytical commentary.
- **Specific Action - Re-integrate Detrending Comparison Table (`tab_detrending_comparison.tex`):**
  - *Context:* The diagnostic table comparing detrending models across the 75-year panel (Linear OLS vs. Quadratic vs. Cubic vs. Splines vs. HP filter, evaluating $R^2$, AIC, BIC, ADF unit-root $p$-values, and Out-of-Sample expanding-window RMSE on $6{,}075$ county-year evaluations) was identified as a valuable asset ("era carina").
  - *Strategic Role:* Re-inserting this table alongside `fig_yield_trajectory.pdf` eliminates multiple paragraphs of dry econometric text explaining the detrending choice. The table immediately and objectively proves to the reader why Linear OLS is chosen (lowest OOS RMSE of 5.889 bu/ac, minimizing Runge boundary oscillations).
- **Section-by-Section Visual Action Plan for Chapter 3:**
  - *Section 3.1 & 3.2 (Study Domain & Panel Assembly):* Keep `fig_study_area_map.pdf` and `tab_panel_selection.tex`. Trim narrative prose on historical selection filters.
  - *Section 3.3 (Yield Trajectories & Detrending):* Anchor around `fig_yield_trajectory.pdf`, `fig_state_yield_distributions.pdf`, `fig_dispersion_dynamics.pdf`, and re-integrated `tab_detrending_comparison.tex`. Slash text discussing polynomial degrees.
  - *Section 3.4 (ERA5-Land Grid & Spatial Architecture):* Anchored by `fig_era5_spatial_structure.pdf` (already optimized and lean, 43 words of text).
  - *Section 3.5 & 3.6 (Temporal Structure & Derived Indicators):* Keep `tab_weather_variables.tex` and mathematical definitions.
  - *Section 3.7 & 3.8 (Agro-Climatic EDA & Weather-Yield Sensitivity):* Consolidate the cumulative vs. non-cumulative narrative. Prepare the implementation of the faceted multi-panel spatial comparison ("Small Multiples" of historical shock footprints or compound variables with bottom diagnostic share charts, as defined in Section 8). Eliminate text redundancy between sections.


