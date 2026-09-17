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
| 2026-09-17 | consolidated | Thesis manuscript in English and LaTeX format | Conforms to international academic standards and Unibo Vademecum (12pt, 1.5 line spacing, 2.5cm margins) | output/thesis/ |
| 2026-09-17 | open | Past yield and previous-campaign weather lookback for early horizons | Because long lead times (H=12 down to H=8) precede the current crop year, models necessarily rely on past observed yield data; the exact number of antecedent weather months from the previous campaign to include remains an open decision | Feature engineering & model fitting |

## Reorganization implementation choices

Existing ERA5 algorithms were extracted into modules. Strict calendar/schema checks,
failures that stop invalid output, latest-record checksum semantics and isolated
preflight directories address verified implementation gaps. The area-intersection
formula, evaporation permutation and magnitude-guarded unit conversions are retained.
New output columns use descriptive English names; old artifacts remain unchanged.
These engineering corrections do not resolve the open scientific decisions above.
