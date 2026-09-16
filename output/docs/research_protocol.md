# Consolidated research protocol

This is the concise source of truth for the author's consolidated instructions,
recorded on 2026-09-15. The date records receipt, not an invented earlier approval.

- **Question:** How early can county soybean yield be forecast from observed weather,
  with statistically meaningful and stable improvement over historical-trend and
  climatology baselines?
- **Target:** Annual county soybean yield (`yield_bu_per_acre`), or a yield anomaly
  after training-only detrending. No acreage predictor or weighting.
- **Primary sample:** Illinois, Indiana, Iowa, Minnesota, Missouri and Ohio; 135
  counties; 1951–2025 inclusive; 75 years; 10,125 complete county-year measurements.
  The author prioritizes temporal coverage without model-comparison tests.
- **Source preservation:** The full 479-county exports remain unchanged and separate.
  Their 1950–2025 coverage is incomplete. No automatic imputation. Weather production
  targets weather years 1950–2025 to support target years 1951–2025.
- **Information set:** Weather only, observed on or before each origin. No satellite,
  market, seasonal forecast or future observed/climatological/previous-year completion.
- **Horizons:** 12 through 1 calendar months before an explicitly configured harvest
  reference. October is author-confirmed; the exact day remains **open**. Each horizon has separate
  cutoff-specific features and a separate model or model instance.
- **Validation:** Expanding-window, rolling-origin validation with indivisible crop
  years across all counties. Initial training length remains open; no fixed 60/40
  split or implicit 40-year default. Test years expand through 2025 once configured.
- **Training-only operations:** Detrending, scaling, selection, tuning and any later
  classification thresholds. No random county-year cross-validation.
- **Evaluation:** Support RMSE, MAE, out-of-sample R² and optional baseline-relative
  skill. No single primary metric selected; no acreage evaluation weighting.
- **Uncertainty:** Whole-year block bootstrap and earliest stable skill are planned.
  No model family's superiority is assumed.

## Open decisions and implementation boundaries

The harvest date, feature lookback start, metric aggregation/conventions, and target
publication dates are unresolved. A 12-month origin for crop year Y is in Y-1;
therefore a provisional fold containing Y-1 yield may contain a label unavailable
at that origin. The splitter is a crop-year grouping utility, not a claim that such
labels were already published. Resolve that availability rule before model fitting.

The author explicitly excludes the 1950 yield target: its October 1949–October
1950 campaign lacks October–December 1949 weather. The first retained campaign is
October 1950–October 1951. Keep 1950 weather while excluding 1950 yield. Do not
impute missing pre-1950 weather. The exact harvest day and feature lookback remain open.

Acres-harvested continuity is verified for the selected sample. The author reports
an earlier acreage exclusion, but its threshold and reference period are unrecovered. See
[data and target](data_and_target.md) and the [decision log](decisions.md).
