# Research design

## Question and core objective

The core objective is to determine whether adding observed high-resolution weather
information to historical yield baselines improves out-of-sample forecasts of
detrended county-level soybean yield anomalies, and how early and consistently this
incremental skill emerges before harvest.

The research objectives follow an explicit hierarchy:
1. **Predictive Accuracy:** The forecast must first achieve meaningful out-of-sample
   accuracy relative to historical baselines.
2. **Lead-Time Earliness:** The design identifies how early prior to harvest this
   predictive advantage emerges.
3. **Temporal and Spatial Stability:** The detected gain must remain statistically
   stable and robust across subsequent forecast origins, years, and counties.

Failure to detect stable incremental skill, or evidence that skill emerges only
immediately prior to harvest, constitutes an equally valid empirical result that
defines the empirical boundaries of weather-based predictability under strict out-of-sample
validation.

The primary sample is the 1951–2025 balanced panel of 135 counties in six Midwest
states (10,125 observations), prioritizing temporal depth without artificial imputation.
Acreage contributes solely to sample documentation, never as a predictor or weight.

## Model taxonomy and incremental skill

To isolate the genuine contribution of meteorological dynamics, candidate architectures
are organized into three distinct configurations:
1. **Historical baselines:** Informed exclusively by historical yield time series,
   including deterministic technological trends, county climatological averages, and
   admissible antecedent yields (respecting label publication lags).
2. **Weather-only models:** Ablation benchmarks trained exclusively on surface weather
   features observed up to each cutoff.
3. **Historical + weather models:** Integrated models combining historical baselines
   with observed current-season weather features up to the forecast origin.

The genuine value of weather is measured through the out-of-sample comparison between
the integrated models (3) and the historical baselines (1).

## Forecasting experiment

The active protocol uses twelve monthly horizons before a configurable harvest
reference (October confirmed, exact day still open). For target crop year Y and horizon H ($H \in \{12, \dots, 1\}$),
`forecast_origin = harvest_reference(Y) - H calendar months`.
`target_year` is the crop/response year, not necessarily the calendar year of the
weather or origin. $H=12$ through $H=8$ fall in calendar year $Y-1$, while $H=7$ through $H=1$
fall in year $Y$. With $H=1$, the forecast is generated one month before the harvest
reference, ensuring all origins strictly precede harvest.

For each H, a separate feature dataset is constructed keyed by county and target year.
Only daily dates at or before that origin may enter features. A `filter_weather_at_origin`
result is an input subset, not a fitted feature dataset. The associated validator rejects
future weather. No future-season completion or seasonal climate forecasts are permitted.

## Expanded research hypotheses

1. **H1 — Incremental Weather Skill:** Adding observed high-resolution weather features
   to historical yield baselines yields statistically significant reductions in out-of-sample
   prediction error for detrended yield anomalies.
2. **H2 — Lead-Time Dependence and Persistence:** Predictive skill increases monotonically
   as the origin approaches harvest ($H \to 1$). Early predictive advantages must persist
   across subsequent monthly cutoffs without reverting to noise to be considered stable.
3. **H3 — Temporal Aggregation and Derived Stress Features:** High-frequency temporal
   windows (5-day and 10-day aggregations) and agronomically derived non-linear stress
   metrics (e.g., Extreme Degree Days, vapor pressure deficit) capture yield-limiting
   extremes more effectively than coarse 30-day monthly means.
4. **H4 — Model Complexity and Non-Linearity:** Complex sequential architectures (such as
   LSTMs) do not systematically outperform regularized linear models or tree ensembles
   (Random Forest, XGBoost) under strict leak-free temporal validation.
5. **H5 — Early Regime Recognition vs Continuous Accuracy:** Categorical yield regimes
   (e.g., severe shortfalls vs normal/bumper crops) can be classified with positive skill
   at earlier lead times than required for accurate continuous anomaly estimation.
6. **Sensitivity to Detrending Specification:** While the technological trend must be
   removed strictly train-only, results should remain robust across linear, polynomial,
   spline, and rolling-average specifications. Fourier representations are treated as
   exploratory sensitivity checks rather than primary baselines.

See [protocol](research_protocol.md), [validation](validation_and_modeling.md),
[data and target](data_and_target.md), and [references](references.md).

