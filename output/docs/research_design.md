# Research design

## Question and contribution

The thesis asks how early weather observed through a forecast origin adds stable,
statistically meaningful county soybean yield skill beyond historical trend and
climatology. It evaluates incremental information, without presuming superiority
of neural networks or any other model family.

The primary sample is the 1951–2025 balanced panel of 135 counties in six Midwest
states (10,125 observations). The author chose maximum available temporal coverage
without model comparison. Complete original exports remain separate. Acreage
contributes only to sample documentation, never prediction or weighting.

## Forecasting experiment

The active protocol uses twelve monthly horizons before a configurable harvest
reference (October confirmed, exact day still open). For target crop year Y and horizon H,
`forecast_origin = harvest_reference(Y) - H calendar months`.
`target_year` is the crop/response year, not necessarily the calendar year of the
weather or origin. H=12 always falls in Y-1. Calendar-month subtraction clamps an
invalid origin day to that month's last day; each origin is computed independently.
The exact harvest date itself must be valid and explicitly supplied.

For each H, build a separate feature dataset keyed by county and target year and
labelled with H and the origin. Only daily dates at or before that origin may enter
features. A `filter_weather_at_origin` result is an input subset, not a fitted
feature dataset. The associated validator rejects future weather. No future-season
completion is part of this design. Publication latency and target availability are
open operational constraints; calendar filtering alone does not establish them.

## Hypotheses to evaluate later

1. Observed weather can improve on training-only county climatology and trend.
2. The strength and stability of that improvement depend on forecast horizon.
3. Temporal aggregation and model family may affect skill; the direction is not assumed.

Use identical admissible information sets and temporal folds across model families.
Compare planned statistical models, Random Forest, Gradient Boosting and LSTM only
after the data pilots and feature implementation. No results are available yet.

See [protocol](research_protocol.md), [validation](validation_and_modeling.md),
[features](feature_engineering.md) and [references](references.md).
