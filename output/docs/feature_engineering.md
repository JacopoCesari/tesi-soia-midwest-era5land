# Planned feature engineering and implemented cutoff utilities

Only `features/forecast_horizons.py` is implemented. It constructs twelve origins,
returns separate labelled weather subsets, and rejects dates beyond the inclusive
daily UTC cutoff. It does not generate agronomic features or model datasets.

`configs/features.yaml` records October as the harvest month; the exact day and
feature lookback remain unset. Call `configured_forecast_horizons` only after supplying
the explicit day. Target years start in 1951, supported by weather from 1950.
Each month's origin is computed from the harvest reference independently, with
month-end clamping when needed. `target_year` identifies the crop year; weather and
`forecast_origin` may be in its previous calendar year. The weather date is a daily
UTC label, not an assertion about the hour at which a reanalysis file became available.

## Planned daily features

These formulas are inherited methodological specifications, not existing outputs:

- GDD: `max(0, (min(T_max, 30) + max(T_min, 10)) / 2 - 10)`, in °C days.
  Validate the agronomic clipping convention before implementing extreme cases.
- Daily VPD proxy: `max(0, e_s(T_mean) - e_s(T_dewpoint))`, with
  `e_s(T) = 0.61078 * exp(17.27*T / (T + 237.3))`, in kPa.
  This is a proxy from daily temperatures, not the average of hourly VPD.
- Heat days: indicators for maximum temperature >=30 °C and >=35 °C.
- Diurnal temperature range: maximum minus minimum air temperature.
- Rolling precipitation over the preceding 7, 14, 30, 60 and 90 days.
- Rolling water balance: precipitation minus a consistently signed evaporative
  demand measure. ERA5 evaporation retains its original sign in the pipeline;
  the demand/sign convention must be settled before implementing this feature.

## Planned temporal representations

Compare 5-day and 10-day windows, with 30-day windows as a lower-resolution benchmark.
A fixed 30-day window is not identical to a calendar month. Window anchors, lookback
start, partial-window treatment and early-year coverage remain explicit decisions.
Use means for appropriate state variables and totals/counts for accumulations and
event indicators, after specifying each variable's aggregation rule.

For every horizon, filter observations before any aggregation or learned processing.
All feature windows must end no later than that origin. Do not complete the future
season using observed weather, climatology or previous-year weather. Standardization
and selection must use training observations only. The twelve final feature datasets
are still to be built under `data/processed/model_datasets/`.
