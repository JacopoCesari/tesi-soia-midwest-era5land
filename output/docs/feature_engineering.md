# Model features and forecast cutoffs

The maintained replication implementation is `scripts/03_build_feature_matrices.py`
for calendar-month tabular inputs and `scripts/06_train_lstm_model.py` for recurrent
windows. The generic cutoff utility under `src/` is separate from these fitted datasets.

## Predictor dictionary

The models use 17 weather indicators: eight summed exposures (`P`, `SSRD`, `ET0`,
`P_minus_ET0`, `GDD`, `HD30`, `HD35`, `EDD30`) and nine averaged states (`T_mean`,
`T_max`, `T_min`, `T_dew`, `SM1`, `SM2`, `SM3`, `VPD`, `SM_root`).
`HD30` and `HD35` are binary daily indicators at inclusive maximum-temperature
thresholds of 30 and 35 degrees C. `EDD30 = max(T_max - 30, 0)` measures excess
heat intensity in degree-days. Root-zone moisture is `0.07*SM1 + 0.21*SM2 + 0.72*SM3`.
The daily acquisition schema contains auxiliary fields beyond this model subset.

Two normalized coordinates supplement the weather inputs. Recurrent sequences have
19 channels per step. The lagged detrended yield enters the integrated prediction
head separately. Attribution uses 19 groups: 17 weather indicators, one lagged-yield
group, and one jointly permuted coordinate group.

## Campaign and windows

A crop-year campaign runs from November 1 of Y-1 to October 31 of Y, retaining all
365 or 366 dates. H=12 is the October 31 Y-1 zero-anomaly reference; fitted horizons
H=11 through H=1 end in December Y-1 through October Y. The number of elapsed
calendar months is `13-H`; the earliest fitted horizon includes November and December.
Calendar-month matrices therefore have `2 + 17*(13-H)` predictors, plus one lag in
the integrated specification: 206 weather/spatial predictors at H=1, or 207 with lag.

Recurrent windows are consecutive 5-, 10- or 30-day blocks anchored on November 1.
They cross month boundaries and retain an incomplete last block at every cutoff.
Only available dates enter that last block. Leap days are retained. A complete
10-day campaign has 37 steps, with five or six days in its final step. Horizon
arrays are constructed separately: slicing a full-season block cannot implement an
earlier partial-window cutoff. Any trailing tensor padding is excluded from
training scaling and attention and does not represent imputed weather.

The author retains the selected 10-day frequency and head-specific hyperparameters.
The current refresh fits only 2011--2025 test folds; the 1985--1995 selection records
are read-only. Each test fold retains the existing three-year inner early-stopping
split. Missing or duplicate daily county records stop sequence construction.

Regional harvest closure uses September forecasts for Minnesota and October for
other states. No aggregation uses weather beyond its forecast cutoff. Detrending
and standardization use historical training data only.
