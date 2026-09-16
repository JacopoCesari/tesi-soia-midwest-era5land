# Validation and planned modeling

## Implemented temporal splitter

`evaluation/expanding_window.py` yields positional `train_indices`, `test_indices`,
`train_years` and `test_year`. It accepts county-year records and keeps every complete
crop year indivisible. It rejects duplicate keys, missing or noninteger years,
calendar gaps, changing county membership and insufficient history.

`initial_training_years` is unset in `configs/modeling.yaml`; the splitter rejects
an unset value. The author has not chosen the first validation year and explicitly
rejects a fixed 60/40 assumption. Given an approved initial length N, the first
training interval would be 1951 through 1950+N, testing 1951+N, then expanding
annually through 2025, subject to horizon-specific information availability.

All 135 counties from a crop year remain together. Random county-year splits and
ordinary row-based k-fold cross-validation are not accepted alternatives.

## Train-only processing and availability

Detrending, scaling, feature selection, tuning and any later categorical thresholds
must be fitted inside training only. Inner tuning must use earlier complete years
inside the outer training window. Each monthly horizon gets a separate feature set
and model instance using weather no later than its forecast origin.

The fold utility groups target years; it does not assert when their yields were
published. For early origins in Y-1, including yield from Y-1 in training may violate
actual label availability. The author must settle release timing and any resulting
horizon-specific training lag before fitting models. No silent lag or change to the
provisional crop-year folds is implemented here. Historical ERA5 availability also
needs an explicit operational interpretation.

## Planned baselines and candidate models

- County yield climatology: historical mean estimated only from training observations.
- Historical technological trend: linear, polynomial or spline formulation to be settled.
- Parsimonious statistical/econometric benchmark, such as a linear model or GAM.
- Random Forest and Gradient Boosting (XGBoost/LightGBM).
- A sequential LSTM comparison with identical admissible information and folds.

No model is implemented or fitted in this phase. Do not assume deep learning is better.

## Evaluation specification (not implemented)

Support at least RMSE, MAE and out-of-sample R². RMSE is the square root of mean
squared prediction error; MAE is mean absolute prediction error. Out-of-sample R²
requires an explicit reference prediction/denominator convention, to be settled
before implementation. No primary metric is selected. Acreage weights are excluded.
Specify pooled versus year-averaged reporting and treatment of zero baseline loss
before calculating baseline-relative skill, `1 - loss_model / loss_baseline`.

Uncertainty is planned using whole-year block bootstrap, retaining all counties
within each sampled year. A 95% interval and earliest stable skill are methodological
proposals: identify the earliest horizon with positive, significant improvement
that persists at all later origins. Bootstrap settings, temporal dependence between
years, and multiplicity across horizons remain to be finalized; no confidence
intervals or claims of significance are produced by the current code.
