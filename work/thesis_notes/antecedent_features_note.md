# Note: Antecedent Yield and Weather Lookback for Early Lead Times

*Recorded on 2026-09-17.*

## Context and Operational Constraint

The forecasting experiment establishes twelve relative monthly forecast origins ($H=12, \dots, 1$) prior to an October harvest reference ($Y$):
- Origins at long lead times ($H=12$ down to $H=8$) fall in the autumn and winter preceding the crop year (calendar year $Y-1$ or early $Y$, well before planting in May).
- At these early horizons, within-season weather for crop year $Y$ has not yet materialized.
- Consequently, any predictive model operating at these lead times must necessarily incorporate historical yield data (e.g., historical county yield levels, technological trend components, and antecedent yield performance) to establish a baseline.

## Open Operational Questions

1. **Weather lookback window:** How many months of weather from the preceding crop campaign or intervening winter should be included in feature datasets for early origins?
   - Candidates include: preceding autumn post-harvest weather (e.g., October–November $Y-1$), winter precipitation and snowpack recharge, or early spring soil moisture conditions.
2. **Label availability and publication lag:** When using antecedent yield observations (such as yield in year $Y-1$), the operational publication date of official USDA NASS county estimates must be accounted for to ensure strict out-of-sample validity (i.e., whether $Y-1$ county yield was officially published before the $H=12$ origin in October/November $Y-1$).

These specifications will be evaluated and formalized during the feature engineering and model evaluation phases.
