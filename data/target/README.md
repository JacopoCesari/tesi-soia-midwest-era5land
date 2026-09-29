# Thesis target

`soybean_yield_1951_2025.csv` contains the primary balanced panel: 135 counties,
75 years (1951–2025 inclusive), 10,125 observations, with no imputation.
Columns: `county_fips` (five-digit string), `year`, `yield_bu_per_acre`.
Read FIPS as text to preserve leading zeroes.

This is the observed yield target, not a weather-feature dataset. County metadata,
acreage, spatial weights and source hashes are separate in `data/auxiliary/`.
The complete supplied 479-county exports remain unchanged in `../work/data/external/`.
See `docs/data_and_target.md` for selection and `scripts/prepare_thesis_data.py`
for reproducible preparation from those local sources.
