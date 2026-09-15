# Publishable auxiliary data

- `counties.csv`: selected FIPS, state and county names (135 rows).
- `acres_harvested_1951_2025.csv`: acreage for the selected county-year keys;
  sample documentation only, never a predictor or observation/evaluation weight.
- `spatial_weights.csv`: 3,344 county-grid intersections for the selected counties,
  taken unchanged from the supplied matrix, with canonical latitude/longitude names.
  Weights sum to one within each county; county removal does not require renormalization.
- `panel_provenance.json`: author decision, period, counts and source/output SHA-256 hashes.

The modeling target lives in `data/target/`. Original sources and original weights
remain separate and byte-identical. No acreage threshold was added to this selection.
