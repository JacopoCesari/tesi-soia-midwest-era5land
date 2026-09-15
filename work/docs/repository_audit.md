# Repository audit and migration record

This records the earlier reorganization, before the author's subsequent 135-county
1950–2025 selection. Its inventories, paths and test counts describe that stage.
Current layout and commands are in `README.md` and `operations/README.md`;
current selection and source hashes are in `docs/data_and_target.md` and
`data/auxiliary/panel_provenance.json`. Original source bytes remain preserved.

Recorded on 2026-09-15. Work is on `codex/research-project-reorganization`.
The initial tracked worktree was clean and `origin` was fetched before changes.
No commit, push, merge, CDS data download, feature generation, EDA or model fitting
was performed. Installing the local package and Ruff fetched software only.

## Initial audit

- Two root scripts mixed download, response handling, manifest, QC, unit conversion,
  aggregation and CLI orchestration. The original suite had seven regression tests.
- Active documentation was predominantly Italian, used a few seasonal cutoffs, and
  overstated the completeness of preflight and historical pipeline validation.
- Both NASS primary sheets contain 29,219 complete measurement records. The
  workbooks state continuous yield and acreage as a necessary selection condition.
  No original selection script or reproducible acreage threshold was found.
- The supplied preflight consists of eight seven-day NetCDF files. Its reduced
  channel A inventory and same-day boundary request yield six aligned days and 21
  county weather fields in each preserved output, not an annual 37-field dataset.
- The original annual channel B response includes one extra boundary date; applying
  an ordinary 365/366-date validator directly could incorrectly mark it failed.
- Original calendar checks counted rows without checking exact dates. Aggregation
  intersected available days, warned after QC failure and could write invalid output.
  A missing checksum could pass optional verification, and an older PASS could mask
  a newer FAIL. Offline aggregation unnecessarily instantiated a CDS client.
- Dependency declarations mixed basic processing with PyTorch and had two independent
  constraint sets. Raw local preflight samples were ignored rather than tracked.

## Delivered behavior

The package separates data ingestion, ERA5 responsibilities, twelve monthly origins
and complete-year temporal splits. Source algorithms for area weighting, the
evaporation permutation and unit conversions are retained. Numerical regressions
compare both stored county samples; spatial recomputation matches all 11,953 weights.

Explicit date/schema validation, rejection of failed outputs and missing weighted
observations, correctly aligned new preflight dates, per-file manifest updates,
checksum failure handling and lazy client construction address the observed defects.
New output field names are descriptive English; preserved files keep source names.
New downloads and outputs are separate from supplied artifacts.

## Migration table

The table deliberately records historical paths. They are not operational paths.
Every listed destination was checked against a SHA-256 captured before migration;
the exact 40-file mapping and hashes are in `data/external/file_migration.json`.
No original data or PDF file was deleted or rewritten.

| Old location | New location | Contents |
|---|---|---|
| `01_Dati_Soia_e_Target/` | `data/external/usda_nass_and_county_boundaries/` | Workbooks, candidate CSV, Census geometry |
| Root `balanced_panel_candidate_counties.csv` | `data/external/supplied_candidate_counties.csv` | Existing identical supplied CSV retained |
| `era5_land_daily/raw_daily_stats/` | `data/raw/era5_land/supplied_preflight/raw_daily_stats/` | Six original NetCDF files |
| `era5_land_daily/raw_accumulated_boundaries/` | `data/raw/era5_land/supplied_preflight/raw_accumulated_boundaries/` | Two original NetCDF files |
| `era5_land_daily/county_daily/` | `data/interim/county_daily_weather/supplied_preflight/` | Two original Parquet files |
| `era5_land_daily/spatial_weights_479_counties.*` | `data/interim/spatial_weights/` | CSV and Parquet weights |
| `era5_land_daily/manifest.csv` | `data/raw/era5_land/manifest.csv` | Immutable original record |
| `03_Documentazione_e_Paper/Papers/` | `references/papers/` | All 15 scientific PDFs |
| `Vademecum.pdf` | `thesis/university_thesis_guide.pdf` | Original university guide |
| Root ERA5 Python scripts | `src/soybean_yield_forecasting/era5/` and `scripts/` | Extracted implementation and thin entry points |
| `requirements.txt` | `pyproject.toml` and `requirements/` | One authoritative dependency declaration with layered selectors |

The eight small, already-present NetCDF fixtures are now explicitly eligible for
tracking. `.gitignore` still excludes other NetCDF downloads and now ignores new
production/preflight directories, model artifacts, caches and generated report
subdirectories. It permits source documentation and selected final figures/tables.
The two existing county Parquet samples remain available as regression fixtures.

## Preserved sources and deliberate exceptions

All 40 original data/PDF files retain their bytes, including workbook source-language
README cells, Census XML metadata and the university guide. Source metadata language
cannot be translated without violating the explicit integrity requirement; active
repository documentation is English. Original duplicate candidate CSVs are both
retained because deleting original CSV data was prohibited. Existing ignored caches
and the user's credentials/configuration were not changed.

## Verification

- Baseline: **7 passed**. Initial sandbox execution showed 6 passed/1 temporary-file
  permission failure; all seven passed outside the Windows sandbox before refactoring.
- Final suite: **50 passed**, using no live CDS credentials or network responses.
- Mocked ordinary/leap years validate requests, the next-year boundary, all 37
  fields, daily output, manifest/checksum behavior and resume. A mocked new preflight
  covers seven aligned days; real supplied outputs remain numerically unchanged.
- All 40 original files pass SHA-256 verification. Both weight formats agree.
  Recomputed weights have a maximum absolute difference of **0.0** from the original.
- Offline supplied preflight and all historical checksums pass. A JSON QC summary
  is available under `reports/experiments/quality_control/preflight.json` after running
  the documented command (generated and ignored).
- All five script `--help` entry points succeed. All 20 package submodules import
  from the repository root after editable installation. No model dependencies are
  required by the base package.
- Ruff lint and formatting checks pass. Git whitespace checks pass. Operational
  source, configuration, tests and documentation contain no obsolete path references;
  historical paths remain only in this audit, migration metadata and original manifest.
- The seven original assertions were retained in `tests/test_pipeline.py`, with
  English text, package imports and migrated artifact paths.

The installed NetCDF stack emits one NumPy binary-size compatibility warning on
first load, and openpyxl 3.1.2 emits six UTC deprecation warnings. The tests pass;
these upstream environment warnings were not suppressed or treated as proof of
production readiness. During identifier cleanup, tests caught an accidental change
to the standard `Path.stat()` method; it was corrected and the complete suite rerun.

### Environment actually tested

Python 3.12.8; NumPy 1.26.3; pandas 2.2.3; SciPy 1.16.1; xarray 2026.7.0;
netCDF4 1.7.4; pyarrow 24.0.0; cdsapi 0.7.7; geopandas 1.1.4; Shapely 2.1.2;
openpyxl 3.1.2; PyYAML 6.0.1; pytest 9.0.3; Ruff 0.16.7.
The local validation venv reused the machine's installed scientific dependencies.
A fresh dependency solve on another machine has not been tested; package ranges
are authoritative installation constraints, not a fully locked environment.

## Remaining decisions and limits

Harvest month/day is unset. Feature lookback, early 1950 origins requiring 1949
weather, target publication timing at Y-1 origins, and ERA5 real-time availability
need explicit decisions before models. No acreage threshold is inferred from the
provided medians. Metric denominators/aggregation, bootstrap details, derived water
balance signs and feature-window conventions remain open.

Magnitude-guarded unit conversion is preserved, including its limitations for small
energy values. Snow-cover units are not verified by the reduced supplied samples.
Live full-field preflight, ordinary/leap-year CDS pilots, server request limits and
full-domain memory usage remain unverified. Follow the README pilot commands and
acceptance checklist before authorizing the historical run.

## Final directory tree

Generated environments, caches and scratch validation outputs are omitted.
Folders for unimplemented stages contain purpose READMEs, not empty Python modules.

```text
.
├── README.md, AGENTS.md, pyproject.toml, .gitignore
├── configs/ {data, features, modeling}.yaml
├── requirements/ {base, geospatial, machine_learning, deep_learning, development}.txt
├── data/
│   ├── external/
│   │   ├── file_migration.json
│   │   ├── supplied_candidate_counties.csv
│   │   └── usda_nass_and_county_boundaries/ (2 Excel, candidate CSV, Census boundaries)
│   ├── raw/era5_land/
│   │   ├── manifest.csv
│   │   └── supplied_preflight/ {raw_daily_stats, raw_accumulated_boundaries}/
│   ├── interim/
│   │   ├── spatial_weights/ (CSV and Parquet)
│   │   └── county_daily_weather/supplied_preflight/ (2 Parquet files)
│   └── processed/model_datasets/README.md
├── src/soybean_yield_forecasting/
│   ├── __init__.py, configuration.py
│   ├── data/ {__init__, nass, panel}.py
│   ├── era5/
│   │   ├── __init__.py, variables.py, schema.py, file_io.py
│   │   ├── download.py, manifest.py, quality_control.py, conversions.py
│   │   └── spatial_weights.py, county_aggregation.py, pipeline.py, cli.py
│   ├── features/ {__init__, forecast_horizons}.py
│   └── evaluation/ {__init__, expanding_window}.py
├── scripts/
│   ├── download_era5_land.py, compute_spatial_weights.py
│   ├── build_county_daily_weather.py, validate_downloaded_data.py
│   └── run_preflight.py
├── tests/ (original regressions plus focused data, ERA5, horizon and split tests)
├── docs/
│   ├── research_protocol.md, research_design.md, data_and_target.md
│   ├── era5_land_methodology.md, feature_engineering.md, validation_and_modeling.md
│   ├── data_dictionary.md, decisions.md, references.md
│   └── reorganization_plan.md, repository_audit.md
├── notebooks/README.md
├── reports/ {exploratory_analysis, figures, tables, experiments}/
├── thesis/
│   ├── README.md, outline.md, source_map.md, references.bib
│   ├── university_thesis_guide.pdf
│   └── {chapters, figures, tables}/README.md
└── references/papers/ (15 unchanged scientific PDFs)
```

## Complete changed-file inventory

The working tree is deliberately uncommitted and unstaged. Git may display a move
as an old-path deletion plus a new untracked destination until staging; the source
hash map verifies preservation independently of Git's rename detection.

### Modified files

- `.gitignore`
- `AGENTS.md`
- `README.md`
- `docs/data_and_target.md`
- `docs/era5_land_methodology.md`
- `docs/feature_engineering.md`
- `docs/references.md`
- `docs/research_design.md`
- `docs/validation_and_modeling.md`
- `pyproject.toml`
- `tests/test_pipeline.py`

### Added files (excluding moved sources)

- `configs/data.yaml`
- `configs/features.yaml`
- `configs/modeling.yaml`
- `data/README.md`
- `data/external/README.md`
- `data/external/file_migration.json`
- `data/interim/county_daily_weather/README.md`
- `data/interim/spatial_weights/README.md`
- `data/processed/model_datasets/README.md`
- `data/raw/era5_land/README.md`
- `docs/data_dictionary.md`
- `docs/decisions.md`
- `docs/reorganization_plan.md`
- `docs/repository_audit.md`
- `docs/research_protocol.md`
- `notebooks/README.md`
- `reports/README.md`
- `reports/experiments/README.md`
- `reports/exploratory_analysis/README.md`
- `reports/figures/README.md`
- `reports/tables/README.md`
- `requirements/base.txt`
- `requirements/deep_learning.txt`
- `requirements/development.txt`
- `requirements/geospatial.txt`
- `requirements/machine_learning.txt`
- `scripts/build_county_daily_weather.py`
- `scripts/compute_spatial_weights.py`
- `scripts/download_era5_land.py`
- `scripts/run_preflight.py`
- `scripts/validate_downloaded_data.py`
- `src/soybean_yield_forecasting/__init__.py`
- `src/soybean_yield_forecasting/configuration.py`
- `src/soybean_yield_forecasting/data/__init__.py`
- `src/soybean_yield_forecasting/data/nass.py`
- `src/soybean_yield_forecasting/data/panel.py`
- `src/soybean_yield_forecasting/era5/__init__.py`
- `src/soybean_yield_forecasting/era5/cli.py`
- `src/soybean_yield_forecasting/era5/conversions.py`
- `src/soybean_yield_forecasting/era5/county_aggregation.py`
- `src/soybean_yield_forecasting/era5/download.py`
- `src/soybean_yield_forecasting/era5/file_io.py`
- `src/soybean_yield_forecasting/era5/manifest.py`
- `src/soybean_yield_forecasting/era5/pipeline.py`
- `src/soybean_yield_forecasting/era5/quality_control.py`
- `src/soybean_yield_forecasting/era5/schema.py`
- `src/soybean_yield_forecasting/era5/spatial_weights.py`
- `src/soybean_yield_forecasting/era5/variables.py`
- `src/soybean_yield_forecasting/evaluation/__init__.py`
- `src/soybean_yield_forecasting/evaluation/expanding_window.py`
- `src/soybean_yield_forecasting/features/__init__.py`
- `src/soybean_yield_forecasting/features/forecast_horizons.py`
- `tests/conftest.py`
- `tests/test_county_aggregation.py`
- `tests/test_data_panel.py`
- `tests/test_era5_download.py`
- `tests/test_expanding_window.py`
- `tests/test_file_io.py`
- `tests/test_forecast_horizons.py`
- `tests/test_manifest.py`
- `tests/test_quality_control.py`
- `tests/test_spatial_weights.py`
- `thesis/README.md`
- `thesis/chapters/README.md`
- `thesis/figures/README.md`
- `thesis/outline.md`
- `thesis/references.bib`
- `thesis/source_map.md`
- `thesis/tables/README.md`

### Moved files, byte-for-byte preserved

| Old path | New path |
|---|---|
| `01_Dati_Soia_e_Target/balanced_panel_candidate_counties.csv` | `data/external/usda_nass_and_county_boundaries/balanced_panel_candidate_counties.csv` |
| `01_Dati_Soia_e_Target/soybean_acres_479_counties_1950_2025.xlsx` | `data/external/usda_nass_and_county_boundaries/soybean_acres_479_counties_1950_2025.xlsx` |
| `01_Dati_Soia_e_Target/soybean_yield_479_counties_1950_2025.xlsx` | `data/external/usda_nass_and_county_boundaries/soybean_yield_479_counties_1950_2025.xlsx` |
| `01_Dati_Soia_e_Target/census_counties/cb_2020_us_county_500k.cpg` | `data/external/usda_nass_and_county_boundaries/census_counties/cb_2020_us_county_500k.cpg` |
| `01_Dati_Soia_e_Target/census_counties/cb_2020_us_county_500k.dbf` | `data/external/usda_nass_and_county_boundaries/census_counties/cb_2020_us_county_500k.dbf` |
| `01_Dati_Soia_e_Target/census_counties/cb_2020_us_county_500k.prj` | `data/external/usda_nass_and_county_boundaries/census_counties/cb_2020_us_county_500k.prj` |
| `01_Dati_Soia_e_Target/census_counties/cb_2020_us_county_500k.shp` | `data/external/usda_nass_and_county_boundaries/census_counties/cb_2020_us_county_500k.shp` |
| `01_Dati_Soia_e_Target/census_counties/cb_2020_us_county_500k.shp.ea.iso.xml` | `data/external/usda_nass_and_county_boundaries/census_counties/cb_2020_us_county_500k.shp.ea.iso.xml` |
| `01_Dati_Soia_e_Target/census_counties/cb_2020_us_county_500k.shp.iso.xml` | `data/external/usda_nass_and_county_boundaries/census_counties/cb_2020_us_county_500k.shp.iso.xml` |
| `01_Dati_Soia_e_Target/census_counties/cb_2020_us_county_500k.shx` | `data/external/usda_nass_and_county_boundaries/census_counties/cb_2020_us_county_500k.shx` |
| `03_Documentazione_e_Paper/Papers/Ceglar & Toreti (2021) — Seasonal Climate Forecasts for European Agriculture.pdf` | `references/papers/Ceglar & Toreti (2021) — Seasonal Climate Forecasts for European Agriculture.pdf` |
| `03_Documentazione_e_Paper/Papers/Hoffman, Kemanian & Forest (2020) — Growing-Phase Climate and Soybean Yield.pdf` | `references/papers/Hoffman, Kemanian & Forest (2020) — Growing-Phase Climate and Soybean Yield.pdf` |
| `03_Documentazione_e_Paper/Papers/Iizumi et al. (2018) — Global Crop Yield Forecasting Using Seasonal Climate Information.pdf` | `references/papers/Iizumi et al. (2018) — Global Crop Yield Forecasting Using Seasonal Climate Information.pdf` |
| `03_Documentazione_e_Paper/Papers/Iizumi et al. (2021) — Global Within-Season Yield Anomaly Prediction.pdf` | `references/papers/Iizumi et al. (2021) — Global Within-Season Yield Anomaly Prediction.pdf` |
| `03_Documentazione_e_Paper/Papers/Jeong et al. (2016) — Random Forests for Global and Regional Crop Yield Predictions.pdf` | `references/papers/Jeong et al. (2016) — Random Forests for Global and Regional Crop Yield Predictions.pdf` |
| `03_Documentazione_e_Paper/Papers/Khaki & Wang (2019) — Crop Yield Prediction Using Deep Neural Networks.pdf` | `references/papers/Khaki & Wang (2019) — Crop Yield Prediction Using Deep Neural Networks.pdf` |
| `03_Documentazione_e_Paper/Papers/Khaki, Wang & Archontoulis (2020) — A CNN-RNN Framework for Crop Yield Prediction.pdf` | `references/papers/Khaki, Wang & Archontoulis (2020) — A CNN-RNN Framework for Crop Yield Prediction.pdf` |
| `03_Documentazione_e_Paper/Papers/Schlenker & Roberts (2009) — Nonlinear Temperature Effects on U.S. Crop Yields.pdf` | `references/papers/Schlenker & Roberts (2009) — Nonlinear Temperature Effects on U.S. Crop Yields.pdf` |
| `03_Documentazione_e_Paper/Papers/Sharma et al. (2025) — Maize and Soybean Yield Prediction Using ML Systematic Review.pdf` | `references/papers/Sharma et al. (2025) — Maize and Soybean Yield Prediction Using ML Systematic Review.pdf` |
| `03_Documentazione_e_Paper/Papers/Shook et al. (2021) — Crop Yield Prediction Integrating Genotype and Weather Using Deep Learning.pdf` | `references/papers/Shook et al. (2021) — Crop Yield Prediction Integrating Genotype and Weather Using Deep Learning.pdf` |
| `03_Documentazione_e_Paper/Papers/Sweet et al. (2023) — Cross-Validation Strategy Impacts ML Performance and Interpretation.pdf` | `references/papers/Sweet et al. (2023) — Cross-Validation Strategy Impacts ML Performance and Interpretation.pdf` |
| `03_Documentazione_e_Paper/Papers/Torsoni et al. (2023) — Soybean Yield Prediction by Machine Learning and Climate.pdf` | `references/papers/Torsoni et al. (2023) — Soybean Yield Prediction by Machine Learning and Climate.pdf` |
| `03_Documentazione_e_Paper/Papers/Vijverberg, Hamed & Coumou (2023) — Skillful U.S. Soy Yield Forecasts at Presowing Lead Times.pdf` | `references/papers/Vijverberg, Hamed & Coumou (2023) — Skillful U.S. Soy Yield Forecasts at Presowing Lead Times.pdf` |
| `03_Documentazione_e_Paper/Papers/Xie, Huang & Meng (2025) — Soybean Yield Modeling with Weather Dynamics.pdf` | `references/papers/Xie, Huang & Meng (2025) — Soybean Yield Modeling with Weather Dynamics.pdf` |
| `03_Documentazione_e_Paper/Papers/Yin et al. (2026) — Estimating Soybean Yields from High-Temporal-Resolution Multi-Source Data.pdf` | `references/papers/Yin et al. (2026) — Estimating Soybean Yields from High-Temporal-Resolution Multi-Source Data.pdf` |
| `era5_land_daily/manifest.csv` | `data/raw/era5_land/manifest.csv` |
| `era5_land_daily/spatial_weights_479_counties.csv` | `data/interim/spatial_weights/spatial_weights_479_counties.csv` |
| `era5_land_daily/spatial_weights_479_counties.parquet` | `data/interim/spatial_weights/spatial_weights_479_counties.parquet` |
| `era5_land_daily/county_daily/county_daily_1950.parquet` | `data/interim/county_daily_weather/supplied_preflight/county_daily_1950.parquet` |
| `era5_land_daily/county_daily/county_daily_2025.parquet` | `data/interim/county_daily_weather/supplied_preflight/county_daily_2025.parquet` |
| `era5_land_daily/raw_accumulated_boundaries/era5_land_1950_accumulated_00utc.nc` | `data/raw/era5_land/supplied_preflight/raw_accumulated_boundaries/era5_land_1950_accumulated_00utc.nc` |
| `era5_land_daily/raw_accumulated_boundaries/era5_land_2025_accumulated_00utc.nc` | `data/raw/era5_land/supplied_preflight/raw_accumulated_boundaries/era5_land_2025_accumulated_00utc.nc` |
| `era5_land_daily/raw_daily_stats/era5_land_1950_daily_maximum.nc` | `data/raw/era5_land/supplied_preflight/raw_daily_stats/era5_land_1950_daily_maximum.nc` |
| `era5_land_daily/raw_daily_stats/era5_land_1950_daily_mean.nc` | `data/raw/era5_land/supplied_preflight/raw_daily_stats/era5_land_1950_daily_mean.nc` |
| `era5_land_daily/raw_daily_stats/era5_land_1950_daily_minimum.nc` | `data/raw/era5_land/supplied_preflight/raw_daily_stats/era5_land_1950_daily_minimum.nc` |
| `era5_land_daily/raw_daily_stats/era5_land_2025_daily_maximum.nc` | `data/raw/era5_land/supplied_preflight/raw_daily_stats/era5_land_2025_daily_maximum.nc` |
| `era5_land_daily/raw_daily_stats/era5_land_2025_daily_mean.nc` | `data/raw/era5_land/supplied_preflight/raw_daily_stats/era5_land_2025_daily_mean.nc` |
| `era5_land_daily/raw_daily_stats/era5_land_2025_daily_minimum.nc` | `data/raw/era5_land/supplied_preflight/raw_daily_stats/era5_land_2025_daily_minimum.nc` |
| `balanced_panel_candidate_counties.csv` | `data/external/supplied_candidate_counties.csv` |
| `Vademecum.pdf` | `thesis/university_thesis_guide.pdf` |

### Deleted obsolete implementation files

- `compute_spatial_weights.py`: Root monolith replaced by the package spatial module and thin scripts entry point; algorithm and artifacts retained.
- `download_era5_land_daily.py`: Root monolith replaced by the focused ERA5 package modules and CLI entry points; regression-covered behavior retained.
- `requirements.txt`: Redundant independent constraint set replaced by authoritative pyproject extras and requirements selectors.

Only empty superseded directories were removed after moving every source file.
No original CSV, Excel, geometry, NetCDF, Parquet, manifest or PDF was deleted.

## Subsequent author-selected panel and delivery verification

On 2026-09-15 the author replaced the primary sample with 135 complete counties
across 1950–2025 (76 years; 10,260 observations), without model-comparison tests.
Prepared target and auxiliary data, source/output hashes, the selected 3,344-row
weight matrix and active configuration agree. Source reconstruction was run twice;
the second run verified identical outputs. All 40 original source hashes still match.
The 40-year initial training default was removed; the initial window is unset.

Internal tests/docs/notebooks were moved under operations; thesis preparation notes
are separate under thesis/notes. The explicit delivery allowlist produced a 59-file
ZIP in scratch/thesis_delivery.zip, excluding originals, internal tests/artifacts,
PDF library and thesis preparation notes. No source data was removed or rewritten.
The entire Git working tree still includes development material; the ZIP is the
reviewed delivery boundary.

Verification: 53 offline tests passed, Ruff checks/format passed, original preflight
and historical checksums passed. The first sandbox test/preflight runs failed on
Windows temporary-directory permissions; the same offline commands passed with
approved access to temporary files. Nine existing dependency warnings remain
(NumPy binary-size warning and openpyxl datetime deprecation). No model fitting,
CDS download, commit or push was performed. Full weather/feature/model stages remain
pending. No claim is made that fewer counties proportionally reduce rectangular
raw CDS downloads.

## Two-section workspace and 1950 target exclusion

Current layout: output/ is the single maintained clean thesis project; work/ holds
original sources, raw/intermediate weather, tests, internal documentation, literature,
writing notes and generated artifacts. Only Git/environment/agent infrastructure
remains beside those two sections. The original 40 source files still match their
SHA-256 hashes after updating the path map. The root .gitignore was not changed in
this step; nested rules protect generated files in the two sections.

The author explicitly excluded 1950 yield because October 1949 weather is unavailable.
The prepared target is now 1951–2025: 135 counties, 75 years, 10,125 rows. Exact
comparison with the prior derived files verified that only the 135 rows for 1950
were removed from yield and auxiliary acreage. The superseded derived CSVs were
removed after this check; original workbooks remain unchanged. Weather years remain
1950–2025 for the first retained October 1950–October 1951 campaign. October is
confirmed; the precise reference day and initial training length remain open.

Weather requests derive their rectangle from selected weight cell centers:
[45.8, -96.8, 36.0, -80.5], covering all 3,344 weighted intersections. The grid has
16,236 points instead of 20,040 (18.98% fewer per field/date). Completed yearly
outputs resume after source-record/checksum and exact schema/calendar/FIPS checks,
without repeating raw NetCDF decompression or aggregation. No CDS live run or
wall-clock/download-byte performance measurement was performed.

Verification: 55 offline tests passed (9 existing third-party warnings), including
source preservation, exact target reconciliation, complete spatial coverage,
1950 weather retention, resumed-year checks and corrupted-checksum/wrong-FIPS
rejection. Ruff passed. Original preflight and historical checksums passed after
migration. The local editable package was reinstalled from output/ without network
access. Windows temporary/cache permissions required approved local tool access.
The 59-file ZIP work/scratch/thesis_delivery_1951_2025.zip matches current output
files byte for byte, has valid Python and local Markdown links, and contains no
1950 yield or internal work material. No model fitting, commit or push occurred.

## Authorized live preflight and annual pilots: 2026-09-15

The author authorized the new seven-day preflight (1950 and 2025) and full annual
pilots (1950 and 1952), retaining the full-history command for later execution.
The live 1950 preflight passed: 945 rows, 135 counties, 37 fields, four raw PASS
records with SHA-256. Its snow-cover source attribute is `%`. At 22:32 Europe/Rome,
the 2025 daily-mean request was still waiting on CDS; its preflight is incomplete.

The first annual daily-mean request failed with HTTP 403 `cost limits exceeded`.
Channel A now uses validated resumable monthly blocks, recursively bisecting days
only for that explicit cost error. Live CDS also rejected the January 31-day block,
then accepted January 1-15. Acceptance is queue admission, not download completion.
The arbitrary 1 KB resume cutoff was removed: valid small chunks now rely on their
QC record, recorded size and optional checksum. Authentication failures are not
mistaken for cost-limit errors. The public command remains unchanged.

Verification: 57 offline tests passed (9 existing third-party warnings); Ruff passed.
Tests include ordinary/leap calendars, month interruption/resume, cost-limit splitting,
licence-error propagation and corrupt/wrong-county resume rejection. No full annual
pilot has yet passed. Production year 1950 is running; 1952 is queued behind its
successful validation using `work/scratch/continue_pilots.ps1`. The continuation
worker writes `work/scratch/pilots_1950_1952.stdout.log` and `.stderr.log`, stops on
failure, and never starts the full history. These processes need this machine awake
and connected. Originals and prepared target files were not modified in this step.
No commit or push was made. Earlier delivery ZIPs predate this downloader correction.
