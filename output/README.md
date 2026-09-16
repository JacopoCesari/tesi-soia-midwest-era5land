# Weather-only soybean yield forecasting

The thesis asks how early observed weather adds stable county soybean yield skill
beyond historical-trend and climatology baselines.

**Primary target: 135 counties × 75 years (1951–2025) = 10,125 observations.**
1950 yield is excluded because the October 1949–October 1950 campaign lacks its
initial weather. Keep 1950 weather for the first retained campaign, October
1950–October 1951. The original source workbooks remain unchanged outside this folder.

## Clean project

- [Selected yield](data/target/soybean_yield_1951_2025.csv): FIPS, year, BU/ACRE.
- [Auxiliary data](data/auxiliary/README.md): county metadata, acreage, spatial weights and hashes.
- [Protocol](docs/research_protocol.md), [data selection](docs/data_and_target.md)
  and [weather methodology](docs/era5_land_methodology.md).
- [Thesis narrative](thesis/README.md).

```text
configs/          Target/weather periods and methodological settings
data/target/      Selected yield target
data/auxiliary/   County metadata, acreage, weights and provenance
data/processed/   Future model-ready feature datasets
src/              Python research package
scripts/          Data preparation and weather pipeline
docs/             Essential usage and methodology
thesis/           Narrative and bibliography
```

This directory is the single maintained project to share. Its sibling `../work/`
contains originals, raw/intermediate weather, internal tests/docs, paper PDFs,
writing notes and scratch outputs. Configured weather/QC outputs go there, so
running the pipeline does not mix them with the clean target and thesis narrative.

## Installation

Run commands from this directory using Python 3.12 (tested: 3.12.8).
In the full workspace, activate the environment at `../.venv/`. For an independent
copy, create and activate a virtual environment, then run:

```powershell
python -m pip install -e .
```

`pyproject.toml` is the dependency source of truth. Optional extras include
`geospatial`, `machine_learning`, `deep_learning` and `development`.
`requirements/` contains installation selectors without duplicated constraints.

Read and validate the delivered target without original workbooks:

```python
import pandas as pd
from pathlib import Path
from soybean_yield_forecasting.configuration import load_configuration
from soybean_yield_forecasting.data.panel import validate_balanced_panel

config = load_configuration(Path("configs/data.yaml"))
target = pd.read_csv(config["paths"]["target"], dtype={"county_fips": str})
validate_balanced_panel(target, "yield_bu_per_acre",
                        expected_counties=config["expected_counties"], **config["primary_period"])
```

In the full workspace, `python scripts/prepare_thesis_data.py` reproduces or
verifies the target and auxiliaries using original sources under `../work/data/`.
It never rewrites sources or silently replaces conflicting prepared files.

## Weather acquisition

Target years and weather years are separate settings. `--primary-period` retrieves
the configured **weather years 1950–2025**, supporting **yield years 1951–2025**.

The pipeline uses **ECMWF ARCO (Analysis-Ready Cloud-Optimized) Zarr** as its primary
acquisition engine, with direct HTTP chunked access to the 8 official stores on
`https://arco.datastores.ecmwf.int`. The legacy CDS batch API pipeline is deprecated and
preserved under `docs/legacy_cds_pipeline.md` (callable via `--legacy-cds`).

### ARCO Pipeline Workflow
1. **Cloud Access**: Lazy, chunk-aware HTTP slicing over the 135-county bounding box.
2. **Local Daily Aggregations**: Hourly fields aggregated to daily statistics (mean, min, max, sum).
3. **Spatial Aggregation**: Area-weighted county aggregation via EPSG:5070 precomputed weights.
4. **Physical Conversions**: Kelvin to °C, Pa to hPa, m to mm, J/m² to MJ/m².
5. **Derived Meteorological & Agronomic Features**:
   - `wind_speed = sqrt(u10^2 + v10^2)`
   - `relative_humidity` (August-Roche-Magnus formula)
   - `vapor_pressure_deficit` (Tetens formula)
   - `growing_degree_days` (base 10°C, cutoff 30°C)
   - `heat_day_30`, `heat_day_35`
   - `et0_fao56` (FAO-56 Penman-Monteith reference evapotranspiration)
   - `p_minus_et0` (climatic water balance)
6. **Validation & Atomic Parquet Output**: Strict QC schema and physical consistency checks.

### Usage Commands

```powershell
# 1. Discover and inspect available ARCO stores and variable inventory
python scripts/arco_inventory.py

# 2. Run minimal 7-day preflight pilot (June 1–7, 1950) via ARCO
python scripts/download_era5_land.py --pilot

# 3. Offline preflight validation of supplied artifacts
python scripts/run_preflight.py

# 4. Acquire one complete pilot year via ARCO (requires prior authorization)
python scripts/download_era5_land.py --test-year 1950

# 5. Legacy CDS execution (deprecated, preserved for benchmark verification)
python scripts/download_era5_land.py --test-year 1950 --legacy-cds --verify-checksum
```

## Current status

The target, spatial weights, weather pipeline, cutoff utilities and temporal splitter
are implemented. The supplied offline weather fixtures remain reduced samples.
The live 1950 preflight passed on 2026-09-15: seven days, 135 counties and 37 fields.
The 2025 preflight and annual pilots are in progress; none of the annual pilots
has yet passed. No full historical download has started. The 18.98%
figure counts requested grid points; transfer size and runtime savings are unmeasured.

There are twelve monthly horizons before October; the exact harvest day, feature
lookback and initial expanding-window training length remain open. There is no
fixed 60/40 split. All preprocessing and tuning must use training data only.
Acreage is auxiliary documentation, never a predictor or evaluation weight.
Feature generation, model fitting, metrics and thesis results remain future work.
Historical target/ERA5 release timing must be resolved before operational forecast claims.
