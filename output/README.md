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
Requests use all 37 fields on the minimum rectangle containing every selected
weight cell: 16,236 grid points instead of 20,040 (18.98% fewer). Resolution and
within-county weights are unchanged. Completed years resume after source-record,
optional checksum and output validation, without raw NetCDF reloading.
Daily-statistic requests use resumable monthly blocks, splitting their days further
when CDS explicitly rejects a block for exceeding its cost limit.

The commands below contact CDS. They require separate download authorization and
credentials outside the repository:

```powershell
python scripts/run_preflight.py --download --verify-checksum
python scripts/download_era5_land.py --test-year 1950 --verify-checksum
python scripts/download_era5_land.py --test-year 1952 --verify-checksum
# Only after pilot acceptance and explicit historical-download authorization:
python scripts/download_era5_land.py --primary-period --verify-checksum
```

For downloaded files, use `python scripts/build_county_daily_weather.py --test-year
1950 --verify-checksum` and `python scripts/validate_downloaded_data.py --year 1950
--verify-checksum`. Recomputing weights with `python scripts/compute_spatial_weights.py
--output-dir ../work/scratch/recomputed_weights_new` requires the original Census
boundaries and the geospatial extra. Prepared weights are already included here.

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
