# Legacy CDS Acquisition Pipeline

> **Status: Deprecated (Preserved for Provenance and Historical Verification)**  
> As of the migration to ECMWF ARCO (Analysis-Ready Cloud-Optimized) Zarr stores, the Copernicus Climate Data Store (CDS) batch API pipeline is deprecated. It is retained in the codebase under `soybean_yield_forecasting.era5.download.CDSDownloader` and `pipeline.run_year_cds` to ensure full scientific reproducibility, historical benchmark comparisons, and source verification.

---

## 1. Architectural Overview

The legacy pipeline retrieved ERA5-Land reanalysis data from ECMWF via the `cdsapi` Python client in two distinct channels:

```
                            ECMWF Copernicus Climate Data Store (CDS)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
    Channel A: Daily Statistics                     Channel B: Hourly Accumulations
  (derived-era5-land-daily-statistics)                   (reanalysis-era5-land)
                  │                                               │
    • 17 daily means                                • 17 accumulated variables at 00:00 UTC
    • 1 daily minimum (t2m_min)                     • Annual dates + Jan 1 (Y+1) boundary
    • 2 daily maxima (t2m_max, skt_max)             • Day-shift: D+1 at 00:00 assigned to day D
                  │                                               │
                  ▼                                               ▼
         Monthly/Day Bisection                            Single/Two-Part NetCDF
           (HTTP 403 fallback)                             (< 12,000 fields limit)
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                                Merge & Quality Control
                                          │
                                          ▼
                            County Spatial Aggregation
                             (EPSG:5070 Area Weights)
                                          │
                                          ▼
                         county_daily_{year}.parquet (37 fields)
```

---

## 2. CDS Limitations and Operational Bottlenecks

During live pilot testing on 2026-09-15 and historical production runs, several critical operational limitations of the CDS API were identified:

### A. Severe Queue Latencies
CDS API requests are submitted to a shared batch queue. Depending on server load, queue wait times vary unpredictably from several minutes to dozens of hours per request. Acquiring 75 years across multiple channels required hundreds of separate requests, creating substantial scheduling bottlenecks.

### B. HTTP 403 "Cost Limits Exceeded"
In September 2026 trials, CDS introduced strict operational cost limits:
- A single annual daily-statistics request for 17 mean variables was rejected with HTTP 403 (`cost limits exceeded`).
- A monthly request for 31 days was also rejected with HTTP 403.
- Only reduced chunks (e.g. 15 days or fewer) were accepted.
- To handle this, `CDSDownloader._download_daily_chunk()` implemented recursive bisection of calendar days upon encountering HTTP 403, merging chunks locally after successful download.

### C. 12,000-Field Ceilings on Channel B
Accumulation requests at 00:00 UTC had to stay strictly below the CDS 12,000-field limit:
- Main annual request: $365 \times 17 = 6{,}205$ fields (or $6{,}222$ in leap years).
- Boundary request (Jan 1 of $Y+1$): $1 \times 17 = 17$ fields.
- Total per year: 6,222 (or 6,239) fields, safely within the ceiling, but requiring two separate API retrieval jobs per year.

### D. Midnight Accumulation Alignment Mechanics
ERA5-Land accumulated fields represent values accumulated over the preceding 24 hours up to the valid time. Therefore, the value at `00:00 UTC` on date $D+1$ represents the accumulation for calendar day $D$. This required downloading January 1 of year $Y+1$ to complete December 31 of year $Y$, followed by index shifting:
$$\text{date}_D = \text{valid\_time}_{D+1} - 1\text{ day}$$

---

## 3. Evaporation Parameter Mapping Clarification

The legacy pipeline implemented a parameter name substitution for evaporation components in Channel B (`apply_evaporation_swap`):
- `evabs` $\rightarrow$ vegetation transpiration (`evaporation_from_vegetation_transpiration`)
- `evaow` $\rightarrow$ bare soil evaporation (`evaporation_from_bare_soil`)
- `evavt` $\rightarrow$ open water evaporation (`evaporation_from_open_water_surfaces_excluding_oceans`)

### Official ECMWF Parameter Definitions
Audit against the official ECMWF Parameter Database (`paramId`) establishes the exact physical definitions:
- **`evabs` (paramId 260029)**: *Evaporation from bare soil* ($\text{m}$ of water equivalent)
- **`evaow` (paramId 260031)**: *Evaporation from open water surfaces excluding oceans* ($\text{m}$)
- **`evatc` (paramId 260028)**: *Evaporation from the top of canopy* ($\text{m}$)
- **`evavt` (paramId 260030)**: *Evaporation from vegetation transpiration* ($\text{m}$)

The legacy three-way swap was an inverted mapping in earlier scripts. The ARCO migration documents both the legacy naming and the true physical mapping in `configs/feature_manifest_37.yaml` and `feature_manifest.py`.

---

## 4. The Four CDS Wishlist Variables

In the original 37-feature target design, four variables are not currently published in the ECMWF ARCO Zarr stores:
1. `total_evaporation` (`e`, paramId 182)
2. `surface_runoff` (`sro`, paramId 174008)
3. `sub_surface_runoff` (`ssro`, paramId 174009)
4. `surface_net_solar_radiation` (`ssr`, paramId 176)

In `feature_manifest.py`, these are classified as `status="OPTIONAL_CDS_WISHLIST"`. Per protocol rules:
- They are **not artificially synthesized** or reconstructed using non-physical proxies.
- They are optional supplementary predictors that can only be obtained if running the legacy CDS pipeline.
- The primary thesis models, EDA, and feature selection operate entirely on the 20 primary ARCO variables and 8 exact/proxy derived agronomic features without CDS dependency.

---

## 5. Running the Legacy CDS Pipeline

To execute the legacy CDS pipeline for comparison or verification, use the `--legacy-cds` flag or `--engine cds` option on the CLI:

```powershell
# 1. Run minimal 7-day preflight via legacy CDS
python scripts/run_preflight.py --download --legacy-cds

# 2. Run a specific pilot year via legacy CDS
python scripts/download_era5_land.py --test-year 1950 --legacy-cds --verify-checksum

# 3. Python programmatic invocation
from pathlib import Path
from soybean_yield_forecasting.era5.pipeline import run_year

output_parquet = run_year(
    year=1950,
    raw_directory=Path("../work/data/raw/era5_land/production"),
    output_directory=Path("../work/data/interim/county_daily_weather/production"),
    weights=Path("data/auxiliary/spatial_weights.csv"),
    engine="cds",
    expected_counties=135,
)
```

### Manifest Tracking
The legacy CDS pipeline records all requests, MD5/SHA-256 hashes, status codes, and QC notes in `../work/data/raw/era5_land/manifest.csv`. Completed and verified NetCDF files are reused on restart without re-downloading.
