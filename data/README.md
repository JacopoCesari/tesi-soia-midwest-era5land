# Unified Research Data Directory

This directory centralizes all datasets used throughout the thesis, eliminating folder fragmentation and ensuring full transparency.

```text
data/
├── target/                     # Primary research modeling target
│   └── soybean_yield_1951_2025.csv   # Balanced panel: 135 counties × 75 years (10,125 observations)
│
├── auxiliary/                  # Harmonized auxiliary tables & spatial cross-references
│   ├── counties.csv            # 135 selected study counties (FIPS, State, County Name)
│   ├── acres_harvested_1951_2025.csv # County harvested acreage (panel audit only)
│   ├── spatial_weights.csv     # Planar intersection weights (ERA5-Land grid to counties)
│   ├── us_soybean_prices_1960_2025.csv # Historical USDA farm-gate prices (1960–2025)
│   └── panel_provenance.json   # SHA-256 data integrity hashes and selection audit
│
├── weather/                    # Retrospective meteorological data (ECMWF ERA5-Land)
│   └── daily_counties/         # County daily agrometeorological series (1950–2025 Parquet)
│       └── county_daily_YYYY.parquet # 17 standardized daily indicators per county-day
│
└── raw/                        # Original, immutable source inputs
    ├── usda_nass/              # Complete USDA NASS SURVEY workbooks (479 counties)
    ├── census_counties/        # U.S. Census Bureau 1:500k county shapefiles
    └── era5_land/              # Atmospheric reanalysis manifests and preflight pilot NetCDFs
```

---

## Data Usage Rules

1. **Target**: `target/soybean_yield_1951_2025.csv` is the only target dataset for model training and evaluation.
2. **Weather**: `weather/daily_counties/` contains the complete 75-year surface weather forcing (1950–2025) aggregated from ERA5-Land.
3. **Auxiliary**: `auxiliary/` files provide spatial weights and geographic metadata. Crop acreage is kept for panel audit only and is never used as a predictor.
4. **Immutability of Raw**: Files in `raw/` represent original external sources and are kept strictly read-only.
