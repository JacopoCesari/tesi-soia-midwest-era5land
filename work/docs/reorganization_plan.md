# Repository reorganization plan

Internal record of the earlier implementation scope. The author subsequently
selected the 135-county 1950–2025 panel and an explicit professor-delivery boundary.
Use the current root README and research protocol for active instructions.

Recorded on 2026-09-15 from the author's implementation request.

## Goal and scope

Make this a reproducible, English-language thesis research package while preserving
the ERA5 spatial algorithm, request design, evaporation correction and numerical
conversions. Implement only the requested forecast-origin and temporal-split utilities.
No feature generation, model fitting, exploratory analysis or network data downloads.

## Architecture and interfaces

- `src/soybean_yield_forecasting/`: configuration, source ingestion, ERA5 download,
  manifest, QC, conversions, aggregation, horizons and grouped temporal splits.
- `scripts/`: explicit CLI entry points; configuration owns paths and year defaults.
- `data/`: external inputs, raw weather, intermediate products and future model data.
- `docs/`: consolidated protocol and operational instructions; `thesis/`: writing.
- Preserve every source data byte; record SHA-256 before and after each move.
- Preserve the original manifest as evidence; resolve its paths through migration
  metadata and use a separate operational manifest for new runs.

## Acceptance and checks

Baseline: seven existing tests pass outside the Windows sandbox (the sandbox alone
causes one temporary-directory permission failure). Add request, aggregation,
calendar, checksum, cutoff and year-grouping regressions. Run all tests, lint, CLI
help, root imports and read-only checks of local data after migration.

## Risks and open decisions

Harvest month/day remains unset. Feature lookback and observation availability must
be settled before modeling. Verify county selection from workbooks without inventing
an acreage threshold. Full-year and historical CDS operation remain unverified until
separately authorized pilots. Existing preflight files must not be overwritten.

The detailed user request supplies the scope and authorization for this plan.
