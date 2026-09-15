# ERA5-Land acquisition and county aggregation

## Scope and implementation

Production targets 1950–2025 for the selected 135 counties. The original 479-county
sources remain separate. On 2026-09-15 the authorized live 1950 preflight passed
(seven days, 135 counties, 37 fields); the 2025 preflight and annual pilots remain
in progress, not accepted as complete. No full historical run has started.
The pipeline operates in daily UTC dates on the regional
original domain `[47.7, -97.0, 35.8, -80.4]` for stored fixtures. Production requests
derive the smallest rectangle containing the selected weight cell centers:
`[45.8, -96.8, 36.0, -80.5]` (north, west, south, east). Target years are 1951–2025;
weather still starts in 1950 for the October 1950–October 1951 campaign.

| Responsibility | Module under `src/soybean_yield_forecasting/era5/` |
|---|---|
| CDS variable selections and domain | `variables.py` |
| Requests and response extraction | `download.py` |
| Portable NetCDF loading and hashing | `file_io.py` |
| Record append and resume checks | `manifest.py` |
| Calendar, physical and county checks | `quality_control.py` |
| Parameter correction and unit conversion | `conversions.py` |
| English output names | `schema.py` |
| County-cell overlay and weight validation | `spatial_weights.py` |
| Daily weighted aggregation | `county_aggregation.py` |
| Validated year orchestration | `pipeline.py` |
| CLI parsing and report orchestration | `cli.py` |

The scripts in `scripts/` are thin entry points. Paths come from `configs/data.yaml`
or explicit overrides. The CDS client is created lazily on the first retrieval;
offline aggregation and validation do not require credentials.

## Two channels, 37 weather fields

**Channel A:** `derived-era5-land-daily-statistics`; 17 daily means, one minimum
(2 m air temperature), two maxima (2 m air and skin temperature). Requests retain
`frequency=1_hourly` and `time_zone=utc+00:00`. See the dictionary for all 20 fields.

Annual channel A retrieval is assembled from monthly blocks. Live CDS rejected
both a full-year request and a 31-day, 17-variable mean request with HTTP 403
`cost limits exceeded`; a 15-day block was accepted on 2026-09-15. On that specific
error, the downloader bisects the requested days recursively. Other failures are
propagated, including authentication/licence errors; an oversized one-day request
also fails explicitly. No fixed server cost threshold is inferred from this trial.
Each completed block receives calendar/variable-count QC and a manifest checksum;
restarts reuse validated blocks before assembling the annual file. These temporary
production blocks stay under `work/`, outside the professor-facing delivery.

**Channel B:** `reanalysis-era5-land`; 17 accumulated variables requested at 00:00.
The retained algorithm assigns midnight D+1 to weather day D. An annual retrieval
uses two requests: all midnight dates in Y, then January 1 of Y+1. This downloads
one unused initial boundary. Aggregation shifts one day and selects the channel A
calendar, retaining December 31 from the following year's January 1.

With 17 variables, the main request contains 6,205 fields in an ordinary year or
6,222 in a leap year; the boundary request has 17. Totals are 6,222 and 6,239,
respectively. Each channel B request stays below the existing 12,000-field budget.
This preserves the repository's request design; current server behavior, quotas,
variable inventory and limits still need verification in an authorized live pilot.

## Spatial algorithm

The 0.1-degree grid has 120 latitude and 167 longitude coordinates (20,040 cells).
Build square cells centered on those coordinates, project cells and county polygons
to **EPSG:5070**, intersect them, then normalize intersection areas within each county:

`weight[county, cell] = intersection_area / sum(county_intersection_areas)`.

County weather is `sum(weight * grid_value)`. The supplied matrix contains 11,953
active intersections across 479 original counties. The primary subset retains 3,344
intersections across 135 counties in `data/auxiliary/spatial_weights.csv`, without
changing any within-county weights. All county weight sums are approximately
one. EPSG:5070 is an equal-area projection; it should not be described as conformal.
The selected rectangle has 99 × 164 = 16,236 grid points instead of 20,040:
18.98% fewer points per requested date/field, with all contributing cells retained.
This is a grid-volume reduction, not a measured wall-clock or compressed-byte saving.
Geographic subsetting uses the documented CDS
[`area` parameter](https://confluence.ecmwf.int/pages/viewpage.action?pageId=529421364).
No variable, frequency or spatial weight is changed. Readers validate finite positive
weights, unique county/cell pairs, FIPS, geographic
domain and normalized sums. Missing weather in a contributing cell cannot silently
become zero through an aggregation that skips missing values.

## Efficient restart

For an existing yearly county output, the pipeline first checks the latest PASS
record for all four raw inputs and their checksums when requested, then validates
the county output's exact dates, 37-field schema and selected FIPS. A valid completed
year returns without reopening/decompressing raw NetCDF or repeating aggregation.
Missing/failed raw records, checksum differences or invalid output prevent this shortcut.
Annual requests and boundary handling remain unchanged; monthly edge trimming and
alternative request layouts are not claimed as implemented. Live timing remains unmeasured.

## Parameter correction and units

The original three-way evaporation permutation is retained:

- `evabs` -> vegetation transpiration;
- `evaow` -> bare-soil evaporation;
- `evavt` (or `evatp`) -> open-water evaporation.

The original unit conversion guards are preserved: temperature K to °C if column
mean >100; pressure Pa to hPa if mean >50,000; selected water depths m to mm if
maximum absolute value <10; energy J/m² to MJ/m² if mean absolute value >10,000.
Evaporation and heat-flux signs are retained. These magnitude guards are a known
limitation: they can leave small energy fields unconverted and are not a general
unit-aware or safely repeatable conversion API. They must not be applied a second
time to already converted artifacts. Replacing them requires a separately reviewed
unit policy and regression evidence, not a style-only refactor.

Canonical output column names are documented in [the dictionary](data_dictionary.md).
The supplied raw and county files retain their original source names and bytes.

## Manifest, resume and data protection

`data/raw/era5_land/manifest.csv` is the immutable original record. Its original
paths are resolved by `../work/data/external/file_migration.json` without editing the CSV.
Its PASS rows denote the historical reduced preflight, not annual validation.

New `preflight/` and `production/` directories have separate operational manifests.
Record each raw file immediately after its calendar and variable checks. Resume uses
the latest record, PASS status, existence and recorded size; `--verify-checksum`
additionally requires a nonempty matching SHA-256. A newer FAIL cannot be masked
by an older PASS. Files already present without a valid manifest record cause an
explicit failure instead of being overwritten. Investigate or choose a new output
directory before retrying; original source data must remain untouched.

An interrupted partial transfer may be requested again. A raw file finalized before
its manifest append requires review rather than silent adoption. Existing valid
yearly county output is checked and reused. Checksum verification covers raw files;
yearly Parquet output currently has schema/calendar/QC validation, not its own digest.

## Supplied versus new preflight

The supplied eight NetCDF files cover June 1–7 in 1950 and 2025. Channel A has only
four weather fields after distinguishing mean/minimum/maximum temperatures; channel
B has 17. The original identical date selections shift channel B back one day,
leaving June 1–6 in the stored county outputs: 2,874 rows and 21 weather fields each.
The two original Parquet files are retained as numerical regression fixtures.

The default preflight command validates these fixtures offline. A **new** preflight
uses all channel A fields and requests channel B June 2–8, producing seven aligned
June 1–7 days and 37 fields. This corrected request is tested with simulated CDS
responses. The live 1950 run passed with 945 county-day rows and all 37 fields on
2026-09-15; the live 2025 run is still pending. Legacy six-day intersection is
allowed only by the explicit supplied-sample regression option, not annual production.

## QC report and pilot acceptance

Yearly output has a sidecar `.qc.json`. CLI validation writes reports under
`reports/experiments/quality_control/`, or an explicit `--report` path. The report
contains overall `status` and a `checks` list with individual status, notes and
available file/year/count metadata. Failed QC prevents a new county Parquet write.

Before an initial historical run, an authorized ordinary-year and leap-year pilot
must demonstrate:

- Exact calendar: 365 or 366 days, no gaps or duplicate dates.
- 135 selected counties on every day; unique `(county_fips, date)` keys.
- 49,275 ordinary-year rows or 49,410 leap-year rows, with all 37 variables.
- No missing or non-finite county weather; all expected county identifiers present.
- Minimum air temperature <= mean <= maximum.
- Implemented physical checks: nonnegative precipitation and downward solar
  radiation to the retained tolerance; soil water within the retained [-0.05, 1.05]
  tolerance. Physical range checks are not exhaustive for all 37 variables.
- Finite normalized weights and complete grid coverage of contributing cells.
- Correct December 31 accumulation from January 1 of the next year.
- Successful per-file manifest, checksum and restart behavior.

Ordinary tests use mocked CDS responses. Passing them establishes local behavior;
it does not prove live response conventions or full-domain annual memory capacity.
NetCDF loading and spatial aggregation still load full datasets into memory as in
the original implementation. Pilot memory/time measurements remain necessary.
See the [README commands](../README.md) for each phase.
