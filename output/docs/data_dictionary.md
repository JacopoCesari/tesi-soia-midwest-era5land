# Data dictionary

## Identifiers, source measurements and protocol metadata

| Field | Meaning / units | Availability and role |
|---|---|---|
| `county_fips` | Five-digit string; state + county FIPS | Join key for source, weights, weather and target |
| `year` | Integer calendar/crop year | Available NASS annual record year |
| `date` | Daily UTC date, midnight label | Available county weather key |
| `state`, `county` | Supplied English names | NASS metadata; not weather predictors |
| `yield_bu_per_acre` | Soybean yield, bushels per harvested acre | Primary available target |
| `acres_harvested` | Acres harvested | Sample construction/audit only; never predictor or weight |
| `target_year` | Crop year whose yield is forecast | Implemented horizon metadata |
| `forecast_origin` | Inclusive weather cutoff date | Implemented horizon metadata; may be in previous year |
| `months_before_harvest` | Integer 12 through 1 | Implemented monthly horizon label |
| `train_years`, `test_year` | Complete target years in each temporal fold | Implemented split metadata |
| `train_indices`, `test_indices` | Positional indices into input records (`iloc`) | Implemented split indices, independent of DataFrame labels |

The ingestion API returns county identifiers/names, year and one explicitly selected
measurement. Original workbook ancillary fields are preserved only at the source:
`state_ansi`, `county_ansi` (source state/county codes); `ag_district` and
`ag_district_code` (agricultural district); `cv_percent` (reported coefficient of
variation, percent); `suppressed` (source flag in acreage workbook);
`in_balanced_window_1950_2010` (period membership flag).
`county_list_479` and the candidate CSV contain `median_acres_1950_2025`, an acreage
summary whose selection role is unverified. `complete_1950_2010` marks candidate
continuity. None of these ancillary acreage summaries enter model inputs.

Workbook README columns `Voce` and `Descrizione` mean item and description; they are
immutable source metadata, not internal configuration names. Coverage sheets include
`counties_with_survey_yield`, `counties_with_survey_acres`, `balanced_main_period`,
`share_of_479_counties`, and `selected_counties` in the state count sheet.
The main sheets have no missing measurement values, but `cv_percent` is missing.
Recent missing observations are also represented by absent rows.

## Implemented ERA5 weather schema

New county outputs have 37 descriptive weather columns plus `date` and `county_fips`.
The source alias column describes NetCDF/legacy output identifiers at the ingestion
boundary. CDS request names are listed in `era5/variables.py`; the canonical mapping
is `era5/schema.py`. The supplied county Parquet fixtures retain 21 fields with
legacy aliases. `canonicalize_weather` changes names only, never units or values.

The output units below describe the conversion **when its retained magnitude guard
fires**. This refactor does not claim a unit-aware conversion engine. In particular,
small energy fields can remain in raw units. Temperature guard: column mean >100;
pressure guard: mean >50,000; water guard: absolute maximum <10; energy guard:
absolute mean >10,000. Only call conversion once on raw CDS quantities.

| Channel | Source alias | Canonical county field | Raw -> converted units |
|---|---|---|---|
| A | `t2m_mean` | `air_temperature_mean` | K -> degrees C (subtract 273.15) |
| A | `t2m_min` | `air_temperature_minimum` | K -> degrees C (subtract 273.15) |
| A | `t2m_max` | `air_temperature_maximum` | K -> degrees C (subtract 273.15) |
| A | `skt_max` | `skin_temperature_maximum` | K -> degrees C (subtract 273.15) |
| A | `d2m` | `dewpoint_temperature_mean` | K -> degrees C (subtract 273.15) |
| A | `u10` | `eastward_wind_mean` | m/s -> m/s |
| A | `v10` | `northward_wind_mean` | m/s -> m/s |
| A | `sp` | `surface_pressure` | Pa -> hPa (divide by 100) |
| A | `stl1` | `soil_temperature_level_1` | K -> degrees C (subtract 273.15) |
| A | `stl2` | `soil_temperature_level_2` | K -> degrees C (subtract 273.15) |
| A | `stl3` | `soil_temperature_level_3` | K -> degrees C (subtract 273.15) |
| A | `stl4` | `soil_temperature_level_4` | K -> degrees C (subtract 273.15) |
| A | `swvl1` | `volumetric_soil_water_layer_1` | m³/m³ -> m³/m³ |
| A | `swvl2` | `volumetric_soil_water_layer_2` | m³/m³ -> m³/m³ |
| A | `swvl3` | `volumetric_soil_water_layer_3` | m³/m³ -> m³/m³ |
| A | `swvl4` | `volumetric_soil_water_layer_4` | m³/m³ -> m³/m³ |
| A | `src` | `skin_reservoir_content` | m water equivalent -> mm (multiply by 1,000); retain sign |
| A | `snowc` | `snow_cover` | % unchanged, confirmed in live June 1950 preflight attributes; annual verification pending |
| A | `rsn` | `snow_density` | kg/m³ -> kg/m³ |
| A | `sd` | `snow_depth_water_equivalent` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `tp` | `total_precipitation` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `e` | `total_evaporation` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `evatc` | `evaporation_from_the_top_of_canopy` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `pev` | `potential_evaporation` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `sro` | `surface_runoff` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `ssro` | `sub_surface_runoff` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `sf` | `snowfall` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `smlt` | `snowmelt` | m water equivalent -> mm (multiply by 1,000); retain sign |
| B | `ssrd` | `surface_solar_radiation_downwards` | J/m² -> MJ/m² (divide by 1,000,000); retain sign |
| B | `strd` | `surface_thermal_radiation_downwards` | J/m² -> MJ/m² (divide by 1,000,000); retain sign |
| B | `ssr` | `surface_net_solar_radiation` | J/m² -> MJ/m² (divide by 1,000,000); retain sign |
| B | `str` | `surface_net_thermal_radiation` | J/m² -> MJ/m² (divide by 1,000,000); retain sign |
| B | `sshf` | `surface_sensible_heat_flux` | J/m² -> MJ/m² (divide by 1,000,000); retain sign |
| B | `slhf` | `surface_latent_heat_flux` | J/m² -> MJ/m² (divide by 1,000,000); retain sign |
| B | `evaow` after physical-identity correction | `evaporation_from_bare_soil` | m -> mm when water guard fires; retain sign |
| B | `evabs` after physical-identity correction | `evaporation_from_vegetation_transpiration` | m -> mm when water guard fires; retain sign |
| B | `evavt or evatp` after physical-identity correction | `evaporation_from_open_water_surfaces_excluding_oceans` | m -> mm when water guard fires; retain sign |

Air/dewpoint/wind/soil states are daily means except the named air minimum/maximum
and skin maximum. Channel B values are daily accumulations assigned from midnight
D+1 to weather day D. Snow-cover scale is not established by the supplied reduced
samples, which do not contain that variable; verify it in a live full-field pilot.
Evaporation signs remain the CDS signs; an agronomic demand convention is not imposed.

## Spatial weights and boundaries

| Field | Meaning / units |
|---|---|
| Source `GEOID` -> `county_fips` | Five-digit Census geographic identifier |
| Source `NAME`, `STATE_NAME` | Census county and state labels |
| `geometry` | County/cell polygons; EPSG:4326 input grid, EPSG:5070 intersection |
| Source `lat` -> `latitude` | Cell-center degrees north |
| Source `lon` -> `longitude` | Cell-center degrees east; western domain is negative |
| `intersection_area` | Intermediate EPSG:5070 area in m²; not persisted in supplied weights |
| `weight` | Positive intersection fraction, sum 1 within each county |

The original weight files have `county_fips, lat, lon, weight`. The loader exposes
`latitude, longitude` internally; recomputation writes those descriptive names.

## Manifest and quality reports

| Manifest field | Meaning |
|---|---|
| `timestamp` | Record creation time; new records use explicit UTC |
| `year` | Weather year |
| `channel` | A (daily statistics) or B (accumulated) |
| `subgroup` | Statistic or accumulated group |
| `file_path` | Recorded raw file path; historical paths resolved through the migration map |
| `file_size_bytes` | Stored raw file size |
| `sha256` | Raw file SHA-256 digest |
| `qc_status` | PASS or FAIL at record creation |
| `qc_notes` | Scope and result of performed checks |

Historical notes and timestamps are preserved literally and must not be interpreted
as modern full-year certification. `file_migration.json` records `old_path`,
`new_path`, and SHA-256 for each of the 40 preserved source files.

JSON validation reports contain an overall `status` and a `checks` array. Each check
contains `check` and `status`; contextual fields include `notes`, `year`, `file`,
`rows`, `counties`, `weather_variables` and `supplied_sample` where relevant. The
offline preflight explicitly records its reduced scope. A failed annual validation
or pipeline check exits unsuccessfully; no failed county output is silently accepted.

## Planned fields: not implemented

GDD (degrees C days), VPD proxy (kPa), heat-day indicators/counts (30 and 35 degrees C),
diurnal temperature range (degrees C), rolling precipitation and signed water-balance
totals (mm), and 5/10/30-day summaries remain specifications in
`feature_engineering.md`. Detrended yield anomalies, class thresholds, model
predictions, RMSE, MAE, out-of-sample R² and bootstrap intervals do not yet exist as
implemented data products. No planned variable is present in the current weather schema.
