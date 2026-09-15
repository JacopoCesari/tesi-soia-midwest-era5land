# Data, panel selection and target

## Primary thesis panel

On 2026-09-15 the author chose **1951–2025 inclusive**, prioritizing the greatest
available number of crop years without model comparison. Within the supplied
479-county population, **135 counties** have a finite SURVEY yield in every one
of the **75 years**, yielding **10,125 observations**. This selected subset is a
balanced panel. The full 479-county 1950–2025 export remains incomplete.

The 1950 yield is explicitly excluded because its October 1949–October 1950
campaign lacks the initial weather. The first retained yield is 1951; keep weather
from 1950 for the October 1950–October 1951 campaign. Removing 1950 retains the
same 135 counties. Original workbook rows for 1950 remain unchanged.

`data/target/soybean_yield_1951_2025.csv` contains only `county_fips`, `year` and
`yield_bu_per_acre`. There is no imputation, CENSUS substitution or new acreage
threshold. The same selected keys have complete harvested acreage, stored
separately in `data/auxiliary/acres_harvested_1951_2025.csv`.

| State | Selected counties | Original counties |
|---|---:|---:|
| Illinois | 28 | 98 |
| Indiana | 20 | 87 |
| Iowa | 35 | 98 |
| Minnesota | 19 | 63 |
| Missouri | 6 | 77 |
| Ohio | 27 | 56 |
| Total | 135 | 479 |

`scripts/prepare_thesis_data.py` reads the immutable `all_years_survey` sheets,
retains counties with finite yield in every year, verifies exact Cartesian-product
coverage and unique keys, checks matching acreage and subsets the original weights.
`data/auxiliary/panel_provenance.json` records source/output SHA-256 hashes and counts.
Repeated preparation verifies identical files and refuses conflicting outputs.

Selection is retrospective, conditional on reporting through 2025 and on the earlier
479-county population. It does not establish nationwide coverage or a cohort selected
using only information available at historical forecast origins. More calendar years
is the author's design priority, not demonstrated predictive superiority.

## Preserved complete source exports

Original workbooks and Census boundaries remain under
`../work/data/external/usda_nass_and_county_boundaries/`, byte for byte unchanged and excluded
from the professor delivery ZIP. Both workbooks contain **35,487 SURVEY records**
in `all_years_survey`: 29,219 historical rows and 6,268 available 2011–2025 rows.
The 25,362 observations outside the new selected panel are preserved there.

The original `balanced_1950_2010` sheets retain 479 counties × 61 years = 29,219
complete measurements. Their sheet names and source README text are historical
metadata, not the active thesis configuration. The full 1950–2025 export has 917
absent county-year pairs, with identical yield/acreage availability masks.
Ancillary `cv_percent` is missing in the old complete window; completeness refers
to yield/acreage measurements, not every metadata field.

`data/nass.py` reads all supplied records by default; `historical` and `recent`
selectors explicitly inspect the old time windows. Source ingestion alone does
not select the active panel. `data/panel.py` handles selection and validation.
The acreage workbook's `census_optional` sheet is never merged into SURVEY.
The supplied duplicate candidate CSV is also retained, with no new duplicate made.

## Earlier selection provenance

Original notes document six states with historically persistent soybean cultivation,
continuous yield and acres harvested over 1950–2010, and the Ste. Genevieve FIPS
harmonization to `29186`. The author reports an earlier low-acreage exclusion.
Its threshold and reference period, original selection script and preselection
universe have not been recovered. Do not infer that continuity was the only filter,
or present an invented acreage threshold as reproducible fact.

`median_acres_1950_2025` in the supplied candidate list remains at source and
must not become a predictor. This preparation uses no such summary to filter counties.

## Target and auxiliary roles

Yield in BU/ACRE is the continuous target. Conversion to t/ha, detrending and later
class thresholds belong to future training-only modeling stages. Acres harvested
is sample documentation only: never a predictor, target, observation weight or
evaluation weight. County names and the 3,344 selected spatial intersections are
auxiliary data. Weather features will live in `data/processed/model_datasets/`.
Source labels and units are mapped in the [dictionary](data_dictionary.md).
