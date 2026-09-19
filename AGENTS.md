# Instructions for research and coding agents

## Sources of truth

1. `output/docs/research_protocol.md` contains the consolidated protocol. Read the relevant
   research design, data/target, ERA5 methodology, feature, validation, dictionary
   and decision documents before editing their implementation.
2. Source code establishes what is actually implemented. Verify it before describing
   behavior; distinguish local tests, simulated responses and completed live runs.
3. External notes may suggest new directions, but only consolidated author-approved
   methodological decisions belong in active repository documentation.
4. Git history is the version archive. Do not create historical copies or archive folders.

## Attribution, publication boundaries and simplicity

- When a method, formula, implementation or argument clearly draws on another
  paper, explicitly annotate that attribution near the relevant code or text and
  connect it to the reference catalog. Distinguish borrowed material from this
  project's choices; do not invent or imply an unverified citation.
- Keep the final material shared with the professor minimal, clean and easy to
  navigate: working research code, essential usage documentation, publishable data
  and the thesis writing layer.
- Clearly separate internal coding-agent operations, development-only documentation,
  internal validation tests and their artifacts, scratch work and other supporting
  development files. Retain them for development, but exclude them from the material
  intended for publication or delivery to the professor.
- Separate explanatory research notes intended to support thesis writing from both
  internal development notes and the actual thesis narrative.
- Keep publishable auxiliary data distinct from the datasets used by the models.
  Make each area's purpose and publication status explicit; folder separation alone
  must not be mistaken for exclusion from a published repository or delivery.
- Prefer minimal, DRY code and structure: preserve behavior and information while
  avoiding redundant modules, documents and copies of the final product. Keep
  technical documentation concise; thesis-oriented explanations may be longer when
  needed to preserve reasoning, attribution and methodological detail.
- Maintain repository cleanliness and organization: prevent note scattering across
  disconnected files and eliminate redundant text or boilerplate across thesis chapters.
  Centralize working notes, research decisions and methodological parameters in designated
  directories (`work/thesis_notes/` and `output/docs/decisions.md`), ensuring the thesis
  narrative remains concise, structured, and non-repetitive.

## Thesis writing protocol and chapter scoping

- **Strict Chapter Role Demarcation**:
  - *Chapter 1 (Introduction)*: High-level motivation, practical supply-chain value, research questions, high-level objectives, academic contributions, and scope boundaries. Never anticipate detailed experimental recipes, equations, or exhaustive model configurations.
  - *Chapter 2 (Literature Review)*: Critical review of existing agronomic, econometric, and ML literature, theoretical foundations, and formal derivation of hypotheses H1--H5. Focus strictly on what existing literature has established, where methodological fallacies lie (e.g., trend confounding, spatial leakage, lack of parsimonious baselines), and why specific data modalities are excluded. Do not describe internal implementation mechanics or anticipate Chapter 4 pipelines.
  - *Chapter 3 (Data and Study Area)*: Complete empirical provenance, spatial coverage, descriptive acreage, quality control, reanalysis extraction, and exploratory climate-yield statistics.
  - *Chapter 4 (Methodology)*: Mathematical and algorithmic formalization of all workflows: 12 standardized horizons, feature engineering, detrending models, baseline specifications, machine learning architectures, and expanding-window temporal validation protocols.
  - *Chapters 5--7 (Results, Discussion, Conclusions)*: Empirical findings, baseline comparisons, ablations, operational implications, and syntheses.
- **Academic Defense of Methodological Exclusions**:
  - Negative constraints (e.g., weather-only, excluding satellites, management records, static soil features, or dynamical climate forecasts) must be justified through rigorous academic rationale in the thesis narrative: ex-post compilation lags, physiological satellite latency ($H \ge 4$), lack of multidecadal county depth, zero temporal variance of static pedology ($\text{Var}(X_{it})=0$), and avoiding confounded dynamical climate model biases.
- **Text and Table Coordination (No Redundancy)**:
  - When comparative or diagnostic tables synthesize empirical dimensions (such as benchmark comparisons or hyperparameter grids), surrounding text must provide critical commentary and analytical synthesis rather than repeating tabular metrics verbatim.
- **Density Over Expansion (Non-Proliferation Rule)**:
  - Text revisions must maximize conceptual density, clarity, and grammatical cohesion without expanding word or character counts. Avoid slide-like bullet-point lists where fluent, connected academic prose can express the ideas more cohesively.

## Scientific constraints

- The main design is weather-only: no markets, futures, macroeconomic data, remote
  sensing, proprietary crop platforms, seasonal forecasts or future-weather completion.
- The author selected the primary balanced panel 1951–2025: 75 years, 135 counties,
  10,125 finite yield observations, with matching complete auxiliary acreage.
  This supersedes the 1950–2010 primary scope. Never describe the full original
  479-county 1950–2025 export as balanced. No imputation is permitted.
- Weather production targets 1950–2025 and the selected 135-county weights. Preserve
  the complete original exports separately in `work/data/`; prepared targets live
  in `output/data/target/` and publishable auxiliaries in `output/data/auxiliary/`.
  Exclude 1950 yield because October 1949 weather is missing. Keep 1950 weather
  for the first retained October 1950–October 1951 campaign.
- The primary target is yield or a training-only detrended anomaly. Acres harvested
  is for panel construction only, never a predictor, target or weight.
- Use 12 separate monthly horizons before October, confirmed by the author. The
  exact harvest day remains open; never invent it. Weather after the cutoff is invalid.
- Use complete-year expanding windows. The initial training window remains unset;
  there is no approved 40-year default or fixed 60/40 split. Preprocessing, tuning
  and any classification thresholds are train-only.
- No random county-year split is accepted. Do not assume deep learning superiority.

## Implementation transparency

The supplied NetCDF samples have seven June days in 1950 and 2025, with reduced
channel A variables. Their original county Parquet files have six aligned days and
21 fields. New preflight configuration requests seven aligned days and 37 fields.
Offline tests and simulated full-year responses do not establish live CDS production
readiness. Feature engineering, model fitting, metrics and bootstrap are future work.
The authorized live June 1950 preflight passed on 2026-09-15 (945 rows, 135 counties,
37 fields). The 2025 preflight and 1950/1952 annual pilots are not yet complete.
Live channel A cost-limit failures require resumable month/day blocks; do not restore
a single full-year daily-statistics request based on offline tests alone.
There are two content sections: `output/` is the single maintained clean thesis
project; `work/` contains originals, operations, tests and writing notes. Run research
commands from `output/`. `work/export_delivery.py` exports only approved output files.
The full Git tree still contains work material and is not the delivery ZIP.

## Data and repository integrity

- Preserve original CSV, Excel, geometry, NetCDF, Parquet, weights, manifest and PDFs
  byte for byte. The author authorized the documented English-path migration and
  scoped `.gitignore` updates in this reorganization; this is not permission for
  future source-data rewriting or deletion.
- Original workbook/source-document language and external field names are immutable
  source content. Explain them in English at ingestion boundaries and in documentation.
- Keep the supplied historical manifest read-only. Its path map is
  `work/data/external/file_migration.json`; new operations use separate manifests.
- Never read or change credentials unnecessarily. Ordinary tests must work offline.
- Ask before any CDS download. Do not launch a historical run without explicit approval.
- Do not work on external repositories or separate feature branches; always work directly on the `main` branch (the author is the sole contributor). Never commit or push unless expressly requested.
- Make repository-owned documentation, code, messages and comments English. Use clear
  snake_case internally; preserve source codes only at ingestion boundaries.
