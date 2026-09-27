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

### 1. Strict Chapter Role Demarcation
- *Chapter 1 (Introduction)*: Practical supply-chain motivation, research questions, 2 core contributions, and scope boundaries. Never anticipate equations, model hyperparameters, or exhaustive tables.
- *Chapter 2 (Literature Review)*: Critical review of existing agronomic, econometric, and ML literature, theoretical foundations, and formal derivation of hypotheses H1--H5. Focus strictly on established findings, methodological fallacies (trend confounding, spatial leakage, lack of parsimonious baselines), and why specific modalities are excluded.
- *Chapter 3 (Data and Study Area)*: Complete empirical provenance, spatial coverage, descriptive acreage, quality control, reanalysis extraction, and exploratory climate-yield statistics.
- *Chapter 4 (Methodology)*: Mathematical and algorithmic formalization of all workflows: 12 standardized horizons, feature engineering, detrending models, baseline specifications, machine learning architectures, and expanding-window validation protocols.
- *Chapters 5--7 (Results, Discussion, Conclusions)*: Empirical findings, baseline comparisons, ablations, operational implications, and syntheses.

### 2. Literature and Citation Discipline (Author Rule)
- **Primary Literature Perimeter**: The active bibliography is strictly limited to the **13 core papers** from `Literature.zip`.
- **One Paper per Topic**: In narrative reviews, cite **one primary reference paper per mini-topic/paragraph** (e.g., Sharma for tree review, Khaki for sequential models, Yin for sub-monthly aggregation). Avoid dense multi-citation stacking unless explicitly contrasting divergent findings.
- **Reserve Papers in Repository**: The 3 additional local papers in `work/references/papers/` (`Jeong et al. 2016`, `Iizumi et al. 2018`, `Khaki & Wang 2019`) are kept in reserve. Cite them **only** if a technical concept is uniquely present in them and absent from the 13 primary papers. Never cite them gratuitously and never upload them to the supervisor's OneDrive folder unless cited.
- **Zero External Additions**: Never add new papers without explicit author authorization.

### 3. Author Voice, Register, and Zero LLM Markers
- The author is an Italian data-science MSc student writing in professional C1 English.
- Adopt a lean, direct, and "asciutto" (dry) tone. Avoid verbose circumlocutions, nested subordinate clauses, filler transitions, and syntactic complexity.
- **Banned LLM Patterns**:
  - *No contrastive binary structures*: Ban `rather than`, `not X, but Y`, `instead of A, we do B`. State choices assertively and directly.
  - *No dramatic discourse adverbs*: Ban sentence starters like `Crucially,`, `Importantly,`, `Notably,`, `Remarkably,`, `Intriguingly,`. Start directly with the subject or factual observation.
  - *No AI cliché vocabulary*: Ban words such as `delve`, `foster`, `testament`, `tapestry`, `pivotal`, `intricate`, `multifaceted`, `nuanced`, `beacon`, `cornerstone`.
  - *No pre-emptive apologetics*: Ban defensive justifications (`To maintain strict methodological hygiene...`). State research protocols as the normal professional standard.
  - *Enforce burstiness*: Alternate concise, punchy factual statements with analytical periods; discourage subordinate clauses beyond the first degree (in strict compliance with Vademecum p. 5).
- Technical terms in Data Science, ML, and Econometrics must be rigorous and expert-level; define acronyms once upon first occurrence (e.g., *VPD, GDD, OOS, OLS*) and use the acronym exclusively thereafter.
- Use standard scientific authorial "we" (*"We evaluate...", "We observe..."*) and active functional subjects (*"The model estimates..."*); avoid singular "I" and passive convolutions.

### 4. Tables, Figures, and Vademecum Compliance
- **Source Attribution**: Every table and figure must include an explicit, formal source attribution line immediately below (e.g., `\textit{Source: Author's calculation on USDA NASS and ECMWF ERA5-Land data.}`).
- **Self-Contained Captions**: Captions must provide all necessary details for understanding (variables, units, periods, thresholds).
- **Text and Table Coordination**: Surrounding text must provide critical commentary and analytical synthesis instead of repeating tabular numbers verbatim.
- **Academic Defense of Exclusions**: Methodological boundaries (weather-only, excluding satellites, management records, static soil features, or dynamical forecasts) must be justified through rigorous academic rationale in the narrative.



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
