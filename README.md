# Soybean yield thesis workspace

The workspace has two content sections:

- **[output/](output/README.md)** — the clean thesis project: research code,
  configuration, selected data, essential documentation and thesis narrative.
  This is the single maintained product, not a duplicate export of another code tree.
- **[work/](work/README.md)** — complete original datasets, raw/intermediate weather,
  development tests/docs, paper library, thesis preparation notes and scratch work.

The active target is **1951–2025: 135 counties × 75 years = 10,125 observations**.
1950 yield is excluded because the October 1949–October 1950 campaign lacks its
initial weather. Original source workbooks remain unchanged. Weather acquisition
still starts in 1950 to cover the first retained campaign, October 1950–October 1951.

Run research commands from `output/`; see its README. Export a clean ZIP from the
workspace root with `python work/export_delivery.py --output work/scratch/thesis_delivery_1951_2025.zip`.
Git metadata, agent instructions, ignore rules and the local Python environment
remain workspace infrastructure at the root. No internal material is in `output/`.
