# Internal work area

Everything here is excluded from the clean project in `../output/` and its delivery ZIP.

- `data/external/`: complete original USDA NASS exports, Census boundaries and migration map.
- `data/raw/`, `data/interim/`: original weather fixtures/weights and future working weather files.
- `tests/`, `docs/`, `notebooks/`: offline tests, implementation records and exploration.
- `references/papers/`: unchanged scientific PDFs.
- `thesis_notes/`: explanatory writing notes, source map and university guide.
- `reports/`: QC and experimental outputs.
- `scratch/`, `scratch_remaining/`, `cache/`: generated local work; never delivery inputs.
- `export_delivery.py`: explicit allowlist for a clean ZIP of the current output project.

From the workspace root, using the activated local environment:

```powershell
python -m pytest -c output/pyproject.toml work/tests -q
python -m ruff check --config output/pyproject.toml output/src output/scripts work/tests work/export_delivery.py
python work/export_delivery.py --output work/scratch/thesis_delivery_1951_2025.zip
```

Run pipeline commands from `output/`. `python scripts/run_preflight.py --verify-checksum`
checks original 479-county fixtures offline. The explicit `--download` variant uses
the configured 135-county selection and contacts CDS only after authorization.
Original sources retain their SHA-256 hashes; the migration map updates paths only.
Older scratch analyses or ZIPs are working artifacts, not the current thesis dataset.
