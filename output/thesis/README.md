# Thesis narrative

## Format and language specifications

- **Language:** English (formal academic style, clear and concise sentences).
- **Format:** LaTeX (`main.tex` with modular chapters in `chapters/`).
- **Typography & Layout (Unibo Vademecum compliant):**
  - Font: Computer Modern, 12pt (10pt for footnotes).
  - Margins: 2.5 cm on all sides (`\usepackage[margin=2.5cm]{geometry}`).
  - Line spacing: 1.5 (`\usepackage{setspace}`, `\onehalfspacing`).
  - Text alignment: Justified.
  - Length constraint: Total thesis text $\le$ 100,000 characters (~35 pages text, excluding footnotes, front matter, tables, figures, bibliography). Introduction target: ~2 pages (~800–1,200 words).
  - Citations: Author-year format referencing `references.bib`.

Use the [outline](outline.md), `chapters/`, `figures/`, `tables/` and
`references.bib` for the thesis itself.

The methodological source of truth is `../docs/`. Working explanations and the
chapter source map live separately in `../../work/thesis_notes/`; the university
guide and paper library are retained in the work area. They are not thesis outputs.

Only add results backed by completed experiments. Figures and tables should identify
their generating script, configuration and data. Avoid duplicate copies of the same
final asset and never invent thesis results or citations.
