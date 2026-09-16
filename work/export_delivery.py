"""Export only approved delivery paths, leaving the full working repository intact."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Explicit inclusion boundary: never recursively copy the working repository.
INCLUDE = (
    "README.md",
    "pyproject.toml",
    "configs/data.yaml",
    "configs/features.yaml",
    "configs/modeling.yaml",
    "requirements/base.txt",
    "requirements/geospatial.txt",
    "requirements/machine_learning.txt",
    "requirements/deep_learning.txt",
    "src/**/*.py",
    "scripts/*.py",
    "docs/*.md",
    "data/README.md",
    "data/target/README.md",
    "data/target/soybean_yield_1951_2025.csv",
    "data/auxiliary/README.md",
    "data/auxiliary/counties.csv",
    "data/auxiliary/acres_harvested_1951_2025.csv",
    "data/auxiliary/spatial_weights.csv",
    "data/auxiliary/panel_provenance.json",
    "thesis/README.md",
    "thesis/outline.md",
    "thesis/references.bib",
    "thesis/chapters/*.md",
    "thesis/figures/README.md",
    "thesis/tables/README.md",
)


def delivery_files(root: Path) -> list[Path]:
    """Require each approved pattern to resolve, and refuse external/symlink files."""
    root = root.resolve()
    selected = set()
    for pattern in INCLUDE:
        matches = [path for path in root.glob(pattern) if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"Missing delivery input: {pattern}")
        for path in matches:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Delivery input must be an ordinary local file: {path}")
            selected.add(path)
    return sorted(selected)


def export_delivery(root: Path, output: Path) -> int:
    """Create a fresh ZIP; existing exports are never silently overwritten."""
    root = root.resolve()
    files = delivery_files(root)
    if output.suffix.lower() != ".zip":
        raise ValueError("Delivery output must be a ZIP file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Delivery ZIP failed integrity verification")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    total = export_delivery(Path(__file__).resolve().parents[1] / "output", arguments.output)
    print(f"Exported {total} files to {arguments.output}")
