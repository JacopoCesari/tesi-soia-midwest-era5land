"""Append-only download records and portable, read-only historical lookup."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .file_io import compute_sha256


class ManifestManager:
    """Track files; an optional migration map resolves immutable historical records."""

    columns = [
        "timestamp",
        "year",
        "channel",
        "subgroup",
        "file_path",
        "file_size_bytes",
        "sha256",
        "qc_status",
        "qc_notes",
    ]

    def __init__(
        self,
        manifest_csv: Path,
        *,
        path_map: Path | None = None,
        repository_root: Path | None = None,
        read_only: bool = False,
    ) -> None:
        self.path = Path(manifest_csv)
        self.repository_root = Path(repository_root or Path.cwd()).resolve()
        self.read_only = read_only
        self.path_map = {}
        if path_map is not None:
            entries = json.loads(Path(path_map).read_text(encoding="utf-8"))
            self.path_map = {entry["old_path"]: entry["new_path"] for entry in entries}

    def resolve_path(self, recorded_path: str) -> Path:
        """Resolve a recorded path without rewriting the historical CSV."""
        normalized = recorded_path.replace("\\", "/")
        path = Path(self.path_map.get(normalized, normalized))
        return (path if path.is_absolute() else self.repository_root / path).resolve()

    def is_done(self, file_path: Path, verify_checksum: bool = False) -> bool:
        """Accept only the latest PASS record and optionally require a matching digest."""
        file_path = Path(file_path).resolve()
        # Small spatial/day blocks can be valid below 1 KB; use recorded QC and size.
        if not file_path.exists() or file_path.stat().st_size == 0 or not self.path.exists():
            return False
        try:
            records = pd.read_csv(self.path).fillna("")
            matching = records[records.file_path.map(self.resolve_path) == file_path]
            if matching.empty:
                return False
            latest = matching.iloc[-1]
            if latest.qc_status != "PASS" or int(latest.file_size_bytes) != file_path.stat().st_size:
                return False
            if verify_checksum:
                return bool(latest.sha256) and compute_sha256(file_path) == latest.sha256
            return True
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            return False

    def record(
        self, year: int, channel: str, subgroup: str, file_path: Path, qc_status: str, notes: str = ""
    ) -> None:
        """Append a portable record; never change an existing record in place."""
        if self.read_only:
            raise PermissionError("The historical manifest is read-only")
        file_path = Path(file_path).resolve()
        try:
            recorded = file_path.relative_to(self.repository_root).as_posix()
        except ValueError:
            recorded = file_path.as_posix()
        size = file_path.stat().st_size if file_path.exists() else 0
        row = dict(
            zip(
                self.columns,
                [
                    datetime.now(timezone.utc).isoformat(),
                    year,
                    channel,
                    subgroup,
                    recorded,
                    size,
                    compute_sha256(file_path) if size else "",
                    qc_status,
                    notes,
                ],
            )
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row], columns=self.columns).to_csv(
            self.path, mode="a", header=not self.path.exists(), index=False
        )
