"""Resume checks are portable and fail closed on missing or failed integrity records."""

import pandas as pd
from soybean_yield_forecasting.era5.manifest import ManifestManager


def test_latest_failed_record_prevents_resume(tmp_path):
    data = tmp_path / "sample.nc"
    data.write_bytes(b"A" * 2000)
    manifest = ManifestManager(tmp_path / "manifest.csv")
    manifest.record(1950, "A", "daily_mean", data, "PASS")
    manifest.record(1950, "A", "daily_mean", data, "FAIL")
    assert not manifest.is_done(data)


def test_missing_checksum_cannot_pass_verification(tmp_path):
    data = tmp_path / "sample.nc"
    data.write_bytes(b"A" * 2000)
    manifest = ManifestManager(tmp_path / "manifest.csv")
    manifest.record(1950, "A", "daily_mean", data, "PASS")
    records = pd.read_csv(manifest.path)
    records["sha256"] = ""
    records.to_csv(manifest.path, index=False)
    assert not manifest.is_done(data, verify_checksum=True)


def test_historical_manifest_resolves_after_migration(project_root):
    manifest = ManifestManager(
        project_root / "../work/data/raw/era5_land/manifest.csv",
        path_map=project_root / "../work/data/external/file_migration.json",
        repository_root=project_root.parent,
        read_only=True,
    )
    for row in pd.read_csv(manifest.path).itertuples():
        assert manifest.is_done(manifest.resolve_path(row.file_path), verify_checksum=True)
