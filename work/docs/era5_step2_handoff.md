# ERA5 step 2: paused handoff

## STOP override — 2026-09-16 11:09 Europe/Rome

The author stopped all ERA5 acquisition and requested evaluation of ARCO/Zarr.
This notice supersedes every resume instruction below. Do not restart either
pilot or any historical download without renewed authorization.

All local step-2 processes were terminated and their absence verified.
CDS request `67c4c406-3765-4060-bb8c-974bf8523ab5` was explicitly deleted:
the API returned `dismissed`, and a subsequent lookup returned HTTP 404.
It cannot be recovered using the previous snippet. No replacement was submitted.
The February 15-28 block was not downloaded. The four existing production PASS
records and all validated data remain unchanged; 1952 has not started.
The scratch runner `work/scratch/resume_step2_20260916.py` is stopped and must
not be relaunched. Its log is `work/scratch/resume_step2_20260916.log`.
ARCO feasibility is under review only; no replacement pipeline is approved.

Earlier snapshot: 2026-09-16, approximately 08:08 Europe/Rome.

## Scope and stop condition

The author requested a pause before shutting down the computer. All local pilot
processes and the automatic continuation worker were stopped and their absence
verified. Do not resume until the author asks the next chat to continue.
Existing authorization covers only the full-year 1950 and 1952 pilots (step 2).
Finish both with successful validation, then STOP. Never run `--primary-period`
or other years. Do not commit/push without a new request.

Workspace: `C:/Users/JacopoCesari-Aretésr/Desktop/Tesi`.
Branch: `codex/research-project-reorganization`; pushed commit `d6c8e67`.
Read root `AGENTS.md` and `output/docs/era5_land_methodology.md`; older readiness
statements in those files predate this snapshot. Commands run from `output/` using
`../.venv/Scripts/python.exe`. Use `PYTHONDONTWRITEBYTECODE=1`.

## Verified progress

- Step 1 is complete: both real June 1950 and June 2025 preflights passed, each
  with seven days, 135 counties, 945 rows and all 37 weather fields. See
  `work/reports/experiments/quality_control/preflight.json` and the two sidecars in
  `work/data/interim/county_daily_weather/preflight/`.
- The 1950 annual pilot is incomplete. Its 17 daily-mean variables are downloaded
  and validated for January 1-31 and February 1-14. This does NOT mean all 37
  fields are complete for those dates. Minimum/maximum and accumulated annual
  groups remain pending, as do later daily means.
- The 1952 annual pilot has not started. No complete historical run was started.
- The production manifest has four PASS records: January 1-15, January 16-31,
  their merged January block, and February 1-14. All four checksums were checked
  at this pause. Files live under
  `work/data/raw/era5_land/production/raw_daily_stats/chunks/1950/`.
- Windows rebooted at 03:31 on September 16, terminating the first run. A completed
  remote January 16-31 result was recovered without a new CDS request; the local
  pipeline resumed at 07:27. This pause at the author's request is intentional.

## Recover the existing remote request before resuming

The last submitted request is **67c4c406-3765-4060-bb8c-974bf8523ab5**:
1950 February 15-28, daily means, 17 variables, area
`[45.8, -96.8, 36.0, -80.5]`, hourly source sampling, UTC.
At the pause it was `accepted` (queued), created around 07:46 local time.
It was NOT cancelled remotely: CDS can process it while this computer is off.
Do not resubmit the block blindly. First inspect it with
`cdsapi.Client().client.get_remote(request_id)` and check `.status` and `.request`.
If still queued/running, wait on that request. If successful, download its result
and pass it through the existing block validation/manifest routine. The following
Python snippet, run from `output/`, reuses that result without submitting a job:

```python
from pathlib import Path
import cdsapi
from soybean_yield_forecasting.configuration import load_configuration
from soybean_yield_forecasting.era5.download import CDSDownloader
from soybean_yield_forecasting.era5.manifest import ManifestManager
from soybean_yield_forecasting.era5.variables import DATASET_DAILY_STATS

config = load_configuration(Path("configs/data.yaml"))
raw = config["paths"]["raw"]
client = cdsapi.Client(quiet=True, timeout=60, retry_max=2)
remote = client.client.get_remote("67c4c406-3765-4060-bb8c-974bf8523ab5")
assert remote.status == "successful", "Wait or inspect the existing request first"
expected = remote.request
assert expected["year"] == "1950" and expected["month"] == ["02"]
assert expected["day"] == [str(day) for day in range(15, 29)]
assert expected["daily_statistic"] == "daily_mean"
assert len(expected["variable"]) == 17
assert expected["area"] == [45.8, -96.8, 36.0, -80.5]

def existing_result(dataset, request):
    assert dataset == DATASET_DAILY_STATS and request == expected
    return remote

client.retrieve = existing_result
downloader = CDSDownloader(
    raw, ManifestManager(raw / "manifest.csv"),
    verify_checksum=True, area=expected["area"],
)
downloader._client = client
print(downloader._download_daily_chunk(
    1950, 2, "daily_mean", expected["variable"], list(range(15, 29)),
))
```

This is a one-off recovery using the current implementation, not a new public API.
If the remote result has expired or failed, inspect the error before resubmitting.
Preserve all existing valid files and manifest records. Never print credentials.

## Resume only step 2

After recovering the outstanding block and ensuring no duplicate processes exist,
run the following PowerShell from `output/`. Each failure stops the sequence:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
foreach ($pilotYear in @(1950, 1952)) {
    & '..\.venv\Scripts\python.exe' scripts/download_era5_land.py --test-year $pilotYear --verify-checksum
    if ($LASTEXITCODE -ne 0) { throw "Pilot $pilotYear failed; inspect its log." }
    & '..\.venv\Scripts\python.exe' scripts/validate_downloaded_data.py --year $pilotYear --verify-checksum
    if ($LASTEXITCODE -ne 0) { throw "Pilot $pilotYear validation failed." }
}
# STOP HERE. No full-history command.
```

Keep durable stdout/stderr logs under `work/scratch/`. If launched in the background,
use a hidden process and verify it started. Do not reuse old PID numbers or restart
`continue_pilots.ps1` against an old PID. Nothing restarts automatically after boot.
The computer must remain awake and connected for local downloading/aggregation.

Annual acceptance: 135 counties, 37 weather fields, exact calendar, no missing or
non-finite county values, physical checks, raw checksums and correct year-end
accumulation boundaries. Expected rows: 49,275 for 1950; 49,410 for 1952.
Resume already reused validated chunks successfully. The last code suite passed
57 offline tests; those do not substitute for annual live acceptance.

Earlier logs: `work/scratch/pilots_resume_20260916_072722_1950.stderr.log` and
`work/scratch/pilots_resume_20260916_072722_continuation.stdout.log`.
The old `pipeline_1950.json` failure report dated September 15 at 22:19 belongs to
the oversized annual request before the block-splitting fix; do not treat it as a
new failure. No annual output exists at this snapshot.
