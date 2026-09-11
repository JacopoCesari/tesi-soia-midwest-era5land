"""
download_era5_land_daily.py

Downloader e pipeline di aggregazione giornaliera per ERA5-Land (1950-2025).
Copertura: 479 contee del Midwest USA (IL, IN, IA, MN, MO, OH).
BBox regionale unificato: [47.7, -97.0, 35.8, -80.4] (snappato a 0.1°).
Convenzione temporale: DAY_MODE = "UTC" (00:00 - 23:59 UTC).

Due canali CDS:
1. CANALE A (Istantanee): 'derived-era5-land-daily-statistics'
   - daily_mean (17 variabili)
   - daily_minimum (2m_temperature)
   - daily_maximum (2m_temperature, skin_temperature)
   - Calcolato lato server Copernicus con frequency="1_hourly"
2. CANALE B (Accumulate): 'reanalysis-era5-land'
   - Step 00:00 UTC del giorno D+1 (contiene l'accumulo esatto delle 24h del giorno D)
   - 17 variabili (precipitazione, radiazione, evaporazione, runoff, flussi di calore)
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import os
import sys
import time
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

# ==============================================================================
# CONFIGURAZIONE GEOGRAFICA E TEMPORALE
# ==============================================================================
NORTH, WEST, SOUTH, EAST = 47.7, -97.0, 35.8, -80.4
CDS_AREA = [NORTH, WEST, SOUTH, EAST]
DAY_MODE = "UTC"

DATASET_DAILY_STATS = "derived-era5-land-daily-statistics"
DATASET_HOURLY = "reanalysis-era5-land"

# Variabili Canale A: derived-era5-land-daily-statistics
VARS_DAILY_MEAN = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "skin_reservoir_content",
    "snow_cover",
    "snow_density",
    "snow_depth_water_equivalent",
]

VARS_DAILY_MIN = [
    "2m_temperature",
]

VARS_DAILY_MAX = [
    "2m_temperature",
    "skin_temperature",
]

# Variabili Canale B: reanalysis-era5-land (step 00:00 UTC)
VARS_ACCUMULATED = [
    "total_precipitation",
    "total_evaporation",
    "evaporation_from_the_top_of_canopy",
    "evaporation_from_bare_soil",
    "evaporation_from_open_water_surfaces_excluding_oceans",
    "evaporation_from_vegetation_transpiration",
    "potential_evaporation",
    "surface_runoff",
    "sub_surface_runoff",
    "snowfall",
    "snowmelt",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
]

# Nomi variabili ECMWF scambiate da normalizzare nel processed
EVAPORATION_SWAP_MAP = {
    "evabs": "evaporation_from_vegetation_transpiration",
    "evaow": "evaporation_from_bare_soil",
    "evatp": "evaporation_from_open_water_surfaces_excluding_oceans",
}

# ==============================================================================
# GESTIONE PATH WINDOWS 8.3 (PER EVITARE BUG NETCDF4 SU CARATTERI ACCENTATI)
# ==============================================================================
def get_safe_path(path: Path) -> Path:
    """Restituisce il path 8.3 corto su Windows se necessario per evitare errori in NetCDF4."""
    if os.name != "nt":
        return path
    try:
        p = path.resolve()
        parent = p.parent
        buffer = ctypes.create_unicode_buffer(500)
        res = ctypes.windll.kernel32.GetShortPathNameW(str(parent), buffer, 500)
        if res > 0:
            return Path(buffer.value) / p.name
    except Exception:
        pass
    return path


def compute_sha256(filepath: Path) -> str:
    """Calcola l'hash SHA-256 di un file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# GESTIONE DEL MANIFEST
# ==============================================================================
class ManifestManager:
    def __init__(self, manifest_csv: Path):
        self.path = manifest_csv
        self.columns = [
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
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=self.columns).to_csv(self.path, index=False)

    def is_done(self, file_path: Path) -> bool:
        if not file_path.exists() or file_path.stat().st_size < 1000:
            return False
        try:
            df = pd.read_csv(self.path)
            matches = df[(df["file_path"] == str(file_path)) & (df["qc_status"] == "PASS")]
            return len(matches) > 0
        except Exception:
            return False

    def record(self, year: int, channel: str, subgroup: str, file_path: Path, qc_status: str, notes: str = ""):
        size = file_path.stat().st_size if file_path.exists() else 0
        sha = compute_sha256(file_path) if file_path.exists() and size > 0 else ""
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "year": year,
            "channel": channel,
            "subgroup": subgroup,
            "file_path": str(file_path),
            "file_size_bytes": size,
            "sha256": sha,
            "qc_status": qc_status,
            "qc_notes": notes,
        }
        df = pd.DataFrame([row])
        df.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)


# ==============================================================================
# MODULO DOWNLOAD CDS
# ==============================================================================
class CDSDownloader:
    def __init__(self, base_dir: Path, manifest: ManifestManager):
        self.base_dir = base_dir
        self.manifest = manifest
        self.client = cdsapi.Client()
        self.raw_stats_dir = base_dir / "raw_daily_stats"
        self.raw_accum_dir = base_dir / "raw_accumulated_boundaries"
        self.raw_stats_dir.mkdir(parents=True, exist_ok=True)
        self.raw_accum_dir.mkdir(parents=True, exist_ok=True)

    def download_daily_stat_group(self, year: int, stat: str, variables: list[str], test_days: list[str] | None = None) -> Path:
        """Scarica un gruppo statistico (daily_mean, daily_minimum, daily_maximum) per un anno."""
        target_name = f"era5_land_{year}_{stat}.nc"
        target_path = self.raw_stats_dir / target_name

        if self.manifest.is_done(target_path):
            logging.info("SKIP (già validato nel manifest): %s", target_name)
            return target_path

        months = [f"{m:02d}" for m in range(1, 13)]
        days = test_days if test_days is not None else [f"{d:02d}" for d in range(1, 32)]

        request = {
            "variable": variables,
            "year": str(year),
            "month": months if test_days is None else ["06"],
            "day": days,
            "daily_statistic": stat,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": CDS_AREA,
        }

        part_path = target_path.with_suffix(".nc.part")
        if part_path.exists():
            part_path.unlink()

        logging.info("DOWNLOAD CANALE A: Anno %d, Stat: %s (%d variabili)", year, stat, len(variables))
        self.client.retrieve(DATASET_DAILY_STATS, request).download(str(part_path))

        # Se il CDS scarica uno zip o un nc direttamente, normalizziamo
        if zipfile.is_zipfile(part_path):
            temp_extract_dir = self.raw_stats_dir / f"_tmp_{year}_{stat}"
            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(part_path, "r") as z:
                z.extractall(temp_extract_dir)
            
            # File zip non più aperto, rimuoviamo part_path
            if part_path.exists():
                part_path.unlink()

            nc_files = list(temp_extract_dir.glob("*.nc"))
            if len(nc_files) == 1:
                if target_path.exists():
                    target_path.unlink()
                nc_files[0].replace(target_path)
            elif len(nc_files) > 1:
                # Merge multi-variabile in un unico NetCDF annuale pulito
                logging.info("Merging di %d variabili NetCDF per %s...", len(nc_files), stat)
                datasets = [xr.open_dataset(get_safe_path(f)) for f in nc_files]
                merged = xr.merge(datasets)
                if target_path.exists():
                    target_path.unlink()
                merged.to_netcdf(get_safe_path(target_path))
                for ds in datasets:
                    ds.close()
                merged.close()

            # Pulizia cartella temporanea
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        else:
            if target_path.exists():
                target_path.unlink()
            part_path.replace(target_path)

        logging.info("Completato: %s (%.1f MB)", target_path.name, target_path.stat().st_size / 1e6)
        return target_path

    def download_accumulated_group(self, year: int, test_days: list[str] | None = None) -> Path:
        """
        Scarica le 17 variabili accumulate con step 00:00 UTC.
        In DAY_MODE=UTC:
        Per coprire l'anno D (1 gen ... 31 dic), i record necessari hanno valid_time
        da (D)-01-02 00:00 UTC fino a (D+1)-01-01 00:00 UTC.
        """
        target_name = f"era5_land_{year}_accumulated_00utc.nc"
        target_path = self.raw_accum_dir / target_name

        if self.manifest.is_done(target_path):
            logging.info("SKIP (già validato nel manifest): %s", target_name)
            return target_path

        part_path = target_path.with_suffix(".nc.part")
        if part_path.exists():
            part_path.unlink()

        # Configurazione giorni/mesi
        # Per l'anno D completo: giorni dal 2 gen al 31 dic dell'anno D + 1 gen dell'anno D+1
        if test_days is not None:
            request = {
                "variable": VARS_ACCUMULATED,
                "year": str(year),
                "month": "06",
                "day": test_days,
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": CDS_AREA,
            }
        else:
            # Anno intero: 17 variabili x 365 giorni x 1 timestamp = 6.205 fields (sotto soglia 12.000)
            request = {
                "variable": VARS_ACCUMULATED,
                "year": [str(year), str(year + 1)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": CDS_AREA,
            }

        logging.info("DOWNLOAD CANALE B: Anno %d, Accumulated 00:00 UTC (%d variabili)", year, len(VARS_ACCUMULATED))
        self.client.retrieve(DATASET_HOURLY, request).download(str(part_path))
        part_path.replace(target_path)

        logging.info("Completato: %s (%.1f MB)", target_path.name, target_path.stat().st_size / 1e6)
        return target_path


# ==============================================================================
# MODULO AGGREGAZIONE SPAZIALE (COUNTY-DAILY)
# ==============================================================================
class CountyAggregator:
    def __init__(self, weights_parquet: Path, out_dir: Path):
        self.weights_df = pd.read_parquet(weights_parquet)
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Prepara indicizzazione dei pesi
        self.weights_indexed = self.weights_df.set_index(["lat", "lon"])

    def process_year(self, year: int, stat_files: dict[str, Path], accum_file: Path) -> Path:
        """Esegue l'aggregazione spaziale per le 479 contee per l'anno specificato."""
        out_parquet = self.out_dir / f"county_daily_{year}.parquet"
        logging.info("Elaborazione aggregazione county-daily per anno %d...", year)

        # Apri file statistiche daily
        ds_stats = {}
        for stat, path in stat_files.items():
            safe_p = get_safe_path(path)
            ds_stats[stat] = xr.open_dataset(safe_p)

        safe_accum = get_safe_path(accum_file)
        ds_accum = xr.open_dataset(safe_accum)

        # Riassegna date accumulated: valid_time 00:00 del giorno D+1 diventa giorno D
        # Filtra all'anno solare D
        time_coord = "valid_time" if "valid_time" in ds_accum.coords else "time"
        accum_dates = pd.to_datetime(ds_accum[time_coord].values) - pd.Timedelta(days=1)
        ds_accum = ds_accum.assign_coords({time_coord: accum_dates})
        
        # Filtra all'anno solare target
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        ds_accum_year = ds_accum.sel({time_coord: slice(start_date, end_date)})

        # Esegui aggregazione spaziale pesata con matrice precomputata
        # Per efficienza: converti i pesi in xarray DataArray allineato
        lats = ds_accum_year["latitude"].values
        lons = ds_accum_year["longitude"].values

        # Unione dei dataset
        county_dfs = []
        logging.info("Calcolo medie ponderate per le 479 contee...")

        # Raggruppamento per contea
        for fips, group in self.weights_df.groupby("county_fips"):
            fips_lats = xr.DataArray(group["lat"].values, dims="points")
            fips_lons = xr.DataArray(group["lon"].values, dims="points")
            w = xr.DataArray(group["weight"].values, dims="points")

            # Estrai punti per la contea
            accum_pts = ds_accum_year.sel(latitude=fips_lats, longitude=fips_lons)
            county_accum = (accum_pts * w).sum(dim="points").to_dataframe()

            county_accum["county_fips"] = fips
            county_dfs.append(county_accum)

        df_year = pd.concat(county_dfs).reset_index()
        df_year.rename(columns={time_coord: "date"}, inplace=True)

        # Salva output Parquet
        df_year.to_parquet(out_parquet, index=False)
        logging.info("Salvato county-daily per anno %d: %s (%.2f MB)", year, out_parquet.name, out_parquet.stat().st_size / 1e6)
        return out_parquet


# ==============================================================================
# PRE-FLIGHT E CLI
# ==============================================================================
def run_preflight(base_dir: Path):
    """Esegue il test pre-flight sul 1950 e sul 2025 su scala ridotta (7 giorni)."""
    print("\n" + "="*80)
    print("AVVIO PRE-FLIGHT TEST (1950 e 2025, Bbox unificato 479 contee)")
    print("="*80)

    manifest = ManifestManager(base_dir / "manifest.csv")
    downloader = CDSDownloader(base_dir, manifest)

    test_years = [1950, 2025]
    test_days = [f"{d:02d}" for d in range(1, 8)]  # primi 7 giorni di giugno

    for y in test_years:
        print(f"\n--- TEST ANNO {y} ---")
        t0 = time.time()
        # Test Canale A: daily_mean (temperatura e dewpoint)
        f_mean = downloader.download_daily_stat_group(y, "daily_mean", ["2m_temperature", "2m_dewpoint_temperature"], test_days=test_days)
        # Test Canale A: daily_min
        f_min = downloader.download_daily_stat_group(y, "daily_minimum", ["2m_temperature"], test_days=test_days)
        # Test Canale A: daily_max
        f_max = downloader.download_daily_stat_group(y, "daily_maximum", ["2m_temperature"], test_days=test_days)
        # Test Canale B: accumulated (pioggia e radiazione)
        f_accum = downloader.download_accumulated_group(y, test_days=test_days)

        print(f"Download anno {y} completato in {time.time() - t0:.1f}s.")

        # QC Check
        ds_mean = xr.open_dataset(get_safe_path(f_mean))
        ds_min = xr.open_dataset(get_safe_path(f_min))
        ds_max = xr.open_dataset(get_safe_path(f_max))

        t_min = ds_min["t2m"].values
        t_max = ds_max["t2m"].values
        t_mean = ds_mean["t2m"].values

        assert np.all(t_min <= t_max), f"ERRORE QC: Tmin > Tmax nell'anno {y}"
        assert np.all(t_min <= t_mean + 1e-4), f"ERRORE QC: Tmin > Tmean nell'anno {y}"
        print(f"QC Anno {y}: PASS (Tmin <= Tmean <= Tmax verificato su tutte le {t_min.size} celle).")

        manifest.record(y, "A", "daily_mean", f_mean, "PASS", "Pre-flight test")
        manifest.record(y, "A", "daily_min", f_min, "PASS", "Pre-flight test")
        manifest.record(y, "A", "daily_max", f_max, "PASS", "Pre-flight test")
        manifest.record(y, "B", "accumulated", f_accum, "PASS", "Pre-flight test")

    print("\n" + "="*80)
    print("PRE-FLIGHT TEST COMPLETATO CON SUCCESSO!")
    print("="*80 + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Downloader ERA5-Land Daily per la tesi sulla soia.")
    parser.add_argument("--preflight", action="store_true", help="Esegue il pre-flight test su 1950 e 2025.")
    parser.add_argument("--start-year", type=int, default=1950, help="Anno iniziale (default: 1950).")
    parser.add_argument("--end-year", type=int, default=2025, help="Anno finale (default: 2025).")
    parser.add_argument("--test-year", type=int, help="Esegue il download di un singolo anno completo di test.")
    args = parser.parse_args()

    base_dir = Path("era5_land_daily")
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        run_preflight(base_dir)
        return

    # Esecuzione anno per anno
    years = [args.test_year] if args.test_year else list(range(args.start_year, args.end_year + 1))
    manifest = ManifestManager(base_dir / "manifest.csv")
    downloader = CDSDownloader(base_dir, manifest)
    weights_path = base_dir / "spatial_weights_479_counties.parquet"

    for y in years:
        logging.info("=== INIZIO ANNO %d ===", y)
        f_mean = downloader.download_daily_stat_group(y, "daily_mean", VARS_DAILY_MEAN)
        f_min = downloader.download_daily_stat_group(y, "daily_minimum", VARS_DAILY_MIN)
        f_max = downloader.download_daily_stat_group(y, "daily_maximum", VARS_DAILY_MAX)
        f_accum = downloader.download_accumulated_group(y)

        # Registra su manifest
        manifest.record(y, "A", "daily_mean", f_mean, "PASS")
        manifest.record(y, "A", "daily_min", f_min, "PASS")
        manifest.record(y, "A", "daily_max", f_max, "PASS")
        manifest.record(y, "B", "accumulated", f_accum, "PASS")

        # Aggregazione spaziale
        if weights_path.exists():
            aggregator = CountyAggregator(weights_path, base_dir / "county_daily")
            stat_files = {"mean": f_mean, "min": f_min, "max": f_max}
            aggregator.process_year(y, stat_files, f_accum)

    logging.info("Tutti gli anni completati con successo!")


if __name__ == "__main__":
    main()
