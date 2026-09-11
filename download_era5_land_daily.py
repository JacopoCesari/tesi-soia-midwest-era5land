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
   - Totale: 20 campi giornalieri
2. CANALE B (Accumulate): 'reanalysis-era5-land'
   - Step 00:00 UTC del giorno D+1 (contiene l'accumulo esatto delle 24h del giorno D)
   - 17 variabili (precipitazione, radiazione, evaporazione, runoff, flussi di calore)
   - Totale: 17 campi giornalieri
Totale complessivo: 37 campi meteorologici giornalieri.
"""

from __future__ import annotations

import argparse
import calendar
import ctypes
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
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

# Variabili Canale A: derived-era5-land-daily-statistics (20 campi totali)
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

# Variabili Canale B: reanalysis-era5-land (step 00:00 UTC, 17 campi totali)
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

# ECMWF parameter swap map: correzione dei tre parametri di evaporazione scambiati
# Param 228101 (evabs) contiene vegetation transpiration
# Param 228102 (evaow) contiene bare soil evaporation
# Param 228103 (evavt / evatp) contiene open water evaporation
EVAPORATION_SWAP_MAP = {
    "evabs": "evaporation_from_vegetation_transpiration",
    "evaow": "evaporation_from_bare_soil",
    "evavt": "evaporation_from_open_water_surfaces_excluding_oceans",
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


def open_dataset_safe(file_path: Path) -> xr.Dataset:
    """
    Apre in modo robusto un file NetCDF o archivio ZIP scaricato da Copernicus CDS.
    Carica i dati in memoria (ds.load()) e chiude tempestivamente i file handle
    per evitare lock su file su sistemi operativi Windows.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")

    if zipfile.is_zipfile(file_path):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmp)
            nc_files = sorted(Path(tmp).glob("*.nc"))
            if not nc_files:
                raise ValueError(f"Nessun file .nc trovato nell'archivio zip: {file_path}")
            opened = []
            for p in nc_files:
                ds = xr.open_dataset(get_safe_path(p))
                ds.load()
                opened.append(ds)
            if len(opened) == 1:
                merged = opened[0]
            else:
                merged = xr.merge(opened, compat="override", join="override")
            for ds in opened:
                ds.close()
            return merged
    else:
        ds = xr.open_dataset(get_safe_path(file_path))
        ds.load()
        ds.close()
        return ds


def apply_evaporation_swap(ds: xr.Dataset) -> xr.Dataset:
    """
    Riassegna i tre flussi di evaporazione ERA5-Land scambiati da ECMWF alla loro
    corretta identità fisica:
    - evabs (originariamente etichettata bare soil) -> vegetation transpiration
    - evaow (originariamente etichettata open water) -> bare soil evaporation
    - evavt / evatp (originariamente etichettata transpiration) -> open water evaporation
    """
    transp_var = "evavt" if "evavt" in ds.data_vars else ("evatp" if "evatp" in ds.data_vars else None)
    if "evabs" in ds.data_vars and "evaow" in ds.data_vars and transp_var is not None:
        t_transp = ds["evabs"].copy()
        t_bare = ds["evaow"].copy()
        t_water = ds[transp_var].copy()
        ds = ds.drop_vars(["evabs", "evaow", transp_var])
        ds["evaporation_from_vegetation_transpiration"] = t_transp
        ds["evaporation_from_bare_soil"] = t_bare
        ds["evaporation_from_open_water_surfaces_excluding_oceans"] = t_water
    return ds


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

    def is_done(self, file_path: Path, verify_checksum: bool = False) -> bool:
        """
        Verifica se un file e gia stato elaborato con successo.
        Se verify_checksum e True, ricalcola e valida l'hash SHA-256 contro il manifest.
        """
        if not file_path.exists() or file_path.stat().st_size < 1000:
            return False
        try:
            df = pd.read_csv(self.path)
            matches = df[(df["file_path"] == str(file_path)) & (df["qc_status"] == "PASS")]
            if len(matches) == 0:
                return False
            if verify_checksum:
                expected_sha = matches.iloc[-1].get("sha256", "")
                if expected_sha and isinstance(expected_sha, str):
                    actual_sha = compute_sha256(file_path)
                    return actual_sha == expected_sha
            return True
        except Exception:
            return False

    def record(self, year: int, channel: str, subgroup: str, file_path: Path, qc_status: str, notes: str = ""):
        size = file_path.stat().st_size if file_path.exists() else 0
        sha = compute_sha256(file_path) if file_path.exists() and size > 0 else ""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
# CONTROLLI DI QUALITÀ (QC VALIDATION)
# ==============================================================================
def validate_era5_dataset(
    ds: xr.Dataset,
    year: int | None = None,
    is_preflight: bool = False,
) -> tuple[str, str]:
    """
    Controlli di qualita automatici su un dataset NetCDF ERA5-Land.
    Verifica:
    1. Calendario e continuita temporale (365/366 giorni se anno intero).
    2. Monotonia e limiti spaziali del bounding box.
    3. Plausibilita dei limiti fisici.
    4. Coerenza termica (Tmin <= Tmean <= Tmax).
    """
    notes = []

    # 1. Coordinate
    if "latitude" in ds.coords and "longitude" in ds.coords:
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        if len(lats) > 1 and not np.all(np.diff(lats) < 0):
            return "FAIL", "Latitudini non monotonicamente decrescenti"
        if len(lons) > 1 and not np.all(np.diff(lons) > 0):
            return "FAIL", "Longitudini non monotonicamente crescenti"
        if lats.min() < SOUTH - 0.2 or lats.max() > NORTH + 0.2:
            return "FAIL", f"Latitudini fuori dal dominio: [{lats.min()}, {lats.max()}]"
        notes.append(f"Grid: {len(lats)}x{len(lons)}")

    # 2. Calendario
    time_col = "valid_time" if "valid_time" in ds.coords else "time"
    if time_col in ds.coords:
        times = pd.to_datetime(ds[time_col].values)
        if len(times) != len(np.unique(times)):
            return "FAIL", "Timestamp duplicati riscontrati nel dataset"
        if year is not None and not is_preflight:
            expected_days = 366 if calendar.isleap(year) else 365
            if len(times) != expected_days:
                return "FAIL", f"Numero giorni non corrispondente: {len(times)} vs {expected_days} attesi"
        notes.append(f"Days: {len(times)}")

    # 3. Limiti fisici
    # Precipitazione >= 0
    for p_var in ["tp", "total_precipitation"]:
        if p_var in ds.data_vars:
            val_min = float(ds[p_var].min())
            if val_min < -1e-4:
                return "FAIL", f"Precipitazione negativa riscontrata: {val_min}"

    # Radiazione >= 0
    for r_var in ["ssrd", "surface_solar_radiation_downwards"]:
        if r_var in ds.data_vars:
            val_min = float(ds[r_var].min())
            if val_min < -1e-4:
                return "FAIL", f"Radiazione solare negativa: {val_min}"

    # Umidita suolo in [0, 1]
    for s_var in ["swvl1", "swvl2", "swvl3", "swvl4"]:
        if s_var in ds.data_vars:
            val_min, val_max = float(ds[s_var].min()), float(ds[s_var].max())
            if val_min < -0.05 or val_max > 1.05:
                return "FAIL", f"Umidita volumetrica suolo fuori range: [{val_min}, {val_max}]"

    # 4. Coerenza termica (se presenti)
    if "t2m_min" in ds.data_vars and "t2m_max" in ds.data_vars:
        t_min = ds["t2m_min"].values
        t_max = ds["t2m_max"].values
        if np.any(t_min > t_max + 1e-4):
            return "FAIL", "Incoerenza termica: Tmin > Tmax"
        if "t2m_mean" in ds.data_vars:
            t_mean = ds["t2m_mean"].values
            if np.any(t_min > t_mean + 1e-4) or np.any(t_mean > t_max + 1e-4):
                return "FAIL", "Incoerenza termica: Tmin <= Tmean <= Tmax violata"
        notes.append("Thermal QC: OK")

    return "PASS", "; ".join(notes) if notes else "QC OK"


def validate_county_daily_df(
    df: pd.DataFrame,
    expected_counties: int = 479,
    expected_days: int | None = None,
) -> tuple[str, str]:
    """Valida cardinalita, chiavi uniche e assenza di valori nulli nel dataset aggregato."""
    counties_found = df["county_fips"].nunique()
    if counties_found != expected_counties:
        return "FAIL", f"Contee trovate: {counties_found} (attese: {expected_counties})"

    if expected_days is not None:
        expected_rows = expected_counties * expected_days
        if len(df) != expected_rows:
            return "FAIL", f"Righe totali: {len(df)} (attese: {expected_rows})"

    # Unicita chiavi
    dups = df.duplicated(subset=["date", "county_fips"]).sum()
    if dups > 0:
        return "FAIL", f"Trovate {dups} coppie (date, county_fips) duplicate"

    # Valori nulli
    null_count = df.isna().sum().sum()
    if null_count > 0:
        return "FAIL", f"Trovati {null_count} valori mancanti/NaN"

    return "PASS", f"Cardinalita verificata ({len(df)} righe, {counties_found} contee, 0 missing)"


# ==============================================================================
# MODULO DOWNLOAD CDS
# ==============================================================================
class CDSDownloader:
    def __init__(self, base_dir: Path, manifest: ManifestManager, verify_checksum: bool = False):
        self.base_dir = base_dir
        self.manifest = manifest
        self.verify_checksum = verify_checksum
        self.client = cdsapi.Client()
        self.raw_stats_dir = base_dir / "raw_daily_stats"
        self.raw_accum_dir = base_dir / "raw_accumulated_boundaries"
        self.raw_stats_dir.mkdir(parents=True, exist_ok=True)
        self.raw_accum_dir.mkdir(parents=True, exist_ok=True)

    def download_daily_stat_group(
        self,
        year: int,
        stat: str,
        variables: list[str],
        test_days: list[str] | None = None,
    ) -> Path:
        """Scarica un gruppo statistico (daily_mean, daily_minimum, daily_maximum) per un anno."""
        target_name = f"era5_land_{year}_{stat}.nc"
        target_path = self.raw_stats_dir / target_name

        if self.manifest.is_done(target_path, verify_checksum=self.verify_checksum):
            logging.info("SKIP (gia validato nel manifest): %s", target_name)
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
            if part_path.exists():
                part_path.unlink()

            nc_files = list(temp_extract_dir.glob("*.nc"))
            if len(nc_files) == 1:
                if target_path.exists():
                    target_path.unlink()
                nc_files[0].replace(target_path)
            elif len(nc_files) > 1:
                logging.info("Merging di %d variabili NetCDF per %s...", len(nc_files), stat)
                opened = []
                for f in nc_files:
                    ds = xr.open_dataset(get_safe_path(f))
                    ds.load()
                    opened.append(ds)
                merged = xr.merge(opened, compat="override", join="override")
                for ds in opened:
                    ds.close()
                if target_path.exists():
                    target_path.unlink()
                merged.to_netcdf(get_safe_path(target_path))
                merged.close()
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        else:
            if target_path.exists():
                target_path.unlink()
            part_path.replace(target_path)

        logging.info("Completato: %s (%.1f MB)", target_path.name, target_path.stat().st_size / 1e6)
        return target_path

    def download_accumulated_group(
        self,
        year: int,
        test_days: list[str] | None = None,
    ) -> Path:
        """
        Scarica le 17 variabili accumulate con step 00:00 UTC.
        In DAY_MODE=UTC:
        Per coprire l'anno D (1 gen ... 31 dic), i record necessari hanno valid_time
        da (D)-01-02 00:00 UTC fino a (D+1)-01-01 00:00 UTC.

        Per evitare di scaricare due anni completi (12.410 fields), la richiesta
        per l'anno intero e strutturata esattamente in due chiamate:
        1. Anno D: tutti i mesi e giorni a time="00:00" (6.205 / 6.222 fields).
        2. Giorno di confine D+1: solo 1 gen a time="00:00" (17 fields).
        Totale fields richiesti: 6.222 (entro la quota massima di 12.000 fields del CDS).
        """
        target_name = f"era5_land_{year}_accumulated_00utc.nc"
        target_path = self.raw_accum_dir / target_name

        if self.manifest.is_done(target_path, verify_checksum=self.verify_checksum):
            logging.info("SKIP (gia validato nel manifest): %s", target_name)
            return target_path

        part_path = target_path.with_suffix(".nc.part")
        if part_path.exists():
            part_path.unlink()

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
            logging.info("DOWNLOAD CANALE B (Preflight): Anno %d (%d variabili)", year, len(VARS_ACCUMULATED))
            self.client.retrieve(DATASET_HOURLY, request).download(str(part_path))
            self._finalize_download(part_path, target_path, year, "accum")
        else:
            # Anno intero: Chiamata 1 (Anno D) + Chiamata 2 (1 Gen D+1)
            part_main = self.raw_accum_dir / f"_part_main_{year}.nc"
            part_bound = self.raw_accum_dir / f"_part_bound_{year}.nc"

            req_main = {
                "variable": VARS_ACCUMULATED,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": CDS_AREA,
            }
            req_bound = {
                "variable": VARS_ACCUMULATED,
                "year": str(year + 1),
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": CDS_AREA,
            }

            logging.info("DOWNLOAD CANALE B (Parte 1/2): Anno %d (365/366 giorni)", year)
            self.client.retrieve(DATASET_HOURLY, req_main).download(str(part_main))
            ds_main = open_dataset_safe(part_main)

            logging.info("DOWNLOAD CANALE B (Parte 2/2): Boundary 01-01-%d", year + 1)
            self.client.retrieve(DATASET_HOURLY, req_bound).download(str(part_bound))
            ds_bound = open_dataset_safe(part_bound)

            # Concatena lungo la coordinata temporale
            time_dim = "valid_time" if "valid_time" in ds_main.dims else "time"
            ds_full = xr.concat([ds_main, ds_bound], dim=time_dim)

            if target_path.exists():
                target_path.unlink()
            ds_full.to_netcdf(get_safe_path(target_path))

            ds_main.close()
            ds_bound.close()
            ds_full.close()

            if part_main.exists():
                part_main.unlink()
            if part_bound.exists():
                part_bound.unlink()

        logging.info("Completato: %s (%.1f MB)", target_path.name, target_path.stat().st_size / 1e6)
        return target_path

    def _finalize_download(self, part_path: Path, target_path: Path, year: int, tag: str):
        """Estrae archivi ZIP ed esegue merge multi-variabile se necessario."""
        if zipfile.is_zipfile(part_path):
            temp_extract_dir = self.raw_accum_dir / f"_tmp_{year}_{tag}"
            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(part_path, "r") as z:
                z.extractall(temp_extract_dir)
            if part_path.exists():
                part_path.unlink()

            nc_files = list(temp_extract_dir.glob("*.nc"))
            if len(nc_files) == 1:
                if target_path.exists():
                    target_path.unlink()
                nc_files[0].replace(target_path)
            elif len(nc_files) > 1:
                logging.info("Merging di %d variabili NetCDF accumulate...", len(nc_files))
                opened = []
                for f in nc_files:
                    ds = xr.open_dataset(get_safe_path(f))
                    ds.load()
                    opened.append(ds)
                merged = xr.merge(opened, compat="override", join="override")
                for ds in opened:
                    ds.close()
                if target_path.exists():
                    target_path.unlink()
                merged.to_netcdf(get_safe_path(target_path))
                merged.close()
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        else:
            if target_path.exists():
                target_path.unlink()
            part_path.replace(target_path)


# ==============================================================================
# MODULO AGGREGAZIONE SPAZIALE (COUNTY-DAILY A 37 CAMPI)
# ==============================================================================
class CountyAggregator:
    def __init__(self, weights_parquet: Path, out_dir: Path):
        self.weights_df = pd.read_parquet(weights_parquet)
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Normalizza coordinate pesi a 2 decimali per matching esatto
        self.weights_df["lat"] = np.round(self.weights_df["lat"].values, 2)
        self.weights_df["lon"] = np.round(self.weights_df["lon"].values, 2)

    def process_year(self, year: int, stat_files: dict[str, Path], accum_file: Path) -> Path:
        """
        Esegue l'aggregazione spaziale per le 479 contee per l'anno specificato.
        Aggrega sia il Canale A (20 campi) sia il Canale B (17 campi) in un unico
        dataset giornaliero completo.
        """
        out_parquet = self.out_dir / f"county_daily_{year}.parquet"
        logging.info("Elaborazione aggregazione county-daily (Canale A + Canale B) per anno %d...", year)

        # 1. Caricamento e rinomina variabili Canale A
        ds_mean = open_dataset_safe(stat_files["mean"])
        ds_min = open_dataset_safe(stat_files["min"])
        ds_max = open_dataset_safe(stat_files["max"])

        # Evita collisioni sui nomi: differenzia t2m_min, t2m_max, t2m_mean, skt_max
        if "t2m" in ds_mean.data_vars:
            ds_mean = ds_mean.rename({"t2m": "t2m_mean"})
        if "t2m" in ds_min.data_vars:
            ds_min = ds_min.rename({"t2m": "t2m_min"})
        if "t2m" in ds_max.data_vars:
            ds_max = ds_max.rename({"t2m": "t2m_max"})
        if "skt" in ds_max.data_vars:
            ds_max = ds_max.rename({"skt": "skt_max"})

        ds_a = xr.merge([ds_mean, ds_min, ds_max], compat="override", join="override")
        time_coord_a = "valid_time" if "valid_time" in ds_a.coords else "time"
        ds_a = ds_a.assign_coords({time_coord_a: pd.to_datetime(ds_a[time_coord_a].values).normalize()}).rename({time_coord_a: "time"})
        ds_a = ds_a.assign_coords({
            "latitude": np.round(ds_a["latitude"].values, 2),
            "longitude": np.round(ds_a["longitude"].values, 2),
        })

        # 2. Caricamento e correzione Canale B
        ds_b = open_dataset_safe(accum_file)
        ds_b = apply_evaporation_swap(ds_b)

        # Shift temporale: valid_time 00:00 del giorno D+1 rappresenta l'accumulo del giorno D
        time_coord_b = "valid_time" if "valid_time" in ds_b.coords else "time"
        shifted_dates = pd.to_datetime(ds_b[time_coord_b].values).normalize() - pd.Timedelta(days=1)
        ds_b = ds_b.assign_coords({time_coord_b: shifted_dates}).rename({time_coord_b: "time"})
        ds_b = ds_b.assign_coords({
            "latitude": np.round(ds_b["latitude"].values, 2),
            "longitude": np.round(ds_b["longitude"].values, 2),
        })

        # 3. Allineamento temporale (date comuni o anno solare target)
        common_dates = np.intersect1d(ds_a.time.values, ds_b.time.values)
        if len(common_dates) == 0:
            raise ValueError(f"Nessuna data comune tra Canale A e Canale B per l'anno {year}")

        ds_a_sub = ds_a.sel(time=common_dates)
        ds_b_sub = ds_b.sel(time=common_dates)
        ds_all = xr.merge([ds_a_sub, ds_b_sub], compat="override", join="inner")

        # 4. Aggregazione spaziale pesata sulle 479 contee
        logging.info("Calcolo medie ponderate per 479 contee su %d date...", len(common_dates))
        county_dfs = []

        for fips, group in self.weights_df.groupby("county_fips"):
            fips_lats = xr.DataArray(group["lat"].values, dims="points")
            fips_lons = xr.DataArray(group["lon"].values, dims="points")
            w = xr.DataArray(group["weight"].values, dims="points")

            pts = ds_all.sel(latitude=fips_lats, longitude=fips_lons)
            county_agg = (pts * w).sum(dim="points").to_dataframe().reset_index()
            county_agg["county_fips"] = fips
            county_dfs.append(county_agg)

        df_year = pd.concat(county_dfs).reset_index(drop=True)
        df_year.rename(columns={"time": "date"}, inplace=True)

        # Rimuovi eventuali colonne residue di coordinate interne
        cols_to_drop = [c for c in ["points", "latitude", "longitude", "number", "expver"] if c in df_year.columns]
        if cols_to_drop:
            df_year.drop(columns=cols_to_drop, inplace=True)

        # 5. Conversioni di unita fisiche standard (Kelvin -> Celsius, Pa -> hPa, m -> mm, J -> MJ/m2)
        df_year = apply_unit_conversions(df_year)

        # 6. Validazione QC finale sulla cardinalita e unicit
        qc_status, qc_msg = validate_county_daily_df(df_year, expected_counties=479, expected_days=len(common_dates))
        if qc_status != "PASS":
            logging.warning("Attenzione QC County-Daily: %s", qc_msg)
        else:
            logging.info("QC County-Daily: %s", qc_msg)

        # Salva output Parquet
        df_year.to_parquet(out_parquet, index=False)
        logging.info("Salvato county-daily per anno %d: %s (%.2f MB, %d righe)", year, out_parquet.name, out_parquet.stat().st_size / 1e6, len(df_year))
        return out_parquet


def apply_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Applica conversioni fisiche da unita raw CDS a unita agronomiche standard."""
    df = df.copy()

    # Temperature da Kelvin a Celsius (se valori > 100)
    temp_cols = [c for c in df.columns if any(t in c for t in ["t2m", "stl", "temperature", "skt", "d2m"])]
    for c in temp_cols:
        if df[c].mean() > 100:
            df[c] = df[c] - 273.15

    # Pressione da Pa a hPa (se valori > 50000)
    if "surface_pressure" in df.columns and df["surface_pressure"].mean() > 50000:
        df["surface_pressure"] = df["surface_pressure"] / 100.0
    if "sp" in df.columns and df["sp"].mean() > 50000:
        df["sp"] = df["sp"] / 100.0

    # Altezze d'acqua da metri a millimetri (m -> mm = * 1000)
    water_cols = [
        "tp", "total_precipitation",
        "e", "total_evaporation",
        "pev", "potential_evaporation",
        "sro", "surface_runoff",
        "ssro", "sub_surface_runoff",
        "sf", "snowfall",
        "smlt", "snowmelt",
        "src", "skin_reservoir_content",
        "sd", "snow_depth_water_equivalent",
        "evatc", "evaporation_from_the_top_of_canopy",
        "evaporation_from_bare_soil",
        "evaporation_from_vegetation_transpiration",
        "evaporation_from_open_water_surfaces_excluding_oceans",
    ]
    for c in water_cols:
        if c in df.columns and df[c].abs().max() < 10.0:  # in metri e tipicamente < 1m/die
            df[c] = df[c] * 1000.0

    # Radiazione e flussi termici da J/m2 a MJ/m2 (se valori medi > 1e4)
    energy_cols = [
        "ssrd", "surface_solar_radiation_downwards",
        "strd", "surface_thermal_radiation_downwards",
        "ssr", "surface_net_solar_radiation",
        "str", "surface_net_thermal_radiation",
        "sshf", "surface_sensible_heat_flux",
        "slhf", "surface_latent_heat_flux",
    ]
    for c in energy_cols:
        if c in df.columns and df[c].abs().mean() > 10000.0:
            df[c] = df[c] / 1e6

    return df


# ==============================================================================
# PRE-FLIGHT E CLI
# ==============================================================================
def run_preflight(base_dir: Path, verify_checksum: bool = False):
    """Esegue il test pre-flight sul 1950 e sul 2025 su scala ridotta (7 giorni)."""
    print("\n" + "=" * 80)
    print("AVVIO PRE-FLIGHT TEST (1950 e 2025, Bbox unificato 479 contee)")
    print("=" * 80)

    manifest = ManifestManager(base_dir / "manifest.csv")
    downloader = CDSDownloader(base_dir, manifest, verify_checksum=verify_checksum)
    weights_path = base_dir / "spatial_weights_479_counties.parquet"

    test_years = [1950, 2025]
    test_days = [f"{d:02d}" for d in range(1, 8)]  # primi 7 giorni di giugno

    for y in test_years:
        print(f"\n--- TEST ANNO {y} ---")
        t0 = time.time()
        # Test Canale A: daily_mean (temperatura e dewpoint)
        f_mean = downloader.download_daily_stat_group(
            y, "daily_mean", ["2m_temperature", "2m_dewpoint_temperature"], test_days=test_days
        )
        # Test Canale A: daily_min
        f_min = downloader.download_daily_stat_group(y, "daily_minimum", ["2m_temperature"], test_days=test_days)
        # Test Canale A: daily_max
        f_max = downloader.download_daily_stat_group(y, "daily_maximum", ["2m_temperature"], test_days=test_days)
        # Test Canale B: accumulated
        f_accum = downloader.download_accumulated_group(y, test_days=test_days)

        print(f"Download/verifica anno {y} completato in {time.time() - t0:.1f}s.")

        # QC Check
        ds_mean = open_dataset_safe(f_mean)
        ds_min = open_dataset_safe(f_min)
        ds_max = open_dataset_safe(f_max)

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

        # Test aggregazione Canale A + B se i pesi esistono
        if weights_path.exists():
            print(f"Test aggregazione spaziale Canale A + Canale B per anno {y}...")
            aggregator = CountyAggregator(weights_path, base_dir / "county_daily")
            stat_files = {"mean": f_mean, "min": f_min, "max": f_max}
            out_p = aggregator.process_year(y, stat_files, f_accum)
            print(f"Aggregazione completata: {out_p.name} generato con successo.")

    print("\n" + "=" * 80)
    print("PRE-FLIGHT TEST E AGGREGAZIONE CANALE A+B COMPLETATI CON SUCCESSO!")
    print("=" * 80 + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Downloader ERA5-Land Daily per la tesi sulla soia.")
    parser.add_argument("--preflight", action="store_true", help="Esegue il pre-flight test su 1950 e 2025.")
    parser.add_argument("--start-year", type=int, default=1950, help="Anno iniziale (default: 1950).")
    parser.add_argument("--end-year", type=int, default=2025, help="Anno finale (default: 2025).")
    parser.add_argument("--test-year", type=int, help="Esegue il download di un singolo anno completo di test.")
    parser.add_argument("--verify-checksum", action="store_true", help="Ricalcola e valida l'hash SHA-256 nel resume check.")
    parser.add_argument("--aggregate-only", action="store_true", help="Esegue solo l'aggregazione spaziale sui file scaricati.")
    parser.add_argument("--base-dir", type=Path, default=Path("era5_land_daily"), help="Cartella dei dati meteo (default: era5_land_daily).")
    args = parser.parse_args()

    base_dir = args.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        run_preflight(base_dir, verify_checksum=args.verify_checksum)
        return

    years = [args.test_year] if args.test_year else list(range(args.start_year, args.end_year + 1))
    manifest = ManifestManager(base_dir / "manifest.csv")
    downloader = CDSDownloader(base_dir, manifest, verify_checksum=args.verify_checksum)
    weights_path = base_dir / "spatial_weights_479_counties.parquet"

    for y in years:
        logging.info("=== INIZIO ANNO %d ===", y)
        if not args.aggregate_only:
            f_mean = downloader.download_daily_stat_group(y, "daily_mean", VARS_DAILY_MEAN)
            f_min = downloader.download_daily_stat_group(y, "daily_minimum", VARS_DAILY_MIN)
            f_max = downloader.download_daily_stat_group(y, "daily_maximum", VARS_DAILY_MAX)
            f_accum = downloader.download_accumulated_group(y)

            # Validazione QC prima della registrazione
            ds_mean = open_dataset_safe(f_mean)
            qc_status, qc_notes = validate_era5_dataset(ds_mean, year=y)
            manifest.record(y, "A", "daily_mean", f_mean, qc_status, qc_notes)

            ds_min = open_dataset_safe(f_min)
            qc_status, qc_notes = validate_era5_dataset(ds_min, year=y)
            manifest.record(y, "A", "daily_min", f_min, qc_status, qc_notes)

            ds_max = open_dataset_safe(f_max)
            qc_status, qc_notes = validate_era5_dataset(ds_max, year=y)
            manifest.record(y, "A", "daily_max", f_max, qc_status, qc_notes)

            ds_accum = open_dataset_safe(f_accum)
            qc_status, qc_notes = validate_era5_dataset(ds_accum, year=y)
            manifest.record(y, "B", "accumulated", f_accum, qc_status, qc_notes)
        else:
            f_mean = base_dir / "raw_daily_stats" / f"era5_land_{y}_daily_mean.nc"
            f_min = base_dir / "raw_daily_stats" / f"era5_land_{y}_daily_minimum.nc"
            f_max = base_dir / "raw_daily_stats" / f"era5_land_{y}_daily_maximum.nc"
            f_accum = base_dir / "raw_accumulated_boundaries" / f"era5_land_{y}_accumulated_00utc.nc"

        # Aggregazione spaziale completa Canale A + Canale B
        if weights_path.exists():
            aggregator = CountyAggregator(weights_path, base_dir / "county_daily")
            stat_files = {"mean": f_mean, "min": f_min, "max": f_max}
            aggregator.process_year(y, stat_files, f_accum)

    logging.info("Elaborazione completata con successo!")


if __name__ == "__main__":
    main()
