"""
tests/test_pipeline.py

Suite di test automatici di validazione e regressione per la pipeline ERA5-Land:
1. Validazione calendario e anni bisestili (365 vs 366 giorni).
2. Cardinalità county-day e unicità delle chiavi su aggregazione 479 contee.
3. Correttezza dell'evaporation swap (ECMWF parameter mapping).
4. Verifica crittografica SHA-256 nel resume check del manifest.
5. Conversioni di unità fisiche da raw CDS a standard agronomico.
6. Limite fields per richiesta Canale B (sotto soglia 12.000).
"""

import calendar
import sys
import tempfile
from pathlib import Path

# Assicura che la root del repository sia nel percorso di importazione
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from download_era5_land_daily import (
    ManifestManager,
    apply_evaporation_swap,
    apply_unit_conversions,
    compute_sha256,
    validate_county_daily_df,
    validate_era5_dataset,
    VARS_ACCUMULATED,
)


def test_leap_year_calendar_validation():
    """Verifica che il controllo QC distingua correttamente anni comuni (365 gg) e bisestili (366 gg)."""
    # Anno comune 1950 (365 giorni)
    dates_common = pd.date_range("1950-01-01", "1950-12-31", freq="D")
    assert len(dates_common) == 365
    ds_common = xr.Dataset(coords={"valid_time": dates_common})
    status, notes = validate_era5_dataset(ds_common, year=1950, is_preflight=False)
    assert status == "PASS"

    # Anno bisestile 1952 (366 giorni)
    dates_leap = pd.date_range("1952-01-01", "1952-12-31", freq="D")
    assert len(dates_leap) == 366
    ds_leap = xr.Dataset(coords={"valid_time": dates_leap})
    status_leap, notes_leap = validate_era5_dataset(ds_leap, year=1952, is_preflight=False)
    assert status_leap == "PASS"

    # Fallimento se mancano giorni in un anno bisestile
    ds_incomplete = xr.Dataset(coords={"valid_time": dates_common})  # 365 giorni passati come anno 1952
    status_fail, notes_fail = validate_era5_dataset(ds_incomplete, year=1952, is_preflight=False)
    assert status_fail == "FAIL"
    assert "365 vs 366 attesi" in notes_fail


def test_evaporation_swap():
    """Verifica che lo swap dei parametri di evaporazione riassegni correttamente le variabili."""
    dummy_data = np.ones((2, 2))
    ds = xr.Dataset(
        data_vars={
            "evabs": (["lat", "lon"], dummy_data * 10),   # deve diventare traspirazione vegetale
            "evaow": (["lat", "lon"], dummy_data * 20),   # deve diventare evaporazione suolo nudo
            "evavt": (["lat", "lon"], dummy_data * 30),   # deve diventare open water evaporation
        }
    )

    ds_swapped = apply_evaporation_swap(ds)

    assert "evabs" not in ds_swapped.data_vars
    assert "evaow" not in ds_swapped.data_vars
    assert "evavt" not in ds_swapped.data_vars

    assert "evaporation_from_vegetation_transpiration" in ds_swapped.data_vars
    assert "evaporation_from_bare_soil" in ds_swapped.data_vars
    assert "evaporation_from_open_water_surfaces_excluding_oceans" in ds_swapped.data_vars

    assert ds_swapped["evaporation_from_vegetation_transpiration"].values[0, 0] == 10
    assert ds_swapped["evaporation_from_bare_soil"].values[0, 0] == 20
    assert ds_swapped["evaporation_from_open_water_surfaces_excluding_oceans"].values[0, 0] == 30


def test_county_daily_cardinality_and_uniqueness():
    """Verifica che il validatore county-daily segnali anomalie di cardinalita o chiavi duplicate."""
    counties = [f"{i:05d}" for i in range(479)]
    dates = pd.date_range("1950-06-01", periods=6, freq="D")

    records = []
    for d in dates:
        for c in counties:
            records.append({"date": d, "county_fips": c, "t2m_mean": 20.0, "tp": 5.0})

    df_valid = pd.DataFrame(records)
    status, msg = validate_county_daily_df(df_valid, expected_counties=479, expected_days=6)
    assert status == "PASS"

    # Caso anomalo: contea mancante (478 invece di 479)
    df_missing_county = df_valid[df_valid["county_fips"] != "00000"].copy()
    status_fail, msg_fail = validate_county_daily_df(df_missing_county, expected_counties=479, expected_days=6)
    assert status_fail == "FAIL"

    # Caso anomalo: riga duplicata sostituendo un'altra (chiave (date, fips) non unica)
    df_dup = df_valid.copy()
    df_dup.iloc[1] = df_dup.iloc[0]
    status_dup, msg_dup = validate_county_daily_df(df_dup, expected_counties=479, expected_days=6)
    assert status_dup == "FAIL"
    assert "duplicate" in msg_dup


def test_manifest_checksum_validation():
    """Verifica che is_done con verify_checksum identifichi file integri e rilevi alterazioni."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / "manifest.csv"
        manifest = ManifestManager(manifest_path)

        test_file = tmp_dir / "sample.nc"
        test_file.write_bytes(b"A" * 2000)

        # Registrazione con PASS
        manifest.record(1950, "A", "daily_mean", test_file, "PASS")

        # Verifica normale
        assert manifest.is_done(test_file, verify_checksum=False) is True
        # Verifica con checksum
        assert manifest.is_done(test_file, verify_checksum=True) is True

        # File alterato (corruzione silenziosa)
        test_file.write_bytes(b"B" * 2000)
        # Senza verifica checksum passerebbe (stessa dimensione)
        assert manifest.is_done(test_file, verify_checksum=False) is True
        # Con verifica checksum rileva l'alterazione e restituisce False
        assert manifest.is_done(test_file, verify_checksum=True) is False


def test_unit_conversions():
    """Verifica le conversioni fisiche da unita raw CDS a unita agronomiche standard."""
    df_raw = pd.DataFrame({
        "t2m_mean": [293.15],         # 20 °C in Kelvin
        "surface_pressure": [101325.0],# Pa -> hPa
        "tp": [0.025],                 # m -> mm (25 mm)
        "ssrd": [15000000.0],          # J/m2 -> MJ/m2 (15 MJ/m2)
    })

    df_conv = apply_unit_conversions(df_raw)

    assert pytest.approx(df_conv["t2m_mean"].iloc[0], rel=1e-4) == 20.0
    assert pytest.approx(df_conv["surface_pressure"].iloc[0], rel=1e-4) == 1013.25
    assert pytest.approx(df_conv["tp"].iloc[0], rel=1e-4) == 25.0
    assert pytest.approx(df_conv["ssrd"].iloc[0], rel=1e-4) == 15.0


def test_channel_b_request_field_count():
    """Verifica che la richiesta annuale per Canale B non superi la quota massima CDS di 12.000 fields."""
    num_vars = len(VARS_ACCUMULATED)
    assert num_vars == 17

    # Parte 1 (Anno intero): 365 giorni (o 366) x 17 variabili x 1 timestamp
    fields_main_common = 365 * num_vars
    fields_main_leap = 366 * num_vars
    assert fields_main_common == 6205
    assert fields_main_leap == 6222

    # Parte 2 (Boundary 1 Gen D+1): 1 giorno x 17 variabili x 1 timestamp
    fields_bound = 1 * num_vars
    assert fields_bound == 17

    # Totale per anno
    assert fields_main_common + fields_bound == 6222
    assert fields_main_leap + fields_bound == 6239

    # Entrambi ampiamente inferiori al limite CDS di 12.000 fields
    assert fields_main_leap + fields_bound < 12000


def test_spatial_weights_integrity():
    """Verifica la completezza, copertura e somma unitaria della matrice dei pesi spaziali."""
    weights_path = Path("era5_land_daily/spatial_weights_479_counties.parquet")
    assert weights_path.exists()

    df_w = pd.read_parquet(weights_path)
    assert len(df_w) == 11953
    assert df_w["county_fips"].nunique() == 479

    sums = df_w.groupby("county_fips")["weight"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-5)

