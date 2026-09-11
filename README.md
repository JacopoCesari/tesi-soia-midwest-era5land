# Progetto di Tesi: Previsione della Resa della Soia con Machine Learning e Dati Meteorologici

Questo repository contiene il codice, i dati agronomici e la pipeline di acquisizione dei dati meteorologici per la tesi di laurea magistrale sulla previsione della resa della soia su un **balanced panel di 479 contee** del Midwest statunitense (1950–2025).

---

## Struttura delle Cartelle

```text
.
├── 01_Dati_Soia_e_Target/             # Dati storici ufficiali USDA NASS e geometrie
│   ├── soybean_yield_479_counties_1950_2025.xlsx   # Rese storiche (bu/acre) 1950-2025
│   ├── soybean_acres_479_counties_1950_2025.xlsx   # Superfici raccolte (acres harvested)
│   ├── balanced_panel_candidate_counties.csv       # Elenco standard FIPS 479 contee
│   ├── metodologia_selezione_panel_479_contee.md   # Nota metodologica selezione panel
│   └── census_counties/                            # Shapefile US Census Bureau 2020
│
├── 02_Meteo_ERA5_Land/                 # Dati meteorologici giornalieri ERA5-Land
│   ├── raw_daily_stats/               # NetCDF variabili istantanee (Canale A)
│   ├── raw_accumulated_boundaries/    # NetCDF/GRIB variabili accumulate 00 UTC (Canale B)
│   ├── county_daily/                  # Dataset giornalieri aggregati a livello di contea (Parquet)
│   ├── model_features/                # Feature ingegnerizzate (weekly/rolling/cutoff)
│   ├── spatial_weights_479_counties.parquet  # Matrice pesi spaziali (EPSG:5070)
│   └── manifest.csv                   # Registro download e controlli di qualità (SHA-256)
│
├── 03_Documentazione_e_Paper/          # Documentazione metodologica e letteratura scientifica
│   ├── METODOLOGIA_ERA5_LAND_DAILY.md # Specifica completa della pipeline meteo
│   ├── deep-research-report.md        # Report approfondito sulla letteratura e sui modelli
│   ├── Deep Research_ previsione...   # Documento di ricerca iniziale
│   ├── istruzioni_nuovo_script...     # Specifica tecnica originaria (versione estesa)
│   └── Papers/                        # Articoli scientifici di riferimento
│
├── download_era5_land_daily.py         # Script operativo unico di download e processing
├── compute_spatial_weights.py          # Script di calcolo della matrice di pesi spaziali
└── README.md                           # Questo indice generale
```

---

## Guida Rapida all'Uso

### 1. Requisiti e Dipendenze
Assicurarsi che l'ambiente Python contenga le librerie richieste:
```bash
pip install cdsapi xarray netCDF4 pandas pyarrow geopandas shapely
```
Il file di credenziali `.cdsapirc` deve risiedere nella home dell'utente:
`C:\Users\<NomeUtente>\.cdsapirc`

### 2. Pre-flight Test (Verifica Rapida)
Per testare la connessione a Copernicus, i controlli di qualità e la coerenza fisica:
```bash
python download_era5_land_daily.py --preflight
```

### 3. Esecuzione del Download e Processing Annuale
Per scaricare e aggregare un singolo anno di prova (es. 2023):
```bash
python download_era5_land_daily.py --test-year 2023
```
Per avviare la serie storica completa (1950–2025):
```bash
python download_era5_land_daily.py --start-year 1950 --end-year 2025
```
Lo script è completamente **resumable**: se interrotto, riparte dall'ultimo blocco non completato verificando l'integrità tramite hash SHA-256 registrati in `manifest.csv`.
