# Previsione della Resa della Soia con Dati Meteorologici e Machine Learning

Questo repository contiene i dati agronomici, il codice di estrazione e aggregazione meteorologica e la documentazione metodologica per la tesi di laurea magistrale:

> **Previsione della resa della soia tramite modelli di Machine Learning e dati meteorologici ad alta risoluzione**
> *Ambito:* 479 contee del Midwest statunitense (Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio).
> *Autore:* Jacopo Cesari

---

## 1. Domanda di Ricerca e Obiettivo

La domanda di ricerca centrale è:

> *“Quali modelli, utilizzando dati meteorologici osservati, riescono a prevedere la resa della soia con un miglioramento statisticamente significativo e stabile rispetto alle baseline di trend storico e climatologia, e quanto precocemente prima del raccolto emerge tale capacità predittiva?”*

Il lavoro si concentra sulla quantificazione rigorosa del segnale predittivo incrementale apportato dalle variabili meteorologiche rispetto a modelli parsimoniosi di trend secolare e climatologia storica locale.

---

## 2. Quadro Metodologico Sintetico

- **Area Geografica:** 6 stati del Midwest USA (IL, IN, IA, MN, MO, OH), rappresentati da **479 contee** selezionate per continuità storica ininterrotta nelle statistiche USDA NASS.
- **Orizzonti Temporali e Distinzione dei Dataset:**
  - **Balanced Panel Principale (1950–2010):** 61 anni completi, 479 contee, **29.219 osservazioni county-year** senza valori mancanti in resa o superficie.
  - **Anni Recenti (2011–2025):** conservati separatamente con i dati SURVEY disponibili; possibili valori mancanti non imputati. Non fanno parte del balanced panel principale.
  - **Copertura Meteorologica Prevista (1950–2025):** serie giornaliere continue per 76 anni su tutte le contee.
- **Scope delle Covariate:** disegno iniziale strettamente **weather-only** basato su rianalisi ERA5-Land (nessuna variabile di mercato, satellite o modello ibrido nel disegno principale).
- **Target di Previsione:**
  - *Primario:* resa continua (`BU / ACRE` o `t/ha`) e anomalia di resa controllando il trend tecnologico.
  - *Complementare:* classificazione tripartita (*Low / Normal / High*) con soglie stimate **esclusivamente sul training set**.
- **Frequenza Temporale e Cutoff:**
  - Base giornaliera sincronizzata UTC (00:00–23:59 UTC).
  - Aggregazioni intermedie principali a **5 giorni** (pentadi) e **10 giorni** (decadi); 30 giorni come benchmark a bassa risoluzione.
  - Cutoff stagionali progressivi: **31 maggio, 30 giugno, 31 luglio, 31 agosto, 15/30 settembre**.
- **Baseline e Modelli:**
  - *Baseline:* climatologia storica di contea, trend tecnologico secolare, modello additivo parsimonioso (GAM/lineare).
  - *Modelli:* Random Forest, Gradient Boosting (XGBoost/LightGBM), e modello sequenziale ricorrente (LSTM) valutato a parità di set informativo.
- **Validazione:** schema **expanding-window (rolling-origin)** con anni mantenuti uniti tra tutte le 479 contee. Preprocessing, detrending e tuning confinati rigorosamente nel training set (nessun data leakage).

---

## 3. Stato di Avanzamento del Progetto

| Componente | Stato Effettivo | Note Documentali |
|---|---|---|
| **Selezione delle 479 contee** | Completato | FIPS normalizzati e armonizzazione Ste. Genevieve County (MO). |
| **Panel bilanciato 1950–2010** | Completato | 29.219 osservazioni county-year senza valori mancanti. |
| **Pesi spaziali areali** | Completato e riproducibile | Matrice precalcolata (11.953 intersezioni EPSG:5070); script `compute_spatial_weights.py` allineato con CLI. |
| **Preflight ERA5-Land** | Completato su campioni | Test di download, coerenza termica e aggregazione su 7 giorni di giugno per 1950 e 2025. |
| **Download full-year ERA5-Land** | Strutturato | Richiesta Canale B divisa in 2 chiamate (6.222 fields totali, sotto la soglia CDS di 12.000). |
| **Aggregazione Channel A + B** | Implementata | Script integra Canale A (20 campi) e Canale B (17 campi) = 37 campi con evaporation swap. |
| **Feature derivate (GDD, VPD, bilanci)** | Da implementare | Formule definite nella metodologia; script di calcolo da realizzare. |
| **Dataset finale per modellazione** | Da costruire | In attesa del completamento dei download annuali. |
| **Baseline e modelli ML/DL** | Da implementare | Struttura e metriche definite; pipeline di addestramento da realizzare. |
| **Expanding-window evaluation** | Da implementare | Protocollo anti-leakage e block bootstrap annuale definiti. |

---

## 4. Struttura del Repository

```text
.
├── 01_Dati_Soia_e_Target/
│   ├── soybean_yield_479_counties_1950_2025.xlsx    # Rese storiche USDA NASS SURVEY (1950-2025)
│   ├── soybean_acres_479_counties_1950_2025.xlsx    # Superfici raccolte USDA NASS SURVEY
│   ├── balanced_panel_candidate_counties.csv        # Elenco FIPS delle 479 contee candidate
│   └── census_counties/                             # Cartografia US Census Bureau 2020 (shapefile)
│
├── era5_land_daily/                                 # Dati e artefatti meteorologici ERA5-Land
│   ├── manifest.csv                                 # Registro dei download e stato controlli di qualità
│   ├── spatial_weights_479_counties.parquet         # Matrice pesi areali (EPSG:5070, 11.953 intersezioni)
│   ├── spatial_weights_479_counties.csv             # Matrice pesi in formato CSV
│   ├── raw_daily_stats/                             # NetCDF campioni preflight Canale A (1950, 2025)
│   └── raw_accumulated_boundaries/                  # NetCDF campioni preflight Canale B (1950, 2025)
│
├── 03_Documentazione_e_Paper/
│   └── Papers/                                      # 15 articoli scientifici verificati in formato PDF
│
├── docs/                                            # Documentazione metodologica canonica
│   ├── research_design.md                           # Domanda di ricerca, ipotesi e disegno sperimentale
│   ├── data_and_target.md                           # Pulizia dati agronomici, panel e definizione target
│   ├── era5_land_methodology.md                     # Pipeline ERA5-Land a 37 campi, canali e QC
│   ├── feature_engineering.md                       # Indici agronomici (GDD, VPD proxy, heat days, 5/10d)
│   ├── validation_and_modeling.md                   # Expanding-window, baseline, modelli ed earliest skill
│   └── references.md                                # Tabella di raccordo della letteratura e link ai PDF
│
├── tests/                                           # Suite di test automatici (pytest)
│   └── test_pipeline.py                             # Test unitari QC, calendario, pesi e swap
│
├── AGENTS.md                                        # Vincoli e regole operative per agenti AI
├── download_era5_land_daily.py                      # Script di download e aggregazione ERA5-Land (37 campi)
├── compute_spatial_weights.py                       # Script di calcolo della matrice di pesi spaziali (CLI)
├── pyproject.toml                                   # Configurazione progetto e strumenti
├── requirements.txt                                 # Dipendenze con versioni bloccate
└── README.md                                        # Questo indice generale
```

---

## 5. Documentazione Metodologica Dettagliata

Per approfondire ciascun aspetto del progetto, fare riferimento ai documenti dedicati nella cartella [`docs/`](docs/):
- [**Disegno della Ricerca (`docs/research_design.md`)**](docs/research_design.md)
- [**Dati Agronomici e Target (`docs/data_and_target.md`)**](docs/data_and_target.md)
- [**Metodologia ERA5-Land Daily (`docs/era5_land_methodology.md`)**](docs/era5_land_methodology.md)
- [**Ingegneria delle Feature (`docs/feature_engineering.md`)**](docs/feature_engineering.md)
- [**Validazione e Modelli (`docs/validation_and_modeling.md`)**](docs/validation_and_modeling.md)
- [**Riferimenti Scientifici (`docs/references.md`)**](docs/references.md)
- [**Linee Guida per Agenti AI (`AGENTS.md`)**](AGENTS.md)

---

## 6. Note di Data Governance e Riproducibilità

1. **Credenziali API:** L'accesso a Copernicus CDS richiede un account e la configurazione del file `.cdsapirc` nella home dell'utente (`~/.cdsapirc`). Tale file non è versionato nel repository.
2. **Dataset Voluminosi:** I file raw NetCDF di grandi dimensioni e gli archivi completi sono esclusi dal tracciamento Git tramite `.gitignore`. I file NetCDF presenti in `era5_land_daily/` sono campioni minimi di preflight (7 giorni) a scopo di test e verifica di integrità.
3. **Stato del Codice:** La disponibilità degli script non implica che la pipeline end-to-end sia validata per la produzione su scala storica completa. L'esecuzione di download massivi su 1950–2025 è attualmente considerata **sperimentale** e subordinata alla risoluzione dei blocker documentati in [`docs/era5_land_methodology.md`](docs/era5_land_methodology.md).
