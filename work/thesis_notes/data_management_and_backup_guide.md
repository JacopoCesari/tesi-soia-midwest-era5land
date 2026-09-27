# Guida Operativa alla Gestione Dati e Backup su OneDrive

Questo documento riassume lo stato dei dati della tesi, le azioni di pulizia da compiere per liberare spazio su disco e le istruzioni di backup su OneDrive tramite l'archivio ZIP strutturato.

---

## 1. Dati da Eliminare o Spostare su Disco Esterno (Cache RAW Inutile)

### Percorso da gestire:
```text
C:\Users\JacopoCesari-Aretésr\Desktop\Tesi\work\data\raw\era5_land\arco_cache
```

- **Spazio occupato sul disco C:** ~**345.7 GB** suddivisi in **768.284 micro-file**.
- **Cosa contiene:** È la cache intermedia temporanea dei chunk Zarr scaricati dal bucket Google Cloud ARCO-ERA5 per processare gli anni 1950–2025.
- **Perché NON serve più:**
  - Tutti i 76 anni di dati meteo (1950–2025) sono già stati interamente estratti, verificati (Quality Control: `PASS` su tutti i 76 anni) e aggregati a livello di singola contea nei file Parquet.
  - Nessuno script di machine learning, nessuna regressione e nessun grafico della tesi legge mai questa cartella.
- **Azione raccomandata:**
  - **Opzione A (Consigliata):** **Cancellare direttamente** la cartella `arco_cache`. Libererai istantaneamente oltre 345 GB sul tuo disco `C:`.
  - **Opzione B:** Se per scrupolo storico desideri conservare i file grezzi intermedi, taglia e **sposta la cartella su un hard disk esterno** (non caricarla su OneDrive: supererebbe il limite dei singoli file e causerebbe blocchi di sincronizzazione).

---

## 2. Dati Essenziali da Conservare (File ZIP Strutturato per OneDrive)

### Percorso del file ZIP pronto:
```text
C:\Users\JacopoCesari-Aretésr\Desktop\Tesi\tesi_dati_completi_1950_2025.zip
```

- **Dimensione:** **~886 MB** (altamente compresso, leggero e portabile).
- **Cosa contiene:** Il 100% dei dati necessari e sufficienti per riprodurre qualsiasi modello di machine learning o grafico della tesi su qualsiasi computer.
- **Nuova struttura cartelle all'interno dello ZIP (semplice, immediata e pulita):**

```text
tesi_dati_completi_1950_2025.zip
│
├── LEGGIMI.txt                               <- Guida descrittiva sintetica inclusa nell'archivio
│
├── meteo/                                    <- Solo i 76 Parquet annuali (senza file QC di disturbo)
│   ├── county_daily_1950.parquet                (3.747.330 osservazioni giornaliere, 135 contee bilanciate,
│   ├── county_daily_1951.parquet                 32 variabili agrometeorologiche complete, zero dati nulli)
│   └── ... fino a 2025
│
├── rese/                                     <- Target di resa ufficiale
│   └── soybean_yield_1951_2025.csv              (10.125 osservazioni finite, panel 75 anni x 135 contee,
│                                                 con FIPS, acri raccolti, produzione in bushel e resa finale)
│
├── geografia/                                <- Anagrafica, pesi spaziali e confini vettoriali
│   ├── counties.csv                             (Anagrafica ufficiale delle 135 contee del panel)
│   ├── counties_with_data_availability.csv      (Quadro di disponibilità storica 1950-2025)
│   ├── county_weights_era5_land.csv             (Pesi geometrici area-weighted per griglia 0.1° ERA5-Land)
│   ├── spatial_weights.json                     (Configurazione di aggregazione)
│   └── shapefile/                               (Confini vettoriali US Census Bureau cb_2020_us_county_500k.*)
│
└── storico_usda/                             <- File storici originali completi (479 contee candidate)
    ├── soybean_yield_479_counties_1950_2025.xlsx (Foglio di calcolo rese originali USDA NASS)
    ├── soybean_acres_479_counties_1950_2025.xlsx (Foglio di calcolo acri originali USDA NASS)
    ├── balanced_panel_candidate_counties.csv     (Valutazione storica contee candidate)
    └── file_migration.json                       (Mappa di tracciabilità dei file)
```

- **Azione raccomandata:**
  - Copia o sposta `tesi_dati_completi_1950_2025.zip` nella tua cartella **OneDrive** dedicata alla tesi (o su un cloud storage).

---

## 3. Checklist delle Cose da Fare per l'Utente

- [ ] **Pulizia Spazio Disco:** Cancellare la cartella `work/data/raw/era5_land/arco_cache` (oppure spostarla su hard disk esterno). Recuperi **~346 GB** liberi sul disco `C:`.
- [ ] **Backup su OneDrive:** Caricare `tesi_dati_completi_1950_2025.zip` (~886 MB) su OneDrive.
- [ ] **Verifica:** Verificare che lo spazio su disco `C:` sia risalito a un livello ottimale (>800 GB liberi).

---

## 4. Stato Scientifico Attuale della Tesi

1. **Dati Meteo e Target:** 100% completi, puliti e allineati per tutti i 76 anni (1950–2025).
2. **Capitolo 3 (Data and Study Area):**
   - **Tabella 3.5 & Tabella 3.6**: allineate con 32 variabili agrometeorologiche.
   - **Figura 3.6**: climatologia completa su tutti i 75 anni (1951–2025) senza discontinuità.
   - **Figura 3.7**: traiettorie fenologiche cumulative intatte.
   - **Figura 3.8**: profili di stato dinamico intatti.
   - **Figura 3.9**: adottata l'Opzione 3 (2012 Flash Drought vs 2016 Modern Bumper, Precipitazione vs $T_{\mathrm{max}}$ con colormap *inferno* ad altissimo contrasto).
   - **Figura 3.10**: correlazioni mensili meteo-resa su tutti i 75 anni.
   - **PDF Tesi (`output/thesis/main.pdf`)**: compilato con successo (36 pagine).
3. **Prossimo Step di Ricerca (Capitolo 4):**
   - Feature Engineering per i 12 orizzonti mensili previsivi ($H = 12, \dots, 1$) a partire dai file Parquet.
