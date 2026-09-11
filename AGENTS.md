# Istruzioni Operative per Coding Agent (AGENTS.md)

Questo file definisce i vincoli metodologici, le gerarchie delle fonti e le regole operative per qualsiasi agente AI che collabori su questo repository.

---

## 1. Gerarchia delle Fonti (Source of Truth)

1. **Documentazione Canonica (`docs/`):**
   - [`docs/research_design.md`](docs/research_design.md): domanda di ricerca, ambito, ipotesi, cutoff e baseline.
   - [`docs/data_and_target.md`](docs/data_and_target.md): selezione USDA NASS, 479 contee, balanced panel 1950–2010.
   - [`docs/era5_land_methodology.md`](docs/era5_land_methodology.md): 37 campi ERA5-Land, architettura a due canali, QC e limiti correnti.
   - [`docs/feature_engineering.md`](docs/feature_engineering.md): GDD, VPD proxy, heat days, bilanci idrici rolling, aggregazioni 5/10/30 giorni.
   - [`docs/validation_and_modeling.md`](docs/validation_and_modeling.md): expanding-window temporal validation, modelli, bootstrap annuale ed earliest stable skill.
   - [`docs/references.md`](docs/references.md): catalogo dei 15 paper verificati disponibili localmente in `03_Documentazione_e_Paper/Papers/`.
2. **Codice della Repository:**
   - Rappresenta l'effettivo stato di implementazione. **Verificare sempre il codice sorgente prima di descriverne il funzionamento.** Non fidarsi di assunzioni o commenti obsoleti.
3. **Notion e Fonti Esterne:**
   - Appunti o workspace Notion possono contenere orientamenti più recenti del relatore/autore, ma nel repository GitHub devono confluire **soltanto le decisioni metodologiche consolidate e approvate**.
4. **Cronologia Git:**
   - Costituisce l'unico archivio delle versioni precedenti. Non creare cartelle `archive/` né duplicati storici.

---

## 2. Scope Scientifico e Vincoli Inviolabili

- **Disegno Iniziale Rigorosamente Weather-Only:**
  La tesi valuta l'informazione predittiva dei dati meteorologici ERA5-Land. **È vietato introdurre nel disegno principale:**
  - Prezzi, futures, indici macroeconomici;
  - Dati di remote sensing (NDVI, SIF, NIRv, MODIS, Sentinel);
  - Dataset proprietari o piattaforme terze (CropCast, IFAB);
  - Modelli ibridi meteo-satellite o simulazioni biofisiche complesse (DSSAT).
- **Distinzione Temporale Categorica:**
  - **Balanced panel principale:** **1950–2010** (61 anni, 479 contee, 29.219 osservazioni). Zero valori mancanti.
  - **Anni 2011–2025:** conservati separatamente nei file Excel (`all_years_survey`). Possono contenere valori mancanti. **Nessuna imputazione automatica.** Non fanno parte del balanced panel principale.
  - **Copertura meteorologica ERA5-Land:** **1950–2025** (76 anni giornalieri continui).
  - **DIVIETO ASSOLUTO:** Non descrivere mai il periodo 1950–2025 come un panel bilanciato.
- **Definizione dei Target:**
  - Primario: resa continua (`BU / ACRE` o `t/ha`) o anomalia controllando il trend tecnologico.
  - Complementare: classificazione in 3 classi (*Low / Normal / High*), con soglie stimate **esclusivamente sul training set**.
- **Protocollo di Validazione Anti-Leakage:**
  - Schema **expanding-window (rolling-origin)**. Gli anni devono essere mantenuti uniti tra tutte le 479 contee.
  - **Vietato** qualsiasi split casuale o k-fold disgiunto sulle coppie contea-anno.
  - Preprocessing, detrending, standardizzazione e tuning devono essere stimati **esclusivamente sul set di addestramento**.
- **Nessuna Assunzione a Priori sul Deep Learning:**
  Non dichiarare né dare per scontata la superiorità di LSTM o del deep learning rispetto a Random Forest, Gradient Boosting o baseline statistiche/econometriche (GAM).

---

## 3. Regole di Trasparenza sull'Implementazione

- **Trasparenza sullo stato del codice e della pipeline:**
  1. I file NetCDF presenti nel repository rappresentano **campioni di preflight di 7 giorni** (giugno 1950 e 2025), non serie annuali complete.
  2. La pipeline `download_era5_land_daily.py` e `compute_spatial_weights.py` è stata consolidata con aggregazione Canale A+B (37 campi), evaporation swap, gestione CDS sotto 12.000 fields, validazione QC e suite di test in `tests/test_pipeline.py`.
  3. Il calcolo delle feature derivate (GDD, VPD proxy, bilanci rolling) e la modellazione (baseline, ML, LSTM) sono definiti metodologicamente ma devono essere sviluppati nelle fasi successive.

---

## 4. Protezione Dati e Integrità Repository

Non modificare, cancellare o sovrascrivere mai:
- File dati CSV o Excel (`.csv`, `.xlsx`);
- Shapefile e geometrie in `01_Dati_Soia_e_Target/census_counties/`;
- File NetCDF (`.nc`) e Parquet (`.parquet`);
- File di pesi spaziali e `manifest.csv`;
- I 15 PDF scientifici in `03_Documentazione_e_Paper/Papers/`;
- Credenziali e file `.cdsapirc` o configurazioni utente;
- Il file `.gitignore`.

---

## 5. Protocollo Git

- Non committare né pushare direttamente su `main` o `dev`. Lavorare sempre su feature branch dedicati (es. `docs/...` o `fix/...`).
- **Regola Commit:** Non committare mai il codice finché l'utente non lo richiede esplicitamente. Attendere sempre l'autorizzazione o la richiesta esplicita dell'utente prima di eseguire `git commit`.
