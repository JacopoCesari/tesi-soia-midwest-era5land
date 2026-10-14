# Registro TO-DO Operativi Attivi per la Tesi

> **Stato del Documento:** Attivo — Foglio di lavoro operativo per la chiusura della tesi  
> **Finalità Esclusiva:** Questo documento contiene **esclusivamente i TO-DO operativi aperti**, ripuliti da tutte le fasi di calcolo empirico e addestramento già completate con successo.  
> **Riferimenti di Supporto Collegati:**
> - Spunti di riflessione e chiavi interpretative per i capitoli: [`discussion_insights.md`](./discussion_insights.md)
> - Sviluppi futuri e prospettive di ricerca per il Capitolo 6: [`future_developments.md`](./future_developments.md)
> - Memoria metodologica e risultati consolidati: vedi la mappa completa in [`README.md`](./README.md)

---

## 1. Stato di Avanzamento Sintetico dei Lavori

- [x] **Pipeline Empirica e Modellistica (100% Completata)**:
  - Estrazione dati e panel bilanciato 1951–2025 (135 contee, 75 anni, 10.125 osservazioni).
  - Feature matrix assembly a 12 orizzonti con $\text{EDD}_{30}$, $VPD$, bilancio idrico FAO-56.
  - Modelli tabulari addestrati (Naive, ElasticNet, Random Forest, XGBoost) su expanding window out-of-sample.
  - Pipeline deep learning LSTM eseguita (con Temporal Attention, frequenza vincente a 10 giorni, micro-grid teste di output).
  - Modello Ensemble multi-architettura calibrato a pesi di varianza inversa.
  - Test cieco 2011–2025 eseguito su tutti i modelli e le varianti; metriche e tabelle consolidate.

---

## 2. TO-DO Operativi Aperti (Mappa di Lavoro Successivo)

### Fase A: Armonizzazione Grafica di Tutte le Figure
Obiettivo: Rendere tutte le figure visivamente coerenti, pulite e conformi agli standard scientifici di pubblicazione, eliminando l'aspetto disomogeneo.
Tutte le figure devono essere generate unicamente dai due script ufficiali in `scripts/`:

- [ ] **1. Figure del Capitolo 3 (Dati & Area di Studio) — `scripts/08_generate_eda_figures.py`**:
  - Centralizzare tutte le figure di EDA nel singolo script ufficiale.
  - Uniformare palette (tonalità continue scientifiche), font serif omogeneo coerente con il LaTeX, assi puliti con etichette chiare e unità di misura ($\text{bu/ac}$, $^{\circ}\text{C}$, $\text{mm}$, $\text{hPa}$).
  - Verificare e allineare in particolare:
    - *Fig 3.1:* `fig_study_area_map` (mappa geografica delle 135 contee selezionate).
    - *Fig 3.2:* `fig_yield_trajectory` (traiettoria 75 anni e trend secolare; rimuovere il duplicato `fig_yield_trajectory_dual_petrol` da `03_data_and_study_area.tex`).
    - *Fig 3.3:* `fig_state_yield_distributions` (distribuzioni violin di resa per stato).
    - *Fig 3.4:* `fig_era5_spatial_structure` (risoluzione e footprint ERA5-Land).
    - *Fig 3.5:* `fig_intra_seasonal_extremes` (dinamiche infra-stagionali degli estremi termici).
    - *Fig 3.6:* `fig_weather_shocks_footprint` (footprint spaziale degli shock storici).
    - *Fig 3.7:* `fig_spatial_shock_comparison` (confronto spaziale siccità 1988 vs 2012).
    - *Fig 3.8:* `fig_climate_yield_sensitivity` (sensibilità mensile correlazioni meteo-resa).

- [ ] **2. Figure del Capitolo 5 (Risultati Empirici) — `scripts/09_generate_results_figures.py`**:
  - Centralizzare tutte le figure dei risultati nel singolo script ufficiale.
  - Applicare la palette modelli condivisa e standard:
    - Naive: Grigio scuro (`#4b5563`)
    - ElasticNet: Blu (`#1d4ed8`)
    - Random Forest: Ambra/Arancio (`#d97706`)
    - XGBoost: Rosso (`#dc2626`)
    - LSTM: Viola (`#7c3aed`)
    - Ensemble: Smeraldo (`#059669`)
  - Verificare e allineare in particolare:
    - *Fig 5.1:* `fig_r2_horizon_progression` (progressione $R^2$ lungo gli orizzonti $H=12 \dots 1$).
    - *Fig 5.2:* `fig_shock_bias_profiles` (profilo dei residui e bias negli anni di shock estremo).
    - *Fig 5.3:* `fig_decile_calibration_and_residuals` (calibrazione per decili e tail shrinkage).
    - *Fig 5.4:* `fig54_pfi_faceted_bars` (Permutation Feature Importance dei 19 indicatori).
    - *Fig 5.5:* `fig55_pfi_domain_bars` (importanza aggregata per macro-domini bioclimatici).
    - *Fig 5.6:* `fig_county_r2_map` (mappa geografica dell'$R^2$ per contea con evidenziazione regola Minnesota $H^*=2$).

---

### Fase B: Revisione del Testo, Taglio Ripetizioni e De-prolissificazione (Capitoli 1–5)
Obiettivo: Eliminare verbosità, giri di parole inutili, preamboli difensivi e ripetizioni tra capitoli, adottando una voce diretta, asciutta e professionale.

- [ ] **Capitolo 1 (Introduction)**:
  - Snellire le motivazioni industriali; focalizzare nettamente le research question e i 2 contributi chiave.
  - Rimuovere premesse metodologiche o dettagli sui modelli prematuri.
- [ ] **Capitolo 2 (Literature Review)**:
  - Eliminare costruzioni binarie contrastive (*"rather than"*).
  - Assicurare che ogni paragrafo citi un solo paper primario di riferimento senza accumuli ridondanti.
- [ ] **Capitolo 3 (Data and Study Area)**:
  - Rimuovere le parafrasi numeriche delle tabelle (il testo sintetizza i trend e le evidenze, senza ripetere a parole i singoli valori già visibili).
  - Eliminare la riga duplicata del grafico della traiettoria petrol.
- [ ] **Capitolo 4 (Methodology)**:
  - Eliminare formule o spiegazioni ridondanti già introdotte al Capitolo 3 (es. i dettagli descrittivi delle variabili meteo).
  - Rimuovere formule intermedie non essenziali; mantenere definizioni matematiche snelle e dirette.
- [ ] **Capitolo 5 (Empirical Results)**:
  - Eliminare la rilettura testuale dei numeri delle Tabelle 5.1, 5.2, 5.3, 5.4; concentrarsi sull'interpretazione analitica ed economica dei risultati.
  - Integrare l'Ensemble nella discussione comparativa.

---

### Fase C: Stesura e Chiusura dei Capitoli Conclusivi (Capitoli 6 e 7)
Obiettivo: Valorizzare criticamente l'intero percorso di ricerca senza prolissità.

- [ ] **Capitolo 6 (Discussion & Operational Implications)**:
  - Sezione 6.1: Sintesi critica ed economica dei risultati (harvest contamination, Minnesota $H^*=2$, mitigazione del tail shrinkage, parità Diebold-Mariano).
  - Sezione 6.2: Implicazioni per la gestione del rischio di filiera e limiti di impiego reale (latenza ERA5T vs ERA5-Land).
  - Sezione 6.3: Confini metodologici e i 5 sviluppi futuri formalizzati in [`future_developments.md`](./future_developments.md).
- [ ] **Capitolo 7 (Conclusions)**:
  - Chiusura concisa: sintesi delle risposte alle research question, takeaway finali e valore metodologico del framework weather-only privo di data leakage.

---

### Fase D: Audit Formale Finale e Compilazione
- [ ] Compilazione LaTeX (`pdflatex`) con 0 errori, verifica riferimenti incrociati (`\ref`, `\cite`).
- [ ] Verifica conformità Vademecum (linea `\textit{Source: ...}` sotto ogni figura e tabella, didascalie auto-contenute ma concise).
- [ ] Verifica finale conteggio caratteri (controllo amministrativo a posteriori via `count_thesis_characters.py`).
