# Operational TO-DO, Action Plans, Open Tests & Future Developments

> **Document Status:** Active Operational Workbench  
> **Ruolo Esclusivo del File:** Registro unificato di tutti i **TO-DO operativi**, **piani di azione**, **esperimenti da testare ancora**, **ipotesi non confermate/aperte**, **modifiche pendenti** e **sviluppi futuri per il Capitolo 6**.  
> **Regola di Demarcazione Rigida:** La memoria metodologica consolidata e i risultati empirici stabili risiedono esclusivamente in [`output/docs/decisions.md`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/output/docs/decisions.md) e nei documenti tematici dedicati (es. [`deep_learning_lstm_and_tail_modeling.md`](./deep_learning_lstm_and_tail_modeling.md), [`tabular_models_and_econometric_diagnostics.md`](./tabular_models_and_econometric_diagnostics.md), [`extreme_years_benchmark.md`](./extreme_years_benchmark.md), [`literature_benchmarks_and_comparison.md`](./literature_benchmarks_and_comparison.md)).

---

## 1. TO-DO Operativi Immediati

### A. Pipeline Deep Learning / LSTM (Esecuzione in Corso)
- [ ] **Monitoraggio Conclusione Fase A (Grid Search Coarse):**
  - Verificare il completamento delle 44 configurazioni su 10d (attualmente a `cfg25`) e successivamente delle configurazioni a 5d.
  - Estrarre la frequenza campionessa $W^* \in \{30d, 10d, 5d\}$ basata su validation loss media aggregata (1985–1995).
  - Confermare empiricamente che la profondità $l^*=2$ domina $l=1$ e $l=3,4$.
- [ ] **Esecuzione Fase Micro-Grid (4 Teste / Loss):**
  - Eseguire il micro-grid locale ("uno sopra, uno sotto" rispetto a $h^*$ e $dr^*$) sulle 4 varianti:
    1. Standard MLP con attivazione GELU (MSE Loss).
    2. Linear projection baseline head (MSE Loss).
    3. Parametric ReLU (PReLU) head (MSE Loss).
    4. Asymmetric Huber Loss ($\alpha=1.5$ su shock negativi).
  - Verificare l'elezione indipendente dei migliori iperparametri per ciascuna testa strictly su validation (1985–1995).
- [ ] **Esecuzione Fase C (Expanding Test 1996–2025):**
  - Eseguire l'expanding test cieco su tutti gli 11 orizzonti ($H=1 \dots 11$) per ciascuna delle 4 teste.
  - Popolare automaticamente le tabelle di performance:
    - Tabella 5.1: Performance globale aggregata ($R^2_{\text{OOS}}$, RMSE, MAE).
    - Tabella 5.2: Scomposizione per regimi (Anni Normali vs Shock Storico 2012).

### B. Generazione Figure e Artefatti per il Capitolo 5
- [ ] **Figura 5.1 (Curva di Progressione Orizzonti):** Generare il grafico finale $R^2_{\text{OOS}}$ e RMSE per tutti i modelli (Naive, ElasticNet, RF, XGBoost, LSTM champion) lungo $H=12 \dots 1$.
- [ ] **Figura 5.2 (Confronto Teste e Loss LSTM):** Grafico comparativo delle 4 varianti di output/loss, evidenziando il comportamento negli anni di siccità estrema.
- [ ] **Figura 5.3 (Mappe Spaziali dei Residui 2012):** Mappa della distribuzione geografica dell'errore di previsione nel Midwest durante il flash drought del 2012.
- [ ] **Figura 5.4 (Feature Attribution & Attention Weights):** Curve di attenzione temporale LSTM e importanza fenologica TreeSHAP/Gain lungo i 12 mesi di campagna.

### C. Audit Visivo e Formale delle Figure dei Capitoli 1–3
- [ ] **Controllo Risoluzione e Layout Grafico:** Verificare per ciascuna figura in `output/thesis/figures/fig_*.pdf`:
  - Font leggibili (senza sovrapposizioni) e palette colorblind-friendly.
  - Assi con etichette esplicite e unità di misura ($\text{bu/acre}$, $^{\circ}\text{C}$, $\text{mm}$, $\text{hPa}$).
  - Didascalia auto-contenuta con riga di attribuzione formale conforme al Vademecum: `\textit{Source: Author's calculation on USDA NASS and ECMWF ERA5-Land data.}`.
  - Verifica specifica su: `fig_study_area_map.pdf`, `fig_yield_trajectory.pdf`, `fig_state_yield_distributions.pdf`, `fig_dispersion_dynamics.pdf`, `fig_era5_spatial_structure.pdf`, `fig_weather_climatology.pdf`, `fig_intra_seasonal_extremes.pdf`, `fig_weather_shocks_footprint.pdf`, `fig_spatial_shock_comparison.pdf`, `fig_climate_yield_sensitivity.pdf`.

---

## 2. Piani di Azione (Action Plans & Execution Roadmaps)

### A. Action Plan per la Stesura della Tesi (Capitoli 4 & 5)
1. **Capitolo 4 (`04_methodology.tex`)**:
   - Dettaglio completo nel file collegato: [`action_plan_chapter_4.md`](./action_plan_chapter_4.md).
   - *Sezione 4.1:* Formulazione del problema (anomalia continua $\epsilon_{c,Y}$, scomposizione additiva, orizzonti $H=12 \dots 1$).
   - *Sezione 4.2:* Rappresentazione dell'input (tensori 3D sequenziali per LSTM vs vettori tabulari 2D per alberi/lineari, coordinate continue lat/lon).
   - *Sezione 4.3:* Ladder di complessità dei modelli (Naive, ElasticNet, SVR, RF, XGBoost, LSTM con Temporal Attention).
   - *Sezione 4.4:* Protocollo expanding window a 3 fasi (1951–1984 train, 1985–1995 val, 1996–2025 test con Inner Validation split $t-3 \dots t-1$).
   - *Sezione 4.5:* Metriche di errore ($R^2_{\text{OOS}}$, RMSE, MAE) e protocollo di stress-testing su coorti di siccità storica.
   - *Verifica budget caratteri:* Script [`count_thesis_characters.py`](./count_thesis_characters.py) per target 16.000–19.000 caratteri.
2. **Capitolo 5 (`05_empirical_results.tex`)**:
   - *Sezione 5.1:* Baselines e confronto complessità (Tabella 5.1 globale).
   - *Sezione 5.2:* Risoluzione temporale (30d vs 10d vs 5d) e contributo della memoria agronomica (ablazione $\epsilon_{t-1}$).
   - *Sezione 5.3:* Stress-testing sui rischi di coda (annate 1988, 1993, 2012, 2016) e confronto loss simmetrica vs asimmetrica.
   - *Sezione 5.4:* Explainability biologica (analisi SHAP e curve di attenzione temporale $\alpha_t$).
   - *Sezione 5.5:* Sintesi econometrica e validazione formale delle ipotesi H1–H5.

### B. Roadmap Esecutiva Modellistica (Milestones di Completamento)
- Dettaglio completo nel file collegato: [`modeling_execution_roadmap.md`](./modeling_execution_roadmap.md).
- *Step 1–5 (Completati 100%):* Ambiente configurato, matrici multi-orizzonte generate, detrending OLS ermetico validato, ML tabulare eseguito su 1996–2025, diagnostica residui completata.
- *Step 6 (In Corso):* Fine-tuning finale LSTM (coarse grid 44 $\to$ micro-grid 4 teste $\to$ test blind espanso 1996–2025 $\to$ tabelle finali).

---

## 3. Cose da Testare Ancora ed Esperimenti Pendenti

1. **Micro-Grid Locale ("Uno Sopra, Uno Sotto") sulle 4 Teste/Loss**:
   - Da eseguire non appena congelata la frequenza campionessa $W^*$ derivata dalla Grid 44.
   - Parametri da saggiare per ciascuna testa: $h \in \{h^-, h^*, h^+\}$ e $dr \in \{dr^-, dr^*, dr^+\}$ (9 configurazioni per testa).
   - Valutazione su validazione espansa 1985–1995 con calcolo score fenologico pesato ($w=1.5$ per mesi estivi).
2. **Expanding Test Cieco (1996–2025, 4.050 osservazioni per orizzonte)**:
   - Esecuzione finale per ciascuna delle 4 teste su tutti gli 11 orizzonti temporali ($H=1 \dots 11$).
   - Estrazione dei residui per contea-anno per calcolare il miglioramento out-of-sample sulle annate di siccità (2012).
3. **Ablazione Anomalia di Resa Ritardata ($\epsilon_{t-1}$) su LSTM**:
   - Misurare se l'aggiunta dell'anomalia detrendata storica alla testa della rete ricorrente produce il medesimo guadagno incrementale ($+1.0\% \dots +2.0\%$ $R^2$) riscontrato sui modelli tabulari.

---

## 4. Cose Non Confermate e Ipotesi Aperte

### A. Harvest Contamination ad Ottobre ($H=1$) vs Picco a Settembre ($H=2$)
* **Evidenza empirica preliminare:** ElasticNet tocca il picco a $H=2$ ($R^2_{\text{OOS}} = 0.213$), flettendo a $H=1$ ($R^2_{\text{OOS}} = 0.193$), mentre XGBoost rimane stabile ($R^2 = 0.205$).
* **Ipotesi agronomica da discutere:**
  - Nelle contee settentrionali (Minnesota, Wisconsin), la trebbiatura si conclude a metà ottobre.
  - Le variabili meteo di fine ottobre misurano eventi su campi già mietuti, agendo come rumore post-raccolto che penalizza i modelli lineari.
* **Azione per la tesi:** Verificare la coerenza con i report USDA NASS Crop Progress e formalizzare l'argomentazione nel Capitolo 6 (Discussion).

### B. Trade-off Funzione di Perdita: $R^2$ Globale vs Accuratezza su Siccità Estrema
* **Riflessione teorica:** La loss quadratica standard (MSE) produce *tail shrinkage* sui crolli rari (attenua lo shock per non peggiorare l'errore medio sulle annate ordinarie).
* **Ipotesi da confermare:** La Asymmetric Huber Loss ($\alpha=1.5$) sacrificherà una modesta quota di $R^2_{\text{OOS}}$ globale a favore di una riduzione netta della sottostima negli anni 2012 e 1988, fornendo un valore operativo superiore per la gestione del rischio di filiera.

### C. Allineamento Fenologico dei Pesi di Attenzione Temporale (LSTM) e SHAP
* **Ipotesi biologica:** I coefficienti di attenzione temporale $\alpha_t$ e i valori TreeSHAP devono concentrare oltre il $50\%$ del peso decisionale nei mesi di luglio e agosto, confermando l'ipotesi H1 (vulnerabilità critica in fioritura e riempimento del baccello).

---

## 5. Modifiche, Refactoring e Bonifica del Repository

- [ ] **Eliminazione Nomi Informali/Operativi ("Overnight", "v2", "v3", "temp"):**
  - Sostituire nomenclature contingenti negli script e report con termini accademici formali (es. `--mode full_pipeline`, `--mode validation_sweep`, `--mode lag_ablation`, `--mode blind_test`).
  - Rinominare i file di output in `work/reports/experiments/lstm/`:
    - `checkpoint_lstm.json` $\to$ `hyperparameter_validation_matrix.json`.
    - Output di test $\to$ `blind_out_of_sample_results.json`.
- [ ] **Audit e Bonifica Globale delle Scorie Obsolete:**
  - Rimuovere file temporanei o log orfani.
  - Preservare la rigorosa separazione dettata da `AGENTS.md`: `output/` contiene esclusivamente il materiale minimale e accademico per il relatore; `work/` raccoglie il codice operativo, i dati intermedi e le note di lavoro.

---

## 6. Sviluppi Futuri e Prospettive di Ricerca (Capitolo 6: Discussion & Future Work)

Questi punti definiscono i confini della ricerca attuale e costituiscono il nucleo delle prospettive future per la tesi:

1. **Integrazione di Dati Satellitari ad Alta Risoluzione (Multimodal Remote Sensing):**
   - *Limite Attuale:* Il protocollo è rigorosamente "weather-only" per isolare il forcing termopluviometrico e testare la prevedibilità pura da rianalisi climatica.
   - *Sviluppo Futuro:* Integrare serie temporali Sentinel-2 e MODIS (NDVI, EVI, NDRE, SIF - Solar-Induced Chlorophyll Fluorescence) per tracciare la fotosintesi e la chioma vegetale post-emergenza, combinando dinamiche meteorologiche e risposta biologica osservata.
2. **Previsioni Dinamiche delle Date di Semina (Sowing Dates):**
   - *Limite Attuale:* La campagna adotta un calendario solare fisso (Novembre $Y-1$ – Ottobre $Y$).
   - *Sviluppo Futuro:* Introdurre stime dinamiche della data effettiva di semina derivate dai report USDA NASS Crop Progress o da algoritmi fenologici satellitari per allineare l'aggregazione temporale alla fenologia termica reale (giorni dopo la semina) anziché a mesi di calendario.
3. **Accoppiamento con Seasonal Dynamical Climate Forecasts (ECMWF SEAS5):**
   - *Limite Attuale:* I modelli valutano l'informazione osservata fino all'orizzonte $H$; nessun dato futuro è utilizzato.
   - *Sviluppo Futuro:* Alimentare le previsioni precoci ($H \ge 6$) accoppiando le osservazioni pregresse con le proiezioni probabilistiche d'insieme dei modelli climatici stagionali (ECMWF SEAS5, CFSv2), colmando il divario di informazione estiva durante i mesi primaverili.
4. **Modellistica Ibrida Process-Based e Machine Learning (Physics-Informed ML):**
   - *Limite Attuale:* I modelli statistici e di deep learning sono data-driven ed apprendono empiricamente le risposte alle sollecitazioni climatiche.
   - *Sviluppo Futuro:* Accoppiare modelli biofisici di crescita colturale (APSIM, DSSAT) con reti neurali o XGBoost, dove il modello biofisico stima la biomassa potenziale e l'algoritmo ML modella il residuo climatico e gli stress estremi non linearmente simulati.
5. **Attenzione Temporale Continua e Architetture Spazio-Temporali Dedicate:**
   - *Limite Attuale:* L'LSTM elabora la sequenza temporale aggregata a finestre $W \in \{30d, 10d, 5d\}$ e concatena le coordinate geografiche all'output.
   - *Sviluppo Futuro:* Implementare architetture Transformer spazio-temporali o Graph Neural Networks (GNN) che modellano esplicitamente la matrice di adiacenza spaziale tra contee e applicano meccanismi di auto-attenzione su sequenze continue giornaliere.
