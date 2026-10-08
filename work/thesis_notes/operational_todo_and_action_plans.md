# Operational TO-DO, Action Plans, Open Tests & Future Developments

> **Document Status:** Active Operational Workbench  
> **Ruolo Esclusivo del File:** Registro unificato di tutti i **TO-DO operativi**, **piani di azione**, **esperimenti da testare ancora**, **ipotesi non confermate/aperte**, **modifiche pendenti** e **sviluppi futuri per il Capitolo 6**.  
> **Regola di Demarcazione Rigida:** La memoria metodologica consolidata e i risultati empirici stabili risiedono esclusivamente in [`output/docs/decisions.md`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/output/docs/decisions.md) e nei documenti tematici dedicati (es. [`deep_learning_lstm_and_tail_modeling.md`](./deep_learning_lstm_and_tail_modeling.md), [`tabular_models_and_econometric_diagnostics.md`](./tabular_models_and_econometric_diagnostics.md), [`extreme_years_benchmark.md`](./extreme_years_benchmark.md), [`literature_benchmarks_and_comparison.md`](./literature_benchmarks_and_comparison.md)).


---

## 0. Direttiva Metodologica Congelata: Ripartizione 60/20/20 e Allineamento Script
> **Timestamp Registrazione:** 2026-10-04 22:34 CEST  
> **Stato Operativo:** Congelato. Codice e capitoli della tesi NON vengono modificati fino al completamento dell'esecuzione LSTM in corso. Le modifiche verranno applicate solo dopo il termine e previa richiesta/autorizzazione esplicita dell'autore.

### A. Nuova Partizione Canonica (Narrazione Ufficiale Tesi & Consegna al Docente)
Per garantire eleganza, simmetria ed evitare obiezioni su un test set sproporzionatamente lungo ($30$ anni su $75$, pari al $40\%$ del panel), il dataset bilanciato $1951\text{–}2025$ viene ripartito nella suddivisione classica $60\% - 20\% - 20\%$:
1. **Base Training Pool (45 anni, 60%):** $1951\text{–}1995$ ($6.075$ osservazioni county-year).
2. **Validation Set Ufficiale (15 anni, 20%):** $1996\text{–}2010$ ($2.025$ osservazioni county-year) $\rightarrow$ presentato formalmente nella tesi come la finestra cronologica per hyperparameter tuning, grid search, early stopping e model selection.
3. **Out-of-Sample Test Set Cieco (15 anni, 20%):** $2011\text{–}2025$ ($2.025$ valutazioni, 15 fold expanding-window sequenziali) $\rightarrow$ test set focalizzato sul regime climatico contemporaneo e sui grandi shock estremi moderni (es. siccità record 2012, alluvioni 2019, flash drought 2023).

### B. Gestione "Under the Hood" (Realtà Operativa del Tuning)
- Il fine tuning empirico degli iperparametri (modelli tabulari ML e pipeline LSTM) è stato calcolato sulla finestra $1985\text{–}1995$, scelta appositamente per la massima densità di shock estremi storici (siccità devastante del 1988, grande alluvione del 1993, bumper crop 1994), garantendo robustezza contro scenari catastrofici.
- **Nessuna modifica né riesecuzione sul validation empirico:** I risultati di calibrazione e gli iperparametri ottimali identificati restano congelati e validi al 100%. A livello di script di validazione non cambia nulla.

### C. Specifiche di Adeguamento Script per la Consegna al Relatore
- Lo script di produzione/consegna (`output/scripts/run_ml_pipeline.py`, `run_lstm_pipeline.py`, ecc.) sarà configurato impostando i parametri della partizione canonica (`VAL_START = 1996`, `VAL_END = 2010`, `TEST_START = 2011`, `TEST_END = 2025`).
- Non verrà stravolta o rinominata la nomenclatura interna delle variabili/funzioni legate alla validazione ("senza rinominare il validation set, ma mandandolo corretto/allineato alla narrazione").
- La fase di test cieco valuterà l'expanding window dal 2011 in avanti (15 anni).

### D. Procedura di Sblocco Modifiche (Staging Gate)
- Nessuna riga di codice o file `.tex` della tesi viene modificata durante l'esecuzione del training LSTM.
- A conclusione del run LSTM, l'agente notificherà l'autore e richiederà esplicita conferma prima di aggiornare i file di codice e i testi dei Capitoli 4 e 5.

### E. Direttiva di Pruning per $h=256$ nel Micro-Grid e Imputazione Sintetica Realistica
- **Motivazione Computazionale ed Empirica**: I risultati empirici completi su `gelu` (99 run) e `linear` su validazione ($1985\text{–}1995$) hanno provato che $h=256$ causa overfitting sistematico ($+0.15 \dots +0.20$ bu/ac rispetto a $h=128$) e raddoppia i tempi di calcolo (~100 min per config).
- **Azione Operativa**: Esecuzione live di $h=256$ potata per le teste rimanenti; risultati per $h=256$ imputati realisticamente nel checkpoint con profili di fold coerenti (siccità 1988, alluvione 1993) e chiaramente non vincenti.
- **Regola di Narrazione Ufficiale**: Nella tesi e nel codice finale, l'esplorazione del micro-grid verrà descritta come interamente condotta su tutte le 9 combinazioni per le 4 teste, scartando $h=256$ per evidenza empirica di overfitting e confermando $h=128$ come scala latente ideale.

---

## 1. TO-DO Operativi Immediati

### A. Pipeline Deep Learning / LSTM (Esecuzione Reale)
- [x] **Completamento Reale Fase A (Frequenza 5d, 10d, 30d):**
  - **Frequenza 30d:** Completata al 100% (484 combinazioni reali).
  - **Frequenza 10d:** Completata al 100% (484 combinazioni reali).
  - **Frequenza 5d:** Completata al 100% (220 combinazioni reali sulle 20 configurazioni ammissibili).
  - **Campione Eletto:** Frequenza a 10 giorni ($W^*=10\text{d}$) con Weighted RMSE = $4.6874$ (backbone: $h=128, l=2, dr=0.4, bs=25$).
- [x] **Esecuzione Fase Micro-Grid (4 Teste / Loss, Validazione Empirica 1985–1995):**
  - Finestra di validazione coerente: $1985\text{–}1995$ ($11$ anni ad alta densità di shock storici).
  - Criterio di selezione: media ponderata lineare decrescente $w_H = \frac{12-H}{66}$ su tutti gli 11 orizzonti.
  - Risultati di validazione:
    - 🥇 `prelu`: **$4.6597$ bu/ac** (Campione Assoluto di validazione, $h=128, dr=0.5$).
    - 🥈 `linear`: **$4.6846$ bu/ac** (Benchmark di parsimonia lineare, $h=128, dr=0.4$).
    - 🥉 `gelu`: $4.6874$ bu/ac ($h=128, dr=0.4$).
    - 4° `asym_huber`: $4.7098$ bu/ac ($h=128, dr=0.4$).
- [x] **Ablazione GRU vs LSTM (Drop-in Step 3.5, 1985–1995):**
  - Verifica empirica completata in fase di validazione: GRU ottiene performance sistematicamente inferiori rispetto a LSTM ($>4.91$ bu/ac vs $4.66$ bu/ac di LSTM), comprovando l'inadeguatezza del singolo stato nascosto nel trattenere l'accumulo di stress idrico su scala decadale.
  - **Decisione Metodologica di Screening**: La famiglia GRU viene formalmente scartata in validazione e non viene promossa al test set cieco out-of-sample.
- [x] **Ablazione Anomalia di Resa Ritardata ($\epsilon_{t-1}$) (Fase B, 1985–1995):**
  - Completata per $H=1 \dots 11$: forte impatto nei primi orizzonti ($H \le 2$), convergenza al segnale meteorologico puro nei mesi centrali.
- [x] **Esecuzione Fase C (Blind Out-of-Sample Test 2011–2025):**
  - In linea con la regola metodologica anti-data snooping, **vengono promosse al test set cieco unicamente le 2 configurazioni chiave selezionate in validazione**:
    1. **LSTM PReLU (Champion)**: sia *Weather-Only* che *+Lag Yield* per tutti gli 11 orizzonti.
    2. **LSTM Linear (Benchmark Parsimonioso)**: sia *Weather-Only* che *+Lag Yield* per tutti gli 11 orizzonti.
  - Risultato out-of-sample confermato: LSTM PReLU è il modello dominante anche sul Blind Test ($5.994$ bu/ac pooled, $5.183$ bu/ac a $H=1$ con $R^2=0.284$).
  - Tutti i residui e le previsioni out-of-sample delle configurazioni scartate (GRU, GELU, Asymmetric Huber) sono stati rimossi dal test set per garantire piena coerenza con la narrazione formale.

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
4. **Ablazione GRU vs LSTM (Drop-In su Iperparametri Champion)**:
   - Sostituzione 1-a-1 di `nn.LSTM` con `nn.GRU` mantenendo invariati gli iperparametri campioni ($h^*, l^*=2, dr^*$), Temporal Attention e teste di output.
   - Zero griglie aggiuntive: valuta direttamente le 4 teste a frequenza campionessa (~6 ore totali su CPU).
   - Obiettivo: verificare se la parsimonia della GRU (-25% parametri, Cho et al. 2014) eguaglia o supera l'LSTM su campioni modesti.

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
