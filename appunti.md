# Appunti di Ricerca, Metodologia e Risultati del Deep Learning (LSTM)

Data ultimo aggiornamento: 01/10/2026 (Sessione Pre-Overnight Run)
Autore: Jacopo Cesari (Tesi Magistrale in Data Science)
Progetto: Previsione delle anomalie di resa della soia nel Midwest USA con dati ERA5-Land

---

## 1. Sintesi dei Risultati dei Run Finora (Validazione 1985–1995, 11 Fold Espansi)

Tutti i confronti sotto riportati sono eseguiti sulla **medesima partizione di validazione (1985–1995)** con finestra espansa anno per anno, senza mai usare dati futuri e con detrendizzazione OLS train-only.

| Modello | Orizzonte $H$ | Feature Input | Spazialità / Head | Val RMSE (bu/ac) | Val $R^2$ OOS | Note di Performance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Trend** | $H=1$ (Ott) | Nessuna (anomalia = 0) | — | 5.4858 | 0.0000 | Baseline a varianza storica |
| **ElasticNet** | $H=1$ (Ott) | 16 indicatori meteo | Lineare | 5.1214 | 0.1320 | Modello lineare regolarizzato |
| **XGBoost** | $H=1$ (Ott) | 16 indicatori meteo | Alberi | 5.1082 | 0.1365 | Benchmark tabulare non-lineare |
| **LSTM V2** | $H=1$ (Ott) | 17 ind. meteo (con EDD30) | County Emb ($d=16$) + MLP GELU | 4.6825 | 0.2315 | $h=256, l=3, dr=0.4, bs=25$ |
| **LSTM V1** | $H=1$ (Ott) | 16 ind. + `lat_norm, lon_norm` | Head lineare diretta | **4.4953** | **0.2986** | $h=64, l=2, dr=0.4, bs=25$ |
| **LSTM V2** | $H=3$ (Ago) | 17 ind. meteo (con EDD30) | County Emb ($d=16$) + MLP GELU | 4.7120 | 0.2251 | $h=64, l=3, dr=0.2, bs=25$ |
| **LSTM V1** | $H=3$ (Ago) | 16 ind. + `lat_norm, lon_norm` | Head lineare diretta | **4.5230** | **0.2901** | $h=128, l=2, dr=0.4, bs=25$ |
| **LSTM V2** | $H=6$ (Mag) | 17 ind. meteo (con EDD30) | County Emb ($d=16$) + MLP GELU | 5.1646 | 0.1014 | $h=256, l=2, dr=0.0, bs=25$ |
| **LSTM V2** | $H=10$ (Gen) | 17 ind. meteo (con EDD30) | County Emb ($d=16$) + MLP GELU | 5.0072 | 0.1139 | $h=128, l=1, dr=0.0, bs=64$ |

---

## 2. Risultati del Test Comparativo di Spatial Encoding (01/10/2026)

Test empirico diretto eseguito sui medesimi 11 fold di validazione (1985–1995) per isolare l'impatto dell'encoding geografico (architettura comune $h=64, l=2, dr=0.2, bs=25$):

* **$H=01$ (Ottobre, 12 mesi)**:
  * V2 County Embedding discreto ($d=16$ nella testa): **4.7884 bu/ac**
  * V2 `[lat_norm, lon_norm]` nella testa MLP: **4.7040 bu/ac** ($-0.084$ bu/ac)
  * V1 `[lat_norm, lon_norm]` nell'input a ogni time step: **4.5671 bu/ac** ($-0.221$ bu/ac vs embedding!)
* **$H=03$ (Agosto, 10 mesi)**:
  * V2 County Embedding discreto ($d=16$ nella testa): **4.7512 bu/ac**
  * V2 `[lat_norm, lon_norm]` nella testa MLP: **4.6964 bu/ac** ($-0.055$ bu/ac)
  * V1 `[lat_norm, lon_norm]` nell'input a ogni time step: **4.6327 bu/ac** ($-0.119$ bu/ac vs embedding!)

**Conclusione metodologica**: Le coordinate continue (`lat_norm`, `lon_norm`) immesse a ciascun time step permettono alle celle LSTM di modulare l'impatto biofisico dello stress idrotermico in funzione della posizione geografica (es. 32°C al nord del Minnesota ha un effetto diverso rispetto al sud dell'Illinois) e azzerano l'overfitting dei 2.160 pesi discreti dell'embedding.

---

## 3. Confronto con la Letteratura (13 Core Papers + Géron)

1. **Khaki & Wang (2019) e Khaki et al. (2020)**:
   * Riportano un RMSE attorno all'8–11% (~3.8–4.5 bu/ac) su mais e soia con CNN-RNN e DNN.
   * *Differenza cruciale*: Loro lavoravano su resa lorda NON detrendizzata con 10 mappe di suolo. La rete apprendeva il trend tecnologico storico (in salita di 0.5 bu/ac/anno) e la latitudine geografica.
   * Nel nostro lavoro, con target puramente detrendizzato (*weather-only*), il nostro RMSE di 4.56 bu/ac su anomalia è in linea con i limiti fisici dell'informazione climatica.
2. **Hoffman et al. (2020) e Sweet et al. (2023)**:
   * Dimostrano che su pura anomalia meteorologica senza data leakage, il tetto della varianza spiegata ($R^2_{\text{OOS}}$) per modelli *weather-only* è compreso tra **0.18 e 0.24**.
   * Il nostro LSTM su validazione tocca $R^2 = 0.23 - 0.29$, allineandosi al limite teorico superiore della letteratura agronomica.
3. **Yin et al. (2026)**:
   * Dimostrano l'importanza della risoluzione temporale sub-mensile: passando da 30 giorni a 5 giorni (pentadi), l'$R^2$ dell'LSTM cresce da 0.55 a 0.67, perché le ondate di calore estreme (EDD30) e i deficit di umidità radicale avvengono su scale di 3–7 giorni e vengono diluiti dalle medie mensili.
4. **Géron (*Hands-On Machine Learning*, 3ª ed.)**:
   * *Inizializzazione*: Kaiming/He è la scelta corretta per strati con attivazione GELU (Xavier è per tanh).
   * *Ottimizzatore*: AdamW scorpora la regolarizzazione weight decay ($10^{-4}$) rispetto all'Adam classico.
   * *Stabilizzazione ricorrente*: Layer Normalization (`nn.LayerNorm`) prima dell'Attention per sequenze a 5 giorni (71 step).

---

## 4. Risoluzione della Criticità Metodologica: Zero Test Peeking / Zero Leakage

Nel codice V2, durante la valutazione sul test set espanso (1996–2025), il test set dell'anno $t$ veniva monitorato epoca per epoca per attivare l'early stopping (*test peeking*).

**Protocollo Rigoroso Implementato in V3**:
* Per prevedere l'anno di test $t$ (es. $t=2012$):
  * **Training set**: anni $1951 \dots t-4$.
  * **Inner Validation set**: anni $t-3, t-2, t-1$ (i 3 anni storici immediatamente precedenti a $t$).
  * L'early stopping e lo scheduler `ReduceLROnPlateau` monitorano **esclusivamente** l'Inner Validation set.
  * Quando l'Inner Validation smette di migliorare per 5 epoche, il modello viene congelato.
  * La previsione sull'anno di test $t$ viene generata a modello congelato: **100% blind out-of-sample**, esattamente come per i modelli tabulari.

---

## 5. Demarcazione Fenologica degli Orizzonti

* **In-Season ($H \le 6$, da Maggio a Ottobre, 6 mesi)**:
  * $H=6$ (Maggio): Semina ed emergenza.
  * $H=5$ (Giugno): Sviluppo vegetativo.
  * $H=4$ (Luglio): Fioritura e allegagione.
  * $H=3$ (Agosto): Riempimento baccelli (finestra critica di vulnerabilità).
  * $H=2$ (Settembre): Maturazione e senescenza.
  * $H=1$ (Ottobre): Raccolta.
* **Pre-Season ($H \ge 7$, da Novembre ad Aprile)**:
  * Periodo antecedente alla semina; informazione meteo debole, convergenza attesa verso la baseline storica Naive.
* **Metrica di Selezione Prioritaria**:
  $$\text{RMSE}_{\text{In-Season}} = \frac{1}{6} \sum_{H=1}^{6} \text{RMSE}_H$$
  con peso preponderante ai mesi estivi ($H=1, 2, 3, 4$).

---

## 6. Protocollo Definitivo: Grid Search Completa a 44 Configurazioni (Senza Esclusioni Forzate)

Su esplicita indicazione metodologica dell'autore ("non ti preoccupare delle ore di run, dobbiamo fare le cose bene, non forzare troppe esclusioni"), la pipeline adotta lo spazio fattoriale completo a **44 configurazioni**, senza sacrificare profondità o batch size.

### Spazio Iperparametrico delle 44 Configurazioni (`GRID_44`):
* $h \in \{64, 128, 256\}$
* $l \in \{1, 2, 3, 4\}$ ($l \in \{1, 2, 3\}$ per $h=256$)
* $dr \in \{0.0, 0.2, 0.4\}$
* $bs \in \{25, 64\}$
Totale: **44 configurazioni per ciascuna frequenza temporale (30d, 10d, 5d)**.

### Struttura delle Fasi Esecutive:
1. **Fase A (Grid Search a 44 Configurazioni su Tutti gli 11 Orizzonti $H=1 \dots 11$)**:
   * Valutazione su validazione espansa 1985–1995 (11 fold).
   * **Nessun fissaggio arbitrario tra frequenze**: 30d, 10d e 5d eleggono ciascuna in modo indipendente la propria configurazione ottimale $(h^*, l^*, dr^*, bs^*)$.
   * **Scoring Fenologico Pesato**:
     $$\text{Score}(W) = \frac{\sum_{H=1}^{11} w_H \cdot \text{RMSE}_H}{\sum_{H=1}^{11} w_H}$$
     con:
     * $w_H = 1.5$ per $H \in \{1, 2, 3\}$ (Agosto, Settembre, Ottobre: riempimento baccelli e resa finale).
     * $w_H = 1.0$ per $H \in \{4, 5, 6\}$ (Maggio, Giugno, Luglio: semina, sviluppo, fioritura).
     * $w_H = 0.5$ per $H \in \{7 \dots 11\}$ (Novembre – Aprile: pre-season, segnale meteo debole).
   * Elezione del Campione Assoluto $(W^*, \text{config}^*)$.
2. **Fase B (Ablation Anomalia di Resa Ritardata $\epsilon_{t-1}$)**:
   * Test della resa ritardata (`anomaly_lag1`) sulla configurazione campionessa per $H=1 \dots 6$.
3. **Fase C (Test Out-of-Sample Completo 1996–2025 su Tutti gli 11 Orizzonti)**:
   * Valutazione espansa su 30 anni (1996–2025, 4.050 osservazioni) con Inner Validation Split blind ($t-3 \dots t-1$) per tutti gli 11 orizzonti.
   * Calcolo metriche pooled out-of-sample: $R^2_{\text{OOS}}$, RMSE, MAE, Hit Rate direzionale.

### Ottimizzazioni Tecnologiche a Costo Zero (Preservando l'Integrità Scientifica):
* **Zero-Cost Sequence Slicing**: Le sequenze complete per tutti i 75 anni sono caricate una sola volta in RAM e affettate per orizzonte ($X[:, :S_H, :]$) senza alcun I/O su disco.
* **Checkpoint Atomico Resumabile**: Ogni singolo orizzonte viene serializzato immediatamente su `checkpoint_lstm.json`. Il processo può essere interrotto, riavviato a pezzi ("a step") o fatto girare in background continuo senza perdere un solo secondo di calcolo pregresso.
* **Early Stopping con LR Plateau**: Pazienza 5 con decadimento del learning rate per evitare epoche ridondanti sui fold già convergenti.

---

## 7. TODO Metodologico & Repository Hygiene: Formalizzazione Nomi, Percorsi e Bonifica Scorie

*Promemoria prioritario per la consegna della tesi e del codice al Relatore (Professore):*

1. **Eliminazione Totale di Nomi Informali/Operativi ("Overnight", "v2", "v3", ecc.)**:
   * Nel materiale finale da consegnare al relatore non deve comparire alcuna dicitura gergale o contingente (ad es. cartelle, log o opzioni CLI chiamate `overnight`, `v2`, `v3`, `temp`, `scratch`).
   * Sostituire con nomenclatura scientifica formale ed esplicativa:
     * Flag CLI: `--mode full_pipeline`, `--mode validation_sweep`, `--mode lag_ablation`, `--mode blind_test`.
     * Report e cartelle: `work/reports/experiments/lstm/hyperparameter_validation_matrix.json`, `blind_out_of_sample_results.json`.
     * Script: `output/scripts/run_lstm_pipeline.py` diventerà `train_lstm_anomaly_forecaster.py` o analogo modulo consolidato.
2. **Audit e Bonifica Globale delle Scorie Obsolete nel Repository**:
   * Eseguire una scansione approfondita dell'intero albero per:
     * Verificare l'assenza di script o prototipi obsoleti non più utilizzati.
     * Bonificare cartelle orfane o file temporanei non tracciati.
     * Garantire la rigida aderenza alle linee guida di `AGENTS.md`: separazione netta tra la delivery minimale, pulita e accademica (`output/`) e l'area di lavoro/raw data (`work/`).
     * Assicurare la massima razionalità, pulizia, eleganza e realismo professionale dell'intera repository prima della presentazione al docente.
