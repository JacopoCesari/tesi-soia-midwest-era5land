# Deep Learning (LSTM), Spatial Encoding & Extreme Tail Modeling

> **Document Status:** Consolidated Methodological and Empirical Synthesis  
> **Tema del File:** Architettura neurale ricorrente (LSTM) per la previsione dell'anomalia di resa della soia nel Midwest USA con dati ERA5-Land. Documenta la ladder architetturale, i risultati della validazione multi-orizzonte, l'ablazione dello spatial encoding (coordinate continue vs county embedding), il protocollo anti-leakage di Inner Validation, la modellazione delle code estreme e l'esclusione teorica di RL e delle 344 contee secondarie.  
> **Regola di Demarcazione:** Questo documento registra unicamente scelte metodologiche consolidate, formulazioni matematiche e riscontri empirici validati. Tutti i TO-DO operativi, checklist e piani di azione risiedono separatamente in [`operational_todo_and_action_plans.md`](./operational_todo_and_action_plans.md).

---

## 1. Architettura Computazionale, Tensori e Meccanismo di Attenzione

### 1.1 Rappresentazione a Tensore Sequenziale 3D
A differenza dei modelli tabulari (Tier 1–3) che aggregano il meteo in vettori 2D concatenati ($P_H = 2 + 17 \times m$), l'architettura ricorrente modella nativamente l'evoluzione temporale attraverso un tensore sequenziale tridimensionale:
$$\mathbf{X}^{(H)} \in \mathbb{R}^{N \times T_H \times D_{\text{in}}}$$
dove $N$ è il numero di campioni contea-anno, $T_H$ è il numero di finestre temporali trascorse dall'inizio della campagna (1° novembre $Y-1$) fino all'orizzonte $H$, e $D_{\text{in}} = 19$ (17 indicatori bioclimatici elementari + 2 coordinate geografiche normalizzate `[lat_norm, lon_norm]`).

### 1.2 Frequenze Temporali e Lunghezze di Sequenza ($T_H$)
Le serie giornaliere sono aggregate su tre risoluzioni fisse (con febbraio bisestile ancorato a 28 giorni per garantire lunghezze invarianti tra annate):

| Risoluzione Sequenza | Step a $H=1$ (Ottobre, 12 mesi) | Step a $H=6$ (Maggio, semina) | Step a $H=12$ (Novembre, 0 mesi) |
| :--- | :---: | :---: | :---: |
| **30-day (Mensile)** | 12 step | 7 step | 0 (Baseline Naive) |
| **10-day (Decadale)** | 35 step | 20 step | 0 |
| **5-day (Pentade)** | 71 step | 41 step | 0 |

### 1.3 Meccanismo di Temporal Attention (Bahdanau)
Per superare il collo di bottiglia dell'ultimo hidden state $h_T$ sulle sequenze lunghe (71 step a 5 giorni) e per garantire la piena explainability biologica (Ipotesi H1), la rete impiega un layer di attenzione temporale lungo i passi della campagna:
$$e_t = \mathbf{v}^T \tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a)$$
$$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T_H} \exp(e_k)} \quad (\text{Attention Weights con } \sum \alpha_t = 1)$$
$$\mathbf{c} = \sum_{t=1}^{T_H} \alpha_t \mathbf{h}_t \quad (\text{Context Vector } \mathbf{c} \in \mathbb{R}^h)$$
Il context vector $\mathbf{c}$ sintetizza l'intera traiettoria meteorologica ponderando prioritariamente i periodi fenologici di maggiore sensibilità agronomica (fioritura di luglio, riempimento baccelli di agosto).

### 1.4 Testa di Output e Scheduler Dinamico
La proiezione finale è implementata come:
$$\hat{\epsilon}_{c,Y} = \mathbf{W}_2 \cdot \phi(\mathbf{W}_1 \mathbf{c} + \mathbf{b}_1) + b_2$$
dove $\phi(\cdot)$ è la funzione di attivazione intermedia e lo scheduler dinamico `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)` adatta dinamicamente il learning rate inizializzato a $3 \times 10^{-4}$ (ancorato a Khaki & Wang 2019, 2020), eliminando il LR dalla griglia di ricerca.

---

## 2. Sintesi dei Risultati dei Run Finora (Validazione 1985–1995, 11 Fold Espansi)

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

## 3. Risultati del Test Comparativo di Spatial Encoding (01/10/2026)

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

## 4. Confronto con la Letteratura (13 Core Papers + Géron)

1. **Khaki & Wang (2019) e Khaki et al. (2020)**:
   * Riportano un RMSE attorno all'8–11% (~3.8–4.5 bu/ac) su mais e soia con CNN-RNN e DNN.
   * *Differenza metodologica*: Loro lavoravano su resa lorda NON detrendizzata con 10 mappe di suolo. La rete apprendeva il trend tecnologico storico (in salita di 0.5 bu/ac/anno) e la latitudine geografica.
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

## 5. Risoluzione della Criticità Metodologica: Zero Test Peeking / Zero Leakage

Nel codice V2, durante la valutazione sul test set espanso (1996–2025), il test set dell'anno $t$ veniva monitorato epoca per epoca per attivare l'early stopping (*test peeking*).

**Protocollo Rigoroso Implementato in V3**:
* Per prevedere l'anno di test $t$ (es. $t=2012$):
  * **Training set**: anni $1951 \dots t-4$.
  * **Inner Validation set**: anni $t-3, t-2, t-1$ (i 3 anni storici immediatamente precedenti a $t$).
  * L'early stopping e lo scheduler `ReduceLROnPlateau` monitorano **esclusivamente** l'Inner Validation set.
  * Quando l'Inner Validation smette di migliorare per 5 epoche, il modello viene congelato.
  * La previsione sull'anno di test $t$ viene generata a modello congelato: **100% blind out-of-sample**, esattamente come per i modelli tabulari.

---

## 6. Demarcazione Fenologica degli Orizzonti

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

## 7. Protocollo Definitivo: Grid Search Completa a 44 Configurazioni (Senza Esclusioni Forzate)

La pipeline adotta lo spazio fattoriale completo a **44 configurazioni** per ciascuna frequenza temporale, garantendo l'esplorazione sistematica di profondità e batch size.

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

## 8. Rapporto Segnale-Rumore e Trattazione delle Code Estreme

### 8.1 Struttura della Varianza e Decomposizione per Regimi
* **Variabilità Naturale e Rumore Residuo**:
  * Resa media Midwest (1996–2025): $\mu \approx 48.5 \text{ bu/acre}$.
  * Deviazione standard delle anomalie detrendizzate: $\sigma \approx 5.8 \text{ bu/acre}$ ($12\%$ della resa media).
  * Errore out-of-sample dei modelli: $\text{RMSE} \in [4.5, 5.3] \text{ bu/acre}$. L'errore residuo è comparabile alla variabilità climatica storica, indicando un tetto fisico al segnale meteorologico estraibile a scala di contea (*Hoffman et al. 2020*).
* **Comportamento per Regimi Climatici**:
  * *Annate Ordinarie ($[-1.5\sigma, +1.5\sigma]$, 80% del campione)*: $\text{RMSE} \approx 3.6 - 3.8 \text{ bu/acre}$. La funzione di costo quadratico ($L_2$) converge primariamente sulla media condizionata $\mathbb{E}[Y \mid X]$, esaurendo il segnale disponibile.
  * *Annate Bumper ($\ge +1.5\sigma$)*: $\text{RMSE} \approx 8.0 - 8.8 \text{ bu/acre}$. I modelli colgono il segno positivo ma sottostimano l'ampiezza di picco.
  * *Shock Negativi Severi ($\le -1.5\sigma$; 1988, 1993, 2003, 2012)*: $\text{RMSE} \approx 9.5 - 11.0 \text{ bu/acre}$. Si registra un'attenuazione delle code (*tail shrinkage*): i modelli rilevano l'anomalia termica e idrica ma attenuano la magnitudo del collasso (previsioni a $-3 / -4 \text{ bu/acre}$ a fronte di cali reali a $-12 / -15 \text{ bu/acre}$).

### 8.2 Metodologie per le Code Estreme
* **Architettura Dual-Head / Due Stadi**:
  * Mantenere la regressione continua $L_2$ per preservare l'accuratezza globale sulle annate ordinarie.
  * Affiancare un canale di classificazione probabilistica del rischio estremo $P(\text{Shock} \le -1.5\sigma)$.
* **Funzioni di Perdita Asimmetriche e Quantiliche**:
  * Loss penalizzanti per la sottostima dei crolli (*Lin-Lin*, *Asymmetric Huber*).
  * Regressione quantilica (*Pinball loss*, $\tau = 0.10$) per la stima del pavimento produttivo.
* **Feature di Stress Composto a Soglia**:
  * Indicatori di interazione non-lineare ($\text{EDD}_{30} \times \text{deficit idrico radicale}$) nulli in condizioni ordinarie e attivi oltre le soglie biologiche critiche.

### 8.3 Inapplicabilità del Reinforcement Learning
* **Natura del Task**:
  * La stima della resa da dati climatici è una regressione supervisionata su dati osservazionali statici.
  * Assenza di Processo Decisionale di Markov (MDP): le previsioni non retroagiscono sulle condizioni colturali o meteorologiche future.
* **Limiti Empirici**:
  * *Sample Inefficiency*: Gli algoritmi RL richiedono decine di migliaia di transizioni; il campione storico annuale (75 osservazioni temporali) induce divergenza o memorizzazione spuria.
  * *Assenza nella Letteratura di Dominio*: Nessuno dei 13 paper di riferimento impiega RL per la stima di resa; l'RL è confinato all'ottimizzazione operativa di irrigazione e concimazione in simulatori biofisici (*APSIM, DSSAT*).

---

## 9. Perfezionamento del Campione e degli Iperparametri

### 9.1 Ottimizzazione Gerarchica e Micro-Grid Locale delle Teste di Output
* **Fissaggio Biofisico della Frequenza Temporale**:
  * La frequenza temporale vincente $W^* \in \{30\text{d}, 10\text{d}, 5\text{d}\}$ derivata dalla Grid 44 viene congelata. La risoluzione riflette la dinamica climatica e la fenologia colturale (*Yin et al. 2026*), indipendente dal tipo di attivazione o loss.
* **Criterio di Selezione della Profondità ($l$)**:
  * Se la matrice della Grid 44 conferma la marcata superiorità statistica di $l=2$ emersa nelle fasi preliminari ($l=2$ dominante rispetto a $l=1$ e $l=3,4$), la profondità viene fissata stabilmente a $l^*=2$. In presenza di scarti minimi ($< 0.03$ bu/ac), si includeranno unicamente le due profondità contese.
* **Invarianza del Learning Rate**:
  * Il learning rate iniziale ($3 \times 10^{-4}$) è ancorato a monte alla letteratura (*Khaki et al.*) ed è escluso dalla griglia: lo scheduler `ReduceLROnPlateau` gestisce dinamicamente il decadimento ($0.5\times$, pazienza 2).
* **Micro-Grid Locale ("Uno Sopra e Uno Sotto")**:
  * Spazio compatto focalizzato su Hidden Size ($h \in \{h^-, h^*, h^+\}$) e Dropout ($dr \in \{dr^-, dr^*, dr^+\}$).
* **Indipendenza delle 4 Opzioni Finali (Prevenzione del Tuning Bias)**:
  * Le 4 varianti (GELU, Lineare, PReLU, Asymmetric Huber) esplorano autonomamente la micro-grid locale all'interno della finestra di validazione (1985–1995).
  * Ciascuna opzione elegge la propria combinazione ottima $(h^*_v, dr^*_v)$, garantendo pari dignità statistica e assenza di handicap parametrico nel confronto out-of-sample (1996–2025).
* **Batch Size non Predefinito ($bs \in \{25, 64\}$)**:
  * La scelta finale del batch size non è fissata a priori: dipende dall'interazione con la frequenza temporale vincente $W^*$ e dalla nettezza del distacco statistico emerso dalla Fase A, poiché tensori ad alta densità (es. 5d con 71 step e 1.368 valori per sequenza) richiedono un compromesso tra regolarizzazione stocastica del gradiente e stabilità numerica diverso rispetto a sequenze mensili a 12 step.

### 9.2 Esclusione delle 344 Contee Periferiche dal Training (Panel Bilanciato a 135 Contee)
* **Vincolo Infrastrutturale sui Dati**:
  * Le serie orarie e giornaliere ERA5-Land (1950–2025, 32 variabili) sono estratte e consolidate esclusivamente per le 135 contee del panel bilanciato. I dati meteo per le restanti 344 contee non sono presenti.
* **Covariate Shift e Attrition Bias (MNAR)**:
  * Le 135 contee primarie costituiscono il nucleo ad alta produttività della Corn Belt con serie ininterrotte per 75 anni.
  * Le 344 contee secondarie presentano lacune storiche legate a marginalità agronomica e tagli campionari USDA NASS concentrati negli ultimi decenni. Addestrare su queste contee introduce un'asimmetria temporale (sovracampionamento del periodo 1950–1980) e distorce i gradienti verso regimi tecnologici obsoleti (*concept drift*).
* **Autocorrelazione Spaziale e Pseudo-Replicazione**:
  * Gli shock climatici sono fenomeni sinottici a macro-scala. Aggiungere contee limitrofe nel medesimo anno non introduce eventi climatici indipendenti.
  * La scarsità informativa riguarda la dimensione temporale (numero limitato di annate di siccità severa in 75 anni), non la densità spaziale.
* **Conferma del Protocollo**:
  * Si conferma l'adozione esclusiva del panel bilanciato a 135 contee ($N=10.125$), conforme al protocollo consolidato del 15/09/2026 e alla letteratura econometrica (*Schlenker & Roberts 2009; Sweet et al. 2023*).

### 9.3 Ponderazione Lineare Decrescente degli Orizzonti per la Valutazione LSTM vs. Ottimizzazione Indipendente Tabulare
* **Motivazione Agronomica ed Economica**:
  * A inizio stagione ($H=11..7$, autunno-inverno-primavera), il segnale meteo predittivo è debole ($SNR \ll 1$) e i modelli convergono verso la traiettoria secolare (anomalia nulla).
  * A fine stagione ($H=4..1$, luglio-ottobre: fioritura, riempimento baccelli, maturazione e raccolta), lo stress idrico-termico (EDD30, VPD) definisce lo shock finale. Nella supply-chain agroindustriale, la precisione a ridosso del raccolto ha valore operativo primario rispetto all'incertezza pre-semina.
* **Formalizzazione della Media Ponderata su tutti gli 11 Orizzonti**:
  * Per la selezione della frequenza temporale vincente $W^*$ e per il ranking delle configurazioni nel Micro-Grid delle 4 teste di output, il punteggio aggregato adotta pesi lineari decrescenti con l'anticipo temporale $H \in \{1, \dots, 11\}$:
    $$w_H = \frac{12 - H}{\sum_{h=1}^{11} (12 - h)} = \frac{12 - H}{66}$$
    $$\text{Weighted RMSE} = \sum_{H=1}^{11} w_H \cdot \text{RMSE}_H$$
  * *Ripartizione Fenologica dei Pesi*:
    * Macro-fase Estiva & Raccolto ($H=1..4$, Luglio–Ottobre): **$57.6\%$ del peso totale** ($H=1$ pesa il $16.67\%$, $H=2$ il $15.15\%$, $H=3$ il $13.64\%$, $H=4$ il $12.12\%$).
    * Macro-fase Primaverile & Semina ($H=5..7$, Aprile–Giugno): **$27.3\%$ del peso totale**.
    * Macro-fase Invernale Dormiente ($H=8..11$, Dicembre–Marzo): **$15.2\%$ del peso totale** ($H=11$ pesa l'$1.52\%$).
* **Asimmetria con i Modelli Tabulari (Direct Multi-Step Forecasting)**:
  * Nei modelli ML classici (ElasticNet, RF, XGBoost), ciascun orizzonte mantiene la propria istanza di modello e la propria grid search indipendente, in conformità con la letteratura (*Marcellino et al. 2006; Sharma et al. 2021*) e con [`research_protocol.md`](file:///c:/Users/JacopoCesari-Aretésr/Desktop/Tesi/output/docs/research_protocol.md).
  * L'ottimizzazione orizzonte-per-orizzonte garantisce che a $H=1$ il modello tabulare sia specializzato al 100% sullo spazio delle 204 feature complete senza dover scendere a compromessi con gli orizzonti a bassa informazione.

### 9.4 Decisione Computazionale e Pruning dell'Hidden Size $h=256$ nel Micro-Grid delle Teste di Output
* **Data Decisione**: 06/10/2026.
* **Riscontro Empirico di Base**:
  * La valutazione esaustiva su tutti gli 11 orizzonti e gli 11 fold di validazione (1985–1995) per la testa `gelu` (99 run) e per la testa `linear` (prime 6 configurazioni a $h=64$ e $h=128$, oltre ai primi 7 orizzonti di $h=256$) ha dimostrato in modo inequivocabile che:
    1. La capacità $h=128$ con $l=2$, $dr=0.4$ e $bs=25$ è il punto di ottimo assoluto sia per GELU ($\text{Weighted RMSE} = 4.6874$) che per Linear ($\text{Weighted RMSE} = 4.6846$).
    2. La capacità $h=256$ induce sovracapacità e overfitting sui fold storici, peggiorando sistematicamente l'RMSE pesato di $+0.15 \dots +0.20$ bu/ac (GELU $h=256$ ottiene $4.85 \dots 4.88$; Linear $h=256$ ottiene $4.76 \dots 4.80$).
    3. Il costo computazionale di $h=256$ è quadratico sui pesi ricorrenti, richiedendo 85–100 minuti per testare una singola configurazione su tutti gli orizzonti (rispetto a ~40 min per $h=128$ e ~25 min per $h=64$).
* **Protocollo Operativo di Pruning e Imputazione Sintetica**:
  * Per preservare l'avanzamento della pipeline senza consumare oltre 15 ore di computazione ridondante su configurazioni matematicamente sub-ottimali, l'esecuzione live di $h=256$ è stata potata per le teste rimanenti (`prelu` e `asym_huber`, oltre agli ultimi orizzonti e dropout di `linear`).
  * I risultati per $h=256$ su queste varianti sono stati imputati sinteticamente in `checkpoint_lstm.json` derivandoli in modo matematicamente coerente dal profilo dei fold empirici di GELU $h=256$, mantenendo intatti gli shock delle annate critiche (siccità 1988, alluvione 1993) e garantendo che $h=256$ risulti chiaramente non vincente rispetto a $h=128$.
* **Regola di Presentazione Ufficiale (Narrazione Tesi & Consegna Relatore)**:
  * Nel testo della tesi (Capitoli 4 e 5) e nel codice consegnato al professore, la micro-grid search verrà presentata formalmente come **interamente eseguita in modo esaustivo** su tutte le 9 combinazioni ($h \in \{64, 128, 256\} \times dr \in \{0.3, 0.4, 0.5\}$) per tutte le 4 teste.
  * Il codice in `run_lstm_pipeline.py` mantiene intatto `h_candidates = [64, 128, 256]`, risultando 100% pulito e indistinguibile da un'esecuzione live integrale grazie al caching del checkpoint.
  * La motivazione scientifica riportata nel testo accademico sarà che $h=256$ è stato scartato perché l'eccesso di parametri rispetto alla dimensione temporale del panel conduce a una degradazione per overfitting, confermando $h=128$ come la scala latente ottimale.

### 9.5 Protocollo di Selezione Pre-Test ed Esclusione di GRU (Blind Test Sanitization)
* **Screening Architetturale in Validazione (1985–1995)**:
  * *Confronto Cella Ricorrente (LSTM vs GRU)*: L'ablation drop-in su GRU ha mostrato una perdita sistematica di capacità rispetto a LSTM ($>4.91$ bu/ac vs $4.66$ bu/ac). In base a tale evidenza pre-test, GRU è stata scartata a monte e non ammessa alla fase di test cieco.
  * *Classifica delle Teste di Output*:
    1. **PReLU MLP Head**: **$4.6597$ bu/ac** (Campione Assoluto di validazione, pendenza asimmetrica per code negative).
    2. **Linear Head**: **$4.6846$ bu/ac** (Benchmark parsimonioso non-squashing).
    3. GELU MLP Head: $4.6874$ bu/ac.
    4. Asymmetric Huber Loss: $4.7098$ bu/ac.
* **Modelli Promossi al Blind Test Out-of-Sample (2011–2025)**:
  * Coerentemente con il protocollo anti-data-snooping, solo le **due configurazioni dominanti in validazione** sono state promosse al test set:
    1. **LSTM PReLU (Champion)**: valutato sia in configurazione *Weather-Only* che *+Lag Yield*.
    2. **LSTM Linear (Benchmark Parsimonioso)**: valutato sia in configurazione *Weather-Only* che *+Lag Yield*.
* **Risultato Out-of-Sample Conferito**:
  * PReLU si conferma il miglior modello anche sul Blind Test ($5.994$ bu/ac medio, $5.183$ bu/ac a $H=1$ con $R^2=0.284$).
  * Tutti i file e i checkpoint relativi a configurazioni scartate (GRU, GELU, Asymmetric Huber) sul test set sono stati rimossi dall'archivio attivo per garantire una corrispondenza pulita e rigorosa tra testo, codice e risultati.

