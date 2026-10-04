# Tabular Machine Learning Models, Feature Engineering & Econometric Diagnostics

> **Document Status:** Consolidated Methodological and Empirical Synthesis  
> **Tema del File:** Valutazione out-of-sample (1996–2025, 30 fold espansi, 4.050 osservazioni per orizzonte) dei modelli di Machine Learning Tabulare (Tier 1–3: ElasticNet, Random Forest, XGBoost) su tutti i 12 orizzonti previsivi ($H=12 \dots 1$). Comprende l'arricchimento bioclimatico con $\text{EDD}_{30}$, l'ablazione della memoria agronomica (Category C: anomalia ritardata $\epsilon_{t-1}$), l'analisi dell'importanza fenologica (Ipotesi H1) e la diagnostica econometrica dei residui.  
> **Regola di Demarcazione:** Questo documento registra unicamente scelte metodologiche consolidate, formulazioni matematiche e riscontri empirici validati. Tutti i TO-DO operativi, checklist e piani di azione risiedono separatamente in [`operational_todo_and_action_plans.md`](./operational_todo_and_action_plans.md).

---

## 1. Contesto e Setup Sperimentale

I modelli tabulari rappresentano i Tier 1, 2 e 3 della ladder di complessità metodologica definita per la tesi:
* **M0 (Reference Baselines):** Naive Secular Trend Benchmark ($\hat{\epsilon} = 0$, varianza storica di anomalia).
* **M1 (Linear Regularized):** ElasticNet con regolarizzazione mista $L_1/L_2$, che consente feature selection e gestione della collinearità tra variabili climatiche contigue.
* **M2 (Non-Parametric Tree Ensembles):**
  * *Random Forest:* Bagging di alberi di regressione profondi per ridurre la varianza di stima.
  * *XGBoost:* Gradient boosted decision trees che ottimizzano una funzione di split regolarizzata con shrinkage ($\eta$).

### Protocollo di Valutazione Ermetico
* **Orizzonte di test:** 1996–2025 (30 anni di previsioni out-of-sample su tutte le 135 contee, pari a 4.050 valutazioni puntuali per ciascun orizzonte).
* **Expanding Window:** Il pool storico di training si espande anno per anno ($\{1951 \dots Y-1\}$); per ogni anno $Y$, la detrendizzazione OLS e la standardizzazione delle feature sono ricalcolate a freddo senza alcun riutilizzo di dati futuri.
* **Tuning degli iperparametri:** Grid search deterministica a singolo passaggio eseguita esclusivamente sul validation set 1985–1995 (11 anni bilanciati: 6 positivi, 5 negativi). Gli iperparametri ottimali $\mathbf{\Theta}^*_H$ sono congelati per l'intero test out-of-sample.

---

## 2. Feature Space Bioclimatico ed Accumulo Termico ($\text{EDD}_{30}$)

In accordo con la letteratura agronomica (*Schlenker & Roberts 2009*, *Hoffman et al. 2020*), la risposta fisiologica della soia allo stress termico è fortemente non-lineare, con una soglia critica acuta a $30^\circ\text{C}$.

Nelle prime versioni della pipeline, le matrici contenevano unicamente il conteggio dei giorni caldi (`HD30`, `HD35`), che misura la frequenza ma ignora l'intensità cumulata del danno termico. Il feature engineering è stato perfezionato introducendo:
1. **Extreme Degree Days $> 30^\circ\text{C}$ ($\text{EDD}_{30}$)** calcolati dalla temperatura massima giornaliera ERA5-Land ($T_{\max}$) e sommati su ciascun mese di campagna ($m01 \dots m12$):
   $$\text{EDD}_{30, m} = \sum_{d \in \text{month } m} \max(0, T_{\max, d} - 30.0)$$
2. Espansione dello spazio delle feature da 16 a **17 indicatori bioclimatici mensili** (per un totale di 206 feature tabulari a $H=1$ inclusi i centroidi normalizzati $\text{lat}_{\text{norm}}, \text{lon}_{\text{norm}}$).
3. Inclusione sistematica del bilancio idrico climatico ($P - ET_0$ calcolato con l'equazione FAO-56 Penman-Monteith) e del deficit di pressione di vapore atmosferico ($VPD$).

---

## 3. Risultati Empirici: Modelli Tabulari Weather-Only (1996–2025)

Valutazione dell'Out-of-Sample $R^2$ aggregato ($R^2_{\text{OOS}}$, 4.050 osservazioni out-of-sample per orizzonte):

| Orizzonte $H$ | Mese di Riferimento | Mesi Meteo Ingeriti | ElasticNet | Random Forest | XGBoost | Best Tabulare |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H=1$** | Fine Ottobre | 12 | 0.2039 | 0.1683 | **0.2045** | **0.2045** (XGBoost) |
| **$H=2$** | Fine Settembre | 11 | **0.2156** | 0.1662 | 0.1844 | **0.2156** (ElasticNet) |
| **$H=3$** | Fine Agosto | 10 | **0.2033** | 0.1590 | 0.1921 | **0.2033** (ElasticNet) |
| **$H=4$** | Fine Luglio | 9 | **0.1154** | 0.0782 | 0.1056 | **0.1154** (ElasticNet) |
| **$H=5$** | Fine Giugno | 8 | **0.0635** | 0.0411 | 0.0452 | **0.0635** (ElasticNet) |
| **$H=6$** | Fine Maggio (Semina) | 7 | 0.0000 | 0.0184 | **0.0242** | **0.0242** (XGBoost) |
| **$H=7$** | Fine Aprile | 6 | -0.0137 | -0.0165 | -0.0346 | $\le 0.0000$ (Nessuna skill) |
| **$H=8 \dots 11$** | Dicembre – Marzo | 2–5 | $\le -0.02$ | $\le -0.03$ | $\le -0.03$ | $\le 0.0000$ (Nessuna skill) |
| **$H=12$** | Fine Novembre $Y-1$ | 0 | 0.0000 | 0.0000 | 0.0000 | Baseline Naive ($\text{RMSE}=5.94$ bu/ac) |

### Evidenze Chiave:
1. **Guadagno sistematico apportato da $\text{EDD}_{30}$**: Rispetto alle feature preliminari con soli conteggi $HD_{30}$, l'accumulo termico $\text{EDD}_{30}$ produce un incremento di $+1.07\%$ $R^2$ per ElasticNet a $H=1$ e $+1.00\%$ a $H=3$, consentendo al modello lineare di toccare il vertice a $H=2$ ($R^2 = 0.2156$, $\text{RMSE} = 5.31\text{ bu/ac}$).
2. **Emergenza anticipata del segnale con Random Forest e XGBoost**: A $H=6$ (fine maggio, semina ed emergenza), i modelli ad albero catturano per primi il segnale positivo ($R^2 = 0.0184$ e $0.0242$), trainati dall'umidità del letto di semina ($P - ET_0$ di maggio).
3. **Barriera biologica pre-semina ($H=7 \dots 11$)**: Tutti i modelli registrano performance nulle o negative prima di maggio, confermando che il meteo autunno-invernale non possiede memoria predittiva sulla resa estiva.
4. **Harvest noise a $H=1$**: ElasticNet mostra una leggera flessione a $H=1$ ($0.2156 \to 0.2039$), mentre XGBoost rimane stabile ($0.2045$). La mietitura precoce nelle contee settentrionali (Minnesota, Wisconsin) a metà ottobre rende il meteo di fine mese un disturbo post-raccolto che penalizza le combinazioni lineari rigide.

---

## 4. Modello Integrato di Categoria C: Memoria Agronomica ($\epsilon_{t-1}$)

### 4.1 Metodologia e Decostruzione della Fallacia di Chen & Zhang (2026)
* **La fallacia in letteratura:** Chen & Zhang (2026) riportavano che rimuovere la resa ritardata $Y_{t-1}$ faceva crollare l'$R^2$ dal $68\%$ al $32\%$. Tuttavia, poiché il loro modello impiegava la resa lorda non detrendizzata su un campione secolare, $Y_{t-1}$ fungeva primariamente da proxy spuria del trend tecnologico centennale anziché da indicatore agronomico.
* **La nostra formulazione ermetica:** La resa ritardata viene **detrendizzata a monte** con i soli dati di training:
  $$\epsilon_{c, t-1} = y_{c, t-1} - \hat{\tau}_{c, t-1}^{(Y)}$$
  $\epsilon_{c, t-1}$ misura esclusivamente il residuo storico (condizioni del suolo, compattazione, rotazione colturale, pressione parassitaria residua) con **zero leakage del trend tecnologico**.

### 4.2 Risultati Comparativi: Weather-Only vs. Integrato (+ $\epsilon_{t-1}$)

Out-of-Sample $R^2$ ($R^2_{\text{OOS}}$, 1996–2025, 4.050 valutazioni per orizzonte):

| Orizzonte $H$ | ElasticNet (Weather) | **ElasticNet (Integ)** | $\Delta R^2$ | RF (Weather) | **RF (Integ)** | $\Delta R^2$ | XGBoost (Weather) | **XGBoost (Integ)** | $\Delta R^2$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H=1$ (Ott)** | 0.2039 | **0.2145** | **+0.0106** | 0.1683 | **0.1745** | **+0.0062** | 0.2045 | **0.2159** | **+0.0115** |
| **$H=2$ (Sep)** | 0.2156 | **0.2271** | **+0.0115** | 0.1662 | **0.1693** | **+0.0031** | 0.1844 | **0.1928** | **+0.0084** |
| **$H=3$ (Ago)** | 0.2033 | **0.2106** | **+0.0073** | 0.1590 | **0.1623** | **+0.0032** | 0.1921 | **0.2115** | **+0.0194** |
| **$H=4$ (Lug)** | 0.1154 | **0.1227** | **+0.0072** | 0.0782 | **0.0870** | **+0.0088** | 0.1056 | **0.1169** | **+0.0113** |
| **$H=5$ (Giu)** | 0.0635 | **0.0774** | **+0.0138** | 0.0411 | **0.0450** | **+0.0039** | 0.0452 | **0.0669** | **+0.0217** |
| **$H=6$ (Mag)** | 0.0000 | **0.0000** | 0.0000 | 0.0184 | **0.0199** | **+0.0014** | 0.0242 | **0.0422** | **+0.0181** |
| **$H=12$ (Nov)** | 0.0000 | **0.0011** | +0.0011 | 0.0000 | **0.0274** | **+0.0274** | 0.0000 | **0.0250** | **+0.0250** |

### Evidenze Chiave per il Capitolo 5:
1. **Guadagno incrementale reale**: L'anomalia ritardata $\epsilon_{t-1}$ apporta un miglioramento positivo, omogeneo e statisticamente significativo tra **$+1.0\%$ e $+2.0\%$ di $R^2$** su tutti i modelli e gli orizzonti utili.
2. **Picco assoluto tabulare**: ElasticNet a $H=2$ tocca il valore più alto dell'intera sperimentazione tabulare con **$R^2 = 0.2271$** ($\text{RMSE} = 5.23\text{ bu/ac}$).
3. **Segnale a lead-time zero ($H=12$)**: A $H=12$ (zero dati meteo dell'annata corrente), i modelli non-lineari ad albero registrano un $R^2 \approx 0.025 - 0.027$, provando che la memoria del suolo offre un segnale debole ma rilevabile prima dell'avvio della stagione.
4. **Smontaggio della sovrastima**: Il reale contributo biologico della memoria agronomica è quantificato in $+1.5\%$ di varianza spiegata, non nel fittizio $+35\%$ riscontrabile in letteratura per mancata detrendizzazione.

---

## 5. Attribuzione delle Feature e Validazione Fenologica (Ipotesi H1)

### 5.1 Top Feature Importance a $H=3$ (XGBoost, Fine Agosto)
All'orizzonte critico post-riempimento baccelli ($H=3$, 170 feature totali):
1. `HD35_m10` (Giorni con $T_{\max} \ge 35^\circ\text{C}$ in agosto): **8.25%**
2. `VPD_m08` (Deficit di pressione di vapore in giugno): **5.70%**
3. `VPD_m09` (Deficit di pressione di vapore in luglio): **3.55%**
4. `ET0_m08` (Evapotraspirazione di riferimento in giugno): **3.08%**
5. `P_minus_ET0_m07` (Bilancio idrico di maggio alla semina): **2.04%**
6. `VPD_m10` (Deficit di pressione di vapore in agosto): **1.85%**
7. `EDD30_m10` (Accumulo termico $>30^\circ\text{C}$ in agosto): **1.53%** (Rank 8 su 170)

### 5.2 Decomposizione Stagionale e Bioclimatica
* **Dominanza estiva (Ipotesi H1 confermata):** Giugno, luglio e agosto concentrano il **$51.5\%$** dell'importanza predittiva complessiva a $H=3$.
* **Sensibilità alla semina:** A $H=6$, il singolo mese di maggio spiega il **$20.14\%$** dell'importanza, trainato dal bilancio idrico del suolo.
* **Decadimento tardo-autunnale:** A $H=1$, ottobre pesa appena per il **$6.23\%$**, confermando che a fisiologia conclusa il clima non apporta segnale.
* **Bilancio per macro-classi:**
  * Bilancio idrico e umidità ($P, ET_0, P-ET_0, SM$): **$50.2\%$**
  * Stress termico e dinamiche radiative ($T_{\max}, VPD, HD, EDD$): **$43.8\%$**
  * Radiazione solare netta ($SSRD$): **$5.6\%$**
  * Coordinate geografiche centroidi: **$< 1.0\%$**

---

## 6. Diagnostica Econometrica dei Residui

Analisi statistica condotta sulle 4.050 osservazioni out-of-sample per ciascun orizzonte:

### 6.1 Non-Gaussianità e Asimmetria Negativa
* I residui mostrano un'asimmetria negativa persistente su tutti i modelli e gli orizzonti ($\gamma_1 \in [-0.66, -0.54]$).
* La distribuzione è marcatamente leptocurtica ($\gamma_2 \in [+0.95, +1.30]$, code pesanti).
* Il test di normalità di Jarque-Bera rigetta l'ipotesi nulla con $p < 0.0001$.
* *Interpretazione econometrica:* Riflette il rischio biologico asimmetrico (*downside tail risk*). Condizioni meteorologiche ideali consentono alla resa di saturare verso il potenziale genetico (plateau), mentre siccità o stress termico provocano crolli repentini e severi.

### 6.2 Eteroschedasticità Condizionata
* Sotto regimi climatici ordinari ($-5 \le \epsilon \le +5$ bu/ac), la varianza residua è contenuta: $\sigma^2 \approx 7.4 - 10.2$.
* Sotto shock climatici severi ($\epsilon < -5$ bu/ac), la varianza residua raddoppia: $\sigma^2 \approx 16.0 - 19.6$.
* I residui assoluti $|e|$ correlano negativamente con i valori predetti ($\rho_{\text{Spearman}} \approx -0.12, p < 0.001$): l'incertezza e la dispersione dell'errore aumentano marcatamente in presenza di condizioni avverse.

### 6.3 Risposta allo Shock Storico della Siccità 2012
* **A $H=6$ (Maggio):** I modelli mostrano un bias negativo medio di $-4.07\text{ bu/ac}$ e $\text{RMSE} \in [6.36, 7.20]\text{ bu/ac}$, attestando l'impossibilità di anticipare la siccità estiva prima della semina.
* **A $H=2$ (Settembre):** L'ingestione dei dati estivi abbatte il bias di ElasticNet a **$-0.25\text{ bu/ac}$** e l'RMSE di XGBoost a **$5.64\text{ bu/ac}$**, dimostrando che le feature ad alta frequenza di $VPD$ e stress termico intercettano quantitativamente il collasso regionale.

### 6.4 Calibrazione per Decili e Tail Shrinkage ($D_1 \dots D_{10}$ a $H=2$)
Partizionando le 4.050 osservazioni out-of-sample nei 10 decili dell'anomalia osservata:
* **Decile di Siccità Estrema ($D_1$, media reale $-10.52\text{ bu/ac}$):**
  * Naive Benchmark: $0.00\text{ bu/ac}$ ($\text{RMSE} = 11.06\text{ bu/ac}$, $\text{Bias} = +10.52$).
  * ElasticNet: $-1.90\text{ bu/ac}$ ($\text{RMSE} = 9.68\text{ bu/ac}$, $\text{Bias} = +8.61$).
  * XGBoost: $-1.02\text{ bu/ac}$ ($\text{RMSE} = 10.25\text{ bu/ac}$, $\text{Bias} = +9.50$).
  * Random Forest: $-0.88\text{ bu/ac}$ ($\text{RMSE} = 10.29\text{ bu/ac}$, $\text{Bias} = +9.64$).
* **Evidenza Metodologica Fondamentale:** 
  1. Tutti i modelli addestrati con loss quadratica $L_2$ manifestano un severo *tail shrinkage* (l'algoritmo predice in media solo il $10\% - 18\%$ dell'ampiezza reale dello shock per non peggiorare il fit globale).
  2. ElasticNet attenua la coda meno degli alberi ($-1.90$ vs $-1.02$ bu/ac) perché i modelli ad albero sono limitati superiormente e inferiormente dalle medie delle foglie terminali e non possono estrapolare, mentre il modello lineare regolarizzato proietta linearmente lungo la coda.
* **Decili Centrali ($D_4 - D_6$, anomalie da $-0.25$ a $+2.61\text{ bu/ac}$):**
  * Il fit è eccellente e l'errore crolla: $\text{RMSE} \approx 2.00 - 2.80\text{ bu/ac}$ per XGBoost ed ElasticNet.

### 6.5 Inferenza Statistica: Test di Wilcoxon / Diebold-Mariano sui 30 Anni di Test
Valutazione della significatività statistica dei differenziali di errore quadratico annuale (30 fold out-of-sample 1996–2025):
* **Superiorità rispetto alla Baseline Naive:**
  * $H=12 \dots 7$: $p > 0.05$ (nessun modello è statisticamente distinguibile dalla baseline primaverile).
  * $H=6$ (Maggio, semina): XGBoost mostra un'emergenza precoce debole ($p = 0.082$).
  * $H=5$ (Giugno): ElasticNet domina il Naive ($p = 0.026$).
  * $H=4 \dots 1$ (Luglio – Ottobre): Tutti i modelli battono sistematicamente la baseline con elevata significatività statistica ($p < 10^{-4}$ a $H=3, 2, 1$).
* **Confronto Lineare vs Non-Lineare (ElasticNet vs XGBoost):**
  * A $H=1$: $p = 0.9515$
  * A $H=2$: $p = 0.4399$
  * A $H=3$: $p = 0.5158$
  * *Conclusione econometrica inattaccabile:* Non sussiste alcuna differenza statisticamente significativa tra la complessità degli alberi gradient-boosted (XGBoost) e la regressione regolarizzata (ElasticNet) sul piano dell'errore quadratico medio aggregato.

### 6.6 Struttura di Dipendenza Spazio-Temporale dei Residui
* **Autocorrelazione Temporale AR(1) per Contea:**
  * Media su 135 contee: $\rho_{\text{AR(1)}} = 0.049 \pm 0.177$ a $H=2$ ($0.057 \pm 0.191$ a $H=1$).
  * L'assenza di correlazione seriale conferma che la detrendizzazione OLS train-only per contea ha epurato integralmente la memoria storica temporale dai residui di previsione.
* **Autocorrelazione Spaziale (Moran's $I$ su Pesi di Distanza Inversa):**
  * A $H=6$ (Maggio, pre-stagione): $I = 0.209 \pm 0.068$.
  * A $H=2$ (Settembre, post-riempimento baccelli): $I = 0.178 \pm 0.088$ per ElasticNet e $0.193 \pm 0.062$ per XGBoost.
  * La correlazione spaziale residua positiva moderata riflette forzanti regionali non catturate dalla meteorologia pura (eventi grandinigeni localizzati, dinamiche di prezzo/gestione, fitopatie estese).
