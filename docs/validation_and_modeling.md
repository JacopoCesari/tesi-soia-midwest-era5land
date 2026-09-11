# Strategia di Validazione, Baseline e Modelli di Previsione

## 1. Obiettivo e Quadro Metodologico

La valutazione della capacità predittiva dei modelli deve simulare esattamente il contesto decisionale operativo di una previsione stagionale in-season. Molti studi di crop yield prediction presentano stime di accuratezza eccessivamente ottimistiche perché adottano schemi di cross-validation casuale o campionamenti k-fold non strutturati. Come dimostrato sperimentalmente da [Sweet et al. (2023)](references.md), la presenza di forte autocorrelazione spaziale e temporale nei dati climatici e agronomici fa sì che la validazione casuale sovrastimi drasticamente la reale capacità di generalizzazione su anni futuri non ancora osservati.

La tesi adotta pertanto uno schema di **validazione temporale expanding-window (rolling-origin)** privo di data leakage.

---

## 2. Schema di Validazione Expanding-Window (Rolling-Origin)

### 2.1 Architettura della Validazione Temporale
Per ogni anno di test target $t \in [t_{\text{start}}, t_{\text{end}}]$ all'interno del panel:
1. **Set di Addestramento:** include tutte le 479 contee per tutti gli anni strettamente precedenti: $\tau < t$.
2. **Inner Validation (Tuning degli Iperparametri):** eventuale selezione degli iperparametri condotta tramite un'ulteriore suddivisione temporale interna confinata all'interno del training set (es. blocchi temporali di anni passati).
3. **Set di Test Out-of-Sample:** include tutte le 479 contee dell'anno $t$.
4. **Espansione Progressiva:** terminata la valutazione per l'anno $t$, le osservazioni di tale anno vengono incorporate nel set di training per la stima del modello sull'anno $t+1$.

```
              SCHEMA DI VALIDAZIONE TEMPORALE EXPANDING-WINDOW
              ────────────────────────────────────────────────
Iterazione 1:
Training: [1950 .................... t-1] ──► Test: [t] (479 contee)

Iterazione 2:
Training: [1950 ........................ t] ──► Test: [t+1] (479 contee)

Iterazione 3:
Training: [1950 ........................... t+1] ──► Test: [t+2] (479 contee)
```

### 2.2 Vincoli Operativi Rigorosi
- **Integrità Annuale:** gli anni devono essere mantenuti uniti tra tutte le 479 contee. È formalmente vietato effettuare split casuali di coppie contea-anno (*county-year*).
- **Isolamento Train-Only:** detrending agronomico, standardizzazione delle feature, selezione delle variabili, calcolo delle soglie di classificazione e tuning degli iperparametri devono essere stimati **esclusivamente sul training set**. Nessuna informazione dell'anno di test o degli anni successivi può intervenire nel preprocessing.

---

## 3. Baseline di Riferimento Obbligatorie

Per quantificare il reale valore aggiunto del segnale meteorologico, i modelli di machine learning devono essere confrontati con baseline parsimoniose e robuste:

1. **Climatologia Storica di Contea:**
   Previsione basata sulla media storica della resa osservata nella specifica contea negli anni precedenti:
   $$\hat{Y}_{c, t}^{\text{clim}} = \frac{1}{t - t_0} \sum_{\tau = t_0}^{t - 1} Y_{c, \tau}$$
2. **Trend Tecnologico Storico:**
   Modello di regressione del trend secolare (lineare, quadratico o spline per contea/stato) calibrato unicamente sugli anni antecedenti a $t$:
   $$\hat{Y}_{c, t}^{\text{trend}} = \hat{\alpha}_c + \hat{\beta}_c \cdot t$$
3. **Modello Statistico / Econometrico (Linear & GAM):**
   Modello additivo generalizzato parsimonioso che combina il trend temporale con le principali feature agroclimatiche (GDD, estremi di calore sopra i 30 °C, pioggia cumulata), ispirato alla formulazione semiparametrica di [Schlenker & Roberts (2009)](references.md). Rappresenta il benchmark statistico non black-box.

---

## 4. Famiglie di Modelli Candidati

Il confronto include approcci a complessità crescente, senza assumere a priori la superiorità di alcun algoritmo:

1. **Random Forest (RF):**
   - Ensemble di alberi non parametrico con comprovata robustezza ed efficacia nella predizione di rese agricole ([Jeong et al., 2016](references.md); [Sweet et al., 2023](references.md)).
   - Tratta non linearità complesse e interazioni tra variabili senza richiedere assunzioni distribuzionali.
2. **Gradient Boosting (XGBoost / LightGBM):**
   - Algoritmo basato su alberi di decisione con boosting sequenziale, altamente performante su feature tabulari e indici rolling ([Torsoni et al., 2023](references.md)).
3. **Modello Sequenziale per Serie Temporali (LSTM):**
   - Architettura ricorrente (*Long Short-Term Memory*) alimentata direttamente dalla sequenza di pentadi o decadi meteorologiche dall'inizio della stagione fino al cutoff ([Shook et al., 2021](references.md); [Khaki et al., 2020](references.md)).
   - **Principio di comparabilità:** il modello LSTM deve essere addestrato e valutato **con lo stesso set informativo e lo stesso identico schema di rolling evaluation** degli altri modelli, senza beneficiare di split casuali o dati futuri.
   - Non si assume a priori che il deep learning superi i modelli ensemble basati su alberi o le baseline statistiche.

---

## 5. Metriche di Valutazione e Inferenza Statistica

### 5.1 Metriche di Errore Puntuale
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (Y_i - \hat{Y}_i)^2}$$
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |Y_i - \hat{Y}_i|$$
- **Coefficiente di Determinazione Out-of-Sample ($R^2_{\text{OOS}}$)**

### 5.2 Forecast Skill Score vs Baseline
Per ogni cutoff stagionale $d$, il guadagno predittivo del modello rispetto alla baseline di solo trend è misurato dallo Skill Score:
$$\text{Skill}(d) = 1 - \frac{\mathcal{L}_{\text{model}}(d)}{\mathcal{L}_{\text{trend}}(d)}$$
dove $\mathcal{L}$ rappresenta la funzione di perdita (es. RMSE o MAE).

### 5.3 Inferenza Statistica e Resampling a Blocchi Annuali
Dato che gli errori di previsione per le 479 contee nello stesso anno sono spazialmente e temporalmente correlati, è metodologicamente scorretto trattare le $479 \times \text{anni}$ osservazioni come campioni indipendenti (il che produrrebbe p-value artificialmente piccoli).

L'inferenza statistica sulla differenza di loss $\Delta \mathcal{L} = \mathcal{L}_{\text{trend}} - \mathcal{L}_{\text{model}}$ viene effettuata tramite:
- **Block Bootstrap per anno solare:** campionando con ripetizione interi anni (mantenendo unite tutte le 479 contee di ciascun anno).
- **Intervalli di confidenza empirici al 95%** su $\text{Skill}(d)$.

---

## 6. Definizione Formale dell'Earliest Stable Skill

Per rispondere alla seconda parte della domanda di ricerca (*"quanto precocemente prima del raccolto emerge tale capacità predittiva?"*), definiamo l'**Earliest Stable Forecast Origin ($d^*$)** come il primo cutoff stagionale del calendario agronomico che soddisfa simultaneamente tre criteri:
1. **Superiorità Media:** $\text{Skill}(d^*) > 0$ (errore medio inferiore alla baseline di trend).
2. **Significatività Statistica:** l'intervallo di confidenza al 95% sulla differenza di loss esclude lo zero ($\Delta \mathcal{L} > 0$).
3. **Stabilità Temporale:** il vantaggio predittivo permane statisticamente significativo in tutti i successivi cutoff della stagione ($d > d^*$) fino al raccolto finale.

Questa formulazione operativa permette di identificare con precisione la soglia temporale oltre la quale le decisioni agronomiche ed economiche possono fare affidamento sull'informazione meteorologica in modo stabile e verificabile.
