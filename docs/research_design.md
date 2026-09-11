# Disegno della Ricerca e Metodologia Generale

## 1. Titolo e Domanda di Ricerca

**Titolo della Tesi:** Previsione della resa della soia tramite modelli di Machine Learning e dati meteorologici ad alta risoluzione.

### Domanda di Ricerca Centrale
> *Quali modelli, utilizzando dati meteorologici osservati, riescono a prevedere la resa della soia con un miglioramento statisticamente significativo e stabile rispetto alle baseline di trend storico e climatologia, e quanto precocemente prima del raccolto emerge tale capacità predittiva?*

Il focus della ricerca non è dimostrare la presunta superiorità algoritmica del deep learning su dati tabulari o grigliati, bensì valutare in modo rigoroso, riproducibile e operativo l'**informazione predittiva incrementale** apportata dalle variabili meteorologiche rispetto a ciò che sarebbe già prevedibile mediante la semplice dinamica tecnologica e la climatologia storica locale.

---

## 2. Contributo Scientifico della Tesi

La letteratura recente sulla previsione delle rese colturali soffre spesso di una sovrapposizione concettuale tra *crop yield estimation* (stima a posteriori a stagione conclusa o con meteo futuro noto) e *in-season yield forecasting* (vera previsione operativa con disponibilità informativa limitata alla data di previsione). Inoltre, numerosi lavori adottano schemi di validazione casuale (random train/test split o k-fold CV non raggruppata) che causano forte data leakage temporale e spaziale, sovrastimando le reali performance out-of-sample ([Sweet et al., 2023](references.md)).

Il contributo di questo lavoro si articola su sei pilastri metodologici integrati:
1. **Previsione county-level ad alta granularità:** modellazione a livello di contea su un panel geograficamente rappresentativo del Midwest statunitense.
2. **Dati meteorologici fisicamente coerenti ed esatti:** impiego del reanalysis ERA5-Land aggregato su base giornaliera (finestra sincronizzata UTC) senza ricorrere a campionamenti spuri.
3. **Confronto multilivello di modelli:** valutazione sistematica che include baseline parsimoniose (trend tecnologico, climatologia storica, modelli lineari/GAM regolarizzati), algoritmi di machine learning ensemble basati su alberi (Random Forest, Gradient Boosting) e architetture per serie temporali sequenziali (LSTM).
4. **Cutoff previsionali progressivi (Lead-time analysis):** simulazione operativa di molteplici punti di previsione lungo la stagione colturale (dalla fase pre-semina fino alla maturazione e raccolta), calcolando a ciascun cutoff esclusivamente le feature derivabili dai dati disponibili a quella data.
5. **Validazione temporale expanding-window (Rolling-origin):** schema di validazione out-of-sample in cui tutti gli anni di test sono temporalmente successivi a quelli di training, con l'intero ciclo di preprocessing, stima del trend, scaling e selezione feature confinato all'interno dell'insieme di addestramento.
6. **Identificazione dell'Earliest Stable Skill:** determinazione formale della prima data del calendario agronomico in cui il modello meteorologico supera la baseline con significatività statistica persistente fino alla raccolta.

---

## 3. Ambito Geografico e Unità di Osservazione

- **Stati inclusi:** 6 stati chiave del Corn Belt / Midwest USA:
  - Illinois (IL)
  - Indiana (IN)
  - Iowa (IA)
  - Minnesota (MN)
  - Missouri (MO)
  - Ohio (OH)
- **Contee selezionate:** 479 contee con continuità storica ininterrotta nelle rilevazioni statistiche ufficiali.
- **Unità di analisi:** contea-anno (*county-year*) per il target; contea-giorno (*county-day*) per le serie meteorologiche intermedie, aggregate a finestre temporali di 5, 10 e 30 giorni.

---

## 4. Orizzonti Temporali e Distinzione dei Dataset

È fondamentale mantenere una chiara distinzione tra la serie temporale agronomica bilanciata e la copertura meteorologica:

1. **Balanced Panel Principale (Agronomico): 1950–2010**
   - 61 anni completi.
   - 479 contee.
   - **29.219 osservazioni county-year** senza alcun valore mancante nel target o nella superficie.
   - Costituisce il nucleo di addestramento e validazione per la modellazione econometrica e di machine learning.
2. **Anni Recenti: 2011–2025**
   - Conservati in file dedicati con le osservazioni USDA NASS SURVEY disponibili.
   - Possono presentare valori mancanti dovuti a cambiamenti nelle politiche di campionamento USDA NASS.
   - **Nessuna imputazione automatica:** questi anni non fanno parte del balanced panel principale e sono riservati per analisi di robustezza fuori campione o test su panel sbilanciati.
3. **Copertura Meteorologica Prevista: 1950–2025**
   - Serie giornaliera continua da ERA5-Land per l'intero arco di 76 anni su tutte le 479 contee.

---

## 5. Definizione dei Target

### Target Primario: Resa Continua e Anomalia di Resa
- **Resa continua osservata ($Y_{c, t}$):** espressa originariamente in bushel per acro (`BU / ACRE`), convertibile in t/ha o kg/ha per coerenza scientifica internazionale.
- **Anomalia di resa controllando il trend tecnologico ($\tilde{Y}_{c, t}$):**
  $$\tilde{Y}_{c, t} = Y_{c, t} - \hat{f}_{\text{trend}}(c, t)$$
  dove $\hat{f}_{\text{trend}}(c, t)$ rappresenta la componente di trend agronomico/tecnologico stimata **esclusivamente sui dati storici antecedenti** all'anno di test.

### Target Complementare: Classificazione Categorica
- Classificazione degli esiti annuali in tre classi qualitative: *Low*, *Normal*, *High*.
- **Regola train-only rigorosa:** le soglie di separazione (es. percentili 33° e 66° o $\pm 1$ deviazione standard dell'anomalia) devono essere stimate unicamente sul set di training dell'expanding-window corrente, senza alcun leakage dai dati futuri o dal set di test.

---

## 6. Forecast Cutoffs Operativi

Per simulare l'ambiente previsionale reale, il modello viene valutato su una sequenza di date fisse (*cutoff origins*) durante la stagione colturale:
1. **31 Maggio:** fine semina / emergenza vegetativa iniziale.
2. **30 Giugno:** pieno sviluppo vegetativo / inizio fioritura.
3. **31 Luglio:** fioritura avanzata / allegagione dei baccelli (*pod setting*).
4. **31 Agosto:** riempimento dei semi (*grain filling*), fase storicamente più sensibile allo stress termo-idrico.
5. **15 / 30 Settembre:** maturazione fisiologica e pre-raccolta.

In ogni cutoff $d$, le covariate meteorologiche includono unicamente le metriche calcolate sui dati osservati fino al giorno $d$.

---

## 7. Ipotesi di Ricerca Verificabili

- **H1 (Skill Incrementale):** L'inclusione delle variabili meteorologiche riduce l'errore quadratico medio di previsione out-of-sample rispetto al modello basato su solo trend storico, ma l'ampiezza di tale miglioramento nei cutoff pre-estivi (maggio-giugno) è limitata.
- **H2 (Finestra Critica Fenologica):** Il guadagno predittivo incrementale subisce una marcata accelerazione tra il cutoff di fine luglio e quello di fine agosto, in coincidenza con le fasi di fioritura e riempimento baccelli.
- **H3 (Complessità del Modello vs Benchmark Parsimonioso):** I modelli non lineari di machine learning (Random Forest, Gradient Boosting) superano le regressioni lineari ingenue, ma il loro vantaggio si riduce quando confrontati con modelli statistici flessibili e ben specificati (GAM con termini non lineari di temperatura ed esposizione estrema).
- **H4 (Leakage e Sovrastima in Random CV):** Uno schema di cross-validation casuale county-year produce metriche di accuratezza artificialmente ottimistiche rispetto alla reale validazione temporale rolling out-of-sample ([Sweet et al., 2023](references.md)).

---

## 8. Perimetro Attuale e Sviluppi Futuri

### Scope Corrente (Weather-Only)
Il disegno sperimentale iniziale della tesi è rigorosamente **weather-only**:
- Input: dati meteorologici ERA5-Land (variabili atmosferiche e del suolo, indici derivati).
- Target: rese storiche USDA NASS SURVEY e superfici raccolte.
- Baseline: trend deterministico/stocastico, medie storiche di contea, modelli lineari.

### Esclusioni Esplicite dal Disegno Iniziale (Future Work)
Per prevenire rischi di *scope creep* e mantenere interpretabile l'attribuzione del segnale predittivo, **non fanno parte del disegno iniziale**:
- Prezzi di mercato, quotazioni futures o variabili macroeconomiche.
- Dati di remote sensing (NDVI, EVI, NIRv, SIF, MODIS, Sentinel).
- Dati fenologici satellitari o approcci ibridi meteo-satellite.
- Modelli di simulazione agronomica biofisica complessa (es. DSSAT, APSIM).
- Dataset proprietari o piattaforme esterne (CropCast, IFAB).
- Previsioni meteorologiche stagionali dinamiche (hindcast GCM/ECMWF SEAS5), che potranno essere considerate solo come estensione successiva.
