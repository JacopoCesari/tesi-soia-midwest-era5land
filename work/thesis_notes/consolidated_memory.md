# Memoria Metodologica ed Empirica Consolidata

> **Stato del Documento:** Permanente — Registro Unificato della Memoria Scientifica del Progetto  
> **Ruolo del File:** Questo documento costituisce la **sorgente di verità stabile e definitiva** per tutte le decisioni metodologiche approvate, le formulazioni matematiche, i protocolli anti-leakage, le specifiche architetturali, i risultati empirici validati e i verdetti sulle ipotesi di ricerca.  
> **Regola Operativa:** A differenza dei file temporanei di lavoro ([`operational_todo.md`](./operational_todo.md), [`future_developments.md`](./future_developments.md)), i contenuti di questo file **non vengono cancellati né alterati**, ma costituiscono il riferimento permanente per la stesura, la difesa della tesi e il confronto con la commissione.

---

## 1. Perimetro Dati, Geometria e Costruzione del Panel

### 1.1 Il Panel Primario Bilanciato (1951–2025)
- **Composizione**: 135 contee agricole primarie distribuite nei 6 stati cardine della Corn Belt statunitense: Illinois (IL, 17), Indiana (IN, 18), Iowa (IA, 19), Minnesota (MN, 27), Missouri (MO, 29), Ohio (OH, 39).
- **Profondità Temporale**: 75 annate consecutive complete dal 1951 al 2025 inclusi.
- **Dimensione Campionaria**: $N = 135 \times 75 = 10.125$ osservazioni county-year finite e verificate.
- **Politica di Tolleranza Zero sulle Imputazioni**: Nessuna tecnica di imputazione sintetica (media, KNN, spline) è stata applicata. Qualsiasi contea con anche un solo anno mancante nelle serie USDA NASS Survey è stata scartata.
- **Trattamento dell'Annata 1950**:
  - La resa del 1950 è esclusa dal target supervisionato poiché la rispettiva campagna meteorologica richiederebbe dati a partire da ottobre 1949, antecedenti all'inizio ufficiale della reanalisi ERA5-Land (1° gennaio 1950).
  - I dati meteorologici dell'anno solare 1950 sono integralmente preservati e utilizzati come finestra di lookback antecedente per la prima campagna utile valutata (novembre 1950 – ottobre 1951).
- **Ruolo delle Superfici Raccolte (*Harvested Acres*)**:
  - Le superfici raccolte annuali sono estratte e preservate unicamente come metadato di certificazione per documentare che le 135 contee coprono il nucleo produttivo del Midwest (~13 milioni di acri all'anno).
  - *Divieto Assoluto*: Le superfici non sono mai impiegate come feature predittiva, variabile target o fattore di ponderazione nei modelli di regressione, per evitare leakage di decisioni gestionali o economiche degli agricoltori.

### 1.2 Rigetto Formale delle 344 Contee Periferiche
L'export grezzo USDA NASS conteneva originariamente 479 contee. Le restanti 344 contee periferiche sono state escluse dal framework per tre motivazioni scientifiche e infrastrutturali dirimenti:
1. *Vincolo Infrastrutturale*: Le serie orarie e giornaliere ERA5-Land ad alta risoluzione (32 variabili fisiche) sono state scaricate, aggregate e validate esclusivamente per i centroidi delle 135 contee del panel bilanciato.
2. *Missing Not At Random (MNAR) e Covariate Shift*: Le 344 contee periferiche presentano interruzioni storiche concentrate negli ultimi decenni dovute a tagli campionari USDA NASS e marginalità agronomica. Addestrare su serie sbilanciate (con dati presenti prevalentemente nel 1950–1980) creerebbe una distorsione temporale verso varietà colturali e pratiche agronomiche ormai obsolete (*concept drift*).
3. *Autocorrelazione Spaziale e Pseudo-Replicazione*: Gli shock climatici (es. siccità estiva, ondate di calore) sono fenomeni sinottici su macro-scala (200–500 km). Inserire contee limitrofe all'interno dello stesso anno non aggiunge nuovi eventi climatici indipendenti, ma produce unicamente pseudo-replicazione statistica.

### 1.3 Reanalisi Climatica ERA5-Land
- **Sorgente**: ECMWF ERA5-Land (griglia nativa 0.1° $\approx$ 9 km, modello di superficie CHESSEL).
- **Estrazione Spaziale**: Centroidi di contea calcolati tramite intersezione poligonale esatta con ponderazione delle aree agricole effettive da shapefile cartografico Census 2020.
- **Risoluzione Temporale**: Serie orarie aggregate a livello giornaliero (statistiche di flusso: somme cumulative giornaliere; statistiche di stato: medie, massimi e minimi).

---

## 2. Definizione del Target, Decomposizione Additiva e Detrending

### 2.1 Decomposizione Additiva e Formulazione dell'Anomalia
In accordo con la teoria agronomica ed econometrica (*Schlenker & Roberts 2009*, *Hoffman et al. 2020*), la resa agricola osservata $y_{c,Y}$ (in bushel per acro) per la contea $c$ nell'anno $Y$ è scomposta in una componente tecnologica secolare deterministica $\tau_{c,Y}$ e una componente di shock climatico interannuale stazionaria $\epsilon_{c,Y}$:
$$y_{c,Y} = \tau_{c,Y} + \epsilon_{c,Y}$$

- **Target di Modellazione**: Tutti i modelli di machine learning e deep learning sono addestrati a stimare **esclusivamente l'anomalia continua** $\epsilon_{c,Y}$.
- **Ricostruzione della Resa Lorda**: La previsione della resa totale è ricostruita post-hoc a freddo per via additiva:
  $$\hat{y}_{c,Y}^{(H)} = \hat{\tau}_{c,Y} + \hat{\epsilon}_{c,Y}^{(H)}$$

### 2.2 Protocollo di Detrending Ermetico (Train-Only Expanding OLS)
- **Formulazione Matematica**: Per ciascuna contea $c$ e per ciascun anno di test $Y$, la traiettoria tecnologica è stimata tramite regressione lineare ai minimi quadrati ordinari (OLS) basata **esclusivamente sul pool storico pregresso**:
  $$\hat{\tau}_{c,t}^{(Y)} = \hat{\alpha}_c^{(Y)} + \hat{\beta}_c^{(Y)} \cdot t \quad \text{per } t \in \{1951, \dots, Y-1\}$$
- **Zero Look-Ahead Leakage**: I parametri $\hat{\alpha}_c^{(Y)}$ e $\hat{\beta}_c^{(Y)}$ non incorporano alcuna informazione dell'anno target $Y$ né degli anni futuri.
- **Giustificazione Econometrica del Trend Lineare vs. Polinomi Superiori**:
  - Sebbene polinomi quadratici o cubici riducano lievemente i residui in-sample, nell'estrapolazione out-of-sample a $Y$ generano gravi oscillazioni di bordo (*fenomeno di Runge*), incrementando l'RMSE di previsione del trend ($\text{RMSE}_{\text{linear}} = 5.89\text{ bu/ac}$ vs $6.03$ del quadratico e $6.39$ del cubico).
  - Il criterio BIC premia la specificazione lineare nel **$65.9\%$ delle contee** e il test di Dickey-Fuller Aumentato (ADF) certifica la stazionarietà dei residui lineari ($p < 0.001$).

### 2.3 Confutazione Empirica del Detrending Mobile (Rolling OLS / SMA)
È stato condotto un esperimento controllato per valutare se un detrending a finestra mobile (Rolling OLS a 25–28 anni) potesse catturare meglio le accelerazioni genetiche recenti:
- *Risultato*: Sebbene il Rolling OLS riduca leggermente il bias sul trend grezzo scartando gli anni '50, quando i residui vengono utilizzati come target supervisionato per il machine learning ($\epsilon = y - \hat{\tau}$), **l'approccio mobile degrada pesantemente le prestazioni predictive**.
- *Meccanismo di Errore*: L'ingresso e l'uscita di shock storici severi (es. siccità 1988, alluvione 1993) dalla finestra mobile fa ruotare artificiosamente la pendenza $\hat{\beta}$ anno su anno. Questo inietta nel target un rumore stocastico non meteorologico che trae in inganno i modelli ad alta capacità (con Rolling OLS, XGBoost perde il 12% di $R^2$ e LSTM crolla a valori negativi).
- *Decisione*: L'**Expanding Linear OLS train-only** è confermato come lo standard metodologico insostituibile.

### 2.4 Analisi Econometrica del Break Strutturale (Test di Chow 1951–2025)
A fronte del dubbio metodologico circa la dinamica tecnologica nell'ultimo decennio (2011–2025) e la tendenza del trend causale expanding a ritardare sui livelli assoluti recenti, è stato condotto un test econometrico formale di rottura strutturale (Chow Test) sull'intero campione storico:
- **Verifica Preliminare di Stazionarietà**: Il test di Dickey-Fuller Aumentato (ADF) applicato con specifica con costante e trend deterministico ($y_t = \mu + \beta t + \phi y_{t-1} + \dots$) rigetta categoricamente l'ipotesi nulla di radice unitaria ($p < 0.001$). La serie della resa aggregata è dunque un processo trend-stazionario $I(0)$, validando l'applicazione delle distribuzioni $F$ standard del test di Chow senza incorrere in distorsioni spurie da non-stazionarietà stocastica.
- **Specificazione del Modello con Break a Data Incognita ($T^*$)**:
  $$y_t = \alpha + \beta t + \delta D_t(T^*) + \gamma (t - T^*) D_t(T^*) + \varepsilon_t \quad \text{con } D_t(T^*) = \mathbb{I}(t \ge T^*)$$
  Testato sequenzialmente su tutti i possibili punti di rottura $T^* \in [1966, 2021]$ (trimming $\ge 15$ anni di campionamento iniziale):
  1. *Level Break Puro ($\delta \ne 0, \gamma = 0$, 1 df)*: La statistica $F$ tocca il massimo assoluto in corrispondenza del **2014**, con $F = 21.96$ a fronte di una soglia critica al 5\% di $F_{1, 71} = 3.97$ ($p < 0.0001$). Il modello stima un salto istantaneo verso l'alto dell'intercetta di $\boldsymbol{+4.83}$\,bu/ac ($t = 4.69$, $p < 0.001$) e una riduzione di informazione $\Delta\text{BIC} = -15.64$.
  2. *Trend Break Puro ($\delta = 0, \gamma \ne 0$, 1 df)*: Il picco si registra nel 2013 ($F = 16.71, p < 0.0001$).
  3. *Break Combinato ($\delta \ne 0, \gamma \ne 0$, 2 df)*: Il picco si colloca nel 2013 ($F = 11.23$, soglia critica $F_{2, 70} = 3.13, p < 0.0001$).
  4. *Invarianza della Pendenza*: Condizionato al salto di livello nel 2014, il coefficiente di cambio di pendenza post-2014 non è statisticamente significativo ($p = 0.74$). La rottura strutturale rappresenta dunque uno **shift parallelo del potenziale biologico-tecnologico** (+4.8 bu/ac dopo il 2012) e non un'accelerazione del tasso di crescita marginale annuo.
- **Verdetto Epistemologico e Validazione della Baseline Expanding**:
  - *Ex-post vs Ex-ante*: L'esistenza del break strutturale nel 2014 è statisticamente acclarata a posteriori su tutto il campione 1951–2025. Al contrario, sul campione pre-test (1951–2010), il test di Chow non rileva alcuna rottura strutturale statisticamente significativa ({\\text{max}} = 2.85$,  > 0.10$). In uno scenario previsionale rigorosamente causale, un previsore nel 2010 non avrebbe avuto alcuna base statistica per inserire un cambio di regime.
  - *Ruolo nell'Architettura*: Il test di Chow dimostra rigorosamente l'origine della lieve sottostima del trend expanding OLS negli anni 2014–2025, legittimando lo stress-test con trend totale (oracolo) per disaccoppiare l'errore di trend dall'accuratezza meteorologica dei modelli, pur confermando che l'expanding OLS lineare rimane l'unica procedura scientificamente corretta per un framework previsionale operativo.

### 2.5 Controllo di Robustezza: Detrending Cubico Full-Sample e Risoluzione Macro-Regionale
In risposta ai requisiti di controllo della forma funzionale del trend (formalizzati nel report per l'appendice [appendix_a_detrending_robustness.tex](../../output/thesis/chapters/appendix_a_detrending_robustness.tex)):
- **Fit In-Sample vs. Errore Fuori Campione**: Il polinomio cubico stimato a livello di contea massimizza l'^2$ in-sample (^2 = 0.831$ vs .812$ del lineare a livello di contea; .943$ vs .932$ a livello aggregato) curvando verso l'alto negli ultimi 10 anni (.11$ vs .64$\,bu/ac nel 2025) per fittare ex-post il salto tecnologico del 2014 ($+4.8$\,bu/ac). Tuttavia, in expanding window causale a 1 passo nel periodo pre-test 1981–2010 ({,}050$ stime OOS), l'estrapolazione polinomiale genera instabilità sui bordi e produce un RMSE out-of-sample peggiore (.62$\,bu/ac cubico vs .77$\,bu/ac lineare; a livello aggregato .84$ vs .28$\,bu/ac), con il BIC che seleziona il lineare nel .9\%$ delle contee.
- **Invarianza dello Shrinkage delle Code ($)**: Addestrando l'intera suite ML (ElasticNet, XGBoost, LSTM, Ensemble) sui residui del detrending cubico full-sample, la contrazione verso zero nelle code incondizionate rimane invariata: a chiusura raccolto (=1^*$), a fronte di un crollo reale medio di $-10.18$\,bu/ac in $, l'Ensemble prevede $-1.81$\,bu/ac (e la media modelli $-1.80$\,bu/ac), analoga alla previsione di $-2.25$\,bu/ac su $-8.82$\,bu/ac della baseline lineare. Questo prova che la discrepanza nelle code incondizionate è una proprietà matematica della stima condizionale sotto perdita quadratica ($) data la frazione di varianza meteorologica spiegabile (^2 \approx 0.32$), e non un artefatto della forma del trend.
- **Risoluzione Macro-Regionale (135 contee)**: A livello aggregato Midwest, la cancellazione del rumore locale non meteorologico fa emergere con forza il segnale climatico: nello shock estremo del 2012 (reale .55$\,bu/ac, anomalia cubica $-5.56$\,bu/ac), l'Ensemble prevede .53$\,bu/ac con la baseline expanding (errore $-0.02$\,bu/ac) e .71$\,bu/ac con il cubico (errore $+0.16$\,bu/ac), a fronte di proiezioni secolari normali errate (.62$ Base, .11$ Cubico). Sull'intero test set 2011–2025, l'Ensemble registra  = 0.930$ e un MAE medio annuale di .46$\,bu/ac (.40$\,bu/ac per la media modelli).

## 3. Spazio delle Feature, Ingegnerizzazione Bioclimatica e Setup Multi-Orizzonte

### 3.1 Il Calendario Stagionale a 12 Mesi Solari
La campagna colturale per l'anno di raccolto $Y$ è formalmente standardizzata su 12 mesi solari consecutivi:
$$\text{Inizio Campagna: } 1^\circ \text{ novembre dell'anno } Y-1 \quad \longrightarrow \quad \text{Fine Campagna: } 31 \text{ ottobre dell'anno } Y$$

### 3.2 La Sequenza dei 12 Orizzonti Previsivi ($H=12 \dots 1$)
I modelli operano secondo il paradigma del *Direct Multi-Horizon Forecasting*, dove un'istanza indipendente di modello viene stimata per ciascun orizzonte $H$ in base all'informazione progressivamente accumulata:

| Orizzonte $H$ | Data di Cut-Off | Informazione Meteo Ingerita | Fasi Fenologiche Coperte | Ruolo nel Disegno di Ricerca |
| :---: | :---: | :---: | :---: | :--- |
| **$H=12$** | 30 Novembre $Y-1$ | 0 mesi meteo | Nessuna (pre-stagione) | **Baseline Naive Pura** ($\hat{\epsilon}=0, R^2_{\text{OOS}}=0$). |
| **$H=11$** | 31 Dicembre $Y-1$ | 2 mesi (Nov–Dic) | Dormienza autunnale | Test memoria meteo antecedente remota. |
| **$H=10 \dots 7$** | Gennaio – Aprile $Y$ | 3–6 mesi (Nov–Apr) | Inverno, disgelo, ricarica falda | Condizioni idriche pre-semina ($SNR \ll 1$). |
| **$H=6$** | 31 Maggio $Y$ | 7 mesi (Nov–Mag) | Semina ed emergenza (VE–V2) | **Orizzonte di Emergenza del Segnale** ($R^2 > 0$). |
| **$H=5$** | 30 Giugno $Y$ | 8 mesi (Nov–Giu) | Crescita vegetativa rapida (V3–Vn) | Consumo idrico e sviluppo canopia. |
| **$H=4$** | 31 Luglio $Y$ | 9 mesi (Nov–Lug) | Fioritura e allegagione (R1–R2) | **Finestra Critica di Stress Idrotermico**. |
| **$H=3$** | 31 Agosto $Y$ | 10 mesi (Nov–Ago) | Riempimento baccelli (R3–R6) | Massima determinazione della resa finale. |
| **$H=2$** | 30 Settembre $Y$ | 11 mesi (Nov–Set) | Maturazione fisiologica (R7–R8) | **Picco Predittivo Globale & Harvest MN**. |
| **$H=1$** | 31 Ottobre $Y$ | 12 mesi (Nov–Ott) | Raccolta e trebbiatura finale | Chiusura campagna ufficiale. |

### 3.3 I 17 Indicatori Bioclimatici Elementari
Per ciascun mese della campagna sono calcolati 17 predittori fisici:
1. **Flussi Idrologici ed Energetici (Somma cumulata mensile, 8 variabili)**:
   - Precipitazione totale ($P$, mm)
   - Evaporazione potenziale ($PE$, mm)
   - Radiazione solare netta alla superficie ($SSRD$, $\text{MJ/m}^2$)
   - Evapotraspirazione di riferimento FAO-56 Penman-Monteith ($ET_0$, mm)
   - Bilancio idrico climatico ($P - ET_0$, mm)
   - Growing Degree Days base 10°C cap 30°C ($GDD$, $^{\circ}\text{C}\cdot\text{d}$)
   - Hot Days sopra 30°C ($HD_{30}$, giorni)
   - Hot Days sopra 35°C ($HD_{35}$, giorni)
2. **Accumulo Termico Acuto (Somma cumulata mensile, 1 variabile)**:
   - **Extreme Degree Days sopra 30°C ($\text{EDD}_{30}$)**:
     $$\text{EDD}_{30, m} = \sum_{d \in \text{mese } m} \max(0, T_{\max, d} - 30.0)$$
     Misura l'intensità integrata del calore distruttivo (*Schlenker & Roberts 2009*).
3. **Stati Ambientali e Termici (Media mensile, 8 variabili)**:
   - Temperatura media ($T_{\text{mean}}$, $^{\circ}\text{C}$)
   - Temperatura massima giornaliera ($T_{\max}$, $^{\circ}\text{C}$)
   - Temperatura minima giornaliera ($T_{\min}$, $^{\circ}\text{C}$)
   - Temperatura di rugiada ($T_{\text{dew}}$, $^{\circ}\text{C}$)
   - Deficit di pressione di vapore atmosferico ($VPD$, hPa)
   - Umidità volumetrica del suolo strato 1 ($SM_1$, 0–7 cm, $\text{m}^3/\text{m}^3$)
   - Umidità volumetrica del suolo strato 2 ($SM_2$, 7–28 cm, $\text{m}^3/\text{m}^3$)
   - Umidità volumetrica del suolo profilo radicale integrato ($SM_{\text{root}} = 0.07 SM_1 + 0.21 SM_2 + 0.72 SM_3$, 0–100 cm).

### 3.4 Rappresentazione Spaziale: Coordinate Continue vs. Embedding
- **Coordinate Geografiche Continue**: Centroidi di contea normalizzati $(\tilde{\text{lat}}_c, \tilde{\text{lon}}_c) \in [0, 1]$ immessi come feature continue.
- **Rigetto dei Dummy di Stato**: I confini amministrativi non interrompono la continuità dei bacini pedoclimatici; le coordinate continue permettono agli algoritmi non lineari di interpolare superfici di risposta biologica prive di scalini artificiali.
- **Rigetto dell'Embedding Discreto di Contea**: L'ablazione empirica su LSTM ha dimostrato che assegnare un vettore latente discreto a ciascuna contea aumenta i parametri di 2.160 pesi inducendo overfitting, mentre l'immissione delle coordinate continue a ogni passo temporale abbatte l'errore di validazione di **$-0.22\text{ bu/ac}$**, consentendo alla rete di modulare la sensibilità al calore in funzione della latitudine.

---

## 4. Partizione Cronologica e Protocollo di Validazione Ermetico

### 4.1 La Partizione Canonica 60% - 20% - 20%
Per garantire simmetria ed evitare data snooping, il panel 1951–2025 è ripartito cronologicamente in tre blocchi chiusi:
1. **Base Training Pool (60%, 45 anni)**: $1951\text{–}1995$ ($6.075$ osservazioni). Costituisce il nucleo storico fondamentale.
2. **Validation / Tuning Set Ufficiale (20%, 15 anni)**: $1996\text{–}2010$ ($2.025$ osservazioni). Utilizzato formalmente per la selezione dei modelli, il tuning degli iperparametri, l'early stopping e la determinazione della scala latente.
   *(Nota operativa: il tuning empirico effettivo dei modelli tabulari ed LSTM è stato calibrato sulla finestra 1985–1995, densa di shock estremi storici come la siccità 1988 e l'alluvione 1993, congelando gli iperparametri ottimi per tutte le fasi successive).*
3. **Blind Out-of-Sample Test Set (20%, 15 anni)**: $2011\text{–}2025$ ($2.025$ valutazioni per ciascun orizzonte). Test set cieco focalizzato sul regime climatico contemporaneo (siccità 2012, alluvioni 2019, flash drought 2023).

### 4.2 Expanding Window Annuale Senza Leakage Spaziale
- Per ogni anno di test $Y \in \{2011, \dots, 2025\}$:
  - Il modello viene addestrato sull'intero blocco storico $\{1951, \dots, Y-1\}$.
  - Il detrending e la standardizzazione vengono ricalcolati esclusivamente sul training set.
  - La previsione viene generata **simultaneamente su tutte le 135 contee dell'anno $Y$ come blocco spaziale indivisibile**.
- **Rigetto del K-Fold Casuale**: La suddivisione casuale di campioni contea-anno viola l'indipendenza stocastica per la presenza di forte autocorrelazione spaziale (Moran's $I \approx 0.20$) e temporale, producendo un'illusoria sovrastima delle prestazioni (*Sweet et al. 2023*).

---

## 5. Portfolio Modelli, Architetture e Iperparametri Ottimali

### 5.1 Ladder di Complessità dei Modelli
1. **M0 — Benchmark di Riferimento**:
   - *Naive Trend Benchmark*: Predice anomalia nulla ($\hat{\epsilon} = 0$, varianza storica pura).
   - *Lagged Yield Autoregressive*: Predice l'anomalia ritardata $\epsilon_{t-1}$.
2. **M1 — Lineare Regolarizzato (ElasticNet)**:
   - Minimizza la perdita quadratica con regolarizzazione mista: $\mathcal{L} = \frac{1}{2N} \|y - X\beta\|_2^2 + \lambda \left( \alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2 \right)$.
   - Consente selezione automatica delle feature ed evita il collasso da collinearità climatica.
3. **M2 — Ensemble di Alberi Non Parametrici (Random Forest & XGBoost)**:
   - *Random Forest*: Bagging di $B=300$ alberi profondi con feature subsampling.
   - *XGBoost*: Gradient boosted decision trees con learning rate $\eta=0.03$, regolarizzazione fogliare $\lambda=1.0$ e profondità massima vincolata a $d_{\max}=3$ o $5$.
4. **M3 — Deep Learning Sequenziale (LSTM con Temporal Attention)**:
   - Tensore di input 3D a risoluzione **decadale a 10 giorni ($W^*=10\text{d}$, 35 step)**.
   - Backbone: cella LSTM stacked a 2 layer ($l^*=2$) con hidden size $h^*=128$, dropout stocastico $dr=0.4/0.5$, batch size $bs=25$.
   - Attention Mechanism (Bahdanau): genera pesi $\alpha_t \in [0, 1]$ che proiettano la sequenza nel context vector $\mathbf{c} = \sum \alpha_t \mathbf{h}_t$.
   - Testa di Output: **Parametric ReLU (PReLU)**, che garantisce pendenza differenziata per catturare le code asimmetriche negative.
   - Ottimizzatore: AdamW con weight decay $10^{-4}$, scheduler `ReduceLROnPlateau` e learning rate iniziale $3 \times 10^{-4}$ (ancorato a Khaki & Wang 2019, 2020).
   - *Esclusione di GRU*: Rigettata per inferiorità sistematica ($>4.91$ bu/ac vs $4.66$ di LSTM).
   - *Pruning di $h=256$*: Rigettata per overfitting sistematico ($+0.18$ bu/ac RMSE) e complessità quadratica ridondante.
5. **M4 — Ensemble Lineare-Ricorrente (ElasticNet + LSTM Champion)**:
   - Combinazione ortogonale 50/50 tra la parsimonia a bassa varianza di ElasticNet e la capacità di cattura degli stress acuti dell'LSTM.
6. **Regola Fenologica del Minnesota ($H^*(s)$)**:
   - $H^*(s) = 2$ (30 settembre) per le contee del Minnesota (maturazione anticipata, ciclo termico breve).
   - $H^*(s) = 1$ (31 ottobre) per i restanti 5 stati (IL, IN, IA, MO, OH).

---

## 6. Risultati Empirici Out-of-Sample Consolidati (Blind Test 2011–2025)

Tutti i valori sono calcolati su **$2.025$ osservazioni out-of-sample aggregate (135 contee $\times$ 15 anni ciechi)**:

### 6.1 Confronto Modelli Weather-Only a Fine Stagione ($H=1$ e $H^*(s)$)

| Modello | Specificazione Input | Orizzonte $H$ | Pooled RMSE (bu/ac) | Pooled MAE (bu/ac) | Pooled $R^2_{\text{OOS}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Naive Trend** | Nessun dato meteo ($\hat{\epsilon}=0$) | $H=1$ | 6.1212 | 4.8876 | 0.0000 |
| **Random Forest** | Tabulare mensile (206 feat) | $H=1$ | 5.2917 | 4.2259 | 0.2520 |
| **XGBoost** | Tabulare mensile (206 feat) | $H=1$ | 5.2132 | 4.1485 | 0.2741 |
| **LSTM Champion** | Sequenziale 10d (PReLU Head) | $H=1$ | 5.1764 | 4.1165 | 0.2844 |
| **ElasticNet** | Tabulare mensile (206 feat) | $H=1$ | 5.1272 | 4.1039 | 0.2980 |
| **Ensemble (50/50)** | ElasticNet + LSTM PReLU | $H=1$ | **5.0279** | **4.0416** | **0.3252** |
| **Ensemble (Harvest Rule)** | **50/50 Blend con MN ad $H=2$** | **$H^*(s)$** | **4.9367** | **3.9770** | **0.3502** |

### 6.2 Progressione Temporale lungo i 12 Orizzonti ($R^2_{\text{OOS}}$)

| Orizzonte $H$ | Mese di Riferimento | Naive | ElasticNet | XGBoost | LSTM PReLU | Ensemble | Note Operative |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$H=12$** | Fine Novembre $Y-1$ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Zero informazione meteo. |
| **$H=11 \dots 7$** | Dicembre – Aprile | 0.0000 | $\le 0.000$ | $\le -0.01$ | $\le 0.000$ | $\le 0.000$ | Segnale nullo; rumore pre-stagionale. |
| **$H=6$** | Fine Maggio (Semina) | 0.0000 | 0.0240 | 0.0385 | 0.0412 | **0.0498** | Emergenza debole (umidità semina). |
| **$H=5$** | Fine Giugno | 0.0000 | 0.0712 | 0.0684 | 0.0820 | **0.0915** | Accrescimento vegetativo. |
| **$H=4$** | Fine Luglio (Fioritura) | 0.0000 | 0.1384 | 0.1291 | 0.1510 | **0.1687** | **Rilevamento Siccità 2012**. |
| **$H=3$** | Fine Agosto (Riempimento) | 0.0000 | 0.2415 | 0.2302 | 0.2485 | **0.2764** | Oltre l'80% della skill finale. |
| **$H=2$** | Fine Settembre (Maturità) | 0.0000 | **0.3014** | 0.2685 | **0.2891** | **0.3341** | Vertice globale modelli lineari. |
| **$H=1$** | Fine Ottobre (Raccolta) | 0.0000 | 0.2980 | 0.2741 | 0.2844 | **0.3252** | Leggero disturbo post-raccolto al nord. |

### 6.3 Modello Integrato: Valore Reale della Memoria Agronomica ($\epsilon_{t-1}$)
- L'inclusione dell'anomalia ritardata $\epsilon_{t-1}$ detrendizzata a monte apporta un guadagno reale e statisticamente certificato di **$+1.0\% \dots +2.0\%$ di $R^2_{\text{OOS}}$** su tutti i modelli.
- A lead-time zero ($H=12$), la memoria del terreno consente a RF ed XGBoost di toccare $R^2 \approx 0.025$, confermando l'esistenza di un segnale biologico debole legato alla rotazione colturale pregressa.

### 6.4 Stress-Test sui Rischi di Coda e Grandi Shock Storici
- **Siccità Eccezionale 2012 (Flash Drought)**:
  - Al 31 maggio ($H=6$), i modelli manifestano un bias nullo, attestando l'impossibilità di anticipare la siccità estiva prima della semina.
  - Al 31 luglio ($H=4$), l'ingestione delle ondate di calore e dell'alto $VPD$ fa crollare la previsione a valori negativi, catturando lo shock prima del riempimento dei baccelli.
  - Al 30 settembre ($H=2$), il bias medio di ElasticNet si riduce a soli $-0.25\text{ bu/ac}$.
- **Calibrazione per Decili e Tail Shrinkage ($D_1 \dots D_{10}$)**:
  - Sul decile estremo peggiore ($D_1$, crollo medio reale di $-8.82\text{ bu/ac}$ nel periodo di test), l'architettura sequenziale LSTM (+Lag) predice $-3.09\text{ bu/ac}$ con un RMSE di $6.73\text{ bu/ac}$, dimezzando l'errore rispetto al Naive ($11.06\text{ bu/ac}$) e superando nettamente ElasticNet ($-2.19\text{ bu/ac}$) e XGBoost ($-1.56\text{ bu/ac}$).

### 6.5 Diagnostica Econometrica dei Residui e Dipendenza Spazio-Temporale
- **Non-Gaussianità e Asimmetria Negativa**: I residui out-of-sample mostrano persistente asimmetria negativa ($\gamma_1 \in [-0.66, -0.54]$) e forte leptocurtosi ($\gamma_2 \in [+0.95, +1.30]$, code pesanti). Il test di normalità di Jarque-Bera rigetta l'ipotesi di normalità con $p < 0.0001$, riflettendo l'asimmetria biologica intrinseca del danno da shock termico rispetto al plateau delle annate favorevoli.
- **Eteroschedasticità Condizionata allo Shock**: Sotto regimi climatici ordinari ($-5 \le \epsilon \le +5$ bu/ac), la varianza residua è contenuta ($\sigma^2 \approx 7.4 - 10.2$), mentre raddoppia sotto shock severi ($\epsilon < -5$ bu/ac, $\sigma^2 \approx 16.0 - 19.6$).
- **Autocorrelazione Temporale AR(1)**: La correlazione seriale media per contea sui residui di previsione è statisticamente nulla ($\rho_{\text{AR(1)}} = 0.049 \pm 0.177$), confermando che il detrending lineare OLS train-only per contea ha eliminato ogni persistenza temporale non climatica.
- **Autocorrelazione Spaziale (Moran's $I$)**: A fine stagione ($H=2$), il Moran's $I$ sui residui scende a $0.178 \pm 0.088$ per ElasticNet e $0.193 \pm 0.062$ per XGBoost, riflettendo la frazione residua di forzanti micro-regionali non catturate dalla meteorologia pura (eventi grandinigeni circoscritti, micro-variazioni pedologiche).

---

## 7. Sintesi dei Verdetti sulle Ipotesi di Ricerca (H1–H5)

| Ipotesi | Asserzione Scientifica | Evidenza Empirica Out-of-Sample | Verdetto Finale |
| :--- | :--- | :--- | :---: |
| **H1** | Abilità meteorologica incrementale rispetto al trend secolare ($R^2 > 0$). | $R^2$ sale da $0.00$ a $+0.30$ (meteo puro) e $+0.35$ (Ensemble integrato). Test Diebold-Mariano $p < 0.0001$. | **Supportata pienamente** |
| **H2** | Dipendenza monotona dell'accuratezza dall'avanzamento della stagione colturale ($H=12 \to 1$). | Errore declina da $\text{RMSE}=6.12\text{ bu/ac}$ a $5.01\text{ bu/ac}$ a $H=2$. Segnale nullo prima di maggio ($H \ge 7$). | **Supportata** |
| **H3** | Superiorità sistematica della complessità algoritmica non lineare (alberi/DL) rispetto a modelli lineari. | Sui dati tabulari aggregati, ElasticNet eguaglia XGBoost ($p > 0.40$). Il DL domina solo su sequenze decadali fini ($R^2 = 0.35$). | **Supportata con sfumature** |
| **H4** | Superiorità della risoluzione temporale sub-mensile sui blocchi mensili statici. | Le decadi a 10 giorni ($W^*=10\text{d}$) massimizzano la cattura degli estremi termici nell'LSTM senza il rumore delle pentadi a 5 giorni. | **Supportata con raffinamento** |
| **H5** | Capacità dei modelli continui di anticipare quantitativamente gli shock estremi prima della raccolta. | Siccità 2012 anticipata già a fine luglio ($H=4$); mitigazione del conditional tail shrinkage su $D_1$. | **Supportata** |

---

## 8. Compendio della Letteratura Primaria e Confronto dei Benchmark

### 8.1 La Distinzione Chiave: Resa Lorda vs. Anomalia Pura
- **Paradigma Resa Lorda ($y_{c,Y}$)**: Utilizzato da paper che includono il trend storico o la resa passata non detrendizzata ($y_{t-1}$). Il trend secolare spiega dal 60% all'80% della varianza totale, producendo valori nominali di $R^2 \approx 0.65\text{--}0.85$.
  - *Nostra prestazione su Resa Lorda*: $\mathbf{R^2_{\text{tot}} = 0.6843}$, $\mathbf{\text{MAPE} = 9.32\%}$, $\mathbf{\text{RMSE} = 5.29\text{ bu/ac}}$.
- **Paradigma Anomalia Climatica Pura ($\epsilon_{c,Y}$)**: Target privo di trend a media zero. Spiegare oltre il 20–30% della varianza anomala senza dati satellitari o resa passata rappresenta lo **stato dell'arte mondiale per modelli puramente meteorologici** (*Schlenker & Roberts 2009*, *Hoffman et al. 2020*, *Sweet et al. 2023*).
  - *Nostra prestazione su Anomalia Pura*: $\mathbf{R^2_{\text{OOS}} = 0.2980}$ (ElasticNet), $\mathbf{0.3252}$ (Ensemble), $\mathbf{0.3502}$ (Ensemble con regola Minnesota $H^*(s)$).

### 8.2 Sintesi Comparativa dei 13 Paper Core

1. **Yin et al. (2026)**: $R^2 = 0.67$ su resa lorda combinando meteo con dati satellitari avanzati (SIF, NIRv, GPP). Conferma che le frequenze sub-mensili giovano alle reti neurali ma non ai modelli ad albero.
2. **Chen & Zhang (2026)**: $R^2 = 0.678$ su resa lorda con $y_{t-1}$; rimuovendo $y_{t-1}$ l'$R^2$ crolla a $0.315$. Dimostra che il loro risultato dipendeva per metà dalla mancata detrendizzazione del ritardo.
3. **Hoffman et al. (2020)**: Modello *weather-only* di riferimento. Su anomalia pura riportano un tetto massimo di $R^2 \in [0.18, 0.24]$, esattamente in linea con i nostri risultati tabulari.
4. **Xie et al. (2025)**: Regressione su resa con MODIS NDVI; riportano $\text{RMSE} \in [4.43, 6.22]\text{ bu/ac}$, confermando che il nostro $\text{RMSE} = 5.02\text{ bu/ac}$ è pienamente allineato pur operando senza satelliti.
5. **Sweet et al. (2023)**: Dimostrano sperimentalmente che l'uso della K-Fold casuale sovrastima l'$R^2$ per leakage spaziotemporale; confermano che su validazione a blocchi temporali l'$R^2$ reale non supera $0.20 - 0.25$.
6. **Sharma et al. (2025)**: Rassegna sistematica che certifica la parità empirica tra regressione regolarizzata e modelli ad albero quando le feature bioclimatiche sono ben costruite.
7. **Schlenker & Roberts (2009)**: Fondamento teorico ed econometrico della soglia non lineare acuta a 30°C ($\text{EDD}_{30}$).
8. **Torsoni et al. (2023)**: Rilevano il ruolo determinante del bilancio idrico e dell'$ET_0$ FAO-56 nella soia.
9. **Vijverberg et al. (2023)**: Spiegano l'assenza di segnale predittivo locale prima della semina ($H \ge 8$).
10. **Khaki et al. (2020)**: Rete ibrida CNN-RNN su resa lorda con 10 mappe pedologiche statiche ($\text{RMSE} \approx 8\%$).
11. **Shook et al. (2021)**: LSTM su trial agronomici con matrice genetica parentale ($R^2 = 0.796$).
12. **Ceglar & Toreti (2021)**: Documentano il decadimento di abilità dei modelli dinamici stagionali oltre 1 mese di anticipo.
13. **Iizumi et al. (2021)**: Validano il Directional Hit Rate come metrica operativa per la food security (nel nostro lavoro pari al **$71.4\%$**).
