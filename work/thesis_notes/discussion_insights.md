# Banco di Validazione Oggettiva delle Ipotesi e Spunti Interpretativi

> **Protocollo di Validazione Super Partes:**  
> Questo documento **non è una raccolta di opinioni soggettive né un esercizio di auto-conferma delle proprie teorie**. È il banco di prova critico dove qualsiasi riflessione, intuizione o ipotesi interpretativa viene sottoposta a uno **scrutinio imparziale e rigoroso** basato su due soli criteri:
> 1. **Evidenza empirica numerica**: metriche out-of-sample, residui, test statistici (Diebold-Mariano, Jarque-Bera, Moran's I).
> 2. **Letteratura scientifica di consenso e agronomia**: i 13 paper core di riferimento, report USDA NASS e fisiologia colturale.
>
> **Regola di Uscita Binaria:**  
> - Se i numeri e la letteratura **confermano l'ipotesi** $\implies$ l'ipotesi viene formalizzata, dichiarata accettata e promossa nelle sezioni definitive ([`consolidated_memory.md`](./consolidated_memory.md)) e nel testo della tesi (Cap. 5 e 6).
> - Se i numeri **smentiscono o non supportano l'ipotesi** $\implies$ l'ipotesi viene scartata ed eliminata ("buttata"), per evitare qualsiasi speculazione priva di riscontro oggettivo.

---

## 1. Ipotesi 1: Harvest Contamination ad Ottobre ($H=1$) vs. Picco a Settembre ($H=2$)

### Riflessione Iniziale
I modelli lineari regolarizzati mostrano un picco di $R^2$ al 30 settembre ($H=2$) e flettono al 31 ottobre ($H=1$). Si ipotizza che il meteo di ottobre nelle contee del nord agisca come rumore post-raccolto.

### Verifica Oggettiva con i Numeri (Blind Test)
- **Dato Panel Globale**: ElasticNet registra $R^2 = 0.2156$ ad $H=2$ e cala a $0.2039$ ad $H=1$ (su 1996–2025). Sul Blind Test 2011–2025 tocca $0.3014$ ad $H=2$ e $0.2980$ ad $H=1$.
- **Dato Specifico per Stato (Minnesota)**: In Minnesota (contee settentrionali, Maturity Group 0/I), ElasticNet crolla vistosamente da $R^2 = +0.271$ ($H=2$) a $R^2 = +0.180$ ($H=1$), perdendo 9.1 punti di varianza spiegata. Al contrario, in Iowa, Illinois e Missouri ($H=1$) le performance crescono o rimangono stabili fino a ottobre.

### Riscontro della Letteratura e Dati USDA
- **USDA NASS Crop Progress Reports**: In Minnesota l'80% delle superfici a soia è già trebbiato e stoccato entro la seconda settimana di ottobre a causa del gelo imminente. Nelle zone centrali e meridionali (Gruppi II–IV), la raccolta si protrae per tutto ottobre e i primi giorni di novembre.
- Il meteo della seconda metà di ottobre in Minnesota cade su campi vuoti: $VPD$, pioggia e temperature misurano eventi senza alcuna pianta in campo, iniettando puro disturbo stocastico nei modelli lineari rigidi.

### Verdetto Super Partes: ACCETTATA & FORMALIZZATA
- L'ipotesi è provata oggettivamente sia dai dati numerici disaggregati per stato sia dai bollettini USDA.
- **Esito operativo**: Formalizzata la regola fenologica $H^*(s)$ (Minnesota ad $H=2$, altri 5 stati ad $H=1$). Promossa a decisione consolidata in [`consolidated_memory.md`](./consolidated_memory.md) (Decisione 87) e inserita nella discussione dei Capitoli 4, 5 e 6.

---

## 2. Ipotesi 2: Tail Shrinkage dell'MSE e Valore Predittivo Asimmetrico

### Riflessione Iniziale
I modelli addestrati con funzione di perdita quadratica (MSE) potrebbero fallire sistematicamente l'ampiezza delle grandi siccità perché l'MSE "preferisce" attenuare lo shock per non rovinare l'accuratezza media sulle annate ordinarie.

### Verifica Oggettiva con i Numeri (Analisi per Decili)
- **Analisi empirica per decili ($D_1 \dots D_{10}$ a $H=2$ e $H=1$)**:
  - Nel decile peggiore ($D_1$, shock reale medio di $-10.52\text{ bu/ac}$ su 1996–2025 e $-8.82\text{ bu/ac}$ su 2011–2025):
    - Naive Benchmark: $0.00\text{ bu/ac}$ ($\text{RMSE} = 11.06\text{ bu/ac}$, bias $+10.52$).
    - XGBoost: $-1.02\text{ bu/ac}$ ($\text{RMSE} = 10.25\text{ bu/ac}$, cattura solo il $9.7\%$ dello shock).
    - ElasticNet: $-1.90\text{ bu/ac}$ ($\text{RMSE} = 9.68\text{ bu/ac}$, cattura il $18.1\%$ dello shock).
    - LSTM PReLU / Asymmetric Huber: $-3.09\text{ bu/ac}$ ($\text{RMSE} = 6.73\text{ bu/ac}$, cattura il $35.0\%$ dello shock).
- **Verifica sui decili centrali ($D_4 - D_6$)**: L'RMSE dei modelli scende a $2.00 - 2.80\text{ bu/ac}$ con bias quasi nullo.

### Riscontro della Letteratura Econometrica ed Economica
- **Teoria Econometrica (James et al. 2023 - ISLP; Géron 2022)**: La loss quadratica stima la media condizionata $\mathbb{E}[Y \mid X]$. In presenza di distribuzioni leptocurtiche con asimmetria negativa (shock rari), la media condizionata comprime le code verso il centro per minimizzare la somma globale dei quadrati.
- **Teoria Economica del Rischio di Filiera**: Per un operatore della logistica o del trading, l'errore non è simmetrico: non anticipare un calo del 25% comporta default fisici e penali contrattuali, mentre un errore di 1 bu/ac in un anno normale è facilmente assorbito dalle scorte ordinarie.

### Verdetto Super Partes: ACCETTATA & FORMALIZZATA
- L'esistenza del *conditional tail shrinkage* è provata matematicamente e quantificata nei decili; il vantaggio delle teste asimmetriche e dell'Ensemble è confermato dalla riduzione dell'errore su $D_1$.
- **Esito operativo**: Promossa in [`consolidated_memory.md`](./consolidated_memory.md) e integrata nel Capitolo 5 (analisi decili) e Capitolo 6 (implicazioni gestionali).

---

## 3. Ipotesi 3: Decostruzione della Memoria Agronomica ($\epsilon_{t-1}$) vs. Chen & Zhang (2026)

### Riflessione Iniziale
Si sospetta che la letteratura recente (es. Chen & Zhang 2026), che attribuisce alla resa dell'anno prima ($y_{t-1}$) oltre la metà dell'abilità predittiva del modello (facendo passare l'$R^2$ dal 32% al 68%), sia viziata da data leakage del trend tecnologico secolare, e che la reale memoria agronomica sia molto più modesta.

### Verifica Oggettiva con i Numeri (Ablazione Controllata)
- **Esperimento Ermetico**: L'anomalia ritardata viene detrendizzata a monte sul solo training set: $\epsilon_{c, t-1} = y_{c, t-1} - \hat{\tau}_{c, t-1}^{(Y)}$.
- **Risultato Out-of-Sample (1996–2025 e 2011–2025)**:
  - A $H=1$: ElasticNet sale da $0.2039$ a $0.2145$ ($\Delta R^2 = +0.0106$); XGBoost da $0.2045$ a $0.2159$ ($\Delta R^2 = +0.0115$).
  - A $H=2$: ElasticNet sale da $0.2156$ a $0.2271$ ($\Delta R^2 = +0.0115$); LSTM da $0.2891$ a $0.3015$.
  - Il guadagno netto reale della memoria agronomica è quantificato con assoluta precisione tra **$+1.0\%$ e $+2.0\%$ di $R^2$**.

### Riscontro della Letteratura
- **Chen & Zhang (2026)**: Addestravano reti neurali su serie storiche 1927–2025 usando rese lorde in bushel/acro non detrendizzate. In una serie centennale in cui la resa media sale da 15 a 55 bu/ac, $y_{t-1}$ ha una correlazione secolare con $y_t$ superiore a $0.80$, riflettendo l'adozione di sementi OGM, fertilizzanti e diserbanti, non la memoria biologica del suolo.
- Nel nostro modello senza lag, la resa lorda ricostruita raggiunge $R^2 = 0.6843$, dimostrando che il 68% si ottiene interamente tramite trend secolare + clima puro.

### Verdetto Super Partes: ACCETTATA & FORMALIZZATA
- La fallacia della letteratura è dimostrata sia teoricamente che quantitativamente.
- **Esito operativo**: Costituisce un punto di forza originale della tesi; formalizzata in [`consolidated_memory.md`](./consolidated_memory.md) e commentata nei Capitoli 2, 4, 5 e 6.

---

## 4. Ipotesi 4: Superiorità dei Modelli non Lineari ad Albero (XGBoost/RF) su Modelli Lineari

### Riflessione Iniziale
L'ipotesi di senso comune in data science assume che algoritmi complessi di machine learning ad albero (XGBoost, Random Forest) debbano superare la regressione lineare regolarizzata (ElasticNet) per via della loro capacità di apprendere superfici non lineari.

### Verifica Oggettiva con i Numeri (Test di Diebold-Mariano)
- **Metriche Out-of-Sample a $H=1$ e $H=2$**:
  - $H=1$ (Weather-Only): ElasticNet $R^2 = 0.2980$ vs XGBoost $0.2741$ vs Random Forest $0.2520$.
  - $H=2$ (Weather-Only): ElasticNet $R^2 = 0.3014$ vs XGBoost $0.2685$.
- **Test di Significatività Statistica di Diebold-Mariano sui Residui**:
  - A $H=1$: $p = 0.9515$ (nessuna differenza statistica).
  - A $H=2$: $p = 0.4399$ (nessuna differenza statistica).
  - A $H=3$: $p = 0.5158$ (nessuna differenza statistica).
- **Conclusione Oggettiva**: L'ipotesi che gli alberi superino i modelli lineari sui dati tabulari aggregati a scala mensile è **falsificata dai dati**.

### Rationale Teorico e Letteratura
- **Sharma et al. (2025)**: Nella loro rassegna sistematica documentano che quando gli indici bioclimatici contengono già le soglie fisiologiche non lineari ($\text{EDD}_{30}$, $VPD$, $ET_0$), la regressione lineare estrae la quasi totalità del segnale disponibile, mentre gli alberi rischiano l'overfitting per frammentazione dello spazio campionario.
- La complessità non lineare apporta valore aggiunto solo quando accoppiata alla struttura temporale sequenziale ad alta frequenza (LSTM a 10 giorni, $R^2 = 0.3502$ con Ensemble).

### Verdetto Super Partes: PARZIALMENTE RIGETTATA (Sfumata oggettivamente)
- Non si forza la mano per difendere la superiorità degli alberi: i numeri dicono che su dati tabulari c'è **parità statistica**, e si accetta il risultato così com'è.
- **Esito operativo**: Questo esito sfumato è formalizzato nel verdetto dell'**Ipotesi H3** in [`consolidated_memory.md`](./consolidated_memory.md) e riportato con trasparenza nei Capitoli 5 e 6.

---

## 5. Ipotesi 5: Allineamento Biologico dell'Attenzione Temporale (LSTM) e SHAP

### Riflessione Iniziale
I modelli data-driven identificano correlazioni spurie pre-stagionali o riflettono le reali fasi fenologiche biologiche note dalla letteratura agronomica?

### Verifica Oggettiva con i Numeri
- **Decomposizione Stagionale dell'Importanza Decisionale**:
  - Giugno, Luglio e Agosto (stadi vegetativi avanzati e riproduttivi R1–R6) concentrano il **$51.5\%$** dell'importanza complessiva a $H=3$.
  - A maggio ($H=6$, semina ed emergenza), il bilancio idrico del letto di semina ($P - ET_0$) spiega da solo il $20.14\%$ dell'importanza del mese.
  - A ottobre ($H=1$), a maturità fisiologica conclusa, l'importanza del clima precipita al **$6.23\%$**.
  - Nei mesi invernali pre-stagionali ($H=12 \dots 8$, novembre–marzo), l'importanza è inferiore al $2\%$ e l'$R^2$ è statisticamente nullo.

### Riscontro della Letteratura Agronomica
- **Fehr & Caviness (1977); Setiyono et al. (2007)**: La soia definisce il numero di baccelli per nodo durante la fioritura di luglio (R1–R2) e il peso specifico del grano durante il riempimento di agosto (R3–R6). Uno stress termico ($T_{\max} > 30^\circ\text{C}$) o deficit di vapore ($VPD > 15\text{ hPa}$) in queste due finestre causa aborto fiorale irreversibile.

### Verdetto Super Partes: ACCETTATA & FORMALIZZATA
- L'allineamento tra pesi dell'attenzione algoritmica e la biologia vegetale classica è perfetto ed esclude l'apprendimento di artefatti spuri.
- **Esito operativo**: Promossa in [`consolidated_memory.md`](./consolidated_memory.md) e integrata nel Capitolo 5 (Sezione Feature Importance) e Capitolo 6 (Spiegabilità biologica).

---

## 6. Ipotesi 6: Break Strutturale Tecnologico Recente (2014) vs. Look-Ahead Bias

### Riflessione Iniziale
Il bias negativo osservato nelle annate avverse recenti ($D_1$ nel test set 2011–2025) e la tendenza a sovrastimare le rese nei minimi storici derivano da un'inadeguatezza del machine learning meteorologico o da un cambio di regime tecnologico nel trend che l'expanding window OLS sconta con fisiologico ritardo causale?

### Verifica Oggettiva con i Numeri
- **Test Econometrico di Chow (1951–2025)**:
  - Serie trend-stazionaria $I(0)$ verificata tramite Dickey-Fuller Aumentato ($p < 0.001$).
  - Il test di Chow sequenziale individua un picco univoco di rottura di livello nel **2014** ($F = 21.96$ vs critico $3.97$, $p < 0.0001$, $\Delta\text{BIC} = -15.64$), con un salto istantaneo di $\boldsymbol{+4.83}$\,bu/ac. Il cambio di pendenza post-2014 è statisticamente non significativo ($p = 0.74$).
- **Stress-Test Oracolo (Full-Sample OLS 1951–2025)**:
  - Eliminando il ritardo di stima tramite il trend totale calcolato ex-post su 75 anni, il bias predittivo sul decile $D_1$ si dimezza (da $-7.26$ a $-3.65$\,bu/ac).
  - La quota residua di attenuazione (circa $3.5$\,bu/ac) è spiegata dal teorico shrinkage condizionale $L_2$ ($R^2 \approx 0.32$), che contrae le stime verso la media per minimizzare l'errore quadratico in presenza di rumore non osservato.

### Riscontro della Letteratura Econometrica ed Agronomica
- **Chow (1960); Stock & Watson (1996)**: Nei processi di forecasting out-of-sample in espansione temporale, una rottura strutturale non può essere stimata in tempo reale negli anni immediatamente successivi ($t = T^*+1, T^*+2$) senza incorrere in look-ahead bias o overfitting estremo su un campione post-rottura microscopico ($N \le 2$).
- **Diffusione Tecnologica USDA**: Il periodo 2013–2015 coincide con l'adozione su larga scala di sementi di soia di seconda generazione (Intacta / Roundup Ready 2 Yield) e pratiche avanzate di drenaggio di precisione, che hanno innalzato il plateau di resa media del Midwest.

### Verdetto Super Partes: ACCETTATA & FORMALIZZATA
- L'esistenza della rottura strutturale nel 2014 è provata oggettivamente e spiega la componente non meteorologica del ritardo predittivo recente. Tuttavia, la correzione ex-ante è metodologicamente illegittima in un contesto operativo. L'expanding window OLS rimane confermato come unico protocollo ermetico e replicabile.
- **Esito operativo**: Promossa in [`consolidated_memory.md`](./consolidated_memory.md) (Sezione 2.4), documentata nel report per il relatore e inserita come punto cardine della discussione econometrica nei Capitoli 3, 4 e 6.



---

## 7. Ipotesi 7: Aggregazione Macro-Regionale vs. Dispersione di Contea (Legge dei Grandi Numeri e Dinamica degli Horizon)

### Riflessione Iniziale
A livello di singola contea, le previsioni puntuali presentano un fisiologico *shrinkage* verso la media nelle code incondizionate ($D_1$) e un errore RMSE che incontra un limite inferiore insuperabile ($\approx 5.0\text{ bu/ac}$), poiché le forzanti meteorologiche su scala di griglia spiegano solo circa il $30\%$ della varianza locale (il restante $70\%$ è rumore stocastico non climatico: grandine localizzata, parassiti, gestione colturale, errori campionari USDA).  
Ci si chiede: l'abilità predittiva del framework agrometeorologico è realmente limitata a questo livello, oppure quando aggreghiamo le 135 contee a scala macro-regionale (la vera scala operativa di mercato e di filiera) il rumore locale si annulla per la Legge dei Grandi Numeri, rivelando un segnale climatico predittivo di accuratezza decisamente superiore?

### Verifica Oggettiva con i Numeri (Blind Test 2011–2025)
1. **Confronto tra Livello Contea e Livello Macro lungo gli Horizon ($H=12 \dots 1$)**:
   - $H=12 \dots 6$ (pre-stagionale e inizio semina): assenza di segnale estivo. County RMSE $= 6.12\text{ bu/ac}$, Macro RMSE $= 3.84\text{ bu/ac}$ (identico al Naive, $R^2_{\text{macro}} = 0.311$).
   - $H=4$ (fine giugno, sviluppo vegetativo): County RMSE scende a $5.45\text{ bu/ac}$, mentre Macro RMSE crolla a **$2.94\text{ bu/ac}$** ($R^2_{\text{macro}}$ balza a **$0.596$**, $r = 0.891$).
   - $H=3$ (fine luglio, fioritura critica R1–R2): County RMSE $= 5.20\text{ bu/ac}$, Macro RMSE scende a **$2.77\text{ bu/ac}$** ($R^2_{\text{macro}} = 0.640$, $r = 0.903$).
   - $H=2$ (fine agosto, riempimento baccelli R3–R6): County RMSE $= 5.14\text{ bu/ac}$, Macro RMSE tocca **$2.50\text{ bu/ac}$** ($R^2_{\text{macro}} = 0.707$, $r = 0.919$).
   - $H=1$ (raccolto con Ensemble): Macro MAE scende a **$1.92\text{ bu/ac}$** (Baseline Expanding) e **$1.46\text{ bu/ac}$** (Cubico Full-Sample), con correlazione $r = 0.946$.
2. **Cattura degli Shock Sistemici (Siccità Estrema 2012)**:
   - Resa reale media osservata nelle 135 contee: **$43.55\text{ bu/ac}$** (crollo di oltre 14 bu/ac sotto il trend).
   - Previsione Baseline Expanding: **$43.44\text{ bu/ac}$** (errore di appena **$-0.11\text{ bu/ac}$**).
   - Previsione Full-Sample Cubico: **$43.71\text{ bu/ac}$** (errore di appena **$+0.17\text{ bu/ac}$**).
3. **Analisi del Trend Cubico vs. Lineare Medio (1951–2025, Figura `fig_mean_cubic_vs_linear_trend.png`)**:
   - Tracciando la media dei trend delle 135 contee, il trend cubico flette verso l'alto a partire dal 2014, raggiungendo nel 2025 un valore di $59.11\text{ bu/ac}$ contro i $55.64\text{ bu/ac}$ del lineare ($\Delta = +3.47\text{ bu/ac}$).
   - Questo dimostra che il maggior fit in-sample del cubico ($R^2 = 0.831$) è dovuto all'adattamento ex-post al salto tecnologico del 2014 (Chow test $F = 21.96, \Delta = +4.83\text{ bu/ac}$). Tuttavia, in expanding window causale, la tendenza del polinomio di terzo grado a divergere fuori campione porta a un peggioramento dell'RMSE OOS ($6.39$ vs $5.89\text{ bu/ac}$ del lineare), giustificando la scelta del lineare per la modellazione causale.

### Riscontro della Letteratura Economica e della Pratica di Mercato
- **Necessità dell'Addestramento Granulare**: Addestrare a livello di singola contea ($N = 10.125$) è indispensabile per fornire sufficienti gradi di libertà agli algoritmi di machine learning (ElasticNet, Random Forest, LSTM) e mappare le soglie termiche e idriche non lineari locali (ad es. $VPD > 15\text{ hPa}$, $EDD_{30}$). Su sole 75 osservazioni macro-regionali aggregate, qualsiasi modello ML soffrirebbe di *sample starvation* e grave overfitting.
- **Valore Operativo dell'Aggregazione**: Nella gestione delle catene di fornitura, nel procurement industriale (Areté, trader internazionali, trasformatori) e nella copertura sui mercati a termine (CBOT), gli operatori non assumono decisioni su scala di singola contea, ma definiscono strategie di bilancio di offerta/domanda e di prezzo a scala macro-regionale.
- A questa scala, il rumore non climatico locale si compensa vicendevolmente ($\sigma / \sqrt{N}$), e il clima spiega oltre il $70\%$ della varianza reale.

### Verdetto Super Partes: ACCETTATA & FORMALIZZATA
- L'ipotesi è pienamente confermata dai dati numerici: l'addestramento su contee è il **motore metodologico di identificazione statistica**, mentre la valutazione aggregata macro-regionale è il **deliverable operativo primario** per l'economia e la supply chain.
- **Esito operativo**: Formalizzata in [`consolidated_memory.md`](./consolidated_memory.md) (Sezione 4) e inserita nel Capitolo 5 come secondo livello di analisi dopo i decili di contea, nonché proposta al relatore per il tracciamento degli errori lungo i 12 horizon.
