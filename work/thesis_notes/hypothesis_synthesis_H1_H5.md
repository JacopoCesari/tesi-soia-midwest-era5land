# Hypothesis Synthesis (H1–H5): Empirical Verdicts & Notes

Questo documento raccoglie la sintesi empirica formale delle cinque ipotesi di ricerca (H1–H5 formalizzate nel Capitolo 2) valutate sui risultati del test cieco 2011–2025. Questa sintesi funge da materiale di lavoro per la redazione dei Capitoli 6 e 7 (Discussion, Operational Implications & Conclusions).

---

## 1. Valutazione Dettagliata delle Ipotesi

### H1 (Incremental weather skill over secular trend)
* **Asserzione formale:** L'ingestione di forzanti agrometeorologiche ad alta risoluzione (ERA5-Land) genera un potere predittivo incrementale statisticamente significativo rispetto all'estrapolazione del trend deterministico secolare ($R^2_{\text{OOS}} > 0.00$).
* **Verdetto:** **Supportata pienamente** (Nulla di abilità nulla rigettata con $p < 0.0001$).
* **Evidenza empirica:**
  - L'orizzonte di emergenza del segnale meteorologico è identificato a $H^* = 6$ (31 maggio, epoca di semina/emergenza), dove i modelli superano il trend nullo ($R^2_{\text{OOS}} \approx +0.05$).
  - La capacità esplicativa cresce progressivamente durante il ciclo vegetativo fino a raggiungere $R^2_{\text{OOS}} = 0.30$ sotto forzante meteorologica pura ($H = 2$, 30 settembre) e $R^2_{\text{OOS}} = 0.35$ ($\text{Skill}_{\text{RMSE}} = +19.3\%$) quando combinata con la memoria agronomica di resa ritardata ($\epsilon_{t-1}$).
  - I test di Diebold-Mariano certificano la superiorità predittiva rispetto alla baseline ingenua al livello di significatività dello 0.1% ($p < 0.0001$).

### H2 (Monotone lead-time dependence)
* **Asserzione formale:** L'accuratezza predittiva cresce monotonicamente man mano che la stagione colturale avanza e si riduce il lead time rispetto al raccolto ($H = 12 \to H = 1$).
* **Verdetto:** **Supportata**.
* **Evidenza empirica:**
  - L'errore fuori campione declina sistematicamente da $\text{RMSE} = 6.12\text{ bu/ac}$ a $H = 12$ fino a $5.01\text{ bu/ac}$ a $H = 2$ ($4.94\text{ bu/ac}$ con specifica integrata).
  - Gli orizzonti invernali e primaverili precoci ($H \ge 7$, novembre–aprile) contengono segnale azionabile nullo ($R^2_{\text{OOS}} \le 0$).
  - La predicibilità si concentra quasi integralmente nella finestra riproduttiva critica di luglio–agosto (stadi R1–R6: fioritura e riempimento baccelli), dopodiché le metriche si stabilizzano alla maturità fisiologica (ottobre).

### H3 (Model complexity value: Non-linear ML/DL vs Parsimonious Linear)
* **Asserzione formale:** La complessità algoritmica non lineare (alberi decisionali, deep learning) fornisce vantaggi predittivi sistematici rispetto a modelli lineari regolarizzati parsimoniosi (ElasticNet).
* **Verdetto:** **Supportata con sfumature (Nuanced / Partially Supported)**.
* **Evidenza empirica:**
  - Su dati tabulari aggregati mensilmente, gli ensemble di alberi (Random Forest, XGBoost) non superano la regressione lineare regolarizzata: ElasticNet raggiunge $R^2 = 0.30$ contro $0.27$ di XGBoost a $H = 2$ in modalità weather-only, e $0.31$ vs $0.31$ in modalità integrata. La flessibilità non lineare sui dati aggregati tabulari non offre vantaggi di rilievo.
  - Al contrario, quando applicata a traiettorie sequenziali a 10 giorni (decadi), la rete neurale ricorrente LSTM domina i modelli tabulari, raggiungendo $R^2 = 0.35$ con memoria storica. La complessità algoritmica premia solo quando accoppiata a un'architettura che preserva la struttura temporale fine.

### H4 (Temporal aggregation representation: Pentads vs Decads vs Monthly)
* **Asserzione formale:** La risoluzione temporale sub-mensile migliora la cattura dei picchi di stress fisiologico rispetto ai blocchi mensili statici.
* **Verdetto:** **Supportata con raffinamento**.
* **Evidenza empirica:**
  - Mentre i blocchi mensili garantiscono stabilità dimensionale per i modelli tabulari appiattiti ($P_1 = 194$), le decadi a 10 giorni ($W^* = 10\text{d}$, 35 passi temporali) costituiscono la rappresentazione fisiologica ottimale per le architetture sequenziali.
  - Le decadi isolano gli impulsi acuti di stress termico ($HD_{35}, VPD$) che verrebbero diluiti nelle medie mensili, evitando al contempo l'inflazione parametrica e il rumore stocastico delle pentadi a 5 giorni.

### H5 (Tail-risk shock anticipation)
* **Asserzione formale:** I modelli di regressione continua anticipano quantitativamente gli shock estremi di resa (siccità e bumper) prima della raccolta.
* **Verdetto:** **Supportata**.
* **Evidenza empirica:**
  - Durante la siccità eccezionale del 2012, i modelli identificano la contrazione severa già al 31 luglio ($H = 4$, bias ridotto a fine stagione).
  - La calibrazione per decili dimostra che l'architettura sequenziale LSTM (+Lag) mitiga il fenomeno del *conditional shrinkage*, riducendo l'errore di stima sul decile peggiore ($D_1$, media $-8.82\text{ bu/ac}$) a $\text{RMSE} = 6.73\text{ bu/ac}$ e predicendo $-3.09\text{ bu/ac}$, rispetto a $-1.56\text{ bu/ac}$ di XGBoost e $-2.19\text{ bu/ac}$ di ElasticNet.

---

## 2. Tabella Sinottica di Riepilogo

| Ipotesi | Asserzione Formale | Verdetto & Evidenza Chiave | Stato |
| :--- | :--- | :--- | :---: |
| **H1** | Incremental weather skill | Nulla di abilità nulla rigettata: $R^2_{\text{OOS}}$ raggiunge $+0.30$ (meteo puro) e $+0.35$ (integrato), test Diebold-Mariano $p < 0.0001$. | $\checkmark$ |
| **H2** | Monotone lead-time dependence | Crescita monotona da semina ($H=6$) a maturità ($H=2$); errore declina da $6.12$ a $5.01\text{ bu/ac}$. | $\checkmark$ |
| **H3** | Non-linear complexity value | Vantaggio dell'LSTM sequenziale a decadi sui modelli tabulari ($R^2 = 0.35$ vs $0.31$ per XGBoost/ElasticNet). | $\checkmark$ / $\sim$ |
| **H4** | Temporal aggregation representation | Risoluzione a 10 giorni (decadi) ottimale: isola le ondate di calore estive senza rumore delle pentadi né diluizione mensile. | $\checkmark$ |
| **H5** | Tail-risk shock anticipation | Siccità 2012 anticipata a $H=4$; LSTM (+Lag) mitiga il tail shrinkage su $D_1$ ($\text{RMSE} = 6.73\text{ bu/ac}$). | $\checkmark$ |
