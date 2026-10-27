# Registro di Revisione Globale della Tesi (Note dell'Autore)

> **Finalità:** Questo documento raccoglie tutti gli appunti, le annotazioni e i passaggi da correggere/riformulare durante la lettura completa della tesi (`output/thesis/main.pdf`).  
> Una volta completata la lettura dell'intera tesi, tutte le modifiche verranno applicate in modo organico e sincronizzato sui sorgenti LaTeX.  
>  
> **REGOLA METODOLOGICA OPERATIVA (Disciplina dello Scope):**  
> 1. Verranno riscritti o modificati **esclusivamente** i paragrafi e le sezioni esplicitamente segnalati come non soddisfacenti in queste note.  
> 2. Tutto il resto del testo della tesi **non deve essere toccato né ristrutturato opportunisticamente**.  
> 3. Le parti non segnalate potranno subire minime correzioni **solo ed esclusivamente se strettamente necessario** per garantire l'allineamento logico, grammaticale o dei rimandi incrociati rispetto alle modifiche richieste sui paragrafi segnalati.


---

## Capitolo 1: Introduction

### Sezione 1.1: Problem, Practical Value, and Motivation (Pagina 1)
- **Nota 1 (Ristrutturazione logica Sezione 1.1 - Dal generale al particolare):**
  - **Problema rilevato:** La struttura attuale salta avanti e indietro: inizia sul generale (resa osservabile solo a raccolto), butta lì la soia (*"Soybean supplies..."*), parla dei prezzi e shock della soia, e poi torna su concetti totalmente generici di forecasting (*"Forecast value depends on lead time... We assess three criteria..."*).
  - **Ristrutturazione richiesta:**
    1. **Parte 1 (Generale sulle rese agricole):** Aprire con il valore operativo della previsione anticipata delle rese agricole per il risk management; spostare qui il blocco generico sul valore del lead time, sui diversi orizzonti decisionali (pre-semina vs in-season) e sui 3 criteri di valutazione (accuratezza sulla distribuzione, timing di emersione del segnale, persistenza tra aggiornamenti mensili $H=12 \dots 1$).
    2. **Parte 2 (Focus specifico su Soybean e Corn Belt):** Transizione fluida verso la soia come coltura elettiva per questo studio (ruolo sistemico globale per mangimi zootecnici e scambi commerciali, centralità del Corn Belt USA). Approfondire i costi economici asimmetrici degli shock (siccità estive, impatto sui prezzi CBOT con elasticità $+7.5\%$ per $-10\%$ di resa, colli di bottiglia logistici sul Mississippi) e perimetrare l'oggetto d'indagine sulle anomalie di resa a livello di contea.

### Sezione 1.2: Research Question and Target Formulations (Pagina 2)
- **Nota 2 (Raccordo naturale sul trend e demistificazione del Downside Risk):**
  - **Problema rilevato:**
    1. Il primo paragrafo introduce il trend tecnologico in modo generico e sembra staccato dalla domanda di ricerca a cui dovrebbe invece condurre.
    2. La descrizione della *"Downside Tail-Risk Anticipation"* è inutilmente ampollosa, quasi a voler sembrare una teoria a parte. In realtà stimiamo normali modelli di regressione continua; valutare come performano sulle code estreme (es. siccità, decile $D_1$) è semplicemente il test fondamentale di buon senso: se un modello prevede bene solo vicino alla media storica (dove le oscillazioni contano poco) e fallisce sui casi estremi (che causano tutti i disastri economici), lo studio è inutile.
  - **Intervento richiesto:**
    1. **Raccordo Trend $\to$ RQ:** Riformulare il primo paragrafo in modo che introduca il trend come premessa metodologica immediata alla domanda di ricerca (per isolare quanto e quando il meteo spiega le rese, dobbiamo prima depurare le serie dal trend tecnologico secolare).
    2. **Semplificazione Downside Risk:** Ridurre l'enfasi accademica artificiale; spiegare in modo chiaro e asciutto che i modelli continui vengono stress-testati sulle code negative perché sono i casi estremi a guidare i rischi reali di mercato e a determinare se un modello ha vera utilità pratica.
    3. **Ultimo paragrafo:** Confermato valido così com'è (obiettivi e link a H1--H5); non toccarlo.

### Sezione 1.3: Methodological Contributions and Scope Boundaries (Pagina 2--3)
- **Nota 3 (Distinzione concettuale sui contributi e rimozione titoletto AI):**
  - **Problema rilevato:**
    1. Il primo contributo metodologico accorpa confusamente due aspetti che sono concettualmente distinti: (a) la mappatura temporale a 12 orizzonti (*quanto prima si riesce a prevedere lungo l'anno*); (b) il vincolo operativo di non usare dati di previsione meteo o indicatori non disponibili in tempo reale (*realismo operativo e assenza di look-ahead*).
    2. Il titoletto in linea `\paragraph{Scope Boundaries and Methodological Discipline.}` sa di cliché da AI e appesantisce la lettura.
  - **Intervento richiesto:**
    1. **Sdoppiare/chiarire bene i due aspetti nel Contributo 1:**
       - *Dimensione Temporale:* Mappatura sistematica dell'anticipo previsionale ($H=12 \dots 1$) su 12 cutoff mensili da novembre a ottobre per capire quando il segnale emerge davvero.
       - *Realismo Operativo Causale (No Forecast Inputs):* Valutare rigorosamente solo i dati che un previsore ha effettivamente in mano alla data di cutoff, senza "barare" retrospettivamente con dati futuri e senza inserire previsioni meteo numeriche (che portano errori di calibrazione e drift), lasciando l'integrazione di seasonal forecasts come eventuale sviluppo futuro.
    2. **Punto 2 (Confronto specifiche ML vs lineari):** Confermato valido così com'è.
    3. **Rimozione titoletto `\paragraph`:** Togliere il titoletto; iniziare direttamente il paragrafo con *"We evaluate this framework on soybean..."*.
    4. **Allineamento testo finale:** Raccordare il testo finale dell'1.3 enfatizzando questo principio di realismo empirico (panel 75 anni non imputato, associazioni predittive senza pretese di identificazione causale strutturale).

---


## Capitolo 2: Literature Review

### Sezione 2.1: Introduction and Organization of the Literature (Pagina 5)
- **Nota 4 (Eliminazione Sezione 2.1 - Ridondanza introduttiva):**
  - **Problema rilevato:** La Sezione 2.1 (*"Introduction and Organization of the Literature"*) è costituita da appena 3 frasi generiche di meta-testo che ripetono quanto già anticipato nella Sezione 1.4 (*"Structure of the Study"*). Avere una sottosezione introduttiva all'inizio del Capitolo 2 è superfluo e sa di riempitivo.
  - **Intervento richiesto:** Eliminare completamente la Sezione 2.1 (`\section{Introduction and Organization of the Literature}`). Far iniziare il Capitolo 2 direttamente con la sostanza agronomica ed econometrica: `\section{Technological Trends, Weather Variability, and Soybean Agronomic Sensitivity}` (che diventerà la nuova Sezione 2.1).


### Sezione 2.2: Technological Trends, Weather Variability, and Soybean Agronomic Sensitivity (Pagine 5--6)
- **Nota 5 (Unificazione indici non lineari ed eliminazione doppione fenologia 2.2.4):**
  - **Problema rilevato:**
    1. Le sottosezioni 2.2.1 (GDD), 2.2.2 (EDD), 2.2.3 (VPD e Bilancio Idrico) e 2.2.4 (Finestre fenologiche) sono 4 mini-sottosezioni frammentate di appena 1-2 paragrafi ciascuna, che spezzettano inutilmente la lettura.
    2. La sottosezione 2.2.4 (*"Phenological Vulnerability Windows"*) ripete concetti botanici e fasi fenologiche della soia già trattati in modo completo e dettagliato nella Sezione 3.2 del Capitolo 3 (*"Soybean Phenology and Critical Agro-Climatic Sensitivity Windows"*).
  - **Intervento richiesto:**
    1. Fondere 2.2.1, 2.2.2 e 2.2.3 in un'unica sottosezione solida ed elegante: `\subsection{Non-Linear Agro-Climatic Stress Indicators}`, incentrata sulla giustificazione teorico-statistica del feature engineering da letteratura (tempo termico GDD, soglie di danno estremo EDD $>30^\circ\text{C}$, deficit di pressione di vapore VPD e bilancio climatico $P - ET_0$).
    2. Eliminare la sottosezione 2.2.4 come blocco autonomo, inserendo a chiusura degli indici una breve frase di raccordo che richiama la concentrazione del danno durante la finestra riproduttiva estiva e rimanda la dettagliata descrizione agronomica delle fasi fenologiche alla Sezione 3.2.


### Sezione 2.3: Empirical Design Dimensions: Lead Times, Target Formulations, and Baseline Benchmarking (Pagine 6--7)
- **Nota 6 (Sottosezione 2.3.1: Contrasto sintetico tra cutoff di agosto in Chen & Zhang e continuum a ottobre):**
  - **Problema rilevato:** In 2.3.1 si cita Chen & Zhang (2026) che si ferma ad agosto, mentre il nostro studio prosegue fino ad ottobre ($H=1$, o settembre $H=2$ per il Minnesota). Manca una breve spiegazione del perché di questa differenza, utile a prevenire dubbi del professore o dei revisori.
  - **Intervento richiesto:** Integrare in 2.3.1 una spiegazione concisa (1-2 frasi asciutte):
    - *Perché Chen & Zhang si fermano ad agosto:* Nel Corn Belt la finestra fenologica critica di formazione della granella (fioritura R1-R2 e riempimento dei semi R5-R6) si conclude fisiologicamente a fine agosto; gli studi focalizzati strettamente sulla risposta agronomica considerano quindi concluso l'accumulo del segnale biologico.
    - *Perché noi proseguiamo fino ad ottobre ($H=1$ / $H=2$ per MN):* Estendiamo la valutazione su un continuum completo a 12 mesi per coprire l'intera finestra decisionale della filiera fino alla vigilia della raccolta commerciale e per verificare empiricamente (Ipotesi H2) se il meteo tardivo aggiunge effettivo valore predittivo o solo rumore/overfitting.

- **Nota 7 (Eliminazione Sezione 2.3.2 ed emendamento titolo Sezione 2.3):**
  - **Problema rilevato:**
    1. La sottosezione 2.3.2 (*"Target Formulations: Continuous Yields and Downside Risk Anticipation"*) è una duplicazione superflua: i concetti di stima continua e stress-testing sulle code ($D_1$) sono già stati chiariti in 1.2 e formalizzati in Cap. 4; le citazioni di Vijverberg (2023) e Iizumi (2021) sono già riprese in modo approfondito nella Sezione 2.4 sui benchmark.
    2. Contiene frasi su figure professionali (*"grain elevator operators, processors, and insurers"*) orientate alla logistica aziendale, non adatte a una tesi quantitativa.
  - **Intervento richiesto:**
    1. Eliminare integralmente la sottosezione 2.3.2 (`\subsection{Target Formulations: Continuous Yields and Downside Risk Anticipation}`).
    2. Riformulare il titolo della Sezione 2.3 togliendo il riferimento al target: `\section{Empirical Design Dimensions: Lead Times and Baseline Benchmarking}`.
    3. Il recupero di testo libera circa 800 caratteri utili a salvaguardare il tetto massimo dei 100.000 caratteri.


### Sezione 2.6 e 2.7: Benchmark Studies e Literature Gap Synthesis (Pagine 7--9)
- **Nota 8 (Riorganizzazione Benchmark 2.6 e Grande Sintesi del Literature Gap in 2.7.1):**
  - **Problema rilevato:**
    1. Nelle sottosezioni dei 4 benchmark (2.6.1 Chen & Zhang, 2.6.2 Vijverberg, 2.6.3 Yin, 2.6.4 Khaki), le frasi di chiusura contengono accenni parziali e ripetitivi a ciò che fa il nostro studio. In particolare, la chiusura di 2.6.1 (riga 96) svende il nostro lavoro, riducendolo a un semplice cambio di dataset ERA5 e orizzonti $H=12 \dots 1$, senza evidenziare il superamento del limite strutturale di Chen & Zhang (l'illusione del lag $Y-1$ e il test set di soli 2 anni).
    2. La sottosezione 2.7.1 (*"Literature Consensus and the Identified Opportunity"*) è attualmente ridotta a un "moncone" vuoto di sole 4 righe generiche (righe 111--113), privo di mordente analitico.
  - **Intervento richiesto (Opzione B):**
    1. **Nelle sottosezioni 2.6.1--2.6.4:** Mantenere intatte le descrizioni dei paper ma rifinire solo le frasi conclusive per focalizzarle sui limiti oggettivi di ciascun lavoro (Chen & Zhang: distorsione da lag autoregressivo su rese grezze e test set minimale di soli 2 anni; Vijverberg: confinamento alla classificazione binaria arbitraria; Yin: panel corto di soli 21 anni; Khaki: assunzione non verificata della superiorità del Deep Learning in assenza di test di parsimonia contro modelli lineari). Rimuovere le micro-ripetizioni frammentate di "noi invece facciamo...".
    2. **Nella sottosezione 2.7.1:** Sostituire il trafiletto vuoto di 4 righe con la sintesi organica dei 4 gap metodologici colmati dal nostro framework:
       - *Disaccoppiamento anomalia-trend:* isolamento di $\varepsilon_{it}$ da $T_{it}$ con detrending rigorosamente train-only, eliminando l'inerzia del lag $Y-1$;
       - *Continuum a 12 orizzonti unificato:* integrazione del segnale pre-semina e in-season senza look-ahead né previsioni numeriche;
       - *Granularità a 10 giorni su 75 anni:* estensione dell'aggregazione decadale su un panel storico multidecennale;
       - *Validazione temporale rigorosa e parsimonia:* expanding window out-of-sample su 15 campagne (2011--2025) con test formali di Diebold-Mariano tra lineari, alberi e LSTM.
    3. **Sezione 2.7.2 (Ipotesi H1--H5):** Raccordare l'introduzione alle ipotesi come formalizzazione diretta di questi 4 gap.

- **Nota 9 (Sezione 2.7.2: Impostazione delle Ipotesi come Risposta alle 5 Domande Chiave della Tesi):**
  - **Problema rilevato:** Nell'elenco attuale, le 5 ipotesi sono presentate come un elenco formale accademico secco, senza far emergere con immediatezza la specifica domanda di ricerca intuitiva che ciascuna di esse va a testare per guidare l'analisi empirica.
  - **Intervento richiesto:** Strutturare la presentazione di ciascuna ipotesi in 2.7.2 esplicitando per ciascuna la domanda metodologica di fondo a cui vogliamo rispondere:
    1. *Domanda 1 (Esistenza del segnale meteo):* Il meteo aggiunge reale valore predittivo rispetto a una stima basata unicamente sui dati storici passati (media e trend)? $\to$ **H1 (Incremental Weather Skill)**
    2. *Domanda 2 (Anticipo temporale e persistenza):* Con quanto anticipo prima del raccolto emerge un segnale affidabile lungo i 12 orizzonti, e tale segnale persiste stabilmente o si degrada in rumore? $\to$ **H2 (Lead-Time Dependence and Persistence)**
    3. *Domanda 3 (Risoluzione temporale su lungo periodo):* Una granularità sub-mensile a 10 giorni apporta vero valore rispetto alle medie mensili a 30 giorni quando testata su un panel storico multidecennale di 75 anni? $\to$ **H3 (LSTM Temporal Aggregation)**
    4. *Domanda 4 (Complessità dei modelli e parsimonia):* Reti neurali complesse (LSTM) battono davvero modelli lineari regolarizzati (Ridge) o alberi (XGBoost) nel prevedere le anomalie sotto validazione temporale rigorosa, o vince la parsimonia? $\to$ **H4 (Model Complexity under Strict Validation)**
    5. *Domanda 5 (Stress test sui disastri climatici):* Modelli di regressione continua sanno anticipare i grandi collassi produttivi (il peggior decile $D_1$) o soffrono di appiattimento (*shrinkage*) verso la media storica? $\to$ **H5 (Tail-Risk and Downside Shock Anticipation)**

---


## Capitolo 3: Data and Study Area

### Sezione 3.1: Agronomic Setting and County Panel Assembly (Pagine 11--12)
- **Nota 10 (Sottosezione 3.1.1: Eliminazione frase tronca "Observed yield values are preserved"):**
  - **Problema rilevato:** A riga 19, dopo aver chiarito che il panel richiede serie annuali complete 1951--2025 senza imputazione (*"without imputation"*), la frasina finale *"Observed yield values are preserved."* risulta superflua, ambigua e dal tono burocratico.
  - **Intervento richiesto:** Eliminare la frasina *"Observed yield values are preserved."*. Il concetto di integrità del dato è già espresso in modo rigoroso e sufficiente dalla dicitura *"without imputation"*.

- **Nota 11 (Figura 3.1: Rimozione di "Balanced" da titolo, legenda e caption):**
  - **Problema rilevato:** Nel grafico della mappa (`fig_study_area_map`), il titolo recita *"Geographic Distribution of Balanced Study Counties (1951–2025)"* e la legenda riporta *"Balanced Panel Counties (N=135)"*. Inoltre la caption LaTeX ripete goffamente "balanced" due volte (*"135 balanced study counties [...], illustrating balanced coverage..."*). L'uso dell'aggettivo "balanced" riferito alle contee è una distorsione del gergo econometrico (*balanced panel*), suona male in inglese e appesantisce la figura.
  - **Intervento richiesto:**
    1. In `scripts/08_generate_eda_figures.py`:
       - Titolo del grafico: cambiare in `Geographic Distribution of the 135 Study Counties (1951–2025)`.
       - Legenda del grafico: cambiare in `Study Counties (N=135)`.
       - Rigenerare la figura `fig_study_area_map.pdf` e `.png`.
    2. Nel sorgente LaTeX `03_data_and_study_area.tex` (riga 32):
       - Modificare la caption in: `\caption{Geographic distribution of the 135 study counties across the six Midwestern states (1951--2025), illustrating spatial coverage across the primary agro-ecological subregions of the Corn Belt.}`.

- **Nota 12 (Tabella 3.1: Pulizia toni difensivi/enfatici in caption e note a piè tabella):**
  - **Problema rilevato:**
    1. Nella caption della Tabella 3.1 compare di nuovo *"balanced study counties"*.
    2. Nelle note a piè tabella (`\footnotesize{\textit{Notes:} ...}`) compare un testo enfatico e difensivo da AI (*"unbroken 75-year continuity under a strict zero-imputation protocol... total finite county-year yield records"*), che suona melodrammatico e artificioso.
  - **Intervento richiesto:**
    1. In `output/thesis/tables/tab_panel_selection.tex`:
       - Caption: cambiare in `\caption{Descriptive summary of the 135 study counties across the six Midwestern states (1951--2025).}`.
       - Note: sostituire con testo asciutto e oggettivo:  
         `\footnotesize{\textit{Notes:} Counties ($N$) indicates the number of counties with complete annual records from 1951 to 2025 without imputation. Mean Yield and Std. Dev. are in bushels per acre. \textit{Source:} Author's calculation on USDA NASS Quick Stats data (1951--2025).}`.

- **Nota 13 (Sottosezione 3.1.2: Compattazione estrema ed eliminazione dell'elenco botanico pedante):**
  - **Problema rilevato:** La sottosezione 3.1.2 (*"Soybean Phenology and Critical Agro-Climatic Sensitivity Windows"*) contiene un elenco puntato pedante di fasi agronomiche/botaniche (emergenza, allungamento del fittone, chiusura della chioma, ecc.) che non ha utilità diretta per una tesi quantitativa in statistica e data science e appesantisce la lettura.
  - **Intervento richiesto:** Eliminare completamente l'elenco puntato botanico e compattare la sottosezione in **due soli brevi paragrafi operativi** incentrati esclusivamente sui risvolti modellistici:
    1. *Effetti tempo-dipendenti e concentrazione del danno:* Le variabili meteorologiche hanno impatti che variano radicalmente per segno e intensità a seconda del momento fenologico; mentre le finestre di semina e vegetative tollerano compensazioni fisiologiche, la finestra riproduttiva estiva (luglio-agosto) costituisce il vero collo di bottiglia in cui si concentra la varianza spiegabile (giustificando la discretizzazione a 12 orizzonti mensili e a blocchi decadali di 10 giorni).
    2. *Chiusura operativa e differenziazione regionale (Minnesota $H^*=2$):* Il gradiente latitudinale (Maturity Groups) determina una maturazione e una raccolta anticipate a fine settembre per il Minnesota settentrionale (Group I), fondando la giustificazione agronomica per la chiusura anticipata al 30 settembre ($H^*(s)=2$) rispetto al 31 ottobre ($H^*(s)=1$) degli altri stati, evitando rumore post-raccolta.
    3. Il taglio elimina oltre 800 caratteri di testo botanico superfluo, espandendo ulteriormente il margine sotto il limite dei 100.000 caratteri.


### Sezione 3.2: Exploratory Data Analysis of County Soybean Yields (Pagine 12--15)
- **Nota 14 (Sottosezione 3.2.1 e raccordo con 5.4.4: Spiegazione intuitiva dell'$R^2$ del Trend su 75 anni vs 15 anni di test):**
  - **Problema rilevato:** Il lettore potrebbe percepire un'apparente discrepanza tra il Capitolo 3 (dove il trend tecnologico da solo spiega l'81.2% della varianza storica delle contee e il 93.2% a livello regionale aggregato su 75 anni) e il Capitolo 5 (dove nel test set 2011--2025 il trend da solo spiega solo il 47.4% e il modello completo trend+meteo arriva al 67.3%).
  - **Intervento richiesto:**
    1. *In Sottosezione 3.2.1 (dopo l'equazione 3.2):* Lasciare una trattazione snella del trend a 75 anni.
    2. *In Sottosezione 5.4.4 (Capitolo 5):* È la sede primaria naturale per la spiegazione: spiegare matematicamente la differenza di denominatore tra i 75 anni storici (dove la triplicazione secolare $+40\text{ bu/ac}$ domina la varianza portando il trend all'81%) e il test set out-of-sample di 15 anni 2011--2025 (dove il trend cresce di appena $+7\text{ bu/ac}$ e pesano le grandi siccità come il 2012, riducendo la quota del trend al 47.4% e aprendo lo spazio al valore aggiunto del meteo $+20\%$, fino al 67.3%).

- **Nota 15 (Sottosezione 3.2.1: Giustificazione ex-ante del Trend Lineare sul pre-test 1951--2010 e rimando all'Appendice A):**
  - **Problema rilevato:** La selezione del trend lineare in Tabella 3.3 (Panel B) è basata rigorosamente sul periodo pre-test (1951--2010) per evitare data snooping / look-ahead bias verso il test set (2011--2025). Nel testo manca un raccordo esplicito che colleghi questa scelta metodologica alla successiva comparsa del break strutturale del 2014 e al ruolo dell'Appendice A.
  - **Intervento richiesto:** Integrare a chiusura del commento alla Tabella 3.3 una frase asciutta di raccordo metodologico ed econometrico:
    - Esplicitare che per rispettare la causalità temporale ed evitare look-ahead bias, la scelta del modello di trend deve basarsi rigorosamente sul campione pre-test (1951--2010), periodo in cui non vi era evidenza statistica di structural break (Chow test $p=0.093$) e il modello lineare OLS era strettamente preferito da BIC e out-of-sample RMSE.
    - Spiegare che l'analisi dell'intero campione storico 1951--2025 (in cui emerge uno scalino strutturale nel 2014) e la valutazione di specifiche alternative flessibili (come il trend cubico oracle) sono relegate in **Appendice A come stress-test di robustezza**, poiché incorporare a priori il break del 2014 nei modelli operativi della tesi costituirebbe un illegittimo bias retrospettivo.

- **Nota 16 (Sottosezione 3.2.1: Rimozione della pedanteria accademica fuorviante su ADF e stazionarietà):**
  - **Problema rilevato:** A riga 96, la frase finale *"The reported Augmented Dickey-Fuller test rejects a unit root in the detrended series ($p<0.001$); this result does not establish strict stationarity."* crea un'inutile confusione: dopo aver rigettato la radice unitaria, la postilla pedante insinua nel lettore il dubbio che la serie non sia utilizzabile o non sia stazionaria.
  - **Intervento richiesto:** Riformulare in modo diretto, positivo e scientificamente ineccepibile, allineandolo a quanto già espresso in Appendice A (riga 32):  
    `The Augmented Dickey-Fuller test firmly rejects the null hypothesis of a unit root in the detrended series ($p < 0.001$), confirming that the yield anomalies are stationary ($I(0)$).`  
    Eliminare del tutto l'inciso superfluo su *"strict stationarity"*.

- **Nota 17 (Sottosezione 3.2.2: Riconversione statistica dell'eterogeneità spaziale ed eliminazione di speculazioni agronomiche):**
  - **Problema rilevato:** Nel testo di 3.2.2 (righe 102 e 113) compaiono affermazioni qualitative di taglio agronomico/pedologico non osservate dai modelli (es. *"favored by deep soils and reliable thermal regimes"*, *"northward commercial adoption of cold-tolerant varieties"*, *"frequent exposure to midsummer heat and drought"*).
  - **Intervento richiesto:** Riscrivere i due paragrafi eliminando qualsiasi speculazione agronomica/biologica e reimpostando l'analisi in chiave puramente statistica ed econometrica:
    1. *Eterogeneità degli intercette e delle medie base:* Le differenze di resa storica tra stati (Iowa/Illinois a 39 bu/ac vs Missouri a 29 bu/ac) riflettono una marcata eterogeneità spaziale cross-sezionale ($\alpha_c$), che giustifica formalmente l'adozione di fixed effects / detrending county-specific rispetto a un modello pooled uniforme.
    2. *Eterogeneità dei coefficienti di crescita (slope del trend):* Le pendenze del trend lineare variano sensibilmente nello spazio (da $+0.509\text{ bu/ac/year}$ in Minnesota a $+0.386\text{ bu/ac/year}$ in Missouri), fondando la necessità statistica di stimare pendenze $\beta_c$ individuali a livello di contea anziché un trend aggregato comune.
    3. *Eterogeneità della dispersione cross-contea:* Stati come Indiana e Ohio mostrano una varianza interna molto più contenuta rispetto alla forte dispersione tra contee del Minnesota o dell'Illinois, motivando l'analisi disaggregata della performance predittiva a livello locale.

- **Nota 18 (Tabella 3.4: Sostituzione di "farm-gate" con "commodity price"):**
  - **Problema rilevato:** La dicitura *"farm-gate prices"* nella caption e nelle note della Tabella 3.4 è un termine specialistico di economia agraria che può risultare poco immediato.
  - **Intervento richiesto:** Mantenere la colonna dei prezzi per illustrare l'impatto economico degli shock climatici, ma sostituire *"farm-gate prices"* con il termine standard e universale **"commodity prices"**:
    1. In `output/thesis/tables/tab_yield_shocks.tex`:
       - Caption: `\caption{Historical macro-climatic shock years, commodity price responses, and associated drivers (1951--2025).}`.
       - Note: `Max Price Swing: peak percentage departure in national commodity prices over the 12-month campaign window.`.

- **Nota 19 (Sottosezione 3.2.3: Eliminazione del paragrafo di chiusura fuori contesto e del rimando rotto):**
  - **Problema rilevato:** Il paragrafo finale di 3.2.3 (riga 125, *"While major shortfalls are primarily associated with midsummer drought and heat, the historical record also contains losses from unseasonal frost and excessive wetness. We analyze these daily mechanisms in Section 3.4. Because synoptic weather shocks span multi-state regions, cross-county yield correlations exceed 0.75, necessitating strict expanding-window temporal validation to prevent spatial information leakage (Sweet et al., 2023)."*) presenta due gravi difetti:
    1. Contiene un rimando a una non-esistente *"Section 3.4"*;
    2. Salta prematuramente dalla descrizione storica delle rese alla metodologia di validazione del machine learning (expanding window e spatial leakage), che appartiene invece al Capitolo 4.
  - **Intervento richiesto:** Eliminare integralmente questo paragrafo di due frasi a riga 125. La Sezione 3.2.3 si chiude in modo perfetto e rigoroso sul paragrafo precedente (riga 122), che formalizza l'asimmetria biologica delle anomalie e la coda spessa delle perdite (skewness $-1.12$, Jarque-Bera $p<0.001$). Il taglio elimina una contraddizione di numerazione e risparmia circa 350 caratteri.


### Sezione 3.3: Surface Weather Data: ECMWF ERA5-Land (Pagine 15--18)
- **Nota 20 (Sottosezione 3.3.1: Rimozione della parentesi superflua sulla temperatura del suolo):**
  - **Problema rilevato:** A riga 139, l'inciso tra parentesi *"soil temperatures (strongly collinear with ambient air temperatures)"* genera un fastidioso sovraccarico di parentesi all'interno della stessa frase (dove c'è già `$SM_4 > 1\text{ m}$`) e spezza il ritmo dell'elenco.
  - **Intervento richiesto:** Eliminare la parentesi `(strongly collinear with ambient air temperatures)`, lasciando scorrere la frase in modo fluido e asciutto: *"Deep soil horizons beyond mature soybean taproot depths ($SM_4 > 1\text{ m}$), soil temperatures, and snowpack dynamics are excluded a priori to prevent dimensionality inflation and avoid introducing uninformative predictors during the summer vegetative cycle."*.

- **Nota 21 (Tabella 3.5: Ristrutturazione a 3 colonne ariose, incorporazione unità di misura e fine delle parole spezzate):**
  - **Problema rilevato:** La Tabella 3.5 è compressa orizzontalmente: la colonna `Unit` sottrae $1.3\text{ cm}$ di larghezza utile solo per poche lettere (`°C`, `mm/d`, `kPa`), costringendo la prima colonna `Category` a essere troppo stretta ($2.4\text{ cm}$) e a spezzare le parole a metà sillaba in modo antiestetico (*Hy-drological*, *Pe-dological*).
  - **Intervento richiesto:**
    1. In `output/thesis/tables/tab_weather_variables.tex`:
       - Eliminare la colonna `Unit` separata;
       - Incorporare l'unità tra parentesi direttamente accanto al simbolo/nome della variabile nella seconda colonna, rinominata in `\textbf{Variable and Unit}` (es. `$T_{\text{mean}}, T_{\text{max}}, T_{\text{min}}$ ($^\circ\text{C}$)`, `$P$ ($\text{mm/d}$)`, `$VPD$ ($\text{kPa}$)`, ecc.);
       - Allargare la colonna `Category` a $\approx 3.4\text{ cm}$ in modo che tutte le categorie abbiano spazio per stare su una riga o andare a capo pulite tra parole intere senza mai spezzarsi a metà sillaba;
       - Riassegnare tutto lo spazio residuo $X$ alla colonna della descrizione fisica e agronomica.

- **Nota 22 (Sottosezione 3.3.2: Copertura 1:1 di tutte le 8 metriche derivate in modo compatto ed essenziale):**
  - **Problema rilevato:** Nella Sottosezione 3.3.2 alcuni indicatori derivati del Panel B della Tabella 3.5 sembrano mancare o sono nascosti: in particolare $EDD_{30}$ non compare nel titolo del paragrafo ed è relegato a una frasetta tra parentesi graffe senza formula dedicata, mentre $ET_0$ e $P - ET_0$ sono accavallati.
  - **Intervento richiesto:** Riorganizzare i paragrafi della 3.3.2 in modo che coprano tutte le 8 metriche del Panel B con perfetto parallelismo 1:1, mantenendo il testo ultra-asciutto ed essenziale (1-2 frasi secche per metrica, zero "papiri"):
    1. $VPD$ (Eq. 3.3 e 3.4);
    2. $ET_0$ e $P - ET_0$ (Eq. 3.5 e 3.6);
    3. $GDD$ (Eq. 3.7);
    4. $HD_{30}, HD_{35}$ ed esplicitazione dell'equazione di $EDD_{30} = \max(0, T_{\text{max}} - 30)$;
    5. $SM_{\text{root}}$ (Eq. 3.9).
    Il testo deve rimanere snello e compatto per non appesantire il capitolo e bilanciare la Tabella 3.5.

- **Nota 23 (Sottosezione 3.3.3: Rimozione della struttura contrastiva/apologetica "replaces naive centroid sampling"):**
  - **Problema rilevato:** A riga 198, la frase *"this area-weighted polygon integration replaces naive centroid sampling, preserving the total precipitation volume..."* usa una struttura binaria contrastiva/apologetica vietata dalle linee guida (*"replaces naive X"*). Non dobbiamo giustificarci rispetto a cosa non facciamo, ma enunciare il metodo in modo assertivo e scientifico.
  - **Intervento richiesto:** Riscrivere la frase in modo assertivo e diretto:  
    `As illustrated in Figure~\ref{fig:era5_spatial_structure}, area-weighted polygon integration preserves the continuous precipitation volume, radiative flux, and atmospheric mass within each county polygon without geometric distortion.`


### Sezione 3.4: Exploratory Agro-Climatic Analysis and Yield Sensitivity (Pagine 18--22)
- **Nota 24 (Sottosezioni 3.4.1 e 3.4.2: Cifra stilistica statistica, shock composti e non-linearità vs botanica):**
  - **Direttiva metodologica e cifra stilistica:** Nella Sezione 3.4 (confronto shock storici e correlazioni climatiche) l'analisi deve rimanere rigorosamente ancorata alla prospettiva del **data scientist ed econometrista quantitativo**, evitando dettagli biologico-cellulari non osservati (es. *"root asphyxia, impaired nitrogen fixation"*):
    1. *Non-linearità e non-monotonicità (Concavità della risposta):* Il fatto che sia il 2012 (siccità estrema, $-11.7\%$) sia il 1993 (alluvione estrema, $-10.5\%$) abbiano causato crolli identici di resa è la prova statistica inconfutabile che la funzione di risposta alla pioggia e all'idrologia è non-lineare e concava (una retta OLS semplice fallirebbe perché non può catturare il danno da eccesso idrico e quello da deficit con lo stesso segno). Questo giustifica l'uso di alberi decisionali non-parametrici (Random Forest, XGBoost) e feature soglia.
    2. *Shock composti e forte collinearità multivariata:* Nel 2012 l'esplosione contemporanea di $VPD > 1.8\text{ kPa}$, $T_{\text{max}} > 33^\circ\text{C}$ e il crollo di $SM_{\text{root}} < 0.18$ mostrano che le forzanti climatiche agiscono in modo congiunto ed altamente correlato, motivando la penalizzazione convessa (ElasticNet) e le interazioni non lineari.
    3. *Eteroschedasticità temporale del segnale (Timing dello shock):* I profili rolling evidenziano che a maggio-giugno le annate bumper e le annate di disastro sono statisticamente indistinguibili (rapporto segnale/rumore $SNR \approx 0$), mentre il segnale predittivo si concentra quasi per intero nella finestra luglio-agosto, giustificando la dinamica a 12 orizzonti.
    4. *Limite delle correlazioni bivariate (Sezione 3.4.2):* Le correlazioni di Pearson tra resa e singoli indici mensili (Fig. 3.8) offrono una visione marginale utile ma insufficiente, motivando il passaggio alla modellistica predittiva congiunta multivariata out-of-sample del Capitolo 4.

- **Nota 25 (Sottosezioni 3.4.1 e 3.4.2: Commento analitico diretto delle Figure 3.5--3.8 e riscrittura del paragrafo finale):**
  - **Problema rilevato:** Le quattro figure analitiche della Sezione 3.4 (Figura 3.5 traiettorie cumulative, Figura 3.6 medie mobili a 15 giorni, Figura 3.7 mappe spaziali 2012 vs 2016, Figura 3.8 correlazioni fenologiche mensili) sono ricche di evidenze quantitative ma attualmente sono accompagnate da un testo troppo stringato che non ne estrae il valore statistico. Inoltre, il paragrafo conclusivo (righe 289--292) scade in un meta-testo burocratico privo di sostanza (*"in Chapter 4 formalizziamo i modelli e in Chapter 5 i risultati"*).
  - **Intervento richiesto:**
    1. Arricchire il commento a ciascuna figura con osservazioni chiare, sobrie e dirette (senza "papiri" e senza cliché da AI):
       - *Figura 3.5:* Evidenziare numericamente il contrasto idrico tra il 2012 ($\sum(P - ET_0)$ a $-300\text{ mm}$ con impennata di $HD_{30}$) e il 1993 ($+400\text{ mm}$ di surplus con zero caldo estremo), dimostrando visivamente la concavità della risposta alla risorsa idrica.
       - *Figura 3.6:* Sottolineare la durata e la persistenza continua dello shock nel 2012 (4 settimane consecutive di $VPD > 1.8\text{ kPa}$ e $T_{\text{max}} > 33^\circ\text{C}$ con umidità del suolo sotto $0.18$), giustificando l'adozione di architetture sequenziali ricorrenti (LSTM) o finestre a 10 giorni rispetto a medie mensili statiche.
       - *Figura 3.7:* Illustrare la scala sinottica e macro-regionale della siccità 2012 (calore omogeneo su Illinois, Indiana, Missouri e Iowa centro-meridionale, con relativo risparmio del Minnesota), fondando visivamente la presenza di autocorrelazione spaziale che impone la validazione temporale per interi blocchi d'anno.
       - *Figura 3.8:* Commentare la progressione temporale delle correlazioni di Pearson, da deboli e vicine allo zero ad aprile-maggio ($|r| < 0.10$) fino al picco estivo di luglio-agosto ($r \approx -0.45$ per $VPD$ e $+0.40$ per pioggia/suolo), prima del decadimento a settembre-ottobre.
    2. *Riscrittura del paragrafo finale (righe 289--292):* Sostituire il generico meta-testo con una sintesi analitica di chiusura: spiegare che la convergenza tra concavità idrica (Fig. 3.5), persistenza dinamica dello stress (Fig. 3.6), coerenza sinottica regionale (Fig. 3.7) e concentrazione estiva del segnale (Fig. 3.8) fornisce la rigorosa giustificazione empirica per l'architettura metodologica del Capitolo 4 (modelli non-lineari, sequenze a 10 giorni, 12 orizzonti predittivi e validazione temporale causale).

---

## Capitolo 4: Methodology
*(In attesa di note...)*

---

## Capitolo 5: Empirical Results
*(In attesa di note...)*

---

## Capitolo 6: Discussion and Conclusions
*(In attesa di note...)*

---

## Appendice A / Altro
*(In attesa di note...)*
