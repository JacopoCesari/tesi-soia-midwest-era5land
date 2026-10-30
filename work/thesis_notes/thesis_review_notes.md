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

- **Nota 38 (Sottosezione 1.1: Sostituzione di "farm-gate prices" con "commodity prices"):**
  - **Problema rilevato:** A riga 12 compare ancora la dicitura specialistica agraria *"farm-gate prices"* (*"each 10\% negative departure [...] associated with an average 7.5\% increase in annual farm-gate prices"*), in contrasto con la scelta di universalità e sobrietà adottata nella Nota 18 per la Tabella 3.4.
  - **Intervento richiesto:** Allineare la frase sostituendo con il termine standard:  
    `associated with an average 7.5\% increase in annual commodity prices ($p = 0.014$)`

- **Nota 39 (Sottosezione 1.2: Riconciliazione perfetta 1:1 tra obiettivi introduttivi e le 5 ipotesi H1--H5):**
  - **Problema rilevato:** A riga 33 si legge *"We structure our investigation around four primary objectives: [...] We derive formal hypotheses (H1--H5) addressing these objectives in Chapter 2."* C'è una discrepanza numerica e concettuale tra i *"quattro obiettivi"* e le **cinque ipotesi formali (H1--H5)** derivate nel capitolo successivo.
  - **Intervento richiesto:** Riformulare la frase a riga 33 allineandola direttamente alle cinque ipotesi formali:  
    `We structure our investigation around five core research questions and formal hypotheses (H1--H5, derived in Chapter~\ref{chap:literature_review}): evaluating incremental weather skill over secular trends, mapping lead-time persistence across horizons ($H=12 \dots 1$), assessing sub-monthly temporal aggregation, comparing linear parsimony against non-linear tree and deep learning architectures, and stress-testing the anticipation of downside tail risks.`

- **Nota 45 (Sottosezione 1.1: Semplificazione assertiva e rimozione della struttura circolare/negativa):**
  - **Problema rilevato:** A riga 14, la frase *"We focus on county-level yield anomalies (in bushels per acre), which isolate weather sensitivity from secular technology and cross-county acreage allocation, rather than predicting production totals or modeling farm-gate price responses directly."* contiene una struttura circolare e contrastiva vietata (*"rather than..."*) e ripropone la dicitura specialistica agraria *"farm-gate price"*.
  - **Intervento richiesto:** Sostituire con una formulazione assertiva, pulita e diretta:  
    `We focus specifically on county-level yield anomalies (in bushels per acre), which isolate climate sensitivity from secular technology and cross-county acreage shifts.`


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

- **Nota 40 (Sottosezione 2.2: Snellimento del danno termico cellulare in chiave macro-agronomica):**
  - **Problema rilevato:** A riga 30, descrivendo l'impatto termico sopra i $30^\circ\text{C}$, si indugia in un elenco pedante di biologia cellulare (*"via floral abscission, pollen sterility, stomatal closure, and impaired photosynthesis"*). In una tesi quantitativa di data science è preferibile mantenere la terminologia a livello macro-agronomico ed empirico osservabile.
  - **Intervento richiesto:** Sostituire con una formulazione più asciutta:  
    `via reproductive heat stress, floral abortion, and accelerated canopy senescence.`

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

- **Nota 41 (Sottosezione 3.1.1: Razionale fluido dell'avvio del panel al 1951 vs 1950):**
  - **Problema rilevato:** A riga 23, la motivazione per cui il panel parte dal 1951 anziché dal 1950 è spiegata in modo macchinoso e contorto (*"The first retained crop year is 1951. Its weather campaign begins in November 1950; the October 31 origin ($H=12$) provides the zero-anomaly reference. ERA5-Land begins in January 1950, leaving the antecedent months required for crop year 1950 unavailable."*).
  - **Intervento richiesto:** Riformulare in modo diretto e naturale:  
    `While ERA5-Land begins in January 1950, constructing the complete 12-month campaign for crop year 1950 would require reanalysis records from November 1949. Consequently, 1951 is the first crop year with complete antecedent weather data.`

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

- **Nota 43 (Figura 3.2: Allineamento temporale del panel a 1951--2025 in caption e script):**
  - **Problema rilevato:** Nella caption della Figura 3.2 (`fig_yield_trajectory_dual_petrol`, riga 68) compare ancora l'intervallo temporale `(1950--2025)`, mentre il panel confermato e ripulito parte rigorosamente dal 1951 (come stabilito nella Nota 41).
  - **Intervento richiesto:**
    1. In `output/thesis/chapters/03_data_and_study_area.tex` (riga 68): correggere `(1950--2025)` in `(1951--2025)`.
    2. In `scripts/08_generate_eda_figures.py`: verificare e allineare coerentemente il titolo o l'intervallo temporale della figura.


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

### Sezione 4.1: Problem Formulation and Multi-Horizon Forecast Setup (Pagine 23--25)
- **Nota 26 (Sottosezione 4.1.1: Rimozione avverbio difensivo "strictly"):**
  - **Problema rilevato:** A riga 24, la frase *"Detrending parameters are estimated strictly on training observations to prevent look-ahead leakage"* contiene l'avverbio enfatico/difensivo *"strictly"*, che appesantisce la frase con un tono da cliché AI.
  - **Intervento richiesto:** Rimuovere *"strictly"*, lasciando la frase pulita e assertiva:  
    `Detrending parameters are estimated on training observations to prevent look-ahead leakage.`

- **Nota 27 (Sottosezione 4.1.1: Esplicitazione del criterio di stima OLS via minimizzazione dell'MSE):**
  - **Problema rilevato:** A riga 24, la frase che introduce l'equazione (4.2) si limita a dire che i parametri del trend vengono stimati sulle campagne storiche senza nominare a parole il criterio statistico di ottimizzazione (la minimizzazione della somma dei quadrati / MSE).
  - **Intervento richiesto:** Completare la frase prima dei due punti esplicitando il criterio statistico:  
    `For harvest year $Y$, county-specific linear trend parameters $(\hat{\alpha}_c^{(Y)}, \hat{\beta}_c^{(Y)})$ are fitted over historical campaigns $\mathcal{T}_{\text{train}}(Y) = \{1951, \dots, Y-1\}$ by minimizing the mean squared error:`

- **Nota 28 (Sottosezione 4.1.3: Chiarimento baseline $H=12$, eliminazione elenco ridondante e semplificazione stilistica della persistenza):**
  - **Problema rilevato:**
    1. A riga 54, la frase *"The October 31 origin in $Y-1$ ($H=12$) is the zero-anomaly reference."* risulta ambigua, potendo far credere che vengano usate le osservazioni meteo dell'ottobre precedente. In realtà, $H=12$ rappresenta il benchmark "a bocce ferme" alla vigilia dell'annata agraria (31 ottobre di $Y-1$), in cui non si dispone di alcuna informazione meteo ($m=0, P_{\text{met}} = 0$) e la predizione di anomalia è identicamente nulla ($\hat{\epsilon}^{(12)} = 0$), basandosi unicamente sul trend tecnologico secolare ($\hat{y}_{c,Y}^{(12)} = \hat{\tau}_{c,Y}$).
    2. A riga 61, la frase *"To isolate the marginal value of multi-year agronomic memory against pure meteorological forcing, we formulate an integrated predictor space ingesting historical yield persistence:"* è infarcita di gergo altisonante e pomposo. Il concetto è semplicemente verificare se aggiungere la resa dell'anno prima apporti reale valore predittivo rispetto alle sole variabili meteo correnti.
    3. A riga 72 (Eq. 4.6), il caso $H=12$ riporta erroneamente `(m = 1, \text{ zero weather})` anziché $m=0$.
    4. Subito sotto la formula (4.6), le righe 77--81 inseriscono un elenco puntato che ripete pedestremente quanto già detto in apertura di sottosezione e che introduce l'ulteriore refuso fuorviante `Horizon $H = 12$ ($m = 1$, Nov 30)`, creando confusione temporale.
  - **Intervento richiesto:**
    1. In apertura di 4.1.3, formalizzare la timeline in modo compatto, lineare e privo di ambiguità:
       `The crop campaign spans November 1 of year $Y-1$ to October 31 of harvest year $Y$. The pre-season origin at October 31 of $Y-1$ ($H=12$) ingests zero weather predictors ($m=0, P_{\text{met}}=0$), serving as an uninformed benchmark that outputs a zero anomaly ($\hat{\epsilon}_{c,Y}^{(12)} = 0$) and relies entirely on the secular trend ($\hat{y}_{c,Y}^{(12)} = \hat{\tau}_{c,Y}$). Operational horizons $H \in \{11, \dots, 1\}$ correspond to monthly cutoffs from December 31 of $Y-1$ through October 31 of $Y$, ingesting $m = 13 - H \in \{2, \dots, 12\}$ elapsed campaign months.`
    2. A riga 61, snellire il testo eliminando il gergo artificioso:  
       `To evaluate whether past yield persistence provides incremental skill over current weather conditions, we specify an integrated feature vector that appends the prior-year detrended yield anomaly ($\epsilon_{c,Y-1}$):`
    3. A riga 72 (Eq. 4.6), correggere il caso $H=12$ impostando $m=0$:  
       `P_{\text{geo}} = 2, & \text{for } H = 12 \quad (m = 0, \text{ zero weather})`
    4. **Eliminare integralmente l'elenco puntato ridondante delle righe 77--81** (`\begin{itemize} ... \end{itemize}`), sostituendolo con una singola frase fluida che raccorda l'espansione delle feature da $P_{11}=36$ a $P_1=206$ (o 207 con il lag) direttamente alla regola di chiusura fenologica dello stato (Eq. 4.7). Questo elimina la contraddizione su `Nov 30`, toglie inutili duplicazioni e risparmia oltre 350 caratteri.

- **Nota 29 (Sottosezione 4.1.3: Chiarimento inequivocabile sulla regola di Harvest Closure $H^*(s)$ per il Minnesota):**
  - **Problema rilevato:** La definizione dell'orizzonte di chiusura fenologica nell'Equazione (4.7) ($H^*=2$ per il Minnesota e $H^*=1$ per gli altri 5 stati), accompagnata dalla sola frase laconica a riga 92 (*"Freezing inputs at $H^*(s)=2$ for Minnesota reflects early physiological maturity..."*), può ingenerare il grave dubbio che il Minnesota sia scalato di un mese su tutti gli orizzonti dell'anno (ad es. $H=2 \implies H=3$). Nel testo manca l'esplicitazione che da $H=12$ a $H=2$ il calendario è identico al 100% per tutti e che il congelamento interviene solo ed esclusivamente all'ultimo cut-off pre-raccolto ($H=1$).
  - **Intervento richiesto:** Integrare subito dopo l'Equazione (4.7) una spiegazione cristallina e assertiva:  
    `Across all preliminary horizons $H \in \{12, \dots, 2\}$, meteorological features are aligned to identical calendar month-ends across all six states (October 31 of $Y-1$ through September 30 of harvest year $Y$). The state-specific closure rule $H^*(s)$ applies exclusively at the final pre-harvest horizon: at October 31 ($H=1$), forecasts for Minnesota are frozen at their September 30 state ($H=2$) to eliminate post-harvest weather noise from already harvested fields, while models for central and southern states ingest October weather.`

- **Nota 30 (Sottosezioni 4.1.2 e 4.1.5: Eliminazione del riferimento spurio al 1995 e allineamento rigoroso al Test Set 2011--2025):**
  - **Problema rilevato:** A riga 38 (*"With 6,075 observations in the historical pool through 1995..."*) e a riga 107 (*"Separate county-by-county estimation with 45 to 74 historical observations... The baseline 1951--1995 partition provides 6,075 county-years, expanding to 9,990 observations at the final 2025 fold."*) si fa riferimento al range 1951--1995. Questo genera grande confusione: 1985--1995 è la fase di validazione/tuning (Phase 2), mentre la valutazione empirica del capitolo è sul **test set out-of-sample 2011--2025** (Phase 3). Nel test set il primo fold (2011) parte con il training 1951--2010 ($60\text{ anni} \times 135 = 8{,}100\text{ county-years}$), che si espande fino al 2024 ($74\text{ anni} \times 135 = 9{,}990\text{ county-years}$). Le serie per singola contea hanno quindi da 60 a 74 osservazioni (non da 45 a 74). Non ha alcun senso confondere le dimensioni della validazione con quelle del test set in questa sezione introduttiva.
  - **Intervento richiesto:**
    1. A riga 38, togliere il riferimento al 1995 e concentrarsi sul problema della dimensionalità:  
       `Ingesting unconstrained daily weather across 365 days generates approximately 6,205 features per county-year, inducing severe multicollinearity and variance inflation.`
    2. A riga 107, eliminare il riferimento spurio a 1951--1995 e 45 osservazioni, descrivendo con precisione la partizione di test out-of-sample:  
       `Separate county-by-county estimation with 60 to 74 historical observations causes parameter collapse ($P_H > T$) against feature vectors with up to 206 variables. Pooling expands sample size to $N_{\text{train}} = 135 \times |\mathcal{T}_{\text{train}}(Y)|$ observations. The initial training partition (1951--2010) provides $8{,}100$ county-years, expanding to $9{,}990$ observations at the final 2025 fold.`  
    La fase di validazione 1985--1995 viene descritta separatamente e rigorosamente nel posto a essa deputato, ovvero la Sezione 4.4.


### Sezione 4.2: Spatial Covariates and Input Representation (Pagine 25--26)
- **Nota 31 (Sottosezione 4.2.1 e 4.4.2: Eliminazione dell'inutile dicitura "training bounds" per le coordinate geografiche):**
  - **Problema rilevato:** A riga 128 (*"Coordinates are scaled to $[0, 1]$ using training bounds"*) e a riga 348 (*"Spatial normalization bounds are extracted solely from training counties"*) si applica un'inutile formula difensiva da manuale di machine learning. Poiché il nostro panel bilanciato è composto dalle stesse identiche 135 contee per l'intera finestra storica 1951--2025 (la geografia è immutabile e non cambia nel tempo, né facciamo out-of-sample spaziale su contee non osservate), i limiti $\min$ e $\max$ delle coordinate sono semplicemente gli estremi geografici dell'area di studio. Parlare di "training bounds" insinua nel lettore il dubbio errato che le contee varino tra train e test.
  - **Intervento richiesto:**
    1. A riga 128, riformulare in modo pulito e realistico:  
       `Coordinates are scaled to $[0, 1]$ across the geographic domain of the 135 study counties:`  
       dove $\min_i$ e $\max_i$ definiscono i confini spaziali fissi del panel.
    2. A riga 348, eliminare il punto elenco superfluo sulle coordinate: l'isolamento causale rigoroso riguarda le grandezze tempo-variabili (parametri del trend e momenti $\mu, \sigma$ delle variabili meteorologiche).

- **Nota 32 (Sottosezione 4.2.2: Eliminazione integrale della sottosezione fantasma):**
  - **Problema rilevato:** La Sottosezione 4.2.2 (*"Agrometeorological Predictor Space and Functional Partitioning"*, righe 139--145) consta di sole tre righe che si limitano a rimandare alla Sezione 3.3.2, alla Tabella 3.5 e all'Equazione (4.4). Non introduce alcuna nuova formula, nessun nuovo concetto e nessuna analisi, ripetendo pedestremente quanto già formalizzato in dettaglio in 4.1.2.
  - **Intervento richiesto:**
    1. Eliminare integralmente la Sottosezione 4.2.2 e il relativo titoletto;
    2. Ristrutturare la Sezione 4.2 come sezione compatta a tema unico sulle coordinate spaziali continue:  
       `\section{Spatial Encoding: Continuous Centroid Coordinates}`  
       eliminando la frammentazione interna in micro-sottosezioni e risparmiando circa 350 caratteri.


### Sezione 4.3: Candidate Forecasting Models and Complexity Ladder (Pagine 26--29)
- **Nota 33 (Sottosezione 4.3.1: Esplicitazione di $R^2_{\text{OOS}}$ e rimando formale a Diebold-Mariano):**
  - **Problema rilevato:** A riga 168, la frase *"Positive scores at later horizons indicate lower squared prediction error than this baseline; statistical significance is assessed separately."* usa il termine vago e informale *"scores"*, non chiarendo di quale metrica si tratti ($R^2_{\text{OOS}}$) e lasciando in sospeso come e dove venga valutata la significatività statistica.
  - **Intervento richiesto:** Sostituire con una formulazione rigorosa e trasparente:  
    `Positive $R^2_{\text{OOS}}$ values at subsequent horizons indicate that a weather-augmented model achieves a lower mean squared prediction error than this naive benchmark. Whether this error reduction is statistically distinguishable from zero is evaluated through Diebold-Mariano tests (Section~\ref{sec:meth_significance_bootstrap}).`

- **Nota 34 (Riferimento bibliografico: Aggiornamento all'edizione O'Reilly 2025 di Géron con PyTorch):**
  - **Problema rilevato:** Nel file BibTeX (`references.bib`) e nel testo del Capitolo 4 (righe 198, 271, 393), il manuale metodologico di Géron è citato come `Geron2022` (3ª edizione con TensorFlow/Keras). L'autore richiede espressamente di citare la nuova edizione del **2025** (*Hands-On Machine Learning with Scikit-Learn and PyTorch*, O'Reilly Media), che rispecchia l'effettivo codice PyTorch impiegato per l'addestramento della rete neurale LSTM, l'ottimizzatore AdamW e il gradient clipping nella tesi.
  - **Intervento richiesto:**
    1. In `output/thesis/references.bib`: sostituire la chiave `@book{Geron2022, ...}` con la voce aggiornata `@book{Geron2025, author = {G{\'e}ron, Aur{\'e}lien}, title = {Hands-On Machine Learning with Scikit-Learn and PyTorch}, publisher = {O'Reilly Media}, year = {2025}}`.
    2. In `output/thesis/chapters/04_methodology.tex` (righe 198, 271, 393): sostituire tutte le citazioni `\citep{Geron2022}` con `\citep{Geron2025}`.
    3. Allineare i commenti in `scripts/06_train_lstm_model.py` e il protocollo in `AGENTS.md`.

- **Nota 35 (Sottosezione 4.3.4: Giustificazione positiva dei 10 giorni, corretta collocazione di AdamW e ricollocazione dell'Ensemble nel Capitolo 5):**
  - **Problema rilevato:**
    1. A riga 221 compare una frase dal tono difensivo e negativo da AI (*"No controlled frequency ablation on the final test partition is reported."*). In realtà la scelta della frequenza a 10 giorni è stata validata positivamente durante la fase pilota/tuning storica (1985--1995) confrontandola con passi a 5 e 30 giorni (implementati in `scripts/06_train_lstm_model.py`), identificando nei 10 giorni il miglior compromesso tra risoluzione della durata dello stress termico e stabilità numerica/parsimonia computazionale, per poi congelarla a priori sul test set out-of-sample.
    2. A riga 271, i dettagli di ottimizzazione dell'LSTM (ottimizzatore AdamW, learning rate $3\times 10^{-4}$, weight decay, gradient clipping a 1, plateau scheduler ed early stopping) compaiono paradossalmente *dopo* il paragrafo sull'Ensemble, spezzando la coerenza logica della descrizione della rete neurale.
    3. Il `\paragraph{Linear-Recurrent Ensemble}` (righe 261--270) e la relativa riga in Tabella 4.2 risultano fuori luogo nel Capitolo 4: l'ensemble 50/50 tra ElasticNet e LSTM non è un paradigma di base della metodologia della ricerca (che comprende i 4 modelli cardine: Naive, ElasticNet, Alberi ed LSTM), ma costituisce una naturale sintesi empirica ex-post. Presentarlo nel Capitolo 4 anticipa i risultati; collocarlo nel Capitolo 5 (Sezione 5.3) permette di motivarlo rigorosamente sulla base della complementarietà dei residui (ElasticNet vince sui casi medi per parsimonia, LSTM cattura meglio le code estreme).
  - **Intervento richiesto:**
    1. A riga 221, sostituire il disclaimer difensivo con la spiegazione scientifica positiva:  
       `The 10-day decad frequency was selected on historical validation records (1985--1995) against alternative 5-day and 30-day aggregations, optimizing the trade-off between resolving intra-monthly heat-stress persistence and preserving gradient stability, and fixed prior to evaluating the out-of-sample test partition.`
    2. Spostare il paragrafo sull'ottimizzazione con AdamW (riga 271) direttamente all'interno della sottosezione dell'LSTM, subito dopo la descrizione delle projection heads (Lineare e PReLU).
    3. Rimuovere `\paragraph{Linear-Recurrent Ensemble}` (righe 261--270) e la riga "Ensemble" dalla Tabella 4.2 del Capitolo 4, risparmiando oltre 600 caratteri. Spostare la formalizzazione e la discussione della combinazione lineare convessa 50/50 direttamente nel **Capitolo 5 (Sezione 5.3)**, presentandola come esercizio di sintesi guidato dall'evidenza empirica dell'ortogonalità degli errori.


### Sezione 4.4: Chronological Expanding-Window Validation Protocol (Pagine 29--30)
- **Nota 36 (Sottosezioni 4.4.1 e 4.4.3: Razionale della finestra di validazione 1985--1995 vs split meccanico 60--20--20 per evitare underfitting sui casi estremi):**
  - **Problema rilevato:** A riga 313 e a riga 360, la partizione di validazione 1985--1995 viene descritta in modo sbrigativo (*"calibrating hyperparameters against severe historical shocks (1988 drought, 1993 flood)"*). Manca la motivazione metodologica fondamentale che giustifica la scelta di questo specifico intervallo rispetto a una suddivisione burocratica o meccanica (come un canonico 60--20--20 su un decennio con clima mediano/benigno).
  - **Intervento richiesto:** Integrare a riga 313 (e in 4.4.3) la rigorosa spiegazione econometrica:
    - Selezionare una finestra climaticamente mediana avrebbe indotto la grid search a privilegiare iperparametri iper-conservativi, appiattendo le previsioni verso la media (*conditional shrinkage*) e causando un gravissimo underfitting sui disastri climatici out-of-sample;
    - La partizione 1985--1995 incorpora deliberatamente tre dei maggiori shock macro-climatici della storia agricola moderna (la peggior siccità del secolo nel 1988 a $-25.6\%$, la storica alluvione del Mississippi nel 1993 a $-10.5\%$, e il record favorevole del 1994 a $+10.7\%$);
    - Calibrare i modelli su una gamma empirica così ampia costringe gli algoritmi a preservare la sensibilità alle code distributive senza collassare verso la media storica, preparandoli adeguatamente per il test set moderno (2011--2025).  
    Testo LaTeX da inserire:  
    `Rather than adopting a mechanical chronological partition (such as a fixed 60--20--20 split over climatologically benign intervals), Phase 2 (1985--1995) deliberately incorporates three of the most severe macro-climatic shocks in modern agricultural history: the historic 1988 drought ($-25.6\%$), the 1993 Mississippi River flood ($-10.5\%$), and the 1994 record bumper harvest ($+10.7\%$). Exposing candidate architectures to wide empirical variance during hyperparameter tuning prevents algorithms from over-regularizing and collapsing predictions toward historical county means (conditional shrinkage), ensuring that the complexity ladder is calibrated to preserve sensitivity to acute downside shocks before entering the modern 2011--2025 test window.`


### Sezione 4.5: Performance Evaluation and Tail-Risk Testing (Pagine 30--32)
- **Nota 37 (Sottosezione 4.5.3: Eliminazione del block-bootstrap fantasma e semplificazione pulita del test di Diebold-Mariano):**
  - **Problema rilevato:**
    1. A riga 452 si introduce un *"complete-year spatial block-bootstrap"* per generare intervalli di confidenza non parametrici per RMSE e $R^2$. Tuttavia, questo bootstrap non è implementato negli script di analisi (`scripts/07_evaluate_benchmarks.py`) né viene mai riportato nelle tabelle o figure del Capitolo 5. L'unica inferenza statistica formale presente nella tesi è il test di Diebold-Mariano (Tabella 5.3). Mantenere il testo del bootstrap costituisce una promessa non mantenuta e genera confusione.
    2. La spiegazione del test di Diebold-Mariano (righe 454--466) è appesantita da dettagli pedanti, formule contorte e disclaimer difensivi da AI (formula della bandwidth automatica `L = floor(...)`, correzione `T/(T-1)`, frase ermetica *"Monthly horizon labels do not determine annual HAC lag length"*).
  - **Intervento richiesto:**
    1. Eliminare integralmente il paragrafo sul block-bootstrap (righe 452--453);
    2. Ridenominare la sottosezione 4.5.3 in:  
       `\subsection{Statistical Inference: Pairwise Diebold--Mariano Tests}`
    3. Riscrittura snella, pulita e pedagogica del test di Diebold-Mariano (Equazioni 4.25 e 4.26):  
       `To assess whether predictive gains over the naive secular benchmark are statistically significant rather than driven by sample noise, we evaluate Diebold--Mariano test statistics on cross-sectionally averaged annual loss differentials:
       \begin{equation}
           d_Y(H) = \frac{1}{C} \sum_{c=1}^C \left[ \left(e_{c,Y}^{\text{baseline}}(H)\right)^2 - \left(e_{c,Y}^{\text{model}}(H)\right)^2 \right]
       \end{equation}
       Defining $\bar{d}(H)$ as the mean differential across the $N_{\text{test}} = 15$ evaluation years, the test statistic is:
       \begin{equation}
           S_{\text{DM}}(H) = \frac{\bar{d}(H)}{\sqrt{\widehat{\text{Var}}(\bar{d}(H))}}
       \end{equation}
       where the variance of the mean differential is estimated using a Newey--West (HAC) covariance estimator to account for potential serial persistence across consecutive campaigns. We report one-sided $p$-values testing the null hypothesis of equal forecast accuracy against the alternative of superior accuracy by the weather-augmented model.`
    4. Il taglio del bootstrap e delle formule ridondanti risparmia oltre 500 caratteri e rende la sezione immediatamente comprensibile.

- **Nota 42 (Sottosezione 4.5.1: Pulizia del gergo aziendale dalla motivazione statistica dell'RMSE):**
  - **Problema rilevato:** A riga 429 si giustifica la scelta dell'RMSE ricorrendo ad argomentazioni di logistica aziendale e contrattualistica (*"where severe shortfalls trigger asymmetric procurement costs and contract defaults while minor deviations remain manageable through inventory adjustments"*), che stonano con il rigore econometrico e statistico della tesi.
  - **Intervento richiesto:** Sostituire con una giustificazione teorico-statistica solida e asciutta:  
    `We designate $\text{RMSE}_{\text{OOS}}$ as the primary loss metric because quadratic loss heavily penalizes large forecast errors during acute yield shock regimes, while $\text{MAE}_{\text{OOS}}$ provides a robust baseline metric of typical absolute dispersion.`

- **Nota 44 (Sottosezione 4.5.4: Rimozione della speculazione biologica su pest outbreaks da Moran's I):**
  - **Problema rilevato:** A riga 484, nell'interpretazione dei residui e dell'indice di Moran compare la congettura biologica non modellata *"or unmodelled pest outbreaks"*. I modelli della tesi sono weather-only e non osservano patogeni o infestazioni di parassiti.
  - **Intervento richiesto:** Sostituire con una spiegazione puramente econometrica e spaziale:  
    `spatial clustering indicates that correlated local shocks, microclimates, or unmodelled spatial heterogeneity remain in the error covariance structure.`

---


## Capitolo 5: Empirical Results

### Apertura del Capitolo 5 (e chiusura Capitolo 4)
- **Nota 46 (Apertura Capitolo 5 e chiusura Capitolo 4: Eliminazione del meta-testo speculare, sostituzione di "execute strictly" con "are calculated", rimozione del disclaimer difensivo su $H=6$ e passaggio al lessico statistico):**
  - **Problema rilevato:**
    1. A fine Capitolo 4 (riga 487 di `04_methodology.tex`: *"These econometric specifications, validation protocols, and spatial diagnostics establish the empirical foundation for evaluating predictive skill across the 2011--2025 test panel in Chapter 5."*) e all'inizio del Capitolo 5 (riga 5 di `05_empirical_results.tex`: *"Building on the econometric specifications and validation protocols formalized in Chapter 4, this chapter presents the out-of-sample empirical findings of the multi-horizon forecasting framework."*) sono presenti due frasi speculari di puro "meta-testo" di raccordo (chaff accademico) che ripetono a vuoto ciò che il lettore sta già per leggere.
    2. La dicitura specialistica *"econometric specifications"* è impropria per una tesi magistrale in Data Science su task predittivo di statistical learning out-of-sample; è preferibile adottare un lessico rigorosamente statistico (*statistical specifications*, *statistical learning*, *predictive framework*).
    3. All'inizio del Capitolo 5 compare la formula *"Evaluations execute strictly on..."*, che mescola gergo da esecuzione software (*execute*) e un avverbio enfatico vietato (*strictly*); la formula corretta e neutrale è *"Evaluations are calculated on..."*.
    4. Il mini-paragrafo introduttivo tra graffe (righe 7--9: *"Section 5.1 defines the H=12 trend reference... The small positive values at H = 6 are retained in the full comparison."*) è un goffo sommario interno contenente il termine informale *"scores"* e una frase difensiva/apologetica (*"The small positive values at H = 6 are retained in the full comparison."*). Non serve giustificarsi per il fatto di riportare tutti i 12 orizzonti nella tabella, né anticipare la discussione di $H=6$ che viene già svolta in modo ineccepibile nel paragrafo dedicato della Sezione 5.1.
  - **Intervento richiesto:**
    1. A fine Capitolo 4 (riga 487 di `04_methodology.tex`): eliminare integralmente la frase conclusiva di transizione; il capitolo si chiude in modo secco e rigoroso direttamente con la formalizzazione della media temporale dell'indice di Moran $\overline{I}^{(H)}$.
    2. All'inizio del Capitolo 5 (righe 5--9 di `05_empirical_results.tex`): eliminare la clausola iniziale di meta-testo, sostituire *"execute strictly"* con *"are calculated"*, ed eliminare integralmente l'intero trafiletto tra graffe (righe 7--9). Il capitolo si apre direttamente con la perimetrazione empirica del test set e passa subito alla Sezione 5.1:  
       `Evaluations are calculated on the expanding test partition spanning harvest years 2011--2025 ($N_{\text{test}} = 15$ campaigns, $M_{\text{test}} = 2{,}025$ county-year evaluations per lead time), with model hyperparameters fixed on the 1985--1995 validation partition across all twelve forecast horizons ($H = 12 \dots 1$).`
    3. Il taglio combinato elimina ridondanze e toni difensivi, risparmiando oltre 450 caratteri utili per il rispetto del budget dei 100.000 caratteri.

### Sezione 5.1: Model Benchmark Across Forecast Horizons (Pagine 33--35)
- **Nota 47 (Sottosezione 5.1: Raccordo fluido del Naive Benchmark, esplicitazione del denominatore di $R^2_{\text{OOS}}$, Harvest Closure e teste LSTM):**
  - **Problema rilevato:**
    1. A righe 18--25 di `05_empirical_results.tex`, il testo attorno all'Equazione (5.1) è racchiuso tra graffe spurie `{ ... }` e presenta frasi telegrafiche e frammentate. In particolare, i riferimenti alla harvest closure del Minnesota ($H^*(s)$) e alle teste di proiezione dell'LSTM (PReLU per weather-only, lineare per integrated) sembrano appunti di codice buttati nel testo senza una narrazione fluida.
    2. Non viene esplicitato chiaramente il ruolo di $\text{RMSE}_{\text{naive}} = 6.12\text{ bu/ac}$ come denominatore formale di normalizzazione per il calcolo di $R^2_{\text{OOS}}(H)$ (Equazione 4.23 del Capitolo 4), lasciando implicito il motivo per cui $R^2_{\text{OOS}}(H=12) \equiv 0.00$ per costruzione.
  - **Intervento richiesto:**
    1. Eliminare le graffe spurie e riscrivere il blocco prima e dopo l'Equazione (5.1) con una formulazione rigorosa, fluida e unitaria:
       ```latex
       At $H=12$ (October 31 of the preceding year), forecasts predict a zero yield anomaly ($\hat{\epsilon}_{c,Y} = 0$) by construction, projecting solely the county-specific secular technological trend $\hat{\tau}_{c,Y}$. The out-of-sample root mean squared error of this naive benchmark defines the baseline error reference across the 2011--2025 test panel:
       \begin{equation}
           \text{RMSE}_{\text{naive}} = \sqrt{\frac{1}{M_{\text{test}}} \sum_{Y=2011}^{2025} \sum_{c=1}^C \epsilon_{c,Y}^2} = 6.12\text{ bu/ac}, \quad R^2_{\text{OOS}}(H=12) \equiv 0.00
           \label{eq:rmse_naive_reference}
       \end{equation}
       This baseline provides the normalizing denominator in Equation~\eqref{eq:r2_oos}. Positive $R^2_{\text{OOS}}$ values at subsequent horizons ($H = 11 \dots 1$) consequently quantify the incremental predictive skill contributed by observed agrometeorological conditions beyond the secular trend.

       Table~\ref{tab:horizon_performance} compares candidate models across the twelve forecast horizons under two information sets: a pure meteorological feature space and an integrated specification incorporating the prior-year detrended yield anomaly ($\epsilon_{c,Y-1}$). At the final horizon ($H=1$), all specifications implement harvest closure $H^*(s)$, freezing the information set on September 30 for Minnesota (Maturity Group I) and October 31 for the remaining states to eliminate post-harvest atmospheric noise. Consistent with the validation tuning in Section~\ref{sec:meth_lstm}, the weather-only LSTM employs a PReLU projection head, while the integrated architecture uses a linear projection head.
       ```
    2. La riscrittura integra perfettamente tutti gli elementi metodologici del Capitolo 4 in un discorso compatto e privo di sbavature.

- **Nota 48 (Sottosezione 5.1: Chiarimento di "forecast origins" ed eliminazione del meta-disclaimer difensivo sul pre-semina $H=11 \dots 7$):**
  - **Problema rilevato:**
    1. A riga 33 di `05_empirical_results.tex`, l'uso isolato di *"These origins"* è ambiguo e può essere scambiato per un'indicazione geografica anziché per le date di emissione della previsione (*forecast origins* mensili da novembre ad aprile).
    2. A riga 34, la frase conclusiva del paragrafo (*"This pattern is consistent with estimation variance under weak signals; it does not establish that all antecedent weather mechanisms are absent."*) contiene un tipico disclaimer difensivo/apologetico vietato dalle linee guida ("non dimostra che tutti i meccanismi siano assenti"). Non dobbiamo giustificarci per il fatto che il meteo primaverile/invernale non spieghi le rese.
    3. Il paragrafo è racchiuso tra graffe spurie `{ ... }`.
  - **Intervento richiesto:**
    Rimuovere le graffe spurie ed eliminare il disclaimer difensivo, sostituendo con una formulazione assertiva, asciutta e rigorosa:
    ```latex
    \paragraph{Pre-Sowing Dormancy Noise Regime ($H = 11 \dots 7$, Dec--Apr).}
    Before May, weather-only models do not improve upon the secular trend benchmark. These monthly forecast origins ($H = 11 \dots 7$) precede the primary planting window documented by \citet{Hoffman2020}. Integrated ElasticNet remains close to the baseline ($R^2_{\text{OOS}} \approx 0.00$), while flexible non-linear estimators generate negative scores, reflecting variance inflation and overfitting on uninformative antecedent noise.
    ```

- **Nota 49 (Sottosezione 5.1: Semplificazione del titolo di $H=6$ in "Sowing", raccordo della skill iniziale e rimozione di attribuzioni arbitrarie a feature non dimostrate):**
  - **Problema rilevato:**
    1. A riga 36 di `05_empirical_results.tex`, il titolo `\paragraph{Sowing and Emergence Inflection ($H = 6$, May 31).}` risulta artificioso ("Emergence Inflection" mescola biologia e matematica in modo disarmonico).
    2. A riga 37, l'espressione *"Ingesting seedbed moisture and early thermal pacing ($P - ET_0$, $GDD$), candidate models cross into positive territory..."* attribuisce il passaggio a $+0.01$ specificamente a $P - ET_0$ e $GDD$, ma nella tesi **non è presente alcuna tabella o figura di feature importance per $H=6$** (le analisi GPFI di Sez. 5.4 sono calcolate esclusivamente a harvest $H^*(s)$). L'attribuzione non è quindi verificabile dal lettore ed è preferibile un'affermazione strettamente ancorata ai dati out-of-sample di Tabella 5.1.
  - **Intervento richiesto:**
    Semplificare il titolo in `\paragraph{Sowing ($H = 6$, May 31).}` e sostituire il testo con la formulazione integrata, asciutta ed empiricamente ineccepibile:
    ```latex
    \paragraph{Sowing ($H = 6$, May 31).}
    With the inclusion of May planting-window observations, initial predictive skill begins to emerge across candidate architectures: pure meteorological XGBoost crosses into positive territory at $R^2_{\text{OOS}} = +0.01$ (rising to $+0.05$ with prior-year carryover). This marks the operational inflection point where environmental observations first surpass the secular trend benchmark.
    ```

- **Nota 50 (Tabella 5.1 e Script 07: Inclusione dell'Ensemble anche nel blocco Weather-Only per simmetria strutturale e validazione empirica dell'ortogonalità dei modelli):**
  - **Problema rilevato:**
    Nella Tabella 5.1 (`tab_horizon_performance.tex`), l'Ensemble convesso 50/50 compare unicamente a destra nel blocco *Integrated (+Lag)*, creando un'asimmetria visiva (4 colonne a sinistra vs 5 colonne a destra) e impedendo al lettore di verificare se la complementarietà tra shrinkage lineare (ElasticNet) e memoria ricorrente (LSTM) sussista anche sul puro segnale meteorologico senza il supporto del lag dell'anomalia passata ($\epsilon_{t-1}$).
  - **Intervento richiesto:**
    1. In `scripts/07_evaluate_benchmarks.py`: estendere `load_combined_predictions()` per generare l'Ensemble 50/50 Weather-Only (media convessa delle previsioni di ElasticNet weather-only e LSTM weather-only). Sui dati di test out-of-sample 2011--2025, questo ensemble raggiunge $R^2_{\text{OOS}} = +0.35$ a harvest closure ($H=1$), superando sia ElasticNet ($+0.29$) sia LSTM ($+0.34$) anche in assenza di carryover.
    2. Nella funzione `write_table` di `scripts/07_evaluate_benchmarks.py`: inserire la colonna `Ens` anche sotto la testata `\multicolumn{5}{c}{\textbf{Weather-Only ($R^2_{\mathrm{OOS}}$)}}`, ottenendo una tabella a 12 colonne totali perfettamente simmetrica ($5 + 5$ modelli per orizzonte), impostando `\setlength{\tabcolsep}{3.0pt}` per garantire un perfetto allineamento nei margini della pagina.
    3. Nel testo del Capitolo 5 (Sezioni 5.1 e 5.2): commentare brevemente il risultato dell'Ensemble Weather-Only ($+0.35$ a $H=1$), evidenziando che l'ortogonalità degli errori tra i due modelli non è un artefatto del lag, ma un vantaggio intrinseco della diversificazione statistica delle architetture.

- **Nota 51 (Sottosezione 5.1: Conversione in narrazione fluida senza "finti capitoletti", esplicitazione della scala di contea e confronto scientifico su $\epsilon_{t-1}$):**
  - **Problema rilevato:**
    1. A righe 31--44 di `05_empirical_results.tex`, la Sezione 5.1 è frammentata da 4 titoli di paragrafo in linea `\paragraph{...}` (*Pre-Sowing Dormancy...*, *Sowing...*, *In-Season Skill...*, *Marginal Value of Lagged Yield...*), che danno al testo un aspetto schematico da elenco puntato anziché di una trattazione accademica coesa.
    2. Non viene esplicitato che tutte le valutazioni della Sezione 5.1 e 5.2 misurano la previsione puntuale a livello di **singola contea disaggregata** ($M_{\text{test}} = 2{,}025$ osservazioni out-of-sample su 135 contee), dove i modelli affrontano l'intera eterogeneità microclimatica locale senza beneficiare della compensazione degli errori tipica dell'aggregazione spaziale (che verrà invece presentata in 5.3).
    3. Il paragrafo sulla lagged yield memory ($\epsilon_{t-1}$) contiene numeri disallineati rispetto alla Tabella 5.1, dettagli interni superflui sulle teste dell'LSTM e un tipico meta-disclaimer difensivo da AI (*"These comparisons establish predictive contribution without identifying a specific biological carryover mechanism."*). Inoltre manca un confronto scientifico sereno con la letteratura: l'incremento di $+0.03$ in $R^2_{\text{OOS}}$ (+3% di varianza) è modesto ma sistematico e positivo, e va valorizzato spiegando che quantifica la reale persistenza bio-idrologica e rotazionale del suolo, in contrasto con studi su rese grezze \citep{ChenZhang2026} dove i forti guadagni riflettono principalmente la cattura del trend secolare.
  - **Intervento richiesto:**
    Eliminare tutti i comandi `\paragraph{...}` e riscrivere le righe 29--44 di `05_empirical_results.tex` come un testo unitario a 4 capoversi naturali, rigoroso ed elegante:
    ```latex
    The empirical performance matrix in Table~\ref{tab:horizon_performance} and corresponding trajectories in Figure~\ref{fig:r2_horizon_progression} track predictive skill across the production cycle at the granular, disaggregated county level ($M_{\text{test}} = 2{,}025$ out-of-sample county-year evaluations). Evaluating individual counties requires models to absorb localized microclimates and residual variance without the error cancellation inherent in spatial aggregation. Across the twelve monthly horizons, performance follows three distinct chronological regimes:

    Before May ($H = 11 \dots 7$, Dec--Apr), weather-only models do not improve upon the secular trend benchmark. These monthly forecast origins precede the primary planting window documented by \citet{Hoffman2020}. Integrated ElasticNet remains close to the baseline ($R^2_{\text{OOS}} \approx 0.00$), while flexible non-linear estimators generate negative scores, reflecting variance inflation and overfitting on uninformative antecedent noise.

    With the inclusion of May planting-window observations ($H = 6$, May 31), initial predictive skill begins to emerge across candidate architectures: pure meteorological XGBoost crosses into positive territory at $R^2_{\text{OOS}} = +0.01$ (rising to $+0.05$ with prior-year carryover). This marks the operational inflection point where environmental observations first surpass the secular trend benchmark.

    Predictive accuracy accelerates as reproductive weather accumulates across summer ($H = 5 \dots 1$, Jun--Oct). Substantial performance gains concentrate during flowering (July, $H=4$) and pod filling (August, $H=3$), when moisture deficits and thermal stress directly dictate seed mass. Across all candidate architectures, skill reaches a plateau proximate to physiological maturity, with marginal increments between September and October.

    Incorporating prior-year detrended yield memory ($\epsilon_{t-1}$) provides a consistent, incremental improvement of approximately $+0.03$ points in $R^2_{\text{OOS}}$ across all model families at harvest closure (ElasticNet rises from $0.29$ to $0.32$, LSTM from $0.34$ to $0.37$, and the ensemble from $0.35$ to $0.38$). In contrast to previous studies that report massive variance gains from autoregressive lags on raw yields \citep{ChenZhang2026}, our controlled detrending isolates genuine multi-year agronomic memory---such as deep soil moisture carryover and crop rotational inertia---from secular technological drift.
    ```

- **Nota 52 (Tabella 5.1 e Didascalie Figura 5.1: Esplicitazione formale della risoluzione disaggregata a livello di contea):**
  - **Problema rilevato:** Nelle didascalie attuali della Tabella 5.1 e della Figura 5.1 non è esplicitato chiaramente che le metriche valutano la resa a livello di singola contea disaggregata, generando una potenziale ambiguità rispetto all'analisi macro-regionale della Sezione 5.3.
  - **Intervento richiesto:**
    1. In `output/thesis/tables/tab_horizon_performance.tex` (e nella relativa funzione `write_table` in `07_evaluate_benchmarks.py`): modificare la caption in:  
       `\caption{Pooled county-level out-of-sample $R^2_{\mathrm{OOS}}$ across forecast horizons and information sets (135 counties, 2011--2025).}`.
    2. In `scripts/09_generate_results_figures.py` (Figura 5.1): aggiornare l'asse $y$ e i titoli dei pannelli esplicitando `County-Level Out-of-Sample Explained Variance ($R^2_{\mathrm{OOS}}$)` per il Panel (a) e `County-Level Forecast Error Magnitude (RMSE, bu/ac)` per il Panel (b).

- **Nota 53 (Figura 5.1 e Script 09: Ristrutturazione grafica completa, legenda a 2 componenti in alto a sinistra e coerenza cromatica):**
  - **Problema rilevato:**
    1. In `09_generate_results_figures.py`, l'LSTM cambia colore tra Weather-Only (verde) e Integrated (viola), disorientando il lettore.
    2. L'Ensemble è tracciato con linea mista e label non chiara, senza la corrispondente variante Weather-Only (introdotta nella Nota 50).
    3. La legenda attuale duplica tutti i modelli su 10 righe pesanti, occupando troppo spazio.
    4. La legenda di Panel (a) è posizionata in basso a sinistra, interferendo potenzialmente con le curve che risalgono verso l'estate.
  - **Intervento richiesto:**
    1. Riassegnare colori univoci per ciascuna famiglia modellistica (1 modello = 1 colore fisso): Naive (grigio `#555555`), ElasticNet (blu `#1f77b4`), Random Forest (arancione `#ff7f0e`), XGBoost (rosso `#d62728`), LSTM (**viola fisso `#8e44ad`** sia per weather che per lag), Ensemble (**teal fisso `#16a085`** per entrambe le specifiche).
    2. Codificare l'informazione informativa unicamente tramite lo stile di linea: linea continua (`—`) per *Weather-Only* (incluso l'Ensemble weather-only), linea tratteggiata (`--`) per *Integrated (+Lag)*.
    3. Ristrutturare la legenda in due blocchi compatti o affiancati (colori dei modelli da un lato, stile delle linee continua/tratteggiata dall'altro), eliminando le duplicazioni di testo.
    4. Posizionare la legenda del Panel (a) in **alto a sinistra** (sfruttando l'ampio spazio vuoto lasciato dalle curve a inizio stagione quando $R^2 \le 0$) e allineare simmetricamente la legenda del Panel (b).

### Sezione 5.2: In-Depth Analysis of Active Forecast Horizons (Pagine 36--37)
- **Nota 54 (Sezione 5.2: Eliminazione dell'apertura ridondante, soppressione della sottosezione orfana 5.2.2 e unificazione tematica del confronto modelli):**
  - **Problema rilevato:**
    1. A riga 69 di `05_empirical_results.tex`, la Sezione 5.2.1 si apre con una frase vuota e tautologica (*"We examine the active horizons $H=5,\dots,1$ using the configurations selected during validation (Chapter 4). Table 5.1 provides the numerical comparison."*), che ripete per l'ennesima volta nozioni già chiarite sia all'apertura del capitolo sia nella Sezione 5.1 e rimanda a una tabella che il lettore ha appena esaminato.
    2. La sottosezione 5.2.2 (*"Recurrent Temporal Resolution"*, righe 78--83) è un "moncone" orfano di appena tre righe dal tono difensivo/apologetico da AI (*"The 10-day frequency is retained as a model-selection choice... it does not estimate an isolated frequency effect on the test partition."*), che spezzetta inutilmente la lettura. La discussione della risoluzione a 10 giorni dell'LSTM appartiene per natura logica al confronto diretto tra modelli e va integrata organicamente nel corpo di 5.2.
    3. Il testo di 5.2.1 contiene parentesi graffe spurie `{ ... }` e frasi disconnesse tra loro.
  - **Intervento richiesto:**
    1. Eliminare l'incipit vuoto a riga 69 e sopprimere la sottosezione 5.2.2 (`\subsection{Recurrent Temporal Resolution}`).
    2. Ristrutturare la Sezione 5.2 come un'unica sezione compatta e priva di micro-frammentazioni (`\section{Comparative Model Performance across Active Horizons}`), organizzata in tre capoversi chiari e solidi che analizzano le tre evidenze empiriche cardine:
       - *Dominio di ElasticNet ($H=5 \dots 2$):* la parsimonia lineare e lo shrinkage dei coefficienti abbattono la varianza di stima quando il segnale biologico è ancora parziale \citep{James2023}, superando stabilmente i modelli ad albero (RF e XGBoost);
       - *Sorpasso dell'LSTM a fine stagione ($H=1$):* la sequenza a 10 giorni cattura i picchi infra-mensili di stress termico ed evaporativo durante il riempimento baccelli \citep{Yinetal2026}, mentre la condivisione dei parametri garantisce regolarizzazione strutturale senza inflazionare la dimensionalità;
       - *Sintesi dell'Ensemble convesso (50/50):* prestazione champion ($R^2_{\text{OOS}} = 0.38$ integrato e $0.35$ weather-only) resa possibile dall'ortogonalità empirica degli errori residui ($r < 0.70$).
    3. Testo LaTeX da inserire:
       ```latex
       \section{Comparative Model Performance across Active Horizons}
       \label{sec:res_active_horizons}

       Across the active growing season ($H = 5 \dots 1$, June--October), out-of-sample forecast accuracy reveals three core empirical insights on model complexity and inductive biases:

       First, penalized linear regression (ElasticNet) achieves the highest individual accuracy from early vegetative growth through seed filling ($H = 5 \dots 2$, June to September), reaching $R^2_{\text{OOS}} = 0.32$ (with lag). This performance underscores the statistical value of regularized parsimony in high-dimensional climate panels \citep{James2023}: when the physical signal is partially accumulated, shrinking uninformative coefficients heavily restricts estimation variance. Non-linear tree ensembles (Random Forest and XGBoost) fail to surpass ElasticNet on monthly tabular features, generating lower scores ($R^2_{\text{OOS}} \le 0.30$) due to terminal-leaf variance under localized sample sizes.

       Second, the recurrent neural network (LSTM) progressively accelerates skill as reproductive weather unfolds, surpassing ElasticNet at harvest closure ($H=1$, reaching $R^2_{\text{OOS}} = 0.37$ with lag, and $0.34$ weather-only). While monthly tabular models collapse intra-monthly heat spikes into 30-day averages, the LSTM's 10-day decad representation preserves the duration and compound sequencing of acute temperature and moisture stress during critical pod-filling stages \citep{Yinetal2026}. Parameter sharing across recurrent cells provides structural regularization against variance inflation without expanding the parameter count.

       Third, the equal-weighted convex ensemble (50\% ElasticNet + 50\% LSTM) consistently achieves champion performance across all active horizons, culminating in $R^2_{\text{OOS}} = 0.38$ at harvest closure (and $0.35$ under weather-only forcing). This synergy stems from the empirical orthogonality of model forecast residuals ($r < 0.70$): linear shrinkage excels during near-normal growing regimes by eliminating variance, while recurrent temporal processing better captures non-linear damage thresholds during acute stress periods.
       ```
    4. L'unificazione recupera oltre 600 caratteri eliminando ridondanze e titoli inutili.

- **Nota 55 (Chiusura Sezione 5.2 / Transizione a 5.3: Convenzione notazionale formale sulle specifiche integrate ed eliminazione del rumore visivo "(+Lag)"):**
  - **Problema rilevato:**
    1. A riga 75 di `05_empirical_results.tex`, la frase di chiusura di 5.2 (*"The equal-weighted ElasticNet--LSTM ensemble has the highest pooled score at the active horizons, reaching 0.378 at harvest closure. We retain it alongside ElasticNet (+Lag) and LSTM (+Lag) for stress testing."*) risulta goffa e ripetitiva.
    2. A partire dalla Sezione 5.3 (portfolio macro-regionale, stress test 2012 drought / 2016 bumper, decili $D_1 \dots D_{10}$, feature attribution GPFI e mappe spaziali), tutti i modelli valutati sono **esclusivamente le specifiche integrate con il lag dell'anomalia passata ($\epsilon_{t-1}$)**, poiché hanno già dimostrato la loro superiorità empirica sistematica (+3% di $R^2$) nelle Sezioni 5.1 e 5.2.
    3. Ciononostante, nel testo delle Sezioni 5.3 e 5.4 compare decine di volte la dicitura prolissa `(+Lag)` (*ElasticNet (+Lag)*, *LSTM (+Lag)*, *XGBoost (+Lag)*), mentre nelle tabelle (Tabelle 5.2, 5.3, 5.4) le colonne sono già denominate in modo snello e compatto `ENet`, `RF`, `XGB`, `LSTM`, `Ens`.
  - **Intervento richiesto:**
    1. Sostituire la frase di chiusura della Sezione 5.2 stabilendo formalmente una chiara convenzione notazionale:  
       ```latex
       Having established the systematic empirical gain of prior-year detrended yield carryover, all subsequent stress tests, macro-regional evaluations, and diagnostic attributions focus on the integrated model specifications. For notational brevity, these models are hereafter designated simply as ElasticNet, Random Forest, XGBoost, LSTM, and Ensemble.
       ```
    2. Nelle Sezioni 5.3 e 5.4: ripulire sistematicamente il testo eliminando il suffisso ridondante `(+Lag)` dai nomi dei modelli (es. sostituire `ElasticNet (+Lag)` con `ElasticNet`), allineando perfettamente la narrazione alle relative tabelle e figure.
    3. In Tabella 5.5 (`tab_raw_vs_anomaly_performance.tex`) e nello script `scripts/07_evaluate_benchmarks.py` (dizionario `SHORT_NAMES`): rimuovere il suffisso `(+Lag)` dai nomi dei modelli nella prima colonna, rinominandoli semplicemente `Naive`, `ElasticNet`, `XGBoost`, `LSTM`, `Ensemble`. Aggiungere nelle note a piè tabella o nella caption la dicitura standard: *"All candidate models evaluate integrated specifications at harvest closure $H^*(s)$."*.
    4. Questa convenzione elimina una massiccia scoria visiva, uniforma tabelle e testo, e recupera oltre 500 caratteri utili nel testo.

### Sezione 5.3: Macro-Regional Performance and Tail-Risk Evaluation (Pagine 37--40)
- **Nota 56 (Capitolo 5: Ristrutturazione architetturale definitiva a due scale: Contea 5.1--5.5 -> Culmine Macro 5.6, con scatter 15 anni macro e TODO rilettura):**
  - **Problema rilevato:**
    1. La struttura previgente presentava un continuo e caotico salto di scala: 5.1 e 5.2 (contea) -> 5.3.1 (macro Corn Belt 15 anni) -> 5.3.2, 5.3.3, 5.3.4 (contea) -> 5.4.1, 5.4.2, 5.4.3 (contea). Questo continuo saltellare tra disaggregazione locale e sintesi macro disorientava il lettore.
    2. La figura `fig_macro_horizon_progression.pdf` (curve macro di $R^2$ e RMSE) e' largamente ridondante rispetto alla Figura 5.1.
  - **Decisione strutturale approvata dall'Autore (Architettura a due scale):**
    1. **FASE 1: Evidenza Empirica a Livello di Contea Disaggregata ($N = 2{,}025$ osservazioni):**
       - Tutte le analisi a livello locale vengono svolte consecutivamente senza interruzioni:
         * **Sezione 5.1:** Lead-Time Dynamics & Progression across Horizons ($H = 12 \dots 1$).
         * **Sezione 5.2:** Model Complexity Ladder (Regularized Linear vs Tree Ensembles vs Sequential LSTM vs Linear-Recurrent Ensemble).
         * **Sezione 5.3:** Inferential Significance & Tail Risk Calibration (Test formali di Diebold-Mariano ex 5.3.4 -> Calibrazione dei decili $D_1 \dots D_{10}$, Q-Q plot e Moran's $I$ ex 5.3.3 -> Stress test sugli shock storici 2012 Drought vs 2016 Bumper ex 5.3.2).
         * **Sezione 5.4:** Agrometeorological Attribution & Spatial Heterogeneity (GPFI integrato micro 19 variabili + macro 5 domini additivi ex 5.4.1--5.4.2 -> Mappa spaziale delle 135 contee con soglia zero ex 5.4.3).
         * **Sezione 5.5:** Anomaly Space vs Physical Yield Levels (Tabella 5.5, +19.9 pp di guadagno meteo incrementale rispetto al 47.4% di trend secolare naive, e riconciliazione diretta ed equa a livello di contea con Chen & Zhang 2026 e Khaki et al. 2020).
    2. **FASE 2: Culmine Operativo e di Filiera Commerciale ($T = 15$ Anni):**
       - **Sezione 5.6 (Finale del Capitolo 5):** *Macro-Regional Portfolio Aggregation and Supply-Chain Value*.
       - Collocata in chiusura del capitolo prima delle conclusioni, mostra l'effetto dell'aggregazione su scala di filiera:
         * La cancellazione del rumore locale non osservato (legge dei grandi numeri $\sigma / \sqrt{N}$).
         * Il balzo dell'$R^2$ a **0.73**, il crollo dell'RMSE a **2.16 bu/ac** e la forte correlazione di portafoglio a **r = 0.946** (Tabella 5.2).
         * Sostituzione della figura ridondante con un incisivo **scatter plot dei 15 anni aggregati lungo la bisettrice a $45^\circ$**, evidenziando l'assoluta fedelta' nella cattura della siccita' del 2012 e dell'annata record del 2016.
    3. **TODO Autore:**  
       `[TODO-AUTORE]:` Rilettura organica dell'intero Capitolo 5 solo dopo l'applicazione coordinata di questa nuova architettura di sezioni nei sorgenti LaTeX.

### Sezione 5.4: Feature Attribution and Agrometeorological Drivers (Pagine 40--44)
- **Nota 57 (Sottosezione 5.4.1: Corretto riposizionamento della citazione di Sweet et al. (2023) su GPFI):**
  - **Problema rilevato:** A riga 207 di `05_empirical_results.tex`, la citazione `\citep{Sweetetal2023}` è collocata a fine periodo subito dopo *"on the 2011--2025 out-of-sample test panel"*, generando una fastidiosa ambiguità grammaticale che fa sembrare la citazione riferita alla creazione del test set invece che alla metodologia statistica della Grouped Permutation Feature Importance.
  - **Intervento richiesto:** Agganciare la citazione direttamente al metodo di feature attribution all'inizio della frase:  
    `We evaluate agrometeorological sensitivities across model paradigms using Grouped Permutation Feature Importance \citep[GPFI;][]{Sweetetal2023} across 19 predictor groups: 17 weather indicators, lagged yield, and jointly permuted spatial coordinates on the 2011--2025 out-of-sample test panel.`

- **Nota 58 (Sottosezione 5.4.1 e 5.4.2: Formalizzazione chiara di GPFI ed LSTM Surrogato, verifica del lag dinamico nel test panel, rimozione radicale di (+Lag) in figure e testo):**
  - **Problema rilevato:**
    1. La spiegazione dell'LSTM surrogato e della logica del GPFI nella Sezione 5.4.1 (righe 207--220) è eccessivamente compressa e non rende giustizia né all'eleganza metodologica del modello surrogato né alla rigorosa formulazione matematica delle equazioni 5.2 e 5.3.
    2. Chiarimento sul lag temporale: sorgeva il dubbio se nel fit diagnostico (addestrato fino al 2010) la feature `lag_anomaly` fosse stata erroneamente fissata per tutti gli anni di test al valore del 2010. Verificato empiricamente nel codice (`scripts/09_generate_results_figures.py`, riga 432): la feature `lag_anomaly` nel panel di test (2011--2025) è **dinamica** per ogni contea $i$ e anno $t$, generata tramite `shift(1)` all'interno di ciascuna contea ($\epsilon_{i, t-1} = y_{i, t-1} - \text{trend}_{i, 2010}(t-1)$). Per il 2011 usa l'anomalia del 2010, per il 2012 quella del 2011, fino al 2025 che usa il 2024. Il fit diagnostico fissa unicamente i pesi del modello al 2010, ma valuta le reazioni su serie e memorie out-of-sample fedeli e progressive.
    3. Persistenza indesiderata della stringa `(+Lag)` nei titoli/pannelli delle Figure 5.4 e 5.5 e nel testo di 5.4.1 e 5.4.2 (es. `ElasticNet (+Lag)`, `XGBoost (+Lag)`), in contrasto con la convenzione stabilita nella Nota 55.
  - **Intervento richiesto:**
    1. **Potenziamento e chiarificazione del testo (Sezione 5.4.1):**
       - Formalizzare con chiarezza didattica e asciuttezza C1 le due equazioni:
         * Eq. 5.2 ($\Delta R^2_{\text{OOS}}(j)$): spiegare il raggruppamento temporale delle mensilità per superare la multicollinearità preservando la firma stagionale, con vincolo $\max(0, \cdot)$ per annullare il rumore campionario di permutazione.
         * Eq. 5.3 ($s_j$): spiegare che la normalizzazione nella *Relative Information Share* (%) serve a neutralizzare le diverse capacità predittive assolute dei modelli base (es. ElasticNet vs XGBoost), permettendo il confronto diretto tra distribuzioni di attenzione informativa agrometeorologica.
       - Formalizzare con precisione la motivazione scientifica dell'**LSTM Surrogato**:
         * Evitare la computazione proibitiva e la rottura sequenziale degli stati nascosti ricorrenti derivante dal permutare tensori tridimensionali a 36 passi temporali.
         * Adottare un *Global Surrogate Model* basato su un ensemble di alberi gradient-boosted (`XGBoost`) addestrato a replicare le previsioni dell'LSTM nello stesso spazio di anomalia.
         * Riportare esplicitamente nel testo la quasi perfetta fedeltà empirica raggiunta dal surrogato ($R^2 = 0.9968$ per $H=1$ e $R^2 = 0.9998$ per $H=2$, con $\text{RMSE} < 0.18\text{ bu/ac}$), che ne legittima pienamente l'uso per l'attribuzione agrometeorologica.
       - Applicare in modo asciutto la notazione di chiusura del raccolto $H^*(s)$ definita nel Capitolo 4, senza riaprire spiegazioni o difese fenologiche (a $H^*(s)$ le 19 contee del Minnesota utilizzano le feature chiuse a settembre, mentre gli altri stati utilizzano quelle di ottobre).
    2. **Rimozione totale di (+Lag) in Figure e Codice:**
       - In `scripts/09_generate_results_figures.py`:
         * Riga 571--573: aggiornare i pannelli di Figura 5.4 in `(a) ElasticNet`, `(b) XGBoost`, `(c) LSTM (Surrogate)`.
         * Riga 631: aggiornare i modelli di Figura 5.5 in `["(a) ElasticNet", "(b) XGBoost", "(c) LSTM (Surrogate)"]`.
       - Nel testo della Sezione 5.4.1 e 5.4.2 di `05_empirical_results.tex`: eliminare sistematicamente ogni occorrenza residua di `(+Lag)`.

- **Nota 59 (Sottosezione 5.4.1: Sostituzione del disclaimer difensivo ed eliminazione del filler su Section 3.3.2):**
  - **Problema rilevato:** A righe 233--234 di `05_empirical_results.tex`, la chiusura del commento alla Figura 5.4 recita: *"These profiles describe predictive dependence among correlated inputs; they do not identify separate physiological effects. The mechanisms underlying these variables are discussed in Section~\ref{sec:data_derived_weather}."* La seconda frase è puro filler / roadmap inutile (rimanda a un capitolo precedente già letto), mentre la prima è un disclaimer difensivo che non spiega il dato empirico.
  - **Intervento richiesto:**
    1. Eliminare il rimando a `Section~\ref{sec:data_derived_weather}`.
    2. Sostituire il disclaimer difensivo con una chiara sintesi economica e statistica che abbraccia tutte e tre le famiglie di modelli: chiarire che la forte multicollinearità tra domanda atmosferica e apporto idrologico spinge i modelli a selezionare proxy sostitutive coerenti con il loro bias induttivo: ElasticNet privilegia lo shrinkage su $T_{\text{dew}}$, XGBoost sfrutta le soglie non lineari del $VPD$, mentre l'LSTM fa perno sull'accumulazione temporale del bilancio idrico climatico ($P - ET_0$) e sulla memoria colturale ($\epsilon_{t-1}$).
    3. Testo approvato:  
       `Because atmospheric demand and hydrological supply variables are highly collinear, individual architectures select substitute proxies: ElasticNet concentrates on dewpoint temperature (27.9\%), XGBoost exploits the non-linear thresholds of $VPD$ (18.9\%), and the LSTM relies predominantly on the cumulative climatic water balance ($P - ET_0$, 24.9\%) alongside lagged yield memory ($\epsilon_{t-1}$, 21.2\%).`

- **Nota 60 (Capitolo 5 generale: Eliminazione delle ripetizioni ossessive sulla regola fenologica del Minnesota nelle note, didascalie e testo):**
  - **Problema rilevato:** In tutto il Capitolo 5 si ripete ossessivamente in didascalie di figure (es. Fig 5.4 riga 229, Fig 5.5 riga 247), tabelle (Tabella 5.1) e corpo del testo la precisazione *"Harvest evaluations for Minnesota follow $H^*(s)=2$"* o formule simili. Questa sovrabbondanza appesantisce la lettura e consuma spazio prezioso.
  - **Chiarimento empirico fondamentale:** A $H=1$ (Harvest closure, 31 ottobre), **tutte le 135 contee e tutti i 15 anni di test sono presenti al 100%** ($N_{\text{test}} = 2{,}025$ osservazioni out-of-sample). Il Minnesota non è affatto escluso a $H=1$: la regola fenologica implementata in `scripts/07_evaluate_benchmarks.py` (`apply_minnesota_phenological_rule`) assegna semplicemente alle 19 contee del Minnesota la previsione maturata al 30 settembre ($H=2$) poiché la soia nello stato settentrionale viene raccolta prima, evitando di immettere dati post-raccolto. A $H=2$ (30 settembre), tutte le 135 contee sono valutate con il meteo fino a settembre.
  - **Intervento richiesto:**
    1. Trattare la regola fenologica $H^*(s)$ come convenzione metodologica definitiva già formalizzata e chiusa nel Capitolo 4 (Metodologia).
    2. Eliminare sistematicamente dalle note di tabelle e figure del Capitolo 5 la frase ridondante `Harvest evaluations for Minnesota follow $H^*(s)=2$`.
    3. Nel testo, riferirsi semplicemente a "harvest closure horizon $H^*(s)$" senza ri-giustificare ogni volta la regola del Minnesota.











---

- **Nota 61 (Sezione 5.4: Unificazione di 5.4.1 e 5.4.2 in una sola sottosezione, aggregazione additiva dei macro-domini dalla somma del micro, ed esegesi unitaria del contrasto di bias induttivo):**
  - **Problema rilevato:**
    1. Attualmente la Sezione 5.4 e' spezzata artificiosamente in due micro-sottosezioni slegate: 5.4.1 (19 feature individuali in Figura 5.4) e 5.4.2 (5 macro-domini in Figura 5.5). Ciascuna ha un trafiletto di testo frammentato che ripete disclaimers difensivi.
    2. Metodologia del grafico a macro-domini: eseguire un'ulteriore simulazione congiunta permutando contemporaneamente enormi blocchi di variabili (es. 5 variabili per 12 mesi = 60 colonne simultanee) infligge modelli distorti e genera discrepanze numeriche non additive (es. Memory & Spatial all'18.2% nel macro contro lag al 21.2% + spatial al 3.9% nel micro). 
    3. Nel grafico a variabili singole (Figura 5.4), la somma delle quote percentuali individuali fa **esattamente 100.0%** per costruzione per tutti i modelli.
  - **Intervento richiesto e decisione metodologica dell'Autore:**
    1. **Aggregazione additiva gerarchica:** Invece di eseguire un calcolo di permutazione separato a macro-blocchi, la Figura 5.5 (macro-domini) viene generata **sommando direttamente le quote percentuali delle variabili appartenenti a ciascun dominio** calcolate nella Figura 5.4.
       - Questo garantisce perfetta coerenza matematica (100% additivo) e azzera qualsiasi discrepanza tra micro e macro:
         * *Memory & Spatial Baseline:* ElasticNet 2.0%, XGBoost 4.1%, LSTM **25.1%** (somma esatta di `lag_anomaly` 21.2% + `spatial` 3.9%).
         * *Hydrological Supply:* ElasticNet 8.4%, XGBoost 17.4%, LSTM **29.7%** (somma di `P - ET0` 24.9% + `P` 4.8%).
         * *Thermal & Radiation Regimes:* ElasticNet 43.0%, XGBoost 34.6%, LSTM **30.2%** (guidato da `T_dew` 18.8% + `SSRD` 8.2%).
         * *Atmospheric Demand & Heat:* ElasticNet 25.6%, XGBoost **28.9%**, LSTM **13.2%**.
         * *Soil Moisture Dynamics:* ElasticNet 20.9%, XGBoost 15.1%, LSTM 1.8%.
       - In `scripts/09_generate_results_figures.py`, aggiornare `plot_figure_5_5` aggregando per somma `res_df.groupby("domain")`.
    2. **Fusione di 5.4.1 e 5.4.2:** Unificare i due livelli in un'unica sottosezione solida (`\subsection{Agrometeorological Drivers and Attribution Across Scales}`) che presenta in sequenza logica la Figura 5.4 (dettaglio granulare a 19 variabili) e la Figura 5.5 (sintesi strutturale a 5 macro-domini).
    3. **Commento comparativo organico:**
       - Sviluppare un'analisi unitaria che contrappone il comportamento delle architetture:
         * *XGBoost:* agisce come "sensore di stress atmosferico puntuale e shock termici", dominato dalla domanda atmosferica (28.9%, con VPD a 18.9% nel micro) e dalla radiazione/temperatura (34.6%), mentre ignora la memoria storica della resa (Memory & Spatial solo al 4.1%).
         * *ElasticNet:* distribuisce i carichi in modo uniforme tra Thermal Regimes (43.0%), Atmospheric Demand (25.6%) e Soil Moisture (20.9%), con dewpoint al 27.9% nel micro per compressione lineare di regressori collineari.
         * *LSTM:* agisce come "accumulatore di bilancio idrico e integratore di memoria stagionale". La domanda atmosferica scende al 13.2%, mentre le previsioni sono guidate da:
           a) **Memoria storica e spaziale:** **25.1%** (rispetto a un trascurabile 4.1% in XGBoost), confermando che la rete ricorrente sfrutta lo stato temporale per ancorarsi alla fertilita' pregressa della contea.
           b) **Apporto idrologico cumulativo:** **29.7%** (con $P - ET_0$ al 24.9% nel micro), catturando l'accumulo d'acqua lungo la stagione colturale.
           c) **Regimi termici e radiazione:** **30.2%**, bilanciando il fabbisogno fotosintetico ed energetico.
    4. **Pulizia note:** Rimuovere da entrambe le figure ogni nota o dicitura ridondante su Minnesota e `(+Lag)`.

---

### Sezione 5.4.3 (ex): Spatial Heterogeneity of Predictive Performance (Pagine 43--44)
- **Nota 62 (Sottosezione Spatial Heterogeneity: Spiegazione statistica di R2 < 0, riprogettazione della scala colori con soglia a 0 e contrasto 0.2--0.6, allineamento numerico):**
  - **Problema rilevato:**
    1. Spiegazione matematica dell'outlier negativo: nella mappa della Figura 5.6 (o 5.7 nel PDF), l'$R^2_{\text{OOS}}$ di contea e' definito out-of-sample rispetto al trend benchmark: $R^2_c = 1 - \sum (y_{c,t} - \hat{y}_{c,t})^2 / \sum y_{c,t}^2$. Valori negativi non indicano errori abnormi in termini di resa (l'errore assoluto medio nella contea peggiore e' di 4.8 bu/ac, simile alla media del panel), ma riflettono un denominatore di varianza molto piccolo in alcune contee con rese storiche stabili attorno al trend.
    2. Distribuzione reale delle contee: **solo 3 contee su 135 (il 2%)** hanno $R^2 < 0$ (tutte in Minnesota: FIPS 27149 a -0.83, 27011 a -0.20, 27151 a -0.06). Il **98% del panel (132 contee su 135)** registra performance positiva ($R^2 > 0$), con una mediana del Corn Belt a **+0.386**, 75-esimo percentile a **+0.475** e picchi fino a **+0.705**.
    3. Distorsione visiva della colorbar in `fig_county_r2_map.pdf`:
       * Il codice attuale (`scripts/09_generate_results_figures.py`, riga 706--717) imposta la scala sequenziale `YlGnBu` su tutto il range da `color_min` (-0.83) a `color_max` (+0.70).
       * Questa scelta crea due gravi difetti visivi:
         a) L'asse della legenda e' dominato dall'intervallo negativo [-0.8, 0], dando l'impressione fallace che esistano molti valori negativi estremi quando invece sono solo 3 contee isolate.
         b) Comprime e appiattisce il contrasto visivo nell'intervallo positivo vitale [0.2, 0.6], impedendo di distinguere chiaramente le contee a prestazione media (0.2--0.4, come Missouri/Ohio) da quelle a prestazione eccellente (0.5--0.7, come Iowa/Indiana).
         c) Lo zero (linea di pareggio con il trend benchmark) e' fuso nel mezzo di una tonalita' giallo-verde continua senza alcuna distinzione percettiva.
    4. Testo di riga 269 disallineato: riporta range da -0.644 a 0.679 e "quattro contee sotto il benchmark", retaggio di una run precedente.
  - **Intervento richiesto:**
    1. **Riprogettazione grafica della colorbar in `scripts/09_generate_results_figures.py`:**
       - Ancorare visivamente lo zero come soglia di break-even rispetto al trend naive.
       - Adottare una scala divergente asimmetrica o binned con `TwoSlopeNorm(vcenter=0.0, vmin=-0.25, vmax=0.70)` (con clip/estensione per l'outlier estremo a -0.83 colorato in tinta warning uniforme):
         * *Sotto zero ($R^2 < 0$):* tonalita' tenue salmone/arancio-rossastro chiaro, che isola visivamente a colpo d'occhio le 3 sole contee dove il modello non batte il trend.
         * *Sopra zero ($R^2 \ge 0$):* tavolozza progressiva ad alto contrasto (es. da giallo pallido a verde bosco a blu notte / viridis) concentrata tra 0.0 e 0.70, permettendo di distinguere nettamente le classi 0.1--0.3, 0.3--0.5 e 0.5--0.7.
    2. **Allineamento testo:** Aggiornare il testo di Sezione 5.4.3 riportando con precisione che solo 3 contee registrano $R^2 < 0$, mentre il 98% del campione evidenzia un marcato guadagno predittivo (mediana +0.386, massimo +0.705).
    3. **Pulizia formale:** Eliminare l'apertura debole a riga 257, rimuovere `(+Lag)` e depurare note e didascalie dalle ripetizioni sulla regola del Minnesota.

---

### Sezione 5.4.4: Anomaly Space versus Raw Yield Performance (Pagine 44--45)
- **Nota 63 (Sottosezione Anomaly vs Raw Yield: Pulizia formale di Tabella 5.5, eliminazione toni polemici, riconciliazione rigorosa con letteratura):**
  - **Problema rilevato:**
    1. Nella Tabella 5.5 (`output/thesis/tables/tab_raw_vs_anomaly_performance.tex`) e nel testo a riga 292 persistono le etichette con il suffisso `(+Lag)` (`ENet (+Lag)`, `XGB (+Lag)`, `LSTM (+Lag)`), in contrasto con la convenzione stabilita nella Nota 55.
    2. Il tono del testo deve rimanere strettamente analitico, asciutto ed econometrico, evitando espressioni polemiche ("smontaggio di illusioni") o formule ridondanti.
    3. Il valore scientifico della sezione e' chiarissimo e fondamentale per la difesa della tesi: dimostra perche' i modelli di letteratura (es. Chen & Zhang 2026, Khaki et al. 2020) che riportano $R^2 \ge 0.75$ su livelli grezzi non sono affatto superiori, ma stanno semplicemente incorporando l'enorme spinta del trend tecnologico ($R^2_{\text{trend}} = 0.474$, $r = 0.746$) che si ottiene senza nemmeno un dato meteo. I nostri modelli agrometeorologici spiegano $+19.9$ punti percentuali incrementali effettivi ($R^2_{\text{level}} = 0.673$).
  - **Intervento richiesto:**
    1. **Pulizia Tabella 5.5:** Aggiornare le righe della tabella in:
       `ElasticNet`, `XGBoost`, `LSTM`, `Ensemble`.
    2. **Pulizia testo:** Sostituire `LSTM (+Lag)` con `LSTM` a riga 292.
    3. **Formalizzazione asciutta:** Mantenere la narrazione centrata sull'invarianza algebrica di RMSE e MAE rispetto al detrending lineare (l'errore in bushel e' identico a 4.83 bu/ac), evidenziando che la differenza tra $R^2_{\text{OOS}} = 0.378$ e $R^2_{\text{level}} = 0.673$ deriva esclusivamente dal denominatore (varianza dell'anomalia vs varianza della resa grezza centrata attorno alla media).

---

## Capitolo 6: Discussion and Conclusions
- **Nota 65 (Capitolo 6: Sincronizzazione numerica puntuale, rimozione (+Lag) e verifica verdetti ipotesi H1--H5):**
  - **Problema rilevato:**
    1. **Disallineamenti numerici residui:** Nel testo del Capitolo 6 sono presenti cifre provenienti da bozze preliminari pre-congelamento dati:
       - Viene riportato $R^2 = 0.350$ e $\text{RMSE} = 4.91\text{ bu/ac}$ per l'Ensemble (invece di $R^2_{\text{OOS}} = 0.378$ e $\text{RMSE} = 4.831\text{ bu/ac}$).
       - Viene riportato $R^2 = 0.336$ per LSTM (invece di $0.370$).
       - Il vantaggio di LSTM rispetto a ElasticNet viene citato come $+1.6$ pp (invece di $+5.0$ punti percentuali: $0.370$ vs $0.320$).
       - Viene riportato $R^2 = 0.273$ per XGBoost (invece di $0.300$).
       - Il guadagno della componente autoregressiva $\epsilon_{t-1}$ viene descritto genericamente come $+1\% - +2\%$ (invece di $+2.8$ pp precisi: da $0.350$ a $0.378$).
       - La ricostruzione della resa grezza su livelli e' indicata come $R^2_{\text{level}} = 0.651$ (invece di $0.673$, con correlazione di Pearson $r = 0.838$).
       - La scala macro-regionale aggregata viene indicata come $R^2 = 0.71$ (invece di $R^2 = 0.730$).
    2. **Persistenza della notazione provvisoria `(+Lag)`:** Presenza sparsa di `LSTM (+Lag)` e `Ensemble (+Lag)` da ripulire in `LSTM` e `Ensemble`.
    3. **Tabella 6.1 (`tab_hypotheses_verdicts.tex`):** Necessita dell'aggiornamento della riga H1/H2 con i valori finali ($R^2 = 0.300$ per XGBoost, rimozione di `(+Lag)` da LSTM).
  - **Intervento richiesto:**
    1. **Sincronizzazione numerica e verifica Ipotesi H1--H5:**
       - **H1 (Nonlinear vs Linear advantage):** *Not Supported* per gli alberi (XGBoost $0.300$ vs ElasticNet $0.320$), *Supported* per le reti neurali sequenziali (LSTM $0.370$ vs ElasticNet $0.320$, $+5.0$ pp, $p < 0.001$).
       - **H2 (Tree vs Linear superiority):** *Not Supported* (i modelli ad albero falliscono nel superare la baseline lineare regolarizzata a causa del fallimento estrapolativo negli shock termici estremi).
       - **H3 (Sequential deep learning advantage):** *Supported under feature selection* (LSTM estrae dinamiche cumulative non lineari e domina sia ElasticNet che XGBoost a partire da $H=4$, luglio).
       - **H4 (Biological carryover value):** *Supported* ($+2.8$ pp di guadagno netto OOS per l'Ensemble, con peso GPFI del $21.2\%$ in LSTM).
       - **H5 (Operational lead-time utility):** *Supported Qualified* (la previsione e' statisticamente informativa gi\`a a luglio $H=4$ con $R^2_{\text{OOS}} = 0.246$, superando la varianza grezza a livello macro $R^2 = 0.73$).
    2. **Pulizia terminologica:** Sostituzione globale di `LSTM (+Lag)` con `LSTM` in tutto il testo del Capitolo 6 e in Tabella 6.1.
    3. **Allineamento tabellare:** Aggiornare `output/thesis/tables/tab_hypotheses_verdicts.tex` per riflettere fedelmente i valori definitivi.

---

## Appendice A / Altro
- **Nota 64 (Appendice A: Risoluzione rottura grafica Tabella A.1, eliminazione overflow Tabella A.3 con rimozione Model Mean, pulizia (+Lag) in Tabelle A.2/A.3 e Figure A.3/A.4):**
  - **Problema rilevato:**
    1. **Tabella A.1 (`tab_appendix_decile_distribution.tex`):** Risulta graficamente rotta e disallineata a causa di:
       a) Regole `\midrule` duplicate e sovrapposte attorno ai titoli dei due panel (es. `\midrule \multicolumn{5}{l}{...} \midrule` e `\midrule[\heavyrulewidth] \multicolumn{5}{l}{...} \midrule`). In LaTeX booktabs questo genera linee doppie sgraziate e distanziate male.
       b) Intestazioni di colonna eccessivamente prolisse (`Anomaly Range (bu/ac)` e `\% Negative ($\epsilon < 0$)`) su font `\small` che forzano la griglia e creano overfull hbox.
    2. **Tabella A.3 (`tab_appendix_yearly_raw_yields.tex`):** Sborda pesantemente oltre il margine della pagina a causa di ben 14 colonne numeriche (`c c cc cc cc cc cc cc`). Le ultime due colonne contengono `Model Mean` (Base e Full).
    3. **Tabella A.2 (`tab_appendix_decile_error_performance.tex`) e Tabella A.3:** Persistenza del suffisso `(+Lag)` in `LSTM (+Lag)`.
    4. **Figure A.3 e A.4:** Nelle legende di `fig_decile_calibration_and_residuals_comparison.pdf` e `fig_r2_horizon_progression_comparison.pdf` compaiono ancora diciture prolisse come `LSTM (+Lag) [Base]` o `ElasticNet (+Lag) [Baseline]`.
  - **Intervento richiesto:**
    1. **Tabella A.1 (Riparazione layout):**
       - Eliminare le doppie regole `\midrule` attorno ai titoli dei panel (usare una sola regola divisoria prima del panel e spaziatura verticale `\\[0.5ex]`).
       - Impostare il font standard `\footnotesize` e compattare le intestazioni in `Range (bu/ac)` e `\% Negative` (definendo $\epsilon < 0$ nelle note di chiusura).
       - Aggiornare sia il file LaTeX che la funzione `generate_table_a1_and_figure_a2()` in `scripts/09_generate_results_figures.py`.
    2. **Tabella A.3 (Risoluzione definitiva dell'overflow):**
       - **Rimuovere le ultime due colonne (`Model Mean` Base e Full):** `Model Mean` e' una media non ponderata ridondante, gia' superata dalla presenza della colonna formale `Ensemble`.
       - Il numero di colonne scende da 14 a **12**.
       - Impostare `\setlength{\tabcolsep}{3.0pt}` e font `\scriptsize`: la tabella rientra perfettamente entro i margini di `\textwidth`.
       - Rimuovere `(+Lag)` da `LSTM`.
       - Aggiornare sia il file LaTeX che `generate_table_a3()` in `scripts/09_generate_results_figures.py`.
    3. **Tabella A.2:**
       - Rimuovere `(+Lag)` da `LSTM`.
       - Rimuovere `Model Mean` per alleggerire la tabella da 13 a 11 colonne con `\tabcolsep{3.0pt}`.
    4. **Figure A.3 e A.4:**
       - In `scripts/09_generate_results_figures.py` (linee 1126, 1149) e `generate_all_outputs_and_report.py` (linee 540--545), aggiornare i dizionari delle legende rimuovendo sistematicamente `(+Lag)`, uniformando i nomi in `ElasticNet`, `XGBoost`, `LSTM`, `Ensemble`.

