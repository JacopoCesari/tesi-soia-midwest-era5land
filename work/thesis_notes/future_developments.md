# Sviluppi Futuri e Prospettive di Ricerca per il Capitolo 6

> **Stato del Documento:** Temporaneo di lavoro — Guida per la stesura del Capitolo 6  
> **Ruolo Operativo:** Questo file raccoglie i 5 punti di sviluppo futuro e le considerazioni operative per la discussione. Man mano che questi contenuti vengono integrati e scritti nel Capitolo 6 della tesi, vengono smarcati o rimossi da qui, confluendo definitivamente nel testo finale e nella memoria consolidata.  
> **Riferimento Stabile:** Per i dati e il protocollo metodologico permanente, consultare [`consolidated_memory.md`](./consolidated_memory.md).

---

## 1. I Cinque Confini Metodologici e relative Traiettorie Evolutive

### 1.1 Integrazione Multimodale con Remote Sensing ad Alta Risoluzione
- **Confine Attuale**: Il protocollo è rigorosamente "weather-only" per isolare il forcing climatico puro ed evitare endogeneità.
- **Sviluppo Futuro**:
  - Integrare serie temporali satellitari multispettrali ad alta risoluzione spaziotemporale (Sentinel-2 a 10 metri, MODIS/VIIRS a 250–500 metri).
  - Adottare indici di risposta biologica avanzati: **NIRv** (Near-Infrared Reflectance of Vegetation) e **SIF** (Solar-Induced Chlorophyll Fluorescence da sensori TROPOMI/OCO-2), in grado di quantificare l'attività fotosintetica effettiva e il danno da siccità ben prima della comparsa della clorosi fogliare visibile.
  - *Architettura target*: Modello a due rami (Dual-Stream Encoder) con un ramo meteorologico per l'accumulo idrotermico e un ramo ottico per lo stato reale della canopia.

### 1.2 Previsioni Dinamiche delle Date di Semina (Sowing Dates)
- **Confine Attuale**: La campagna previsionale adotta un calendario solare fisso (1° novembre dell'anno $Y-1$ – 31 ottobre dell'anno $Y$).
- **Sviluppo Futuro**:
  - Superare la griglia a mesi solari ancorando l'aggregazione temporale a coordinate fenologiche relative (**Days After Sowing**, DAS, o unità termiche cumulate post-semina).
  - Stimare la data effettiva di semina a scala di contea integrando i bollettini settimanali USDA NASS *Crop Progress* o algoritmi di rilevamento della transizione verde da satellite (*Green-up detection*).
  - *Vantaggio agronomico*: Consente di normalizzare le annate anomale in cui piogge primaverili persistenti ritardano le semine di 3–4 settimane (come avvenuto nel 1974 e nel 2019), evitando che il calore di luglio impatti stadi vegetativi non allineati.

### 1.3 Accoppiamento con Seasonal Dynamical Climate Forecasts (ECMWF SEAS5)
- **Confine Attuale**: I modelli valutano unicamente l'informazione meteorologica osservata fino alla data di cut-off $H$; nessun dato futuro o previsione stagionale è impiegato.
- **Sviluppo Futuro**:
  - Per gli orizzonti pre-stagionali e primaverili precoci ($H \ge 6$, da novembre ad aprile), colmare il vuoto predittivo estivo alimentando le reti con le proiezioni climatiche d'insieme a lungo raggio dei modelli fisici globali (ECMWF SEAS5, NOAA CFSv2, NMME).
  - *Metodologia ibrida*: Ingestione delle distribuzioni probabilistiche di anomalia termica e precipitativa previste per luglio e agosto, consentendo ai modelli ML di quantificare il rischio di siccità estiva prima ancora che la coltura sia seminata.

### 1.4 Modellistica Ibrida Process-Based e Machine Learning (Physics-Informed ML)
- **Confine Attuale**: L'approccio è puramente data-driven: gli algoritmi deducono empiricamente le relazioni tra forzanti meteorologiche e anomalie di resa.
- **Sviluppo Futuro**:
  - Accoppiare modelli biofisici di simulazione di crescita colturale basati su processi (*APSIM, DSSAT*) con reti neurali o gradient boosting.
  - *Schema ibrido*: Il simulatore biofisico calcola la crescita potenziale della pianta, la traspirazione e l'indice di area fogliare (LAI) simulando la fisiologia nota del suolo e dell'azoto; l'algoritmo di machine learning apprende a predire lo scostamento residuo, intercettando gli stress estremi e le interazioni climatiche non riprodotte dai parametri biofisici standard.

### 1.5 Architetture Spazio-Temporali Native (Graph Neural Networks & Spatio-Temporal Transformers)
- **Confine Attuale**: L'LSTM sequenziale elabora la serie temporale e riceve le coordinate geografiche continue concatenate; la continuità spaziale tra contee contigue è appresa implicitamente.
- **Sviluppo Futuro**:
  - Implementare modelli geometrici e a grafo (**Spatial-Temporal Graph Convolutional Networks, ST-GCN**).
  - Modellare esplicitamente la matrice di adiacenza fisica e topologica tra le contee della Corn Belt, permettendo all'informazione di propagarsi nello spazio in funzione delle barriere idrografiche e dei bacini pedologici omogenei.
  - Applicare meccanismi di auto-attenzione spazio-temporale su sequenze continue non aggregate a frequenza giornaliera.

---

## 2. Requisiti e Limiti Operativi per l'Applicazione Industriale Reale

Oltre ai fronti metodologici accademici, il Capitolo 6 discuterà le barriere applicative per l'impiego reale del sistema nelle sale trading e nelle centrali acquisti agroindustriali:

1. **Latenza dei Dati e Reanalisi Operativa**:
   - *ERA5-Land standard* consolida le stime finali con un ritardo di 2–3 mesi, rendendolo inapplicabile per il trading live intra-stagionale.
   - *Soluzione industriale*: Utilizzare lo stream operativo preliminare **ERA5T** (disponibile con una latenza di soli 5 giorni) o rianalisi ad alta frequenza NCEP RTMA / NOAA HRRR, quantificando l'impatto predittivo del disallineamento tra dati preliminari e rianalisi consolidata.
2. **Trasferibilità a Nuovi Bacini Geografici e Altre Colture**:
   - Estendere il framework multi-orizzonte ad altri grandi bacini esportatori (Brasile - Mato Grosso e Paraná; Argentina - Pampa) e al mais (*corn*), verificando se la sensibilità fenologica concentrata ad agosto rimanga valida o se il mais anticipi la finestra critica a luglio durante la fase di emissione del pennacchio (*silking*).
3. **Valutazione del Survivorship Bias nel Panel Selezionato**:
   - Discutere criticamente il trade-off del panel bilanciato a 135 contee: massimizzare la profondità temporale (75 anni continui dal 1951 al 2025 senza dati mancanti) preserva la robustezza statistica sugli shock rari, ma esclude contee marginali della periferia meridionale o occidentale del Midwest, dove la soia viene coltivata su terreni a minore capacità di ritenzione idrica.
