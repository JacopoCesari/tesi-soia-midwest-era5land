# Dati Agronomici, Selezione del Panel e Definizione del Target

## 1. Fonte dei Dati Agronomici e Pulizia

I dati statistici storici a livello di contea provengono dal servizio ufficiale **USDA NASS (National Agricultural Statistics Service) Quick Stats**.

### Variabili Selezionate
- **Variabile Target Primaria:**
  `SOYBEANS - YIELD, MEASURED IN BU / ACRE`
  Rappresenta la resa media della soia per acro raccolto a livello di contea, ottenuta esclusivamente dalle indagini annuali di tipo **SURVEY**.
- **Superficie di Riferimento:**
  `SOYBEANS - ACRES HARVESTED`
  Superficie effettivamente raccolta espressa in acri.

### Criteri di Pulizia ed Esclusioni
1. **Esclusione di unita aggregate non geografiche:** Sono state rimosse le categorie residue aggregate (`OTHER COUNTIES`, `OTHER (COMBINED) COUNTIES`) poiché non riconducibili a una precisa geometria amministrativa.
2. **Esclusione di definizioni eterogenee di resa:** Sono state escluse le rilevazioni su base `NET PLANTED ACRE` o dati parziali non conformi allo standard SURVEY.
3. **Separazione dei dati CENSUS:** I censimenti quinquennali dell'agricoltura (CENSUS) sono mantenuti distinti dalle serie storiche annuali SURVEY e non vengono fusi automaticamente per evitare discontinuità metodologiche.
4. **Normalizzazione dei codici FIPS e armonizzazione storica:** Ciascuna contea è identificata univocamente tramite il proprio codice FIPS a 5 cifre (`STATE_FIPS` + `COUNTY_FIPS`). È stata inoltre normalizzata la discontinuità storica di **Ste. Genevieve County, Missouri**, ricondotta stabilmente al codice FIPS `29186` per evitare che la medesima entità geografica venisse conteggiata come due unità distinte nel tempo.

---

## 2. Selezione dell'Area Geografica e dei Sei Stati

La selezione iniziale non si è basata sulla sola graduatoria produttiva contemporanea (che avrebbe sovrarappresentato la geografia agricola recente a scapito della continuità storica), ma sulla presenza storicamente consolidata e continuativa della coltivazione della soia.

I sei stati selezionati rappresentano il nucleo storico del Corn Belt / Midwest statunitense:
1. **Illinois (IL)**
2. **Indiana (IN)**
3. **Iowa (IA)**
4. **Minnesota (MN)**
5. **Missouri (MO)**
6. **Ohio (OH)**

---

## 3. Costruzione del Balanced Panel 1950–2010

Per evitare le distorsioni tipiche dei panel sbilanciati (in cui la composizione del campione varia annualmente, introducendo variazioni spurie della resa aggregata non imputabili al clima), è stato imposto un vincolo di bilanciamento rigoroso:
- Una contea entra nel dataset principale se e solo se possiede ininterrottamente **sia il dato di yield sia il dato di acres harvested per ogni singolo anno** dell'orizzonte prescelto.

### Risultato del Vincolo di Bilanciamento
Il periodo **1950–2010** (coerente con l'inizio della disponibilità delle reanalisi ERA5-Land nel 1950) massimizza congiuntamente l'ampiezza temporale e il numero di unità geografiche:
- **Anni di osservazione:** 61 (1950–2010).
- **Contee incluse:** **479 contee**.
- **Dimensioni del panel principale:** **29.219 osservazioni county-year**.
- **Valori mancanti:** **0%**. Nessuna imputazione statistica è necessaria per il target o per la superficie nel periodo 1950–2010.

L'elenco univoco delle 479 contee candidate è tracciato nel file `balanced_panel_candidate_counties.csv` e nel foglio `county_list_479` dei file Excel originali.

---

## 4. Gestione Separata degli Anni 2011–2025

A partire dal 2011, modifiche nei protocolli di campionamento e di pubblicazione del programma USDA NASS County Estimates hanno determinato una presenza progressiva di valori mancanti per alcune contee.

Per preservare l'integrità scientifica dell'analisi:
- **Nessuna imputazione automatica:** i dati degli anni 2011–2025 non vengono imputati artificialmente.
- **Conservazione separata:** le osservazioni SURVEY disponibili per le 479 contee negli anni 2011–2025 sono conservate nel foglio dedicato `all_years_survey` dei file Excel:
  - `01_Dati_Soia_e_Target/soybean_yield_479_counties_1950_2025.xlsx`
  - `01_Dati_Soia_e_Target/soybean_acres_479_counties_1950_2025.xlsx`
- **Ruolo metodologico:** il periodo 2011–2025 **non fa parte del balanced panel principale**. È riservato per analisi di sensitività, test di robustezza out-of-sample o future estensioni su panel sbilanciato.
- **Non descrivere mai il periodo 1950–2025 come un panel bilanciato.** La copertura 1950–2025 riguarda esclusivamente la disponibilità delle serie meteorologiche ERA5-Land.

---

## 5. Definizione Operativa dei Target

### 5.1 Target Continuo Primario
Il target primario è la resa continua della soia. Vengono utilizzate due formulazioni complementari:
1. **Resa continua grezza ($Y_{c, t}$):** espressa in `bu/acre` (e convertibile in `t/ha` con il fattore standard $1\text{ bu/acre} \approx 0{,}06725\text{ t/ha}$).
2. **Anomalia di resa de-trendizzata ($\tilde{Y}_{c, t}$):**
   $$\tilde{Y}_{c, t} = Y_{c, t} - \hat{Y}_{\text{trend}}(c, t)$$
   La componente di trend tecnologico e varietale $\hat{Y}_{\text{trend}}(c, t)$ cattura il continuo miglioramento genetico e agronomico avvenuto tra il 1950 e il 2010.

### 5.2 Regola di De-trendizzazione Rigorosamente Train-Only
Per evitare forme subdole di **data leakage temporale**, la stima del trend tecnologico (che sia un trend lineare per stato/contea, quadratico o una spline/GAM) deve essere calibrata **esclusivamente sui dati storici appartenenti al set di addestramento dell'expanding window corrente**:
$$\hat{Y}_{\text{trend}}(c, t_{\text{test}}) = \mathcal{M}_{\text{trend}}\Big(\{(c, \tau, Y_{c, \tau})\}_{\tau < t_{\text{test}}}\Big)$$
È vietato calcolare un trend globale sull'intera serie storica prima della suddivisione temporale train/test.

### 5.3 Target Complementare: Classificazione Tripartita
Accanto alla stima puntuale della resa, viene costruito un indicatore qualitativo su tre classi:
- **Low Yield:** anni di stress severo o calo significativo di resa.
- **Normal Yield:** resa in linea con le attese storiche.
- **High Yield:** annate favorevoli con resa superiore al trend.

**Vincolo anti-leakage sulle soglie:** le soglie quantitative di demarcazione (es. terzili empirici dell'anomalia o deviazioni rispetto alla deviazione standard storica) devono essere calcolate **esclusivamente sul set di training**. In nessun caso le soglie possono essere determinate utilizzando dati di test o distribuzioni dell'intero periodo.

---

## 6. Risoluzione Temporale ed Elaborazioni Meteorologiche Collegate

Per collegare le serie agronomiche annuali ai dati climatici:
- Le variabili meteorologiche di base mantengono una risoluzione giornaliera esatta (finestra sincronizzata UTC 00:00–23:59).
- Il confronto metodologico principale per la modellazione prevede aggregazioni a **5 giorni** e **10 giorni** a partire dal primo giorno della stagione agronomica fino al cutoff di riferimento.
- L'aggregazione a **30 giorni** (mensile) è mantenuta unicamente come benchmark opzionale a bassa risoluzione.
- Non è prevista né richiesta la conservazione di un archivio meteorologico orario persistente.
