# Piano Operativo Colloquio e Chiusura Tesi
*Stato di avanzamento, nodi decisionali per il professore e roadmap esecutiva*
*Data: Ottobre 2026*

---

## 1. Stato di Avanzamento Reale (Cosa è Fatto vs Cosa Manca)

### COMPLETATO AL 100% (Codice & Risultati Empirici)
* [x] **Panel dati & Feature**: 75 anni (1951–2025), 135 contee bilanciate, 17 variabili meteo ERA5-Land riaggregate su 12 orizzonti mensili ($H=12 \dots 1$).
* [x] **Protocollo di Validazione**: Finestra espandibile (expanding window) con test cieco su 15 anni (2011–2025), $2.025$ osservazioni contea-anno per orizzonte. Iperparametri congelati su 1985–1995.
* [x] **Modelli Tabulari & LSTM**: ElasticNet, Random Forest, XGBoost e LSTM sequenziale con attenzione decadica (10 giorni), con e senza lag di carryover ($\epsilon_{t-1}$).
* [x] **Tabella 5.1 (Matrice $R^2$ OOS $H=12 \dots 1$)**: Delineata la Noise Boundary ($H \ge 6$, $R^2 \le 0$) e la transizione attiva ($H \le 5$, LSTM campione a $R^2 = 0.349$).
* [x] **Tabella 5.2 (Stress Test Shock 2012 e 2016)**: Dimostrata l'anticipazione della siccità 2012 a luglio (bias abbattuto da $+4.07$ a $+0.26$ bu/ac).
* [x] **Tabella 5.3 (Performance per Decili $D_1 \dots D_{10}$)**: Calcolata la diagnostica su $10.125$ valutazioni della stagione attiva.
* [x] **Suite di Replicazione**: 9 script Python autonomi e documentati (`thesis_scripts/01`--`09`).
* [x] **Capitoli 1, 2, 3, 4, 5 scritti in LaTeX**: Testo, equazioni, tabelle e figure inseriti e compilati in `output/thesis/`.

### COSA RIMANE DA FARE (Per Chiudere la Tesi)
* [ ] **Capitolo 6 (*Discussion*)**: Da scrivere in LaTeX (`output/thesis/chapters/06_discussion.tex`).
* [ ] **Capitolo 7 (*Conclusions*)**: Da scrivere in LaTeX (`output/thesis/chapters/07_conclusions.tex`).
* [ ] **Decisione su Modelli/Metriche Aggiuntive**: Chiedere al professore se vuole fermare qui la modellazione o aggiungere un'analisi descrittiva di regime/classificazione.
* [ ] **Revisione Finale e Consegna**: Controllo bibliografia, frontespizio ufficiale UNIBO e impaginazione finale.

---

## 2. La Tabella 5.3 al Colloquio: Dati Nudi e Crudi

Porta al professore la **Tabella 5.3**. I numeri dicono esattamente questo:

```
Decile      Range Anomalia (bu/ac)   Naive RMSE   LSTM RMSE   Miglior Modello
─────────────────────────────────────────────────────────────────────────────
D1 (Drought)    [-22.85, -5.22]         9.35        7.17      LSTM  (-2.18 bu/ac)
D2              [-5.15,  -1.95]         3.55        3.68      Random Forest (3.38)
D3 (Centrale)   [-1.92,  +0.02]         1.08        2.62      NAIVE (1.08)
D4 (Centrale)   [+0.02,  +1.71]         1.04        2.23      NAIVE (1.04)
D5 (Centrale)   [+1.72,  +3.15]         2.51        3.11      NAIVE (2.51)
D6              [+3.16,  +4.40]         3.82        3.44      XGBoost (3.41)
D7              [+4.42,  +5.59]         5.03        4.34      XGBoost (4.09)
D8              [+5.60,  +6.83]         6.30        5.25      ElasticNet (5.23)
D9              [+6.84,  +8.83]         7.80        6.22      LSTM (6.22)
D10 (Bumper)    [+8.83, +22.93]        11.21        9.57      ElasticNet (9.31)
─────────────────────────────────────────────────────────────────────────────
OVERALL         [-22.85, +22.93]        6.12        5.26      LSTM (5.26)
```

### Perché il Naive vince al centro ($D_3\text{--}D_5$):
* Negli anni normali la deviazione dal trend è $\pm 1\text{ bu/ac}$. Prevedere zero dà un errore di $1.0\text{ bu/ac}$.
* Il meteo in condizioni ottimali non limita la resa (Legge di Liebig). I modelli ML provano a fittare piccole oscillazioni e introducono varianza di stima ($MSE = Bias^2 + Var$). Poiché il vero segnale meteo è zero, il Naive (che ha varianza zero) vince matematicamente.

### Perché il Machine Learning serve nelle code ($D_1$ e $D_{10}$):
* Negli eventi estremi il clima diventa il fattore vincolante.
* In $D_1$ (siccità) il Naive fa un errore di $9.35\text{ bu/ac}$. L'LSTM lo riduce a $7.17\text{ bu/ac}$ (guadagno netto di oltre $2.1\text{ bu/ac}$).
* Nella gestione del rischio reale (supply chain, trading, stoccaggio), l'errore di $1\text{ bu/ac}$ negli anni normali non genera costi. L'errore di $9\text{ bu/ac}$ nella siccità provoca rotture di stock e inadempienze contrattuali.

---

## 3. Le 3 Domande Operative Chiuse per il Professore

Non aprire discussioni generiche. Fagli queste tre domande dirette per ottenere decisioni operative:

### Domanda 1: Perimetro della Modellazione (Chiudiamo o aggiungiamo?)
> *"Professore, la pipeline di regressione continua sui 15 anni di test è completata e i risultati sono nei Capitoli 4 e 5. Ritiene che la tesi sia completa dal punto di vista computazionale e che possiamo passare direttamente alla stesura dei Capitoli 6 (Discussione) e 7 (Conclusioni), oppure vuole che produca una tabella aggiuntiva?"*

* **Se dice: "Basta così, scrivi i capitoli"** $\to$ Risparmi tempo, passi subito al testo.
* **Se dice: "Vorrei vedere una classificazione o altro"** $\to$ Vai alla Domanda 2.

### Domanda 2: Se vuole la classificazione (Early Warning vs Retraining)
> *"Se ritiene utile una metrica di classificazione di regime, preferisce:*
> *a) Applicare una matrice di Early Warning (Precision/Recall su soglia shock $<-10\%$) direttamente sui valori già predetti a luglio ($H=4$), senza toccare i modelli? (Tempo: 1 ora)*
> *b) Oppure reimpostare un modello ad albero addestrato nativamente con loss di classificazione binaria/multiclasse? (Tempo: 1-2 giorni)"*
*(Nota tecnica: l'opzione a) è preferibile perché confronta direttamente la stessa variabile continua con una soglia operativa senza rischio di etichette arbitrarie).*

### Domanda 3: Struttura del Capitolo 6 (Discussione)
> *"Nel Capitolo 6 intendo concentrare la discussione su 3 punti:*
> *1. L'asimmetria del valore economico (Tabella 5.3: il modello come polizza di assicurazione contro le code, non ottimizzatore della media).*
> *2. Il ritardo operativo di ERA5-Land (5 giorni per ERA5T vs 2-3 mesi per i dati consolidati).*
> *3. Il survivorship bias delle 135 contee bilanciate.*
> *Condivide questo impianto o vuole che dia risalto a qualche altro aspetto specifico?"*

---

## 4. Tabella di Marcia verso la Consegna (Roadmap Operativa)

| Step | Attività Concreta | Output / File Coinvolti | Tempo Stimato |
| :---: | :--- | :--- | :---: |
| **1** | **Colloquio con il Professore** | Presentare Tabelle 5.1, 5.2, 5.3 e ottenere l'ok su Domande 1-3. | 30 min |
| **2** | **Eventuale script di Early Warning (se richiesto)** | Generare matrice di confusione su soglia shock 2012 al 31 luglio. | Mezza giornata |
| **3** | **Scrittura Capitolo 6 (*Discussion*)** | `output/thesis/chapters/06_discussion.tex`: implicazioni economiche, latenza reanalisi, limiti di validità esterna. | 2 giorni |
| **4** | **Scrittura Capitolo 7 (*Conclusions*)** | `output/thesis/chapters/07_conclusions.tex`: risposte formali alle ipotesi H1–H5 e sintesi finale. | 1 giorno |
| **5** | **Revisione Globale LaTeX & BibTeX** | De-commentare include in `main.tex`, verifica formattazione, citazioni e compilazione PDF unico. | 1 giorno |
| **6** | **Esportazione Pacchetto Finale** | Esecuzione script di esportazione, pulizia cartelle e consegna al relatore. | Mezza giornata |

---

## 5. Sintesi Operativa in 3 Battute per il Colloquio

Se il colloquio dura 10 minuti, concentrati su queste 3 frasi:

1. **Stato dell'arte**: *"Tutti i modelli, le feature e la validazione out-of-sample 2011–2025 sono calcolati; i primi 5 capitoli della tesi sono già scritti in LaTeX."*
2. **Risultato chiave (Tabella 5.3)**: *"I dati mostrano che negli anni normali il naive vince perché il meteo non è fattore limitante; il machine learning genera un valore enorme nelle code, riducendo l'errore sulla siccità da 9.3 a 7.1 bu/ac e anticipando lo shock a luglio."*
3. **Decisione per lui**: *"Possiamo considerare chiusa la fase empirica e procedere con i Capitoli 6 e 7 di discussione e conclusioni, oppure c'è una metrica specifica di regime che vuole vedere integrata?"*
