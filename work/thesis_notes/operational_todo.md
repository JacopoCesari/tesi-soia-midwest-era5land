# TO-DO Operativi Attivi (Checklist Temporanea di Lavoro)

> **Stato del Documento:** Temporaneo di lavoro — I task completati vengono cancellati da questo file  
> **Finalità:** Elencare esclusivamente le attività pratiche ancora aperte per chiudere la tesi. Quando un'attività è conclusa e verificata, viene rimossa per mantenere il foglio snello e pulito.  
> **Sorgenti di Riferimento:**
> - Per consultare decisioni, numeri ed evidenze stabili: [`consolidated_memory.md`](./consolidated_memory.md)
> - Per gli spunti interpretativi ed economici: [`discussion_insights.md`](./discussion_insights.md)
> - Per i confini di ricerca da inserire nel Cap. 6: [`future_developments.md`](./future_developments.md)

---

## Task Aperti

### 1. Armonizzazione Grafica di Tutte le Figure
Obiettivo: Rigenerare tutte le figure dai due script canonici ufficiali in `scripts/`, applicando uno stile visivo comune (font `DejaVu Serif` stile LaTeX, palette modelli uniforme, didascalie conformi al Vademecum con `\textit{Source: ...}`).

- [ ] **Capitolo 3 (`scripts/08_generate_eda_figures.py`)**:
  - Centralizzare tutte le figure (3.1–3.8).
  - Verificare assenza di didascalie ridondanti e rimuovere il grafico duplicato `fig_yield_trajectory_dual_petrol` da `03_data_and_study_area.tex`.
- [ ] **Capitolo 5 (`scripts/09_generate_results_figures.py`)**:
  - Centralizzare tutte le figure dei risultati (5.1–5.6).
  - Applicare la palette modelli standard (Naive: grigio, ElasticNet: blu, RF: ambra, XGBoost: rosso, LSTM: viola, Ensemble: smeraldo).
  - Evidenziare nella mappa 5.6 la regola regionale del Minnesota ($H^*=2$).

---

### 2. Revisione del Testo e De-prolissificazione (Capitoli 1–5)
Obiettivo: Eliminare verbosità, giri di parole inutili, preamboli difensivi e ripetizioni tra capitoli, adottando una voce diretta, asciutta e professionale (senza l'ansia del conteggio caratteri, che resta un mero controllo amministrativo a valle).

- [ ] **Capitolo 1 (Introduction)**: snellire premesse, mettere a fuoco le 2 contribution e le RQ.
- [ ] **Capitolo 2 (Literature Review)**: eliminare strutture binarie (*"rather than"*), massimo 1 citazione cardine per mini-argomento.
- [ ] **Capitolo 3 (Data & Study Area)**: eliminare la parafrasi numerica delle tabelle; focalizzarsi sui commenti di sintesi.
- [ ] **Capitolo 4 (Methodology)**: rimuovere ridondanze su definizioni già introdotte al Capitolo 3; formule essenziali e dirette.
- [ ] **Capitolo 5 (Empirical Results)**: evitare di riscrivere nel testo i numeri già stampati nelle Tabelle 5.1–5.4; commentare i pattern economici ed ecofisiologici; integrare l'Ensemble nella discussione.

---

### 3. Stesura dei Capitoli Conclusivi (Capitoli 6 e 7)
- [ ] **Capitolo 6 (Discussion & Limitations)**: attingere dagli spunti in [`discussion_insights.md`](./discussion_insights.md) e integrare i confini metodologici da [`future_developments.md`](./future_developments.md).
- [ ] **Capitolo 7 (Conclusions)**: sintesi asciutta delle risposte alle domande di ricerca e takeaway scientifico finale.

---

### 4. Audit Formale Finale e Compilazione
- [ ] Compilazione LaTeX pulita (`pdflatex`) a zero errori fatali e zero overfull box visibili.
- [ ] Verifica conformità Vademecum: presenza di `\textit{Source: ...}` sotto ogni tabella e figura.
- [ ] Controllo finale amministrativo del conteggio caratteri (`count_thesis_characters.py`).
