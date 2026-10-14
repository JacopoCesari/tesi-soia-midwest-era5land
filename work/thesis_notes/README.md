# Guida e Mappa Tematica dei Documenti di Ricerca (`work/thesis_notes/`)

> **Regole di Progetto e Voce Autoriale:**  
> Tutte le regole vincolanti di scrittura, perimetro bibliografico (13 paper core) e divieto di marker LLM sono formalizzate in [`AGENTS.md`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/AGENTS.md).  
> La documentazione scientifica ufficiale risiede in [`output/docs/`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/output/docs/).  
> Questa cartella (`work/thesis_notes/`) è l'archivio intellettuale, metodologico e operativo del lavoro di tesi.

---

## La Struttura a 4 Riferimenti Chiave

Per garantire trasparenza operativa e consultazione immediata, i documenti sono organizzati in quattro riferimenti distinti:

```
work/thesis_notes/
│
├── 1. consolidated_memory.md      <--- MEMORIA METODOLOGICA ED EMPIRICA CONSOLIDATA [PERMANENTE]
│                                       (Sorgente di verità: panel 135 contee, detrending OLS train-only,
│                                        architetture congelate, tabelle out-of-sample, verdetti H1-H5, letteratura)
│
├── 2. discussion_insights.md      <--- BANCO DI VALIDAZIONE DELLE IPOTESI [OPERATIVO]
│                                       (Le riflessioni e intuizioni vengono vagliate contro numeri out-of-sample
│                                        e letteratura: quelle validate passano in consolidated_memory / tesi,
│                                        quelle smentite vengono scartate senza compiacimenti)
│
├── 3. future_developments.md      <--- SVILUPPI FUTURI E CONFINI [TEMPORANEO DI LAVORO]
│                                       (I 5 punti di sviluppo per la sezione Future Work del Cap. 6;
│                                        man mano che vengono trascritti nella tesi, vengono rimossi da qui)
│
└── 4. operational_todo.md         <--- CHECKLIST TO-DO ATTIVI [TEMPORANEO DI LAVORO]
                                        (I task pratici aperti: allineamento figure, de-prolissificazione,
                                         capitoli finali; i task completati vengono cancellati per pulizia)
```

---

## 1. Memoria Consolidata Permanente vs. Strumenti Operativi Temporanei

| Documento | Natura e Durata | Ruolo Esclusivo |
| :--- | :---: | :--- |
| [`consolidated_memory.md`](./consolidated_memory.md) | **Permanente** | **La sorgente unica di verità scientifica.** Raccoglie in forma compatta e stabile tutte le decisioni approvate, le formule matematiche, i protocolli anti-leakage, i risultati empirici congelati e i verdetti sulle ipotesi H1–H5. **Non viene modificato né cancellato.** |
| [`discussion_insights.md`](./discussion_insights.md) | **Operativo (Filtro Super Partes)** | **Banco di prova e validazione oggettiva delle ipotesi.** Qualsiasi intuizione o riflessione viene vagliata rigorosamente contro evidenza empirica e letteratura: se confermata, viene promossa a fatto accettato nella tesi; se smentita, viene eliminata. |
| [`future_developments.md`](./future_developments.md) | **Temporaneo** | **I 5 confini di ricerca per il Capitolo 6.** Serve come base per la stesura della sezione *Future Research & Operational Perspectives*. Man mano che i punti vengono integrati nel testo finale della tesi, vengono smarcati o rimossi. |
| [`operational_todo.md`](./operational_todo.md) | **Temporaneo** | **La to-do list attiva.** Elenca solo le attività pratiche aperte (figure, testo, capitoli finali, audit formale). Man mano che un task è concluso e verificato, viene cancellato per mantenere il foglio pulito. |

---

## 2. Approfondimenti Tematici Specialistici (Archivio di Dettaglio)

Documenti verticali redatti durante lo sviluppo del progetto, consultabili per analisi specialistiche:

| File | Argomento Specifico |
| :--- | :--- |
| [`tabular_models_and_econometric_diagnostics.md`](./tabular_models_and_econometric_diagnostics.md) | Dettaglio matematico sui modelli tabulari, feature $\text{EDD}_{30}$, diagnostica non-gaussiana dei residui, eteroschedasticità condizionata e confutazione del Rolling OLS. |
| [`deep_learning_lstm_and_tail_modeling.md`](./deep_learning_lstm_and_tail_modeling.md) | Dettaglio dell'architettura ricorrente LSTM, Temporal Attention di Bahdanau, coordinate continue vs. embedding, inner validation anti-leakage e micro-grid. |
| [`extreme_years_benchmark.md`](./extreme_years_benchmark.md) | Catalogo dettagliato delle annate di shock storico (1988, 1993, 2012, 1974, bumper 1994, 2016) e decoupling dal trend tecnologico $R^2=0.81$. |
| [`hypothesis_synthesis_H1_H5.md`](./hypothesis_synthesis_H1_H5.md) | Scheda di dettaglio sulla validazione econometrica e i test di Diebold-Mariano per le ipotesi H1–H5. |
| [`literature_benchmarks_and_comparison.md`](./literature_benchmarks_and_comparison.md) | Analisi comparativa approfondita dei 13 paper core e dei 3 in riserva, distinzione resa lorda vs. anomalia pura. |
| [`literature_details_extracted.txt`](./literature_details_extracted.txt) | Snippet ed estratti testuali estratti direttamente dai PDF dei paper primari. |

---

## 3. Strumenti Istituzionali e Controllo Formale di Ateneo

| File | Funzione |
| :--- | :--- |
| [`university_thesis_guide.pdf`](./university_thesis_guide.pdf) | **Vademecum Ufficiale del Docente Relatore:** Linee guida formali di Ateneo (limite nominale 100.000 caratteri, struttura OneDrive, regole per tabelle e figure con `\textit{Source: ...}`). |
| [`thesis_character_budget.md`](./thesis_character_budget.md) | **Budget Caratteri di Riferimento:** Limite massimo con buffer a 110.000 caratteri. *Nota: il budget è un vincolo amministrativo da verificare a consuntivo, non una guida alla stesura dei contenuti.* |
| [`count_thesis_characters.py`](./count_thesis_characters.py) | **Script di Conteggio Ufficiale:** Script Python che calcola i caratteri effettivi del testo LaTeX escludendo tabelle, figure, equazioni e bibliografia secondo il Vademecum. |
| [`chapter_boundaries.md`](./chapter_boundaries.md) | **Mappa dei Confini tra Capitoli:** Delineazione di cosa appartiene a ciascun capitolo (Cap. 3 dati/EDA, Cap. 4 formule/algoritmi, Cap. 5 risultati/stress-test, Cap. 6 discussione/limiti). |
| [`source_map.md`](./source_map.md) | Mappa di raccordo tra i capitoli della tesi e i documenti tecnici in `output/docs/`. |
| [`data_management_and_backup_guide.md`](./data_management_and_backup_guide.md) | Protocollo di backup e integrità dei dati primari e delle esportazioni per la consegna finale. |
