# Indice delle Note Metodologiche e Strumenti di Stesura (`work/thesis_notes/`)

> **Regole di Progetto e Voce Autoriale:**  
> Tutte le regole vincolanti di scrittura, perimetro bibliografico e divieto di marker LLM sono formalizzate in [`AGENTS.md`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/AGENTS.md).  
> La documentazione scientifica ufficiale e consolidata risiede in [`output/docs/`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/output/docs/).  
> Questa cartella (`work/thesis_notes/`) contiene gli strumenti pratici di controllo, le note metodologiche di supporto e la to-do list operativa per la stesura.

---

## 1. Strumenti Istituzionali e Controllo Formale

| File | Descrizione / Funzione |
| :--- | :--- |
| [`university_thesis_guide.pdf`](./university_thesis_guide.pdf) | **Vademecum Ufficiale di Ateneo (8 pagine):** Linee guida del docente relatore (limite 100.000 caratteri, struttura cartelle OneDrive, regole su tabelle con `Source:` e formattazione). |
| [`thesis_character_budget.md`](./thesis_character_budget.md) | **Budget Caratteri per Capitolo:** Matrice di pianificazione con limite massimo operativo a 110.000 caratteri (+10% buffer) e stato di avanzamento attuale dei capitoli 1--3. |
| [`count_thesis_characters.py`](./count_thesis_characters.py) | **Script di Conteggio Ufficiale:** Script Python che calcola i caratteri effettivi del testo escludendo tabelle, figure, formule e bibliografia in conformità al Vademecum. |

---

## 2. Note Operative e Scoping dei Capitoli

| File | Descrizione / Funzione |
| :--- | :--- |
| [`author_working_notes.md`](./author_working_notes.md) | **To-Do List e Task Aperti:** Decisioni metodologiche differite (es. target continuo vs regimi nel Cap. 5, collocazione Tabella 3.4), re-run delle mappe/curve del 1988 con dati completi, e promemoria citazioni per il Cap. 4. |
| [`chapter_boundaries.md`](./chapter_boundaries.md) | **Mappa dei Confini tra Capitoli:** Delineazione di cosa appartiene a ciascun capitolo (Cap. 3 dati/EDA, Cap. 4 formule/algoritmi, Cap. 5 risultati empirici/stress test, Cap. 6 discussione/implicazioni). |
| [`source_map.md`](./source_map.md) | **Mappa di Raccordo Capitoli--Documentazione:** Matrice che collega ciascun capitolo della tesi ai rispettivi documenti tecnici ufficiali in `output/docs/`. |

---

## 3. Protocolli Metodologici di Supporto per la Modellistica (Cap. 4--5)

| File | Descrizione / Funzione |
| :--- | :--- |
| [`model_progression_tracking.md`](./model_progression_tracking.md) | **Protocollo di Tracciamento Iterazioni Modelli:** Schema per registrare e presentare nel Cap. 5 il percorso incrementale di miglioramento dai benchmark storici semplici (Trend, Climatologia) ai modelli ML/DL (Ridge, Random Forest, XGBoost, LSTM). |
| [`extreme_years_benchmark.md`](./extreme_years_benchmark.md) | **Analisi del Trend e Shock Benchmark:** Fondamenti metodologici sul perché modellare il residuo climatico pur in presenza di un trend tecnologico forte ($R^2=0.81$), con le coorti storiche di shock (1988, 1993, 2012, 1974) per lo stress-test del Cap. 5. |
| [`antecedent_features_note.md`](./antecedent_features_note.md) | **Lookback Invernale e Feature Antecedenti:** Logica di gestione delle feature a lungo anticipo ($H=12 \dots 8$) prima della semina primaverile. |
| [`modeling_execution_roadmap.md`](./modeling_execution_roadmap.md) | **Roadmap Esecutiva Post-Capitolo 4:** Sequenza ordinata di implementazione (feature engineering, detrending ermetico, grid search 1985–1995, expanding test 1996–2025) per produrre i risultati empirici del Cap. 5. |
