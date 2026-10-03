# Guida e Mappa Tematica dei Documenti di Ricerca (`work/thesis_notes/`)

> **Regole di Progetto e Voce Autoriale:**  
> Tutte le regole vincolanti di scrittura, perimetro bibliografico e divieto di marker LLM sono formalizzate in [`AGENTS.md`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/AGENTS.md).  
> La documentazione scientifica ufficiale e consolidata risiede in [`output/docs/`](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/output/docs/).  
> Questa cartella (`work/thesis_notes/`) contiene gli strumenti pratici, le note metodologiche tematiche ("Perché A e non B"), il registro operativo unificato e le roadmap esecutive.
>
> **Regola di Demarcazione Rigida:**  
> 1. I file della **Sezione 1** contengono esclusivamente la memoria metodologica consolidata e i risultati empirici convalidati (ZERO TO-DO, zero elenchi di cose aperte o da testare).  
> 2. Tutti i **TO-DO operativi, piani di azione, cose non confermate, da testare ancora, modifiche pendenti e sviluppi futuri** sono raggruppati esclusivamente nella **Sezione 2** ([`operational_todo_and_action_plans.md`](./operational_todo_and_action_plans.md)).

---

## 1. Memoria Metodologica e Risultati Empirici Consolidati (Stato dell'Arte)

Documenti tematici che registrano le scelte teoriche, le formulazioni matematiche, i test empirici conclusi e le motivazioni "Perché A e non B":

| File | Tema e Rationale Scientifico | Capitoli Tesi |
| :--- | :--- | :---: |
| [`tabular_models_and_econometric_diagnostics.md`](./tabular_models_and_econometric_diagnostics.md) | **Machine Learning Tabulare & Diagnostica Econometrica:** Valutazione out-of-sample (1996–2025, 4.050 valutazioni per orizzonte) dei modelli tabulari (Tier 1–3: ElasticNet, Random Forest, XGBoost) lungo i 12 orizzonti previsivi; arricchimento bioclimatico con $\text{EDD}_{30}$ ($+1.07\%$ $R^2$); memoria agronomica colturale/pedologica via anomalia ritardata detrendata $\epsilon_{t-1}$ ($+1.0\% \dots +2.0\%$ di $R^2$) e decostruzione della fallacia del trend secolare in Chen & Zhang (2026); validazione fenologica estiva $>51\%$ (Ipotesi H1); diagnostica econometrica dei residui (non-normalità da shock asimmetrici, eteroschedasticità con varianza raddoppiata negli shock). | Cap. 4 & 5 |
| [`deep_learning_lstm_and_tail_modeling.md`](./deep_learning_lstm_and_tail_modeling.md) | **Deep Learning Sequenziale & Modellazione delle Code:** Architettura neurale ricorrente LSTM con Temporal Attention (Bahdanau); risoluzioni a 30d, 10d, 5d; ablazione dello spatial encoding (vittoria delle coordinate continue `[lat, lon]` rispetto al county embedding di $-0.29$ bu/ac); protocollo anti-leakage di Inner Validation ($t-3 \dots t-1$); analisi del rapporto segnale-rumore (SNR) e tail shrinkage sui crolli da siccità; micro-grid e formulazione della Asymmetric Huber Loss; rigetto formale del Reinforcement Learning e delle 344 contee periferiche. | Cap. 4 & 5 |
| [`extreme_years_benchmark.md`](./extreme_years_benchmark.md) | **Trend Tecnologico e Benchmark Shock Storici:** Fondamento teorico del perché modellare il residuo climatico volatile pur in presenza di un trend tecnologico dominante ($R^2=0.81$); catalogo degli anni di shock severo (1988, 1993, 2012, 1974, 2016) per lo stress-test del Capitolo 5. | Cap. 3 & 5 |
| [`literature_benchmarks_and_comparison.md`](./literature_benchmarks_and_comparison.md) | **Compendio Critico della Letteratura Primaria:** Rassegna analitica dei 13 paper core; spiegazione del ruolo della K-Fold casuale e del data leakage spaziale negli studi precedenti; isolamento della prevedibilità meteorologica pura rispetto alla resa lorda totale. | Cap. 2 & 5 |

---

## 2. Registro Operativo Unificato: TO-DO, Piani di Azione, Cose da Testare, Ipotesi e Modifiche

Tutto il materiale operativo, non confermato o in corso di esecuzione è centralizzato in un unico documento di lavoro:

| File Principale | Ruolo e Sezioni Contenute |
| :--- | :--- |
| [`operational_todo_and_action_plans.md`](./operational_todo_and_action_plans.md) | **Registro Unificato di Lavoro Operativo:**<br>1. **TO-DO Operativi Immediati** (Checklist esecuzione LSTM, micro-grid 4 teste, expanding test 1996–2025, generazione figure Cap. 5, audit figure Cap. 1–3).<br>2. **Piani di Azione (Action Plans)** (Sintesi esecutiva per la stesura dei Capitoli 4 e 5 e milestones di calcolo).<br>3. **Cose da Testare Ancora & Esperimenti Pendenti** (Micro-grid 4 teste su frequenza campionessa $W^*$, expanding test out-of-sample su 11 orizzonti, ablation lag su LSTM).<br>4. **Cose Non Confermate & Ipotesi Aperte** (Harvest contamination ad ottobre $H=1$, trade-off loss simmetrica MSE vs Asymmetric Huber, verifica biologica pesi attenzione ad agosto).<br>5. **Modifiche, Refactoring e Bonifica Repository** (Eliminazione nomenclatura informale, audit file obsoleti, verifica assenza duplicati).<br>6. **Sviluppi Futuri per il Capitolo 6 (Discussion & Future Work)** (Remote sensing multimodale Sentinel/MODIS, semine dinamiche USDA, seasonal forecasts SEAS5, Physics-Informed ML, GNN). |

### Moduli Specialistici di Supporto Operativo (Allegati di Dettaglio)
- [`action_plan_chapter_4.md`](./action_plan_chapter_4.md): Scheletro matematico e analitico dettagliato per la stesura di `04_methodology.tex`.
- [`modeling_execution_roadmap.md`](./modeling_execution_roadmap.md): Roadmap dettagliata delle fasi di calcolo e dei relativi script di esecuzione.
- [`model_progression_tracking.md`](./model_progression_tracking.md): Tassonomia di progressione della complessità (Ladder A feature, Ladder B modelli, Ladder C ablazioni).

---

## 3. Strumenti Istituzionali e Controllo Formale di Ateneo

| File | Descrizione / Funzione |
| :--- | :--- |
| [`university_thesis_guide.pdf`](./university_thesis_guide.pdf) | **Vademecum Ufficiale di Ateneo (8 pagine):** Linee guida del docente relatore (limite 100.000 caratteri, struttura cartelle OneDrive, regole su tabelle con `Source:` e formattazione). |
| [`thesis_character_budget.md`](./thesis_character_budget.md) | **Budget Caratteri per Capitolo:** Matrice di pianificazione con limite massimo operativo a 110.000 caratteri (+10% buffer) e stato di avanzamento dei capitoli. |
| [`count_thesis_characters.py`](./count_thesis_characters.py) | **Script di Conteggio Ufficiale:** Script Python che calcola i caratteri effettivi del testo escludendo tabelle, figure, formule e bibliografia in conformità al Vademecum. |
| [`chapter_boundaries.md`](./chapter_boundaries.md) | **Mappa dei Confini tra Capitoli:** Delineazione di cosa appartiene a ciascun capitolo (Cap. 3 dati/EDA, Cap. 4 formule/algoritmi, Cap. 5 risultati/stress-test, Cap. 6 discussione/limiti). |
| [`source_map.md`](./source_map.md) | **Mappa di Raccordo Capitoli--Documentazione:** Matrice che collega ciascun capitolo della tesi ai rispettivi documenti tecnici ufficiali in `output/docs/`. |
| [`data_management_and_backup_guide.md`](./data_management_and_backup_guide.md) | **Gestione e Backup Dati:** Linee guida per l'integrità dei dataset primari, parquets ed esportazioni per la consegna finale. |
