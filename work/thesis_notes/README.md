# Master Index delle Note di Lavoro e Documentazione Metodologica
*Registro unificato di tutte le note di ricerca, decisioni operative e documentazione tecnica*

Questo file funge da hub centrale per garantire che nessun appunto, decisione metodologica o appunto di lavoro venga disperso durante la redazione della tesi.

---

## 1. Note di Lavoro & Protocolli di Ricerca (`work/thesis_notes/`)

| File | Argomento / Contenuto Principale | Stato |
| :--- | :--- | :--- |
| [`author_working_notes.md`](./author_working_notes.md) | **Appunti dell'Autore:** Archivio delle scelte metodologiche, parametri aperti (es. sensibilità cutoff), priorità dei target (regressione vs classificazione) e buone pratiche redazionali. | Consolidato / In uso |
| [`model_progression_tracking.md`](./model_progression_tracking.md) | **Progressione dei Modelli:** Protocollo di tracciamento per Ladder A (feature F0-F5), Ladder B (modelli M0-M3) e Ladder C (ablazioni), per documentare l'intero percorso iterativo nel Capitolo 5. | Attivo |
| [`antecedent_features_note.md`](./antecedent_features_note.md) | **Feature Antecedenti:** Logica di inclusione del meteo e rese della campagna precedente ($Y-1$) per i lead-time lunghi ($H=12 \dots 8$). | Attivo |
| [`sample_selection.md`](./sample_selection.md) | **Selezione Campionaria:** Criteri di completezza NASS, resa per acro e bilanciamento delle 135 contee su 6 stati del Midwest (1951–2025). | Consolidato |
| [`source_map.md`](./source_map.md) | **Mappa delle Fonti:** Provenienza, coordinate e riferimenti dei dati USDA NASS, ERA5-Land e confini amministrativi. | Consolidato |
| [`thesis_character_budget.md`](./thesis_character_budget.md) | **Budget Caratteri:** Matrice dei range di flessibilità (70k - 130k) e allocazione caratteri per capitolo (Cap 1 max 10k, Cap 2 max 30k). | Riferimento / In uso |
| [`university_thesis_guide.pdf`](./university_thesis_guide.pdf) | **Linee Guida di Ateneo:** Vincoli formali di redazione (limite 100.000 caratteri / 35 pagine, formattazione 12pt, 13 min presentazione). | Riferimento |

---

## 2. Specifiche Metodologiche e Decision Log Ufficiale (`output/docs/`)

| File | Argomento / Contenuto Principale |
| :--- | :--- |
| [`decisions.md`](../../output/docs/decisions.md) | **Research Decision Log:** Tabella formale di tutte le decisioni metodologiche (stato, data, motivazione, stadi impattati). |
| [`validation_and_modeling.md`](../../output/docs/validation_and_modeling.md) | **Protocollo di Validazione:** Expanding window strettamente cronologica, split train-only, tassonomia dei modelli. |
| [`era5_land_methodology.md`](../../output/docs/era5_land_methodology.md) | **Metodologia ERA5-Land:** Formule di aggregazione spaziale, pesi intersezionali, indici termici e idrologici. |
| [`feature_engineering.md`](../../output/docs/feature_engineering.md) | **Ingegneria delle Feature:** Finestre temporali (5-day, 10-day, 30-day), calcolo GDD, EDD ($>30^\circ\text{C}$), VPD e $P - \text{ET}_0$. |
| [`data_and_target.md`](../../output/docs/data_and_target.md) | **Target e Dati:** Definizione delle anomalie continue di resa, trend tecnologico e formulazione dei regimi categorici. |
| [`research_protocol.md`](../../output/docs/research_protocol.md) | **Protocollo di Ricerca:** Schema operativo complessivo della sperimentazione. |
| [`research_design.md`](../../output/docs/research_design.md) | **Quadro Concettuale:** Inquadramento teorico, motivazione economica e gerarchia degli obiettivi. |

---

## 3. Note Operative e Ingegneristiche di Pipeline (`work/docs/`)

| File | Argomento / Contenuto Principale |
| :--- | :--- |
| [`era5_step2_handoff.md`](../docs/era5_step2_handoff.md) | Note tecniche sul download e pipeline di estrazione da ERA5-Land reanalysis. |
| [`repository_audit.md`](../docs/repository_audit.md) | Audit approfondito sull'integrità dei dati, controlli di parità e verifiche di coerenza. |
| [`reorganization_plan.md`](../docs/reorganization_plan.md) | Piano di architettura della cartella di lavoro e separazione tra raw, interim e processed data. |
