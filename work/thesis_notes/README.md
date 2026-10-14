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

## 2. Strumenti Istituzionali e Controllo Formale di Ateneo

| File | Funzione |
| :--- | :--- |
| [`university_thesis_guide.pdf`](./university_thesis_guide.pdf) | **Vademecum Ufficiale del Docente Relatore:** Linee guida formali di Ateneo (limite nominale 100.000 caratteri, struttura OneDrive, regole per tabelle e figure con `\textit{Source: ...}`). |
| [`count_thesis_characters.py`](./count_thesis_characters.py) | **Script di Conteggio Ufficiale:** Script Python che calcola i caratteri effettivi del testo LaTeX escludendo tabelle, figure, equazioni e bibliografia secondo il Vademecum (*il budget è un vincolo amministrativo da verificare a consuntivo, non una guida alla stesura dei contenuti*). |

