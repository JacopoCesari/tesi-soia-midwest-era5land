# TO-DO Operativi Attivi (Checklist di Lavoro)

> **Finalità:** Elencare esclusivamente le attività pratiche ancora aperte per chiudere la tesi. Quando un'attività è conclusa e verificata, viene rimossa per mantenere il foglio snello e pulito.  
> **Sorgenti di Riferimento:**
> - Current protocol and verified evidence: [`research_protocol.md`](../../output/docs/research_protocol.md) and [`diagnostics`](../../output/data/processed/diagnostics/).
> - Per gli spunti interpretativi ed economici: [`discussion_insights.md`](./discussion_insights.md)
> - Per i confini di ricerca da inserire nel Cap. 6: [`future_developments.md`](./future_developments.md)

---

## Task Aperti (Prossimi Passi)

### LSTM training stability
- [ ] **Repeat the LSTM test evaluation with multiple predefined random seeds**: assess training variability after the complete-day window correction, keeping the selected 10-day frequency, hyperparameters, original validation records and 2011–2025 test folds fixed. Compare all runs with the current results and report their distribution; retain the current run as the reference and avoid selecting a replacement solely because its test score improves. If the adopted results change, regenerate the ensemble, diagnostics, tables, figures and matching thesis text through the canonical scripts.

### Integrazione Analisi Econometrica (Capitolo 3 e 4)
- [ ] **Test di Rottura Strutturale di Chow nel Testo della Tesi**: Integrare la formalizzazione econometrica del test di Chow (1951–2025), la verifica di stazionarietà ADF e l'evidenza del salto di livello nel 2014 ($F = 21.96, p < 0.0001, \Delta = +4.83$\,bu/ac) e la figura a due pannelli [`fig_chow_structural_break.pdf`](./fig_chow_structural_break.pdf). Da inserire nel Capitolo 3 (proprietà empiriche delle rese) o Capitolo 4 (giustificazione del detrending lineare expanding e spiegazione del bias out-of-sample recente). Script di riferimento: [`run_chow_test.py`](./run_chow_test.py).

### Stesura e Chiusura dei Capitoli Conclusivi (Capitoli 6 e 7)
- [ ] **Capitolo 6 (Discussion & Limitations)**: attingere dagli spunti consolidati in [`discussion_insights.md`](./discussion_insights.md) (inclusa l'Ipotesi 6 sul break strutturale del 2014) e integrare i confini metodologici da [`future_developments.md`](./future_developments.md).
- [ ] **Capitolo 7 (Conclusions)**: sintesi asciutta delle risposte alle domande di ricerca e takeaway scientifico finale.

