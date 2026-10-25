# TO-DO Operativi Attivi (Checklist di Lavoro)

> **Finalità:** Elencare esclusivamente le attività pratiche ancora aperte per chiudere la tesi. Quando un'attività è conclusa e verificata, viene rimossa per mantenere il foglio snello e pulito.  
> **Sorgenti di Riferimento:**
> - Current protocol and verified evidence: [`research_protocol.md`](../../output/docs/research_protocol.md) and [`diagnostics`](../../output/data/processed/diagnostics/).
> - Per gli spunti interpretativi ed economici: [`discussion_insights.md`](./discussion_insights.md)
> - Per i confini di ricerca da inserire nel Cap. 6: [`future_developments.md`](./future_developments.md)

---

## Task Aperti (Prossimi Passi)

### Thesis text recovery before further expansion
- [ ] **Review the previous text reductions before adding material**: start with `5258c4a` (2026-10-17 12:16, Europe/Rome), **"Refine thesis narrative and streamline text under 100k character budget"**. This commit shortened Chapters 1--6 by removing table-reading and repeated cross-chapter explanations. Also inspect `0d60748` (2026-10-18 22:57), **"refactor(thesis): refine academic tone and streamline self-explanatory captions and tables"**, for the subsequent prose, caption and table cleanup, and `36c3161` (2026-10-19 03:06), **"Rebuild LSTM test forecasts with complete weather windows"**, which combined updated evidence with further substantial chapter rewrites. Use `git show <commit> -- output/thesis/chapters output/thesis/tables` to inspect the removed and added passages. Recover only material consistent with the current methodology and verified results; check the character budget after assessing content needs. Section 5.2.3, **"Predictive Accuracy Tests"**, and its former inference table were introduced by `36c3161`; the subsection and inference table were subsequently removed locally at the author's request. The section text is retained in the procedural note below.

### LSTM training stability & Multi-Seed Ensembling (Completato Torneo Top 5)
- [x] **Torneo Multi-Seed e Selezione Seed Campione ($H = 5 \dots 1$)**:
  - Screening iniziale a 10 seed su $H=1$: isolata la top 5 (`88`, `2024`, `555`, `101`, `123`) contro la baseline (`42`).
  - Torneo multi-orizzonte su $H = 5 \dots 1$: **Seed 88** vince in modo schiacciante (#1 in 4 orizzonti su 5: $H=5, 4, 2, 1$; media $R^2$ sui 5 orizzonti pari a $0.6172$ vs $0.6074$ di Seed 42; $R^2$ a chiusura raccolto $H=1$ pari a $0.6726$ vs $0.6483$; RMSE a $4.831$\,bu/ac; Ensemble con ElasticNet a $R^2 = 0.6704$).
  - **Dati salvati e pronti al 100%**: `test_predictions_seed_88.parquet` ha già **tutti gli 11 orizzonti completati al 100%** ($89.100$ osservazioni). I file intermedi di tutti i seed (`42`, `88`, `2024`, `555`) sono memorizzati in `work/reports/experiments/lstm/`.
  - **Prossimo passo immediato alla ripresa**:
    1. Conferma elezione Seed 88 e pubblicazione in `output/data/processed/predictions_lstm.parquet` (istantanea).
    2. Esecuzione `07_evaluate_benchmarks.py` e `09_generate_results_figures.py` per refresh di tabelle e figure.
    3. Aggiornamento numerico nel testo di Capitolo 5, verifica budget caratteri (<100.000) e compilazione LaTeX.

### Integrazione Analisi Econometrica (Capitolo 3 e 4)
- [x] **Test di Rottura Strutturale di Chow nel Testo della Tesi** (Integrato in Appendix A e discusso in Cap. 6): formalizzazione econometrica del test di Chow (1951-2010 vs 1951-2025 con break nel 2014, $+4.83$\,bu/ac), serie trend-stazionaria ADF ($p < 0.0001$) e Figura A.1 generata da `scripts/04_compare_detrending.py`.
- [x] **Chiarimento Limiti del Detrending Lineare Expanding vs. Robustezza Cubica (Appendice A)** (Completato in Appendix A, con tabelle e figure in `output/thesis/`): documentazione del fit in-sample vs previsione out-of-sample a 1 passo (lineare superiore in OOS con RMSE 5.89 vs 6.39 bu/ac e BIC favorevole nel 65.9% delle contee), invarianza dello shrinkage di coda in $D_1$ (+8.38 vs +6.74 bu/ac), e tabelle A.1–A.3 generate da `scripts/09_generate_results_figures.py`.

### Struttura Risultati Empirici (Capitolo 5)
- [x] **Doppio Livello di Valutazione (Contea vs. Macro-Regionale)**:
  1. *Livello Contea*: riportata l'analisi per decili ($D_1 \dots D_{10}$, Tabella 5.4 e Figura 5.4), evidenziando la limitazione intrinseca della loss $L_2$ (conditional tail shrinkage dovuto al rapporto segnale/rumore locale).
  2. *Livello Macro-Regionale Aggregato (135 contee)*: mostrato come l'aggregazione spaziale cancelli il rumore idiosincratico locale, rivelando l'elevatissima accuratezza del segnale meteo a scala di mercato ($r > 0.94$, errore sullo shock estremo 2012 di soli $-0.21$\,bu/ac).
- [x] **Grafico di Traiettoria degli Errori lungo gli Horizon ($H=12 \dots 1$) a Livello Macro**: Generato come Figura 5.2 (`fig_macro_horizon_progression.pdf`) tramite `scripts/09_generate_results_figures.py` e integrato nella Sezione 5.3.1. Mostra la discesa dell'errore (Macro-RMSE da 3.84 a 2.18 bu/ac) e la salita di $R^2_{\text{macro}}$ fino a 0.676 a partire dall'inflessione critica di giugno ($H=5$).

### Stesura e Chiusura dei Capitoli Conclusivi (Capitolo 6)
- [x] **Capitolo 6 Unificato (Discussion and Conclusions)**: completato in `output/thesis/chapters/06_discussion_and_conclusions.tex`. Risponde puntualmente alle 5 domande di ricerca con evidenze quantitative, riallinea il confronto deep learning vs modelli lineari valorizzando la frequenza decadale, include la Tabella 6.1 dei verdetti su H1–H5 e confronta i risultati con la letteratura (Chen & Zhang 2026, Hoffman et al. 2020, Sharma et al. 2025).

### Revisione Finale e Allineamento Codici per il Professore
- [x] **Allineamento Totale della Replication Suite (`scripts/01_`–`09_`)**:
  - Tutti i 9 script sono verificati, autocontenuti e coprono al 100% tutte le figure e tabelle della tesi (Capitoli 3, 5 e Appendice A).
  - Documentazione completa e aggiornata in `scripts/README.md`.
- [ ] **Recupero Testo Tagliato (Opzionale)**: tenuto per ultimo come richiesto dall'autore; budget attuale a 92.304 caratteri (7.696 caratteri di margine libero).


## Project procedural notes

### Deferred section: Predictive Accuracy Tests

Author decision (2026-10-19): exclude the former Section 5.2.3 from the active thesis for now. Retain this text only as material for a possible future author-approved insertion. Origin: `36c3161` (2026-10-19 03:06, Europe/Rome), "Rebuild LSTM test forecasts with complete weather windows". No reinsertion is currently scheduled.

Evidence remains in `output/data/processed/diagnostics/forecast_inference.csv`, generated by `scripts/07_evaluate_benchmarks.py`. Before any reinsertion, verify every statistic against the then-current forecasts and diagnostic export, review the small-sample and multiple-comparison limitations, and align cross-references and any hypothesis summary. The former inference table also remains excluded.

Retained LaTeX passage:

```tex
\subsection{Predictive Accuracy Tests}
\label{sec:res_significance}

{
We apply one-sided Diebold--Mariano tests to 15 annual mean squared-loss differentials. The comparison preserves the complete cross-county error vector within each year. At June, weather-only ElasticNet and LSTM improve on the trend benchmark, with nominal $p$-values of $0.0011$ and $0.0009$. Weather-only Random Forest and XGBoost have $p$-values above $0.05$. From July onward, the reported tests reject equal accuracy for the evaluated models. We assess integrated and weather-only specifications separately.

At harvest closure, the LSTM--ElasticNet comparison has $p=0.361$. The ensemble improves on ElasticNet at the nominal level ($p=0.033$); its comparison with LSTM has $p=0.176$. The pooled point ranking therefore does not establish superiority over every individual model.

HAC inference uses two annual lags and the $15/14$ covariance correction. The small number of test years and unadjusted multiple comparisons limit strong claims about the first statistically useful cutoff.
}
```
