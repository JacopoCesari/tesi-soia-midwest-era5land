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

### LSTM training stability
- [ ] **Repeat the LSTM test evaluation with multiple predefined random seeds**: assess training variability after the complete-day window correction, keeping the selected 10-day frequency, hyperparameters, original validation records and 2011–2025 test folds fixed. Compare all runs with the current results and report their distribution; retain the current run as the reference and avoid selecting a replacement solely because its test score improves. If the adopted results change, regenerate the ensemble, diagnostics, tables, figures and matching thesis text through the canonical scripts.

### Integrazione Analisi Econometrica (Capitolo 3 e 4)
- [ ] **Test di Rottura Strutturale di Chow nel Testo della Tesi**: Integrare la formalizzazione econometrica del test di Chow:
  - *Pre-test (1951-2010)*: il test non rifiuta la stabilità del trend (F_max = 2.85 < F_crit = 4.01, p = 0.093), dimostrando che prima del test set non vi era alcuna evidenza statistica di rottura (giustificazione formale del trend lineare causale ex-ante).
  - *Full sample (1951-2025)*: evidenza di un break strutturale post-hoc nel 2014 (F = 21.96, p < 0.0001, Delta = +4.83 bu/ac), serie trend-stazionaria ADF (p < 0.001), e figura a due pannelli [`fig_chow_structural_break.pdf`](./fig_chow_structural_break.pdf). Da inserire nel Capitolo 3 (proprietà empiriche delle rese) o Capitolo 4 (giustificazione del detrending lineare expanding e spiegazione del bias out-of-sample recente). Script di riferimento: [`run_chow_test.py`](./run_chow_test.py).
- [ ] **Chiarimento Limiti del Detrending Lineare Expanding vs. Robustezza Cubica (Appendice)**: Documentare con rigore dove il trend lineare è superiore e dove mostra limitazioni. È superiore strettamente nella previsione **Out-of-Sample a 1 passo**:
  - Nel periodo pre-test 1981-2010 (4.050 fold), l'RMSE del lineare è 5.77 bu/ac contro 5.86 del quadratico e 6.27 del cubico.
  - Sull'intero orizzonte 1981-2025 (6.075 fold), l'RMSE del lineare è 5.89 bu/ac vs 6.03 quadratico e 6.39 cubico, con BIC favorevole nel 65.9% delle contee.
  - Ha invece un fit in-sample inferiore (R^2 = 0.812 vs 0.831 del cubico) perché non flette per incorporare il break post-2014. Il report di appendice [`Report_detrending.pdf`](../experiments/full_sample_detrending/Report_detrending.pdf) dimostra che anche adottando il trend cubico a massimo fit in-sample, lo shrinkage delle code (D_1) e le performance comparative dell'ML restano immutate.

### Struttura Risultati Empirici (Capitolo 5)
- [ ] **Doppio Livello di Valutazione (Contea vs. Macro-Regionale)**:
  1. *Livello Contea*: riportare l'analisi per decili ($D_1 \dots D_{10}$), evidenziando con trasparenza la limitazione intrinseca della loss $L_2$ (conditional tail shrinkage dovuto al fatto che il clima spiega il $\sim 30\%$ della varianza locale, mentre il $70\%$ è rumore non meteo).
  2. *Livello Macro-Regionale Aggregato (135 contee)*: mostrare come l'aggregazione spaziale cancelli il rumore idiosincratico locale, rivelando l'elevatissima accuratezza del segnale meteo a scala di mercato ($r > 0.94$, errore sullo shock estremo 2012 di soli $-0.11$\,bu/ac).
- [ ] **Proposta al Relatore: Valutazione degli Errori lungo gli Horizon ($H=12 \dots 1$) a Livello Macro**: Valutare e discutere l'opportunità di inserire il grafico di discesa dell'errore (Macro-RMSE / Macro-MAE) da $H=12$ a $H=1$ calcolato sul dato medio aggregato Midwest, che rappresenta l'output operativo di riferimento per trading desk e supply chain.

### Stesura e Chiusura dei Capitoli Conclusivi (Capitoli 6 e 7)
- [ ] **Capitolo 6 (Discussion & Limitations)**: attingere dagli spunti consolidati in [`discussion_insights.md`](./discussion_insights.md) (inclusa l'Ipotesi 6 sul break strutturale del 2014) e integrare i confini metodologici da [`future_developments.md`](./future_developments.md).
- [ ] **Capitolo 7 (Conclusions)**: sintesi asciutta delle risposte alle domande di ricerca e takeaway scientifico finale.


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
