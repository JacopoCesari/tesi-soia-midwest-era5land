# Roadmap Esecutiva Post-Capitolo 4: Dalla Metodologia ai Risultati Empirici

Questo documento fissa la sequenza ordinata dei passi operativi da seguire non appena completata la stesura e validazione formale del Capitolo 4 (`04_methodology.tex`).

---

## Gate 0: Validazione e Chiusura Formale del Capitolo 4

1. **Verifica Budget Caratteri Ateneo**:
   - Esecuzione dello script ufficiale: `python work/thesis_notes/count_thesis_characters.py`.
   - **Obiettivo**: Verificare che il Capitolo 4 si attesti rigorosamente tra **16.000 e 19.000 caratteri** (spazi inclusi), salvaguardando il budget per il Capitolo 5 (20.000–25.000 car.).
   - Verificare che tabelle (`table`), figure (`figure`) ed equazioni (`equation`) rimangano escluse dal conteggio come da Vademecum p. 5.
2. **Controllo Compilazione LaTeX**:
   - Compilazione con `latexmk -pdf -interaction=nonstopmode main.tex` in `output/thesis/`.
   - Verifica di zero errori fatali, zero overfull box visibili e corretta risoluzione di tutti i riferimenti incrociati (`\ref{tab:...}`, `\ref{sec:...}`, `\eqref{...}`).

---

## Step 1: Configurazione Ambiente Virtuale ML (`.venv`)

1. **Installazione Dipendenze Machine Learning**:
   - Installare i pacchetti specificati in `output/pyproject.toml` sotto `[project.optional-dependencies]`:
     * `scikit-learn>=1.5.0`
     * `xgboost>=2.1.0`
     * `lightgbm>=4.5.0`
     * (e `torch>=2.4.0` per la componente LSTM sequenziale).
2. **Sanity Check**:
   - Test rapido di importazione e verifica compatibilità numerica (NumPy 2.x / SciPy).

---

## Step 2: Feature Matrix Assembly Multi-Orizzonte (`output/scripts/build_feature_matrices.py`)

1. **Input Dati**:
   - 76 file Parquet giornalieri (1950–2025) in `work/data/interim/county_daily_weather/production/`.
   - Centroidi di contea ponderati su griglia da `output/data/auxiliary/spatial_weights.csv` e coordinate normalizzate $(\tilde{\text{Lat}}_c, \tilde{\text{Lon}}_c) \in [0, 1]$.
   - Target rese storiche da `output/data/target/soybean_yield_1951_2025.csv`.
2. **Algoritmo di Aggregazione Mensile**:
   - Definizione campagna per l'anno di raccolto $Y$: dal 1° novembre $Y-1$ al 31 ottobre $Y$ (12 mesi solari).
   - Per ciascun mese $j \in \{1, \dots, 12\}$ e ciascuna delle 135 contee:
     * **8 Flussi ed Estremi (Somma)**: $P, PE, SSRD, ET_0, P-ET_0, GDD, HD_{30}, HD_{35}$.
     * **9 Stati Ambientali (Media)**: $T_{\text{mean}}, T_{\text{max}}, T_{\text{min}}, T_{\text{dew}}, SM_1, SM_2, SM_3, VPD, SM_{\text{root}}$.
3. **Output Strutturati**:
   - Generazione di 12 matrici tabulari $X^{(H)}$ ($H \in \{12, \dots, 1\}$) con dimensione $P_H = 2 + 17 \cdot (13 - H)$ (da $P_{12}=2$ a $P_1=206$).
   - Salvataggio in `output/data/processed/model_datasets/` (formato Parquet compatto, partizionato o unico indicizzato per `(crop_year, county_fips)`).

---

## Step 3: Pipeline Train-Only di Detrending e Standardizzazione

1. **Modulo Detrending Ermetico**:
   - Funzione che stima la retta OLS $\hat{\tau}_{c,Y} = \hat{\alpha}_c^{(Y)} + \hat{\beta}_c^{(Y)} Y$ per ogni contea $c$ basandosi **esclusivamente** sul pool storico di training $\{1951, \dots, Y-1\}$.
   - Calcolo del target anomalia continua: $\epsilon_{c,t} = y_{c,t} - \hat{\tau}_{c,t}$.
2. **Standardizzazione Hermetica**:
   - Calcolo di $\mu_{\text{train}}$ e $\sigma_{\text{train}}$ sulle sole osservazioni storiche e proiezione a freddo sul blocco di test $Y$.

---

## Step 4: Single-Pass Grid Search su Validation Set (1985–1995)

1. **Protocollo di Tuning**:
   - Addestramento sul set base $\{1951, \dots, 1984\}$ ($N_0 = 4.590$ obs).
   - Valutazione predittiva sulle 11 annate bilanciate del validation set $\{1985, \dots, 1995\}$ ($N_{\text{val}} = 1.485$ obs; 6 anni positivi, 5 negativi inclusi 1988 e 1993).
2. **Spazi di Ricerca (da Tabella 4.3 del Capitolo 4)**:
   - **ElasticNet**: $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0\}$, $\alpha \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ (25 config).
   - **SVR (RBF)**: $C \in \{0.5, 2.0, 10.0\}$, $\epsilon_{\text{tube}} \in \{0.1, 0.5, 1.0\}$, $\gamma \in \{0.1, 1.0, 5.0\} \times \gamma_{\text{scale}}$ (27 config).
   - **Random Forest**: $B=300$, $p_{\text{split}} \in \{\text{'sqrt'}, 0.33, 0.5\} \times P_H$, $n_{\text{leaf}} \in \{5, 15, 30\}$, $d_{\text{max}} \in \{8, 12, \text{None}\}$ (27 config).
   - **XGBoost**: $\eta \in \{0.03, 0.08\}$, $d_{\text{max}} \in \{3, 5\}$, $\text{colsample} \in \{0.6, 0.8\}$, $\lambda_{\text{reg}} \in \{1.0, 10.0\}$ (16 config).
3. **Selezione Iperparametri Ottimi**:
   - Minimizzazione di $\text{RMSE}_{\text{val}}^{(H)}$ per ciascun orizzonte temporale.
   - Esportazione di $\mathbf{\Theta}_H^*$ in `output/data/processed/hyperparameters_optimal.json`.

---

## Step 5: Expanding-Window Out-of-Sample Evaluation (1996–2025)

1. **Ciclo di Valutazione Sequenziale**:
   - Per ciascun anno di test $Y \in \{1996, \dots, 2025\}$ (30 fold, 4.050 valutazioni):
     * Training pool espanso: $\{1951, \dots, Y-1\}$.
     * Calcolo detrending OLS e anomalie storiche.
     * Fit del modello con iperparametri congelati $\mathbf{\Theta}_H^*$.
     * Predizione out-of-sample su tutte le 135 contee dell'anno $Y$ come blocco indivisibile.
     * Ricostruzione resa totale: $\hat{y}_{c,Y}^{(H)} = \hat{\tau}_{c,Y} + \hat{\epsilon}_{c,Y}^{(H)}$.
2. **Calcolo e Salvataggio Metriche**:
   - $\text{RMSE}_{\text{OOS}}$, $\text{MAE}_{\text{OOS}}$, $R^2_{\text{OOS}}$, $\text{Skill Score}$ per tutti i 12 orizzonti.
   - Stress-testing condizionato sulle coorti di shock storico (1988, 1993, 2003, 2012, 2016).
   - Calcolo intervalli di confidenza empirici al 95% via Spatial Block-Bootstrap.
   - Salvataggio delle tabelle predittive e metriche in `output/data/processed/evaluation_metrics.parquet` / CSV per alimentazione immediata del Capitolo 5.

---

## Step 6: Stesura del Capitolo 5 (`05_empirical_results.tex`)

1. **Struttura del Capitolo Risultati**:
   - Sezione 5.1: Benchmark di riferimento e performance globale lungo i 12 orizzonti previsivi.
   - Sezione 5.2: Confronto della scala di complessità (lineari regolarizzati vs kernel vs alberi vs deep learning).
   - Sezione 5.3: Stress-testing sui rischi di coda e capacità di anticipazione degli shock storici estremi.
   - Sezione 5.4: Feature importance (SHAP / Gini) e analisi dei driver agrometeorologici critici lungo il ciclo fenologico.
   - Sezione 5.5: Sintesi dei risultati econometrici e validazione delle ipotesi H1–H5.
