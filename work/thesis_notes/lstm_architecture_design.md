# LSTM Architecture Design — Soybean Yield Anomaly Forecasting

**Last updated**: 2026-09-30  
**Status**: Design consolidated, code ready (run_lstm_pipeline_v2.py), not yet executed.

---

## 1. Motivazione e perimetro

Le LSTM vengono testate come Tier 4 della model complexity ladder (dopo baselines, modelli lineari,
tree ensembles). L'obiettivo è capire se sequenze meteo sub-mensili catturano segnale aggiuntivo
rispetto alle aggregazioni mensili usate dai modelli tabular.

**Target**: anomalia di yield detrended (OLS train-only) — identica ai modelli tabular.  
**Yield grezzo**: fuori perimetro per tutta la Variante A.  
**Frequenze testate**: 30-day (≡ mensile, baseline LSTM), 10-day (decadale), 5-day (pentade).

---

## 2. Varianti architetturali — stato

| Variante | Descrizione | Stato |
|---|---|---|
| **A (lat/lon baseline)** | Input=16 meteo + lat/lon per time step, Xavier init | ✅ Grid v1 archiviata come baseline ablation |
| **A (county embedding + Temporal Attention)** | Input=17 meteo (incl. EDD30), Temporal Attention sui time-step della campagna, county embedding all'output, Dense MLP head | 🎯 Implementato in `run_lstm_pipeline_v2.py` |
| **A (integrated: + lagged yield anomaly)** | Estensione a weather + train-only detrended $\epsilon_{t-1}$ | 🎯 Implementato in `run_lstm_pipeline_v2.py` |
| **B (no detrend)** | No OLS detrend a monte, temporal attention estesa al trend | 📋 Futura estensione esplorativa |

---

## 3. Architettura consolidata — County Embedding, Temporal Attention & Dense Head

### 3.1 Flusso computazionale con Temporal Attention lungo la campagna

```
Input meteo [batch, seq_len, 17] (17 indicatori bioclimatici incl. EDD30)
        ↓
  LSTM(hidden_size, num_layers, dropout)
        ↓
  Tutti gli hidden states [h_1, h_2, ..., h_T]  [batch, seq_len, hidden_size]
        ↓
  Temporal Attention Layer (Bahdanau, lungo la campagna meteo Nov→Ott):
    e_t = v^T tanh(W_a h_t + b_a)
    alpha_t = softmax(e_t)        ← Pesi di attenzione per tempo [batch, seq_len]
                                       (EXPLAINABILITY DIRETTA su giorni/settimane/mesi!)
    context vector c = sum_{t=1}^T (alpha_t * h_t)  [batch, hidden_size]
        ↓
  concat [ context vector c | county embedding e_c (16) (| lag_y (1)) ]
        ↓
  Dense MLP Head: Linear(in_features, 32) → ReLU → Dropout → Linear(32, 1)
        ↓
  Anomalia scalare preditta \hat{epsilon}_{c, Y}
```

### 3.1.1 Vantaggi metodologici ed explainability della Temporal Attention
1. **Explainability nativa (Thesis Chapter 5 & 6)**: I pesi $\alpha_t$ consentono di tracciare grafici temporali dell'attenzione lungo la campagna agricola. Mostrano direttamente se la rete neurale assegna il picco di peso al periodo critico agronomico (fioritura e grain filling di Luglio–Agosto, confermando l'ipotesi H1) rispetto ai mesi invernali e pre-semina.
2. **Superamento del collo di bottiglia dell'ultimo stato $h_T$**: Nelle sequenze a 5 giorni (71 time-steps) e a 10 giorni (35 time-steps), comprimere l'intera stagione nell'ultimo hidden state $h_T$ rischia di affievolire il segnale dei mesi estivi centrali; la temporal attention permette al modello di accedere direttamente agli stati $h_t$ di Luglio e Agosto.

### 3.2 Perché il county embedding all'output e non all'input

Il county embedding viene concatenato all'hidden state **dopo** l'LSTM, non alla sequenza input.
Questo mantiene la sequenza temporale pulita da informazione spaziale — l'LSTM apprende
dinamiche meteo universali, l'embedding corregge il bias spaziale residuo. Se l'embedding
entrasse nel time step input, interferirebbe con la struttura temporale ad ogni passo.

### 3.3 Dimensione dell'embedding — motivazione per d=16 (fisso)

Criteri considerati:
- **N_counties = 135**: regola empirica per categorical embedding → dim ≈ min(50, n//2) = min(50, 67) = 50.
  Ma questo vale per feature tabular; per spazio latente spaziale si usa molto meno.
- **Parametri embedding totali**: 135 × d. Con d=16 → 2,160 parametri. Con ~4,000-10,000
  osservazioni di training, il rapporto obs/params è adeguato (>2).
- **Variabilità spaziale del dataset**: le 135 contee coprono 6 stati con gradiente lat/lon
  relativamente regolare. Uno spazio latente di d=16 è più che sufficiente per catturare
  differenze sistematiche inter-county (suolo, microclima, risposta allo stress termico).
- **Géron (2025)**: per embedding di entità geografiche, dimensioni eccessive rischiano
  overfitting spaziale quando il numero di entità è limitato.

**Decisione**: d=16 fisso. Non incluso nel grid per non far esplodere le combinazioni.

### 3.4 Dense MLP Head & Activation Function

La proiezione finale è implementata come:
`Linear(in_features, 32) → GELU() → Dropout(dropout) → Linear(32, 1)`
dove `in_features = hidden_size + 16` (Weather-Only) oppure `hidden_size + 16 + 1` (Integrated con $\epsilon_{t-1}$).

**Motivazione per GELU (Gaussian Error Linear Unit)**:
1. Il target è un'**anomalia di resa detrendata a media zero** ($\epsilon \in [-25, +20]$ bu/ac). Metà delle osservazioni sono negative (stress idrico, ondate di calore).
2. Con la ReLU standard ($\max(0, z)$), qualsiasi attivazione intermedia negativa ha gradiente nullo (*dying ReLU problem*), riducendo la capacità della rete di modellare shock avversi.
3. GELU è liscia, differenziabile ovunque, non-monotona per piccoli valori negativi e fornisce gradienti non nulli anche per attivazioni negative moderate, risultando la scelta standard nelle architetture neurali moderne. (Il codice supporta anche `--activation leaky_relu` e `--activation relu`).

### 3.5 Learning Rate & Scheduler dinamico

- **Optimizer**: Adam con pesi differenziati (discriminative fine-tuning):
  - LSTM + Attention: $\text{LR} = 3 \times 10^{-5}$
  - County Embedding + MLP Head: $\text{LR} = 3 \times 10^{-4}$
- **Scheduler**: `torch.optim.lr_scheduler.ReduceLROnPlateau(mode='min', factor=0.5, patience=3, min_lr=1e-6)`.
  Dimezza il learning rate quando la validation RMSE si arresta per 3 epoche consecutive. Questo consente al modello di beneficiare di rate di apprendimento progressivamente più fini ($3 \times 10^{-4} \to 1.5 \times 10^{-4} \to 7.5 \times 10^{-5}$) **senza moltiplicare il numero di configurazioni del grid search**.

---

## 4. Pre-training + discriminative fine-tuning

### 4.1 Motivazione

Nell'expanding window, ogni fold riallena il modello da zero (Xavier random). Con i fold
iniziali (es. 1996, training 1951–1995 = 45 anni ≈ 6,075 osservazioni), la convergenza
parte da un punto casuale ogni volta. Il pre-training risolve questo:

- **Pre-training**: LSTM universale + Temporal Attention (senza county embedding) allenato su 1951–1979 (29 anni), validato
  su 1980–1984 (5 anni) con early stopping. Apprende la relazione meteo→anomalia generale su tutto il panel pooled.
- **Fine-tuning**: ogni fold parte dai pesi del pre-training (LSTM + Attention) + county embedding
  inizializzato random (Xavier). Il modello raffina la relazione partendo da un punto già
  "ragionevole" invece che da rumore.

**Garanzia no-leakage**: il pre-training usa solo 1951–1979 con pre-validation 1980–1984, cioè dati rigorosamente precedenti sia alla
validation (1985–1995) che al test (1996–2025). Tutti i fold ricevono gli stessi pesi
iniziali — non c'è trasferimento di informazione inter-fold. Gli anni 1980–1984 vengono poi regolarmente inclusi nel training set dei fold a partire dal 1985.

---

## 5. Grid search — iperparametri e potatura intelligente (Pruned Grid)

### 5.1 Fissi da letteratura e ingegneria

| Param | Valore | Fonte / Motivazione |
|---|---|---|
| Optimizer | Adam | Khaki & Wang (2019); Khaki et al. (2020) |
| LR (head & embedding) | 3×10⁻⁴ | Khaki & Wang (2019, 2020) |
| LR (LSTM & attention fine-tuning) | 3×10⁻⁵ | Discriminative fine-tuning (ULMFiT / Howard & Ruder 2018) |
| LR Scheduler | ReduceLROnPlateau(factor=0.5, patience=3) | Adattamento dinamico fine-tuning |
| Activation | GELU (default, opt: LeakyReLU, ReLU) | Prevenzione dying ReLU su anomalie zero-centered |
| Weight init | Xavier uniform | Khaki & Wang (2019); Khaki et al. (2020) |
| max_epochs | 100 (fine-tuning), 200 (pre-training) | Standard |
| Early stopping patience | 7 (fine-tuning), 15 (pre-training) | Standard |
| Gradient clipping | norm=1.0 | Géron (2025), standard per LSTM |

### 5.2 Strategia Pruned Grid (44 configurazioni)

Tutti i livelli di ciascun iperparametro rimangono pienamente rappresentati:
- `hidden_size`: $\{64, 128, 256\}$
- `num_layers`: $\{1, 2, 3, 4\}$
- `dropout`: $\{0.0, 0.2, 0.4\}$
- `batch_size`: $\{25, 64\}$

**Criteri di esclusione delle sole combinazioni patologiche/ridondanti**:
1. `num_layers = 1` con `dropout > 0.0`: in PyTorch, il dropout interno di `nn.LSTM` opera solo tra strati ricorrenti multipli. Con 1 layer, PyTorch ignora il dropout (lo imposta a 0). Testare `(l=1, d=0.2)` e `(l=1, d=0.4)` significherebbe testare 3 volte lo stesso modello.
2. `num_layers >= 3` con `dropout = 0.0`: una rete profonda senza regolarizzazione su 5.000 campioni agricoli è matematicamente destinata a grave overfitting.
3. `num_layers = 4` con `hidden_size = 256`: genera oltre 1.8M di parametri, con rapporto campioni/pesi $< 0.003$, ingestibile per sequenze di 12 step.

Risultato: **44 configurazioni**, riduzione del ~40% dei tempi di calcolo senza alcuna perdita di informazione.

### 5.3 Orizzonti di Validazione Rappresentativi ($H=1, 3, 6, 10$)

Per il tuning su validation (1985–1995), si valutano gli orizzonti cardinali fenologici:
- $H=1$ (Ottobre, fine stagione, 12 mesi meteo)
- $H=3$ (Agosto, post-fioritura / grain-filling, 10 mesi meteo)
- $H=6$ (Maggio, semina, 7 mesi meteo)
- $H=10$ (Gennaio, pre-stagione / inverno, 3 mesi meteo)

La configurazione ottimale selezionata viene poi applicata per l'addestramento e il test forecast (1996–2025) su **tutti gli 11 orizzonti**, dove ogni orizzonte impara la propria attenzione e il proprio modello sui dati disponibili fino a quel mese.

---

## 6. Expanding window — full retraining vs. continual learning

**Scelta**: full retraining da zero ad ogni fold (identico ai modelli tabular).  
Il pre-training NON è continual learning — i pesi pre-allenati sono FISSI (1951–1979) e
non cambiano tra fold. Ogni fold parte dagli stessi pesi iniziali.

**Alternativa scartata**: continual learning (aggiornare i pesi del fold precedente con i
nuovi dati). Metodologicamente più debole: i pesi del fold Y ereditano implicitamente
informazione da tutti i fold precedenti, rendendo difficile garantire no-leakage. Non
usato in questo progetto.

---

## 7. Checkpoint e comparabilità tra v1 e v2

| Versione | Checkpoint | Architettura | Uso |
|---|---|---|---|
| v1 (lat/lon) | `work/reports/experiments/lstm/grid_checkpoint.json` | input=18 (16+lat+lon), no county emb, Xavier | Ablation: spatial encoding lat/lon |
| v2 (county emb) | `work/reports/experiments/lstm/grid_checkpoint_v2.json` | input=16, county emb d=16, pre-training | Architettura principale |

Le due versioni sono confrontabili orizzontalmente: stesse sequenze, stesso target,
stesso protocollo expanding window. La differenza è solo l'encoding spaziale.

---

## 8. Frequenze e sequence lengths

| Freq | Seq len H=1 | Seq len H=7 | Seq len H=12 |
|---|---|---|---|
| 30-day (mensile) | 12 steps | 6 steps | 0 (naive) |
| 10-day (decadale) | 35 steps | 17 steps | 0 |
| 5-day (pentade) | 71 steps | 35 steps | 0 |

Lunghezze fisse (non-leap Feb=28 per consistenza inter-annuale).
Sequenze cached su disco in `output/data/processed/lstm_sequences/seq_{freq}d_H{H:02d}.npz`.

---

## 9. Ablazioni pianificate (future sessioni)

| Ablazione | Descrizione | Priorità |
|---|---|---|
| **Spatial encoding** | v1 (lat/lon) vs v2 (county emb) | Alta — già in corso |
| **Pre-training delta** | v2 con Xavier vs v2 con pre-training | Alta — flag `--no-pretrain` |
| **Dense head** | Linear(h+16,1) vs Linear(h+16,32)→ReLU→Linear(32,1) | Media |
| **Variante B** | Temporal attention cattura trend (no OLS detrend) | Media |
| **Feature engineering** | Somme cumulate / EMA / extreme-day counts per window | Bassa |
| **Embedding dim** | d ∈ {8, 32} vs d=16 fissato | Bassa |
