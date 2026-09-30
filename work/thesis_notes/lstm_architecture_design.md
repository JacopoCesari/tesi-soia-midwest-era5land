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
| **A (county embedding + Temporal Attention)** | Input=17 meteo (incl. EDD30), Temporal Attention sui time-step della campagna, county embedding all'output | 🎯 Target consolidato (direttiva autore 2026-09-30) |
| **B (no detrend)** | No OLS detrend a monte, temporal attention estesa al trend | 📋 Futura estensione esplorativa |

---

## 3. Architettura consolidata — County Embedding & Temporal Attention

### 3.1 Flusso computazionale con Temporal Attention lungo la campagna

```
Input meteo [batch, seq_len, 17] (17 indicatori bioclimatici incl. EDD30)
        ↓
  LSTM(hidden_size, num_layers, dropout)
        ↓
  Tutti gli hidden states [h_1, h_2, ..., h_T]  [batch, seq_len, hidden_size]
        ↓
  Temporal Attention Layer (lungo la campagna meteo Nov→Ott):
    e_t = v^T tanh(W_a h_t + b_a)
    alpha_t = softmax(e_t)        ← Pesi di attenzione per tempo [batch, seq_len]
                                      (EXPLAINABILITY DIRETTA su giorni/settimane/mesi!)
    context vector c = sum_{t=1}^T (alpha_t * h_t)  [batch, hidden_size]
        ↓
  concat [ context vector c | e_c ]   ← county embedding e_c [batch, 16]
        ↓
  Linear(hidden_size + 16, 1)
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
Ablazione su d ∈ {8, 32} demandata a future sessioni se i risultati lo richiedono.

### 3.4 Nota futura — dense head

Variante non implementata nel grid corrente: sostituire la Linear finale con
`Linear(hidden+16, 32) → ReLU → Linear(32, 1)`. Potenzialmente utile se hidden_size
è grande e la proiezione diretta è troppo brusca. Segnato come TODO nel codice.

---

## 4. Pre-training + discriminative fine-tuning

### 4.1 Motivazione

Nell'expanding window, ogni fold riallena il modello da zero (Xavier random). Con i fold
iniziali (es. 1996, training 1951–1995 = 45 anni ≈ 6,075 osservazioni), la convergenza
parte da un punto casuale ogni volta. Il pre-training risolve questo:

- **Pre-training**: LSTM universale (senza county embedding) allenato su 1951–1979, validato
  su 1980–1984. Apprende la relazione meteo→anomalia generale su tutto il panel pooled.
- **Fine-tuning**: ogni fold parte dai pesi del pre-training (LSTM) + county embedding
  inizializzato random (Xavier). Il modello raffina la relazione partendo da un punto già
  "ragionevole" invece che da rumore.

**Garanzia no-leakage**: il pre-training usa solo 1951–1979, cioè dati precedenti sia alla
validation (1985–1995) che al test (1996–2025). Tutti i fold ricevono gli stessi pesi
iniziali — non c'è trasferimento di informazione inter-fold.

### 4.2 Discriminative fine-tuning — motivazione

Approccio: Howard & Ruder (2018, ULMFiT) applicato al nostro contesto.

| Componente | Learning rate | Motivazione |
|---|---|---|
| LSTM weights (da pre-training) | **3×10⁻⁵** (1/10 del LR base) | Preservare la rappresentazione pre-allenata, aggiornamenti piccoli |
| County embedding (nuovo) | **3×10⁻⁴** (LR base) | Impara da zero, ha bisogno di aggiornamenti più grandi |
| Output head Linear (nuovo) | **3×10⁻⁴** (LR base) | Impara da zero |

L'alternativa più aggressiva — congelare completamente LSTM per N epoche poi sbloccarlo —
è stata scartata perché aggiunge un iperparametro (N) e la differenza empirica rispetto al
discriminative lr è marginale per sequenze brevi (Géron 2025, Cap. 15).

### 4.3 Pesi pre-training salvati per (freq, hidden_size, num_layers)

Un PretrainLSTM separato per ciascuna combinazione unica (hidden_size, num_layers)
= 3 × 4 = 12 modelli × 3 frequenze = 36 file `.pt`. Dropout fisso a 0.2 durante
il pre-training (valore medio, indipendente dalla config del grid).

Path: `output/data/processed/lstm_sequences/pretrain_{freq}d_h{h}_l{l}.pt`

---

## 5. Grid search — iperparametri

### 5.1 Fissi da letteratura

| Param | Valore | Fonte |
|---|---|---|
| Optimizer | Adam | Khaki & Wang (2019); Khaki et al. (2020) |
| LR (pre-training e head) | 3×10⁻⁴ | Identico in entrambi i Khaki |
| LR (LSTM fine-tuning) | 3×10⁻⁵ | Discriminative fine-tuning: LR/10 |
| Weight init | Xavier uniform | Khaki & Wang (2019); Khaki et al. (2020) |
| max_epochs | 100 (fine-tuning), 200 (pre-training) | Standard |
| Early stopping patience | 7 (fine-tuning), 15 (pre-training) | Standard |
| Gradient clipping | norm=1.0 | Géron (2025), standard per LSTM |

### 5.2 Grid search v2 — 72 configurazioni

| Param | Valori | Note |
|---|---|---|
| `hidden_size` | {64, 128, 256} | Khaki 2020 usava 64; range esteso per sequenze più ricche |
| `num_layers` | {1, 2, 3, 4} | Esteso da {1,2} perché num_layers=2 vince sempre in v1 |
| `dropout` | {0.0, 0.2, 0.4} | Nessuna indicazione dalla letteratura; range esplorato |
| `batch_size` | {25, 64} | 25 da Khaki 2020 CNN-RNN; 64 da Khaki 2019 DNN |

3 × 4 × 3 × 2 = **72 config** × 11 folds × 3 freq × 11 horizons = 27,324 fit totali.

### 5.3 Nota su num_layers=3,4

Con sequenze da 12 a 71 steps e ~4k-10k campioni, LSTM a 4 layer è aggressivo.
Il dropout e l'early stopping mitigano l'overfitting. La scelta di testarlo è empirica:
num_layers=2 batte num_layers=1 in tutti i 4 H completati della v1; non si assume
che il vantaggio si arresti a 2 senza evidenza.

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
