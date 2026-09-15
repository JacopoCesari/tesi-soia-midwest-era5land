# Ingegneria delle Feature Agrononiche e Meteorologiche

## 1. Obiettivo e Quadro Generale

La resa finale della soia è fortemente influenzata dalla dinamica temporale con cui le condizioni meteorologiche si manifestano durante le diverse fasi di sviluppo fenologico (emergenza vegetativa, fioritura, allegagione dei baccelli e riempimento dei semi). Semplici medie stagionali indiscriminate tendono ad attenuare gli estremi termici e i deficit idrici concentrati nelle finestre fenologiche più sensibili ([Hoffman et al., 2020](references.md); [Schlenker & Roberts, 2009](references.md)).

L'obiettivo dell'ingegneria delle feature è trasformare la serie di base a 37 campi meteorologici giornalieri in indici agronomici interpretabili e aggregazioni temporali discrete, garantendo la totale assenza di data leakage rispetto ai diversi cutoff previsionali.

---

## 2. Feature Giornaliere Derivate Previste

A partire dai dati giornalieri aggregati a livello di contea, la metodologia prevede il calcolo dei seguenti indicatori fisiologici:

### 2.1 Growing Degree Days (GDD)
Il tempo termico utile allo sviluppo della soia viene quantificato tramite i gradi giorno di crescita con temperatura base di 10 °C e soglia termica superiore di 30 °C:
$$\text{GDD} = \max\left(0, \frac{\min(T_{\max}, 30) + \max(T_{\min}, 10)}{2} - 10\right)$$
- $T_{\text{base}} = 10^\circ\text{C}$: soglia minima al di sotto della quale l'attività fisiologica della soia si arresta.
- $T_{\text{cutoff}} = 30^\circ\text{C}$: temperatura oltre la quale l'accumulo termico non incrementa ulteriormente il tasso di sviluppo.

### 2.2 Vapor Pressure Deficit (VPD) Daily Proxy
Il deficit di pressione di vapore rappresenta la forza motrice dell'evapotraspirazione e dello stress idrico atmosferico. Viene stimato su scala giornaliera mediante la formulazione di Tetens a partire da $T_{\text{mean}}$ e $T_{\text{dewpoint}}$:
$$e_s(T_{\text{mean}}) = 0{,}61078 \exp\left(\frac{17{,}27 \cdot T_{\text{mean}}}{T_{\text{mean}} + 237{,}3}\right)$$
$$e_a(T_{\text{dewpoint}}) = 0{,}61078 \exp\left(\frac{17{,}27 \cdot T_{\text{dewpoint}}}{T_{\text{dewpoint}} + 237{,}3}\right)$$
$$\text{VPD} = \max\big(0, \; e_s(T_{\text{mean}}) - e_a(T_{\text{dewpoint}})\big) \quad [\text{kPa}]$$

### 2.3 Indicatori di Stress Termico Estremo (Heat-Days)
La letteratura agronomica ed econometrica evidenzia come temperature massime prolungate oltre i 30 °C causino aborto fiorale e riduzione drastica della resa nella soia ([Schlenker & Roberts, 2009](references.md)):
- `heat_days_30`: variabile indicatrice $(T_{\max} \ge 30^\circ\text{C})$.
- `heat_days_35`: variabile indicatrice per calore estremo $(T_{\max} \ge 35^\circ\text{C})$.
- `diurnal_temperature_range`: escursione termica giornaliera $\text{DTR} = T_{\max} - T_{\min}$.

### 2.4 Bilanci Idrici e Precipitazioni Rolling
- **Deficit Idrico Giornaliero:**
  $$P_{\text{net}} = \text{total\_precipitation} - \text{potential\_evaporation} \quad [\text{mm}]$$
- **Cumulati di Precipitazione Rolling:** somme mobili a finestre agronomiche di 7, 14, 30, 60 e 90 giorni precedenti.
- **Indici di Siccità Cumulata:** bilancio cumulato $P - \text{PEV}$ sulle medesime finestre mobili per tracciare l'accumulo progressivo di stress idrico nel terreno.

---

## 3. Risoluzione e Aggregazioni Temporali

Il confronto metodologico principale della tesi non richiede un archivio orario persistente, ma si focalizza sull'efficacia predittiva di diverse granularità temporali intermedie:

1. **Aggregazione a 5 giorni (Pentadi):**
   - Rappresentazione fine della dinamica meteorologica intra-stagionale.
   - Cattura onde di calore brevi e singoli eventi di precipitazione intensa.
2. **Aggregazione a 10 giorni (Decadi):**
   - Compromesso ottimale tra dimensionalità del feature set e capacità di seguire le fasi fenologiche della coltura.
   - Riduce la varianza ad alta frequenza mantenendo la tempestività del segnale.
3. **Aggregazione a 30 giorni (Mensile):**
   - Utilizzata come **benchmark a bassa risoluzione**, rappresentativo della maggior parte dei modelli econometrici tradizionali.

Per ciascun intervallo (5, 10 o 30 giorni), le feature aggregate includono: media delle temperature (media, minima, massima, suolo), somma delle precipitazioni e dell'evaporazione, radiazione solare totale incidente, GDD cumulati, numero di giorni con $T_{\max} \ge 30^\circ\text{C}$ e VPD media.

---

## 4. Regole Anti-Leakage e Allineamento ai Cutoff

L'aspetto cardine dell'ingegneria delle feature è il rispetto rigoroso del principio di causalità temporale:

- **Allineamento al Cutoff:** per un cutoff fissato al giorno $d$ (ad esempio 31 luglio):
  - Il vettore di feature $\mathbf{x}_{c, t}^{(d)}$ per la contea $c$ nell'anno $t$ contiene **unicamente** metriche calcolate su dati meteorologici osservati fino al giorno $d$.
  - Tutte le finestre temporali successive (agosto, settembre, ottobre) sono formalmente escluse.
- **Nessuna Imputazione con Informazione Futura:** le metriche rolling o cumulative che cadono a cavallo del cutoff vengono troncate al giorno $d$.
- **Normalizzazione Train-Only:** se le feature richiedono standardizzazione (es. z-score), i parametri di media e varianza devono essere calcolati unicamente sulle osservazioni del training set e applicati per proiezione al set di test.

---

## 5. Stato dell'Implementazione

| Componente | Stato Metodologico | Stato nel Codice |
|---|---|---|
| Formula GDD (10–30 °C) | Definita | Da implementare in script di feature generation |
| Formula VPD (Tetens) | Definita | Da implementare in script di feature generation |
| Heat days (30 °C, 35 °C) | Definiti | Da implementare in script di feature generation |
| Rolling water balance | Definito | Da implementare in script di feature generation |
| Aggregazioni 5 e 10 giorni | Specifica approvata | Da implementare in script di feature generation |
| Dataset finale per modello | In progettazione | Da costruire a valle della validazione del download ERA5-Land |
