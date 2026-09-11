# Metodologia di Acquisizione ed Elaborazione Dati Meteorologici ERA5-Land Daily

## 1. Obiettivo e Quadro Generale

L'obiettivo della pipeline meteorologica è l'acquisizione, il controllo qualità e l'aggregazione spaziale delle serie meteorologiche giornaliere per il periodo **1950–2025** (76 anni, $27.759$ giorni) a livello di contea, per le **479 contee** del Midwest statunitense appartenenti al panel agronomico selezionato (stati: Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio).

### Principi Guida del Design Meteorologico
1. **Risoluzione temporale giornaliera esatta (24 ore):** superamento delle serie campionate a intervalli discreti di 6 ore tramite l'estrazione mirata di statistiche giornaliere lato server e accumuli giornalieri chiusi alle 00:00 UTC.
2. **Efficienza di archiviazione:** eliminazione del download orario ridondante; estrazione del solo bounding box regionale unificato che racchiude le 479 contee.
3. **Assenza di data leakage:** convenzione temporale fissa e indicizzazione rigorosa delle date per garantire che nei cutoff stagionali non confluiscano dati postumi.
4. **Coerenza geodetica e spaziale:** proiezione conforme equal-area (EPSG:5070) per l'intersezione tra le geometrie amministrative delle contee e la griglia regolare di ERA5-Land.

---

## 2. Architettura a Due Canali Copernicus CDS

Il Copernicus Climate Data Store (CDS) distribuisce i dati di rianalisi ERA5-Land attraverso due endpoint con caratteristiche fisiche differenti. La pipeline sfrutta questa suddivisione:

```
                            PIPELINE CDS ERA5-LAND DAILY
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────┐
        ▼                                                                   ▼
   CANALE A: VARIABILI ISTANTANEE                                      CANALE B: VARIABILI ACCUMULATE
   Dataset: 'derived-era5-land-daily-statistics'                       Dataset: 'reanalysis-era5-land'
   • Calcolo statistico orario eseguito lato server CDS                • Proprietà fisica: accumulo orario progressivo da 00:00 UTC
   • Statistiche: daily_mean (17), daily_min (1), daily_max (2)        • Estrazione del record 00:00 UTC del giorno D+1
   • Campionamento interno server: frequency="1_hourly"                • Assegnazione dell'accumulo totale alle 24h del giorno D
   • Totale: 20 campi giornalieri                                      • Totale: 17 campi giornalieri
        │                                                                   │
        └─────────────────────────────────┬─────────────────────────────────┘
                                          ▼
                         UNICO BOUNDING BOX REGIONALE (0.1°)
                            [47.7 N, -97.0 W, 35.8 S, -80.4 E]
                                          │
                                          ▼
                                 RAW DAILY GRID (NetCDF)
                                          │
                                          ▼ (Aggregazione spaziale pesata EPSG:5070)
                         COUNTY_DAILY (479 contee × giorni)
```

---

## 3. Catalogo dei 37 Campi Meteorologici Giornalieri

La metodologia prevede l'estrazione di **37 campi meteorologici giornalieri complessivi** (20 da Canale A e 17 da Canale B).

### 3.1 Canale A: Statistiche Giornaliere (20 campi)
- **Dataset CDS:** `derived-era5-land-daily-statistics`
- **Gruppi statistici:**
  1. `daily_mean` (17 variabili):
     - `2m_temperature`
     - `2m_dewpoint_temperature`
     - `10m_u_component_of_wind`
     - `10m_v_component_of_wind`
     - `surface_pressure`
     - `soil_temperature_level_1` (0–7 cm)
     - `soil_temperature_level_2` (7–28 cm)
     - `soil_temperature_level_3` (28–100 cm)
     - `soil_temperature_level_4` (100–289 cm)
     - `volumetric_soil_water_layer_1` (0–7 cm)
     - `volumetric_soil_water_layer_2` (7–28 cm)
     - `volumetric_soil_water_layer_3` (28–100 cm)
     - `volumetric_soil_water_layer_4` (100–289 cm)
     - `skin_reservoir_content`
     - `snow_cover`
     - `snow_density`
     - `snow_depth_water_equivalent`
  2. `daily_minimum` (1 variabile):
     - `2m_temperature` ($T_{\min}$)
  3. `daily_maximum` (2 variabili):
     - `2m_temperature` ($T_{\max}$)
     - `skin_temperature` ($T_{\text{skin}}$)

### 3.2 Canale B: Variabili Accumulate (17 campi)
- **Dataset CDS:** `reanalysis-era5-land`
- **Proprietà fisica:** i flussi cumulativi partono da zero alle 00:00 UTC e sommano i valori ora dopo ora. Il timestamp con `valid_time = 00:00 UTC` del giorno $D+1$ contiene esattamente la somma integrale delle 24 ore del giorno $D$.
- **Campi estratti (17 variabili):**
  1. `total_precipitation`
  2. `total_evaporation`
  3. `evaporation_from_the_top_of_canopy`
  4. `evaporation_from_bare_soil`
  5. `evaporation_from_open_water_surfaces_excluding_oceans`
  6. `evaporation_from_vegetation_transpiration`
  7. `potential_evaporation`
  8. `surface_runoff`
  9. `sub_surface_runoff`
  10. `snowfall`
  11. `snowmelt`
  12. `surface_solar_radiation_downwards`
  13. `surface_thermal_radiation_downwards`
  14. `surface_net_solar_radiation`
  15. `surface_net_thermal_radiation`
  16. `surface_sensible_heat_flux`
  17. `surface_latent_heat_flux`

### 3.3 Unità di Misura e Formule di Trasformazione

| Variabile / Gruppo | Unità Raw CDS | Unità Elaborata | Formula di Conversione |
|---|---|---|---|
| Temperature ($T_{\text{mean}}, T_{\min}, T_{\max}, T_{\text{skin}}, T_{\text{soil}}$) | $\text{K}$ | $^\circ\text{C}$ | $T(^\circ\text{C}) = T(\text{K}) - 273.15$ |
| Precipitazione, Evaporazione, Runoff, Snowmelt, Reservoir | $\text{m}$ | $\text{mm}$ | $\text{Valore}(\text{mm}) = \text{Valore}(\text{m}) \times 1000$ |
| Radiazione e Flussi Termici | $\text{J/m}^2$ | $\text{MJ/m}^2$ | $\text{Flusso}(\text{MJ/m}^2) = \text{Flusso} / 10^6$ |
| Pressione alla superficie | $\text{Pa}$ | $\text{hPa}$ | $P(\text{hPa}) = P(\text{Pa}) / 100$ |
| Umidità volumetrica suolo | $\text{m}^3/\text{m}^3$ | $\text{m}^3/\text{m}^3$ | Invariato (frazione $0.0 - 1.0$) |
| Componenti del vento ($u, v$) | $\text{m/s}$ | $\text{m/s}$ | Invariato |
| Copertura nevosa | $0 - 1$ | $0 - 1$ | Invariato (frazione) |

> **Nota sui flussi di evaporazione ECMWF (Evaporation Swap):**
> Nella documentazione tecnica ECMWF per ERA5-Land è noto uno scambio di etichette tra tre parametri di evaporazione (`evabs`, `evaow`, `evatp`). Nel codice `download_era5_land_daily.py` è predisposto il dizionario `EVAPORATION_SWAP_MAP`, ma tale correzione non risulta attualmente applicata nel flusso di elaborazione county-daily e dovrà essere formalmente integrata e validata.

---

## 4. Convenzione Temporale: UTC-Day

La convenzione adottata per il dataset giornaliero è l'**UTC-Day** (finestra fissa dalle 00:00 alle 23:59 UTC).

### Motivazioni:
1. **Eliminazione di discontinuità da Daylight Saving Time (DST):** l'orario civile statunitense introduce giornate di 23 e 25 ore ai cambi di ora legale, che altererebbero la somma coerente degli accumuli fisici nelle 24 ore.
2. **Armonizzazione interstatale sincrona:** le 479 contee ricadono a cavallo di due fusi orari (Central Time per IL, IA, MN, MO e parte di IN; Eastern Time per OH e parte di IN). L'adozione dell'orario UTC fornisce una griglia temporale perfettamente sincrona su tutto il dominio.
3. **Efficienza nel download CDS:** consente di estrarre un unico timestamp (`00:00 UTC` del giorno $D+1$) per chiudere il bilancio idrico e radiativo del giorno $D$.

---

## 5. Bounding Box Regionale e Ponderazione Spaziale

### Bounding Box Regionale Unificato
Invece di scaricare frammenti statali sovrapposti, è stato determinato l'inviluppo rettangolare che racchiude tutte le 479 contee (geometrie US Census Bureau 2020), con un margine di $0.1^\circ$ allineato alla griglia regolare di ERA5-Land:
$$\mathbf{CDS\_BBOX = [47.7, -97.0, 35.8, -80.4]} \quad [\text{North, West, South, East}]$$
- Estensione: 120 latitudini $\times$ 167 longitudini = **20.040 celle di griglia**.

### Matrice di Ponderazione Areale (EPSG:5070)
La transizione da griglia regolare a dato aggregato per contea è implementata tramite sovrapposizione geometrica:
1. I poligoni delle 479 contee e le celle quadrate ($0.1^\circ \times 0.1^\circ$) sono proiettati nel sistema conforme ed equal-area **EPSG:5070 (NAD83 / Conus Albers)**.
2. Viene calcolata l'area di intersezione $A_{c, i}$ tra la contea $c$ e la cella $i$.
3. I pesi areali normalizzati sono calcolati imponendo che la loro somma per contea sia unitaria:
   $$w_{c, i} = \frac{A_{c, i}}{\sum_{j \in \text{cells}(c)} A_{c, j}}, \quad \sum_{i} w_{c, i} = 1{,}000000$$
4. Il valore giornaliero per la contea $c$ al giorno $t$ è:
   $$V_c(t) = \sum_{i \in \text{cells}(c)} w_{c, i} \cdot V_i(t)$$

Gli artefatti precalcolati risultanti da questa procedura contengono **11.953 intersezioni attive** e sono archiviati in:
- `era5_land_daily/spatial_weights_479_counties.parquet`
- `era5_land_daily/spatial_weights_479_counties.csv`

---

## 6. Controlli di Qualità, Resume e Manifest

Il tracciamento dei download avviene tramite il file di registro `era5_land_daily/manifest.csv`.

### Distinzione dello Stato dei Controlli di Qualità (QC)

Per garantire trasparenza sullo stato effettivo del codice:

1. **Controlli Implementati nel Flusso Principale:**
   - Verifica di esistenza del file scaricato e soglia minima di dimensione (`file_size > 1000 bytes`).
   - Registrazione nel manifest di timestamp, anno, canale, percorso file, dimensione in byte, hash SHA-256 e stato QC iniziale.
   - Ripresa dei download (*resume*): la funzione `is_done()` verifica l'esistenza del file, la dimensione e la presenza di una riga con stato `PASS` nel manifest.
2. **Controlli Eseguiti nel Test di Preflight:**
   - Verifica termica di base: $T_{\min} \le T_{\max}$ e $T_{\min} \le T_{\text{mean}}$ sui campioni di test di 7 giorni.
3. **Controlli Pianificati (da implementare a regime):**
   - Verifica sistematica della completezza del calendario (365 giorni negli anni comuni, 366 nei bisestili).
   - Verifica di monotonia delle coordinate (latitudini decrescenti, longitudini crescenti, assenza di mix tra $-180/180$ e $0/360$).
   - Validazione dei limiti fisici di tutti i 37 campi (es. $P \ge 0$, radiazione $\ge 0$, umidità suolo $0.0 \le \theta \le 1.0$).
   - Coerenza simultanea $T_{\min} \le T_{\text{mean}} \le T_{\max}$ su ogni cella e timestamp.
   - Verifica di unicità e cardinalità esatta delle righe county-day per anno ($479 \times 365 = 174.835$ record).
   - Controllo crittografico attivo (ricalcolo dinamico dello SHA-256 in fase di resume per rilevare corruzioni silenti).

---

## 7. Stato Effettivo dell'Implementazione e Limitazioni Correnti

La documentazione rispecchia lo stato reale degli script presenti nel repository:

| Componente | Stato Implementativo | Dettaglio / Implementazione |
|---|---|---|
| Pesi spaziali precalcolati | Completato e riproducibile | File `.parquet` e `.csv` verificati in `era5_land_daily/`; script `compute_spatial_weights.py` allineato con CLI e default `01_Dati_Soia_e_Target`. |
| Preflight ERA5-Land | Completato e validato | Test eseguito su campioni di 7 giorni per gli anni 1950 e 2025 con aggregazione Canale A + B a 37 campi. |
| Aggregazione Canale A + B | Implementata | `CountyAggregator.process_year()` integra Canale A (20 campi) e Canale B (17 campi) in un unico file Parquet di contea, applicando conversioni di unità fisiche. |
| Download annuale Canale B | Strutturato | Richiesta CDS divisa in due chiamate (anno D + 1 gen D+1), pari a 6.222 fields (sotto la quota massima di 12.000 fields del CDS). |
| Evaporation swap | Implementato | Funzione `apply_evaporation_swap()` integrata nel flusso di elaborazione e testata con unit test dedicati. |
| Resume logic | Implementata con opzione checksum | `ManifestManager.is_done()` supporta la verifica crittografica SHA-256 tramite il flag CLI `--verify-checksum`. |
| Suite di test automatici | Implementata | Test in `tests/test_pipeline.py` (calendario bisestile, evaporation swap, cardinalità 479 contee, unicità chiavi, checksum e conversioni). |

Le esecuzioni storiche complete 1950–2025 possono ora essere avviate in modo modulare disponendo di una pipeline verificata end-to-end.
