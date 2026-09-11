# Metodologia di Acquisizione ed Elaborazione Dati Meteorologici ERA5-Land Daily (1950–2025)
**Tesi di Laurea Magistrale: Previsione della Resa della Soia tramite Modelli di Machine Learning e Deep Learning**  
*Autore: Jacopo Cesari*  
*Sintesi della specifica tecnica e operativa (sostituisce la documentazione preliminare estesa)*

---

## 1. Obiettivo e Quadro Generale

L'obiettivo è la costruzione di un dataset meteorologico giornaliero continuo e armonizzato per il periodo **1950–2025** (76 anni, $27.759$ giorni) a livello di contea, per le **479 contee** del Midwest statunitense appartenenti al *balanced panel* storico USDA NASS (Illinois, Indiana, Iowa, Minnesota, Missouri, Ohio).

Il dataset è progettato specificamente per alimentare modelli previsionali di resa della soia (*weather-only* e ibridi), garantendo:
1. **Risoluzione temporale giornaliera esatta (24 ore)** senza campionamenti spuri a 6 ore.
2. **Efficienza di archiviazione**: eliminazione del download orario ridondante (volume grezzo ridotto da oltre $1{,}2\text{ TB}$ a soli $30\text{--}60\text{ GB}$, con output finale a livello di contea compresso in circa $2\text{--}4\text{ GB}$).
3. **Assenza di data leakage**: allineamento temporale rigoroso con le date di cutoff stagionale delle previsioni agronomiche.
4. **Coerenza geodetica**: eliminazione delle duplicazioni di longitudine ($-180/180$ vs $0/360$) riscontrate nelle versioni preliminari.

---

## 2. Architettura a Due Canali Copernicus CDS

ERA5-Land separa le variabili atmosferiche in due categorie fisiche distinte nel Copernicus Climate Data Store (CDS). La nostra pipeline sfrutta questa distinzione per evitare il download di dati orari:

```
                            PIPELINE CDS ERA5-LAND DAILY
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────┐
        ▼                                                                   ▼
   CANALE A: VARIABILI ISTANTANEE                                      CANALE B: VARIABILI ACCUMULATE
   Dataset: 'derived-era5-land-daily-statistics'                       Dataset: 'reanalysis-era5-land'
   • Calcolo statistico orario eseguito LATO SERVER                    • Proprietà fisica: accumulo da 00:00 a 24:00 UTC
   • Statistiche: daily_mean, daily_minimum, daily_maximum             • Scarico del solo record 00:00 UTC del giorno D+1
   • Frequenza di campionamento server: frequency="1_hourly"           • Assegnazione del totale al giorno D
   • Volume scaricato: 1 valore/die per punto griglia                  • Volume scaricato: 1 timestamp/die (GRIB/NetCDF)
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
                         COUNTY_DAILY.PARQUET (479 contee × 27.759 gg)
```

### Canale A: Variabili Istantanee
* **Dataset CDS**: `derived-era5-land-daily-statistics`
* **Logica**: Il server Copernicus aggrega i campioni orari originali del modello (`frequency="1_hourly"`) calcolando la media, il minimo e il massimo per ogni giorno.
* **Gruppi di chiamata**:
  1. `daily_mean`: temperatura media a 2m, temperatura di rugiada, componenti del vento $u$ e $v$, pressione superficiale, temperatura del suolo (4 strati), umidità volumetrica del suolo (4 strati), serbatoio superficiale, copertura nevosa, densità neve, equivalente in acqua dello snowpack.
  2. `daily_minimum`: temperatura minima a 2m ($T_{\min}$).
  3. `daily_maximum`: temperatura massima a 2m ($T_{\max}$) e temperatura superficiale del suolo/chioma (`skin_temperature`).

### Canale B: Variabili Accumulate
* **Dataset CDS**: `reanalysis-era5-land` (formato NetCDF/GRIB nativo).
* **Proprietà fisica ECMWF**: In ERA5-Land, le variabili cumulative partono da zero alle 00:00 UTC e sommano progressivamente l'energia o la massa ora per ora fino a 24 ore.
* **Regola di estrazione**: Il timestamp con `valid_time = 00:00 UTC` del giorno $D+1$ **contiene esattamente la somma totale cumulata delle 24 ore del giorno $D$**. Subito dopo (alle 01:00 UTC), il contatore si azzera.
* **Procedura**: Si scarica esclusivamente il timestamp `time = ["00:00"]` e si rietichetta la data sottraendo un giorno ($D = \text{valid\_time} - 1\text{ giorno}$).
* **Variabili estratte**: Precipitazione totale, evaporazione totale, traspirazione della vegetazione, evaporazione da suolo nudo, evaporazione da canopia, evaporazione potenziale, deflusso superficiale, drenaggio profondo, nevicata, fusione nivale, radiazione solare globale incidente, radiazione termica verso terra, radiazione solare netta, radiazione termica netta, flusso di calore sensibile, flusso di calore latente.

---

## 3. Convenzione Temporale: DAY_MODE = "UTC"

La convenzione adottata è l'**UTC-Day** (finestra temporale fissa dalle 00:00 alle 23:59 UTC per tutte le 479 contee).

### Motivazioni Scientifiche e Metodologiche:
1. **Invarianza delle feature agronomiche**: I modelli di resa della soia utilizzano indicatori aggregati su finestre settimanali, decadali, mensili e stagionali (ad esempio Growing Degree Days stagionali, pioggia cumulata a 30 o 90 giorni, indici di siccità SPEI/VPD medio). Su queste finestre temporali, uno scostamento di 5–6 ore rispetto all'orario solare locale produce differenze numericamente trascurabili ($r > 0{,}999$).
2. **Eliminazione di discontinuità da Daylight Saving Time (DST)**: Evita la gestione delle giornate solari civili storiche da 23 e 25 ore legate ai cambi di ora legale negli USA tra il 1950 e il 2025.
3. **Armonizzazione tra fusi orari diversi**: Le 479 contee ricadono in due fusi orari differenti (Central Time per IL, IA, MN, MO e parte dell'IN; Eastern Time per OH e parte dell'IN). L'adozione di un'unica finestra temporale regionale UTC garantisce che un giorno $D$ rappresenti lo stesso istante sincrono per l'intero Midwest.
4. **Massima efficienza e rispetto delle quote CDS**: Richiede 1 solo timestamp/giorno per gli accumulati (6.205 fields/anno per 17 variabili, ampiamente sotto la soglia massima di 12.000 fields/richiesta del CDS).

---

## 4. Catalogo Completo delle Variabili

### Variabili Meteorologiche Estratte (35 campi)
| Variabile | Nome CDS | Statistica | Unità Raw | Unità Processata | Formula di Conversione |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Temperatura Minima** | `2m_temperature` | `daily_minimum` | $\text{K}$ | $^\circ\text{C}$ | $T(^\circ\text{C}) = T(\text{K}) - 273.15$ |
| **Temperatura Massima** | `2m_temperature` | `daily_maximum` | $\text{K}$ | $^\circ\text{C}$ | $T(^\circ\text{C}) = T(\text{K}) - 273.15$ |
| **Temperatura Media** | `2m_temperature` | `daily_mean` | $\text{K}$ | $^\circ\text{C}$ | $T(^\circ\text{C}) = T(\text{K}) - 273.15$ |
| **Temperatura di Rugiada** | `2m_dewpoint_temperature` | `daily_mean` | $\text{K}$ | $^\circ\text{C}$ | $T_d(^\circ\text{C}) = T_d(\text{K}) - 273.15$ |
| **Vento Zonale (U)** | `10m_u_component_of_wind` | `daily_mean` | $m/s$ | $m/s$ | Invariato |
| **Vento Meridionale (V)** | `10m_v_component_of_wind` | `daily_mean` | $m/s$ | $m/s$ | Invariato |
| **Pressione alla Superficie** | `surface_pressure` | `daily_mean` | $\text{Pa}$ | $\text{hPa}$ | $P(\text{hPa}) = P(\text{Pa}) / 100$ |
| **Temp. Superficiale (Skin)**| `skin_temperature` | `daily_maximum` | $\text{K}$ | $^\circ\text{C}$ | $T_s(^\circ\text{C}) = T_s(\text{K}) - 273.15$ |
| **Serbatoio d'Acqua Chioma** | `skin_reservoir_content` | `daily_mean` | $m$ | $mm$ | $h(mm) = h(m) \times 1000$ |
| **Temp. Suolo 1 (0–7 cm)** | `soil_temperature_level_1` | `daily_mean` | $\text{K}$ | $^\circ\text{C}$ | $T_{s1}(^\circ\text{C}) = T_{s1}(\text{K}) - 273.15$ |
| **Temp. Suolo 2 (7–28 cm)** | `soil_temperature_level_2` | `daily_mean` | $\text{K}$ | $^\circ\text{C}$ | $T_{s2}(^\circ\text{C}) = T_{s2}(\text{K}) - 273.15$ |
| **Temp. Suolo 3 (28–100 cm)**| `soil_temperature_level_3` | `daily_mean` | $\text{K}$ | $^\circ\text{C}$ | $T_{s3}(^\circ\text{C}) = T_{s3}(\text{K}) - 273.15$ |
| **Temp. Suolo 4 (100–289 cm)**| `soil_temperature_level_4` | `daily_mean` | $\text{K}$ | $^\circ\text{C}$ | $T_{s4}(^\circ\text{C}) = T_{s4}(\text{K}) - 273.15$ |
| **Umidità Suolo 1 (0–7 cm)** | `volumetric_soil_water_layer_1` | `daily_mean` | $m^3/m^3$ | $m^3/m^3$ | Invariato ($0 - 1$) |
| **Umidità Suolo 2 (7–28 cm)** | `volumetric_soil_water_layer_2` | `daily_mean` | $m^3/m^3$ | $m^3/m^3$ | Invariato ($0 - 1$) |
| **Umidità Suolo 3 (28–100 cm)**| `volumetric_soil_water_layer_3` | `daily_mean` | $m^3/m^3$ | $m^3/m^3$ | Invariato ($0 - 1$) |
| **Umidità Suolo 4 (100–289 cm)**| `volumetric_soil_water_layer_4` | `daily_mean` | $m^3/m^3$ | $m^3/m^3$ | Invariato ($0 - 1$) |
| **Copertura Nevosa** | `snow_cover` | `daily_mean` | $0 - 1$ | $0 - 1$ | Invariato (frazione) |
| **Densità Neve** | `snow_density` | `daily_mean` | $kg/m^3$ | $kg/m^3$ | Invariato |
| **Equivalente Neve (SWE)** | `snow_depth_water_equivalent` | `daily_mean` | $m$ | $mm$ | $SWE(mm) = SWE(m) \times 1000$ |
| **Precipitazione Totale** | `total_precipitation` | `totale 24h` | $m$ | $mm$ | $P(mm) = P(m) \times 1000$ |
| **Evaporazione Totale** | `total_evaporation` | `totale 24h` | $m$ | $mm$ | $E(mm) = E(m) \times 1000$ |
| **Evaporazione Canopia** | `evaporation_from_the_top_of_canopy` | `totale 24h` | $m$ | $mm$ | $h(mm) = h(m) \times 1000$ |
| **Evaporazione Suolo Nudo\***| `evaporation_from_bare_soil` | `totale 24h` | $m$ | $mm$ | *Correzione swap ECMWF* |
| **Evaporazione Specchi Acqua\***| `evaporation_from_open_water_...` | `totale 24h` | $m$ | $mm$ | *Correzione swap ECMWF* |
| **Traspirazione Vegetale\*** | `evaporation_from_vegetation_...` | `totale 24h` | $m$ | $mm$ | *Correzione swap ECMWF* |
| **Evaporazione Potenziale** | `potential_evaporation` | `totale 24h` | $m$ | $mm$ | Pan evaporation ($m \times 1000$) |
| **Deflusso Superficiale** | `surface_runoff` | `totale 24h` | $m$ | $mm$ | $R_{surf}(mm) = R(m) \times 1000$ |
| **Drenaggio Sotterraneo** | `sub_surface_runoff` | `totale 24h` | $m$ | $mm$ | $R_{sub}(mm) = R(m) \times 1000$ |
| **Nevicata Totale** | `snowfall` | `totale 24h` | $m$ eq. | $mm$ | $S(mm) = S(m) \times 1000$ |
| **Fusione Nivale** | `snowmelt` | `totale 24h` | $m$ eq. | $mm$ | $M(mm) = M(m) \times 1000$ |
| **Radiazione Solare Incidente**| `surface_solar_radiation_downwards`| `totale 24h` | $J/m^2$ | $MJ/m^2$ | $R_s(MJ/m^2) = R_s / 10^6$ |
| **Radiazione Solare Netta** | `surface_net_solar_radiation` | `totale 24h` | $J/m^2$ | $MJ/m^2$ | $R_{s,net}(MJ/m^2) = R / 10^6$ |
| **Radiazione Termica Incidente**| `surface_thermal_radiation_downwards`| `totale 24h` | $J/m^2$ | $MJ/m^2$ | $R_t(MJ/m^2) = R_t / 10^6$ |
| **Radiazione Termica Netta** | `surface_net_thermal_radiation` | `totale 24h` | $J/m^2$ | $MJ/m^2$ | $R_{t,net}(MJ/m^2) = R / 10^6$ |
| **Flusso Calore Sensibile** | `surface_sensible_heat_flux` | `totale 24h` | $J/m^2$ | $MJ/m^2$ | $H(MJ/m^2) = H / 10^6$ |
| **Flusso Calore Latente** | `surface_latent_heat_flux` | `totale 24h` | $J/m^2$ | $MJ/m^2$ | $\lambda E(MJ/m^2) = \lambda E / 10^6$ |

*\*Nota ufficiale ECMWF sui tre flussi di evaporazione: nei dati ERA5-Land è documentato uno scambio storico tra tre parametri (bare soil evaporation contiene la traspirazione vegetale; open water evaporation contiene l'evaporazione da suolo nudo; traspirazione vegetale contiene open water evaporation). Il nostro modulo di post-processing riassegna automaticamente i campi alla loro corretta identità fisica.*

### Feature Derivate ad Alto Valore Agronomico
1. **Growing Degree Days (GDD)**:
   $$\text{GDD} = \max\left(0, \frac{\min(T_{\max}, 30) + \max(T_{\min}, 10)}{2} - 10\right)$$
2. **Vapor Pressure Deficit (VPD) Daily Proxy**:
   Calcolato a partire da $T_{\text{mean}}$ e $T_{\text{dewpoint}}$ con la formula di Tetens:
   $$e_s(T) = 0.61078 \exp\left(\frac{17.27 T}{T + 237.3}\right), \quad e_a(T_d) = 0.61078 \exp\left(\frac{17.27 T_d}{T_d + 237.3}\right)$$
   $$\text{VPD} = \max(0, e_s - e_a) \quad [\text{kPa}]$$
3. **Indici di Stress Termico**:
   - `heat_days_30`: Indicatore booleano $(T_{\max} \ge 30^\circ\text{C})$
   - `heat_days_35`: Indicatore booleano $(T_{\max} \ge 35^\circ\text{C})$
   - `diurnal_temperature_range`: $\text{DTR} = T_{\max} - T_{\min}$
4. **Bilanci Idrici Rolling**:
   - Cumulati di precipitazione a 7, 14, 30, 60, 90 giorni.
   - Indice di deficit idrico netto: $P_{\text{net}} = \text{Precipitation} - \text{Potential Evaporation}$.

---

## 5. Geografia, Bounding Box e Pesi Spaziali

### Bounding Box Regionale Unificato
Invece di scaricare box statali sovrapposti, è stato calcolato l'inviluppo convesso rettangolare delle **479 geometrie di contea (US Census 2020)**:
* Estensione originale: $[\text{North}: 47.50^\circ, \text{West}: -96.87^\circ, \text{South}: 35.99^\circ, \text{East}: -80.52^\circ]$.
* Margine applicato: $+0.1^\circ$.
* Snap outward sulla griglia 0.1° ERA5-Land:
  $$\mathbf{CDS\_BBOX = [47.7, -97.0, 35.8, -80.4]} \quad [\text{North, West, South, East}]$$
* Risoluzione griglia: **120 latitudini $\times$ 167 longitudini = 20.040 celle**.

### Matrice di Ponderazione Spaziale per Contea
La riduzione da griglia continua a dato di contea viene eseguita tramite una matrice di pesi areali precomputata:
1. I poligoni delle 479 contee e i riquadri delle celle ERA5-Land ($0.1^\circ \times 0.1^\circ$) sono proiettati nel sistema cartografico conforme ed equal-area **EPSG:5070 (NAD83 / Conus Albers)**.
2. Si calcola l'area di intersezione $A_{c, i}$ tra la contea $c$ e la cella $i$.
3. Il peso assegnato alla cella $i$ per la contea $c$ è:
   $$w_{c, i} = \frac{A_{c, i}}{\sum_{j \in \text{cells}(c)} A_{c, j}}, \quad \text{con } \sum_{i} w_{c, i} = 1.000000$$
4. Per ogni giorno $t$ e variabile $V$, il valore aggregato di contea è:
   $$V_c(t) = \sum_{i \in \text{cells}(c)} w_{c, i} \cdot V_i(t)$$

La matrice di pesi ($11.953$ intersezioni attive) è memorizzata in [spatial_weights_479_counties.parquet](file:///c:/Users/JacopoCesari-Aret%C3%A9sr/Desktop/Tesi/era5_land_daily/spatial_weights_479_counties.parquet) e consente l'aggregazione di un intero anno solare in circa 1,5 secondi.

---

## 6. Qualità dei Dati, Integrità e Validazione

Il file di registro `manifest.csv` memorizza per ogni blocco scaricato: timestamp, anno, canale, path del file, byte totali, hash SHA-256 e stato dei controlli QC.

### Criteri di Accettazione Automatica (QC Gates):
1. **Integrità del Calendario**: Esattamente 365 giorni (o 366 negli anni bisestili). Nessun giorno mancante e nessun duplicato.
2. **Coerenza Termica Fisica**:
   $$T_{\min}(t) \le T_{\text{mean}}(t) \le T_{\max}(t) \quad \forall t, \text{ cella}$$
3. **Plausibilità Fisica**:
   - Precipitazione totale: $P \ge 0$.
   - Umidità volumetrica suolo: $0.0 \le \theta \le 1.0$.
   - Radiazione solare: $R_s \ge 0$.
4. **Monotonia delle Coordinate**:
   Latitudini strettamente decrescenti da Nord a Sud; longitudini strettamente crescenti da Ovest a Est. Nessuna compresenza di coordinate $-180/180$ e $0/360$.

---

## 7. Struttura dei File nel Repository

```
C:\Users\JacopoCesari-Aretésr\Desktop\Tesi\
├── 01_Dati_Soia_e_Target/
│   ├── soybean_yield_479_counties_1950_2025.xlsx    # Rese USDA NASS storiche
│   ├── soybean_acres_479_counties_1950_2025.xlsx    # Superfici raccolte
│   ├── balanced_panel_candidate_counties.csv        # Elenco FIPS 479 contee
│   ├── metodologia_selezione_panel_479_contee.md    # Nota metodologica panel
│   └── census_counties/                             # Geometrie US Census 2020
│
├── 02_Meteo_ERA5_Land/
│   ├── raw_daily_stats/                             # NetCDF giornalieri Canale A
│   ├── raw_accumulated_boundaries/                  # NetCDF 00 UTC Canale B
│   ├── county_daily/                                # Parquet giornalieri per contea
│   ├── model_features/                              # Feature pronte per i modelli
│   ├── spatial_weights_479_counties.parquet         # Pesi areali precomputati EPSG:5070
│   └── manifest.csv                                 # Registro crittografico SHA-256
│
├── 03_Documentazione_e_Paper/
│   ├── METODOLOGIA_ERA5_LAND_DAILY.md               # Questo documento
│   ├── deep-research-report.md                      # Rassegna letteratura
│   ├── Deep Research_ previsione della resa...docx  # Report iniziale
│   └── Papers/                                      # Paper scientifici citati
│
├── download_era5_land_daily.py                      # Script unico operativo
└── README.md                                        # Indice generale del repository
```
