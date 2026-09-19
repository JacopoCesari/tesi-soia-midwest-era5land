# Historical Macro-Climatic Yield Shocks & Extreme Event Benchmark
*Strategic benchmark for out-of-sample model evaluation in Chapter 5*

---

## 1. The R² = 0.81 Trend Dilemma: Why Weather Modeling is Indispensable

A linear secular trend explains $R^2 = 0.81$ of the annual mean yield variance across the 1951–2025 panel. However, this high explanatory power does NOT render weather forecasting models redundant. In fact, it provides the primary justification for them:

1. **Economic Value Lies Entirely in the Residual ($\epsilon_t = y_t - \hat{y}_t$):**
   - The secular trend reflects predictable, long-term technological progress (cultivar breeding, transgenic traits, mechanization, drainage). It is already priced into physical grain forward curves, crop insurance reference prices, and baseline capacity planning months or years in advance.
   - All physical supply-chain disruptions, commercial defaults, logistics bottlenecks, and hedging losses stem from **unexpected departures from the trend**.
   - In 1988, a trend model predicts $37.72\text{ bu/ac}$, whereas actual yield collapses to $28.08\text{ bu/ac}$ ($-25.6\%$). In our 135-county panel (spanning ~13 million acres), this represents an unexpected physical loss of **125 million bushels** (approx. 3.4 million metric tons). A trend model has zero ability to anticipate this shortfall.
2. **Technological Plateauing and Climate Vulnerability in the Literature:**
   - Agronomic and econometric literature (*Schlenker & Roberts 2009*, *Grassini et al. 2013*, *Lobell et al.*) demonstrates that yield growth rates may be approaching biological/physiological plateaus in mature temperate systems.
   - Non-linear thermal stress (temperatures exceeding $30^\circ\text{C}$ to $32^\circ\text{C}$) inflicts sharp quadratic yield damage that increasingly threatens to offset genetic advances under a warming climate.
3. **Decoupling Protocol:**
   - The model design explicitly decomposes yield into a baseline deterministic trend and a weather-driven anomaly ($\Delta y_{c,t} = y_{c,t} - \hat{y}_{c,t}$). The purpose of the machine learning and deep learning models is specifically to forecast this volatile meteorological anomaly.

---

## 2. Catalog of Historical Shock Years for Chapter 5 Stress Testing

These benchmark years will be used in Chapter 5 (Results) to evaluate whether candidate models (at different lead times $H=12, \dots, 1$) successfully predict severe downside shortfalls and upside bumper harvests, or whether they erroneously regress to the mean.

| Year | Type | Observed Mean (bu/ac) | Secular Trend (bu/ac) | Anomaly (bu/ac) | Relative Anomaly (%) | Meteorological Hazard / Ecological Mechanism |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1988** | Adverse (Severe Drought) | 28.08 | 37.72 | -9.64 | **-25.6%** | Historic anticyclonic blocking ridge over the Corn Belt; acute summer moisture deficit and extreme July heat. Deepest shortfall in the 75-year record. |
| **2003** | Adverse (Heat & Aphids) | 35.26 | 44.98 | -9.72 | **-21.6%** | Severe late-summer moisture stress during August pod fill compounded by the first widespread North American invasion of the soybean aphid (*Aphis glycines*). |
| **1974** | Adverse (Cold/Frost/Drought) | 24.84 | 30.93 | -6.09 | **-19.7%** | Abnormally cold and wet spring delaying planting by ~4 weeks, followed by midsummer drought and an extraordinarily early hard freeze in early September. |
| **2012** | Adverse (Flash Drought) | 43.55 | 49.34 | -5.80 | **-11.7%** | Severe summer flash drought and record-breaking July temperatures across the central United States during reproductive stages. |
| **1983** | Adverse (Heatwave/Drought) | 31.27 | 35.29 | -4.02 | **-11.4%** | Prolonged heatwave with temperatures $>35^\circ\text{C}$ and intense drought during July and August across the central Midwest. |
| **1993** | Adverse (Great Flood) | 35.92 | 40.14 | -4.22 | **-10.5%** | Great Upper Mississippi River Flood; continuous torrential precipitation, widespread soil waterlogging, root asphyxiation, nutrient leaching, and root rot. |
| **1952** | Favorable (Bumper) | 22.69 | 20.27 | +2.41 | **+11.9%** | Exceptional vegetative and pod-filling rainfall with moderate temperatures. |
| **2021** | Favorable (Bumper) | 59.50 | 53.70 | +5.79 | **+10.8%** | Timely August rainfall and minimal extreme heat stress throughout the Midwest. |
| **2016** | Favorable (Bumper) | 56.81 | 51.28 | +5.53 | **+10.8%** | Broadly distributed summer precipitation and optimal thermal conditions during pod fill. |
| **1994** | Favorable (Bumper) | 44.95 | 40.62 | +4.33 | **+10.7%** | Near-perfect Midwestern growing season with cool nights and frequent rains. |

---

## 3. Evaluation Protocol in Chapter 5
1. **Lead-Time Tracking for Shocks:** When does the forecast first detect the 1988 or 2012 drought? Does $H=3$ (August 31 cutoff) capture it, or does an early signal emerge at $H=4$ (July 31 cutoff)?
2. **Directional & Categorical Accuracy:** Do classification models trigger a "Severe Shortfall" regime alert in advance of point-prediction accuracy?
3. **Model Robustness Under Stress:** Contrast linear models (Ridge/Lasso) vs. tree ensembles (Random Forest/XGBoost) vs. LSTMs during these non-linear shock years.
