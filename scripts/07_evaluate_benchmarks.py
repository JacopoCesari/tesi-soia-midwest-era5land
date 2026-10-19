"""
07_evaluate_benchmarks.py

Computes verified forecast accuracy and reproduces the Chapter 5 performance,
annual HAC inference, regional, shock, decile and reconstructed-yield tables.
Separates harvest-closure metrics from calendar-horizon LSTM summary exports.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PREDS_LSTM_PARQUET = ROOT / "output" / "data" / "processed" / "predictions_lstm.parquet"

# Evaluation test window
TEST_YEARS = list(range(2011, 2026))
ACTIVE_HORIZONS = list(range(11, 0, -1))

MONTH_NAMES = {
    11: "Dec Y-1 (H=11)",
    10: "Jan Y (H=10)",
    9: "Feb Y (H=9)",
    8: "Mar Y (H=8)",
    7: "Apr Y (H=7, Pre-Planting)",
    6: "May Y (H=6, Emergence)",
    5: "Jun Y (H=5, Vegetative)",
    4: "Jul Y (H=4, Flowering)",
    3: "Aug Y (H=3, Pod-Filling)",
    2: "Sep Y (H=2, Maturity)",
    1: "Oct Y (H=1, Harvest)",
}


def load_combined_predictions():
    """Load explicitly identified information sets; retain one row per forecast."""
    keys = ["H", "year", "county_fips"]
    weather = pd.read_parquet(ROOT / "output/data/processed/predictions_weather_only.parquet")
    integrated = pd.read_parquet(ROOT / "output/data/processed/predictions_integrated.parquet")
    lstm = pd.read_parquet(PREDS_LSTM_PARQUET)
    frames = []
    for frame in (weather, integrated):
        frame["county_fips"] = frame["county_fips"].astype(str).str.zfill(5)
    weather = weather[weather.year.isin(TEST_YEARS) & weather.H.isin(ACTIVE_HORIZONS)].copy()
    reference = weather[weather.model.eq("Naive")][keys + ["y_true_anomaly", "trend"]]
    frames.append(weather)
    integrated = integrated[integrated.year.isin(TEST_YEARS) & integrated.H.isin(ACTIVE_HORIZONS)
                            & integrated.model.ne("Naive")].copy()
    integrated["model"] += " (+Lag)"
    frames.append(integrated)
    variants = [("prelu", False, "LSTM (Weather-Only)"),
                ("linear", False, "LSTM (Linear Weather-Only)"),
                ("linear", True, "LSTM (+Lag)")]
    for head, lag, label in variants:
        frame = lstm[lstm.year.isin(TEST_YEARS) & lstm.frequency.eq(10)
                     & lstm.head_variant.eq(head) & lstm.include_lag.eq(lag)].copy()
        frame = frame.rename(columns={"horizon": "H", "actual_anomaly": "lstm_target",
                                      "pred_anomaly": "y_pred_anomaly"})
        frame["county_fips"] = frame.county_fips.astype(str).str.zfill(5)
        frame = frame.merge(reference, on=keys, how="outer", validate="one_to_one", indicator=True)
        if not frame._merge.eq("both").all():
            raise ValueError(f"Incomplete paired panel for {label}")
        if not np.allclose(frame.lstm_target, frame.y_true_anomaly, atol=1e-5, rtol=0):
            raise ValueError(f"Different target series for {label}")
        frame["model"] = label
        frames.append(frame[weather.columns])
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["model"] + keys).any():
        raise ValueError("Forecast keys must be unique")
    if not np.isfinite(combined[["y_true_anomaly", "y_pred_anomaly", "trend"]].to_numpy()).all():
        raise ValueError("Forecast targets, predictions and trends must be finite")
    enet = combined[combined.model.eq("ElasticNet (+Lag)")]
    recurrent = combined[combined.model.eq("LSTM (+Lag)")]
    ensemble = enet.merge(recurrent[keys + ["y_pred_anomaly"]], on=keys,
                          how="outer", validate="one_to_one", suffixes=("", "_lstm"), indicator=True)
    if not ensemble._merge.eq("both").all():
        raise ValueError("Ensemble forecasts must be fully paired")
    ensemble["y_pred_anomaly"] = (ensemble.y_pred_anomaly + ensemble.y_pred_anomaly_lstm) / 2
    ensemble["model"] = "Ensemble (ENet + LSTM)"
    combined = pd.concat([combined, ensemble[weather.columns]], ignore_index=True)
    for (model, horizon), group in combined.groupby(["model", "H"]):
        if len(group) != 2025 or group.county_fips.nunique() != 135 or set(group.year) != set(TEST_YEARS):
            raise ValueError(f"Incomplete evaluation panel: {model}, H={horizon}")
        annual_loss_differentials(combined, model, horizon)
    return combined


def apply_minnesota_phenological_rule(df):
    """Use H=2 for Minnesota at H=1; missing paired forecasts are errors."""
    if df.duplicated(["model", "H", "year", "county_fips"]).any():
        raise ValueError("Forecast keys must be unique")
    result = df.copy()
    lookup = result[result.H.eq(2)].set_index(["model", "year", "county_fips"]).y_pred_anomaly
    mask = result.H.eq(1) & result.county_fips.astype(str).str.startswith("27")
    index = pd.MultiIndex.from_frame(result.loc[mask, ["model", "year", "county_fips"]])
    values = lookup.reindex(index)
    if values.isna().any():
        raise ValueError("Minnesota harvest closure requires paired H=2 forecasts")
    result.loc[mask, "y_pred_anomaly"] = values.to_numpy()
    return result


def diebold_mariano_test(loss_a, loss_b):
    """One-sided annual DM with Bartlett/Newey-West HAC and T/(T-1) correction.

    The automatic bandwidth matches statsmodels cov_hac(nlags=None):
    floor(4 * (T / 100)**(2/9)). These are annual loss differences, not monthly
    observations: the calendar lead label H is not the HAC bandwidth.
    """
    first, second = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape or len(first) < 3:
        raise ValueError("Loss arrays must have the same one-dimensional shape and at least three years")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Loss arrays must be finite")
    difference = first - second
    count = len(difference)
    centered = difference - difference.mean()
    bandwidth = int(np.floor(4 * (count / 100)**(2 / 9)))
    long_run_variance = np.dot(centered, centered) / count
    for lag in range(1, bandwidth + 1):
        long_run_variance += 2 * (1 - lag / (bandwidth + 1)) * np.dot(centered[lag:], centered[:-lag]) / count
    variance_of_mean = long_run_variance / (count - 1)
    if not np.isfinite(variance_of_mean) or variance_of_mean <= 0:
        raise ValueError("Annual loss differential has no positive HAC variance")
    statistic = float(difference.mean() / np.sqrt(variance_of_mean))
    return statistic, float(stats.norm.sf(statistic))


def annual_loss_differentials(predictions, model, horizon, reference="Naive"):
    """Align county-years first, then average squared-loss differences by year."""
    keys = ["year", "county_fips"]
    first = predictions[predictions.H.eq(horizon) & predictions.model.eq(reference)]
    second = predictions[predictions.H.eq(horizon) & predictions.model.eq(model)]
    if first.duplicated(keys).any() or second.duplicated(keys).any():
        raise ValueError("Paired forecast keys must be unique")
    paired = first.merge(second, on=keys, suffixes=("_reference", "_model"),
                         how="outer", validate="one_to_one", indicator=True)
    if paired.empty or not paired._merge.eq("both").all():
        raise ValueError("Forecasts must have fully paired county-years")
    if set(paired.year) != set(TEST_YEARS) or paired.groupby("year").size().nunique() != 1:
        raise ValueError("The same complete annual panel is required in 2011--2025")
    if not np.allclose(paired.y_true_anomaly_reference, paired.y_true_anomaly_model, atol=1e-5, rtol=0):
        raise ValueError("Paired forecasts have different target values")
    if not np.isfinite(paired[["y_true_anomaly_reference", "y_pred_anomaly_reference",
                               "y_true_anomaly_model", "y_pred_anomaly_model"]].to_numpy()).all():
        raise ValueError("Paired targets and predictions must be finite")
    observed = paired.y_true_anomaly_reference
    paired["loss_difference"] = ((observed - paired.y_pred_anomaly_reference)**2
                                 - (observed - paired.y_pred_anomaly_model)**2)
    return paired.groupby("year").loss_difference.mean().sort_index()


PRIMARY_MODELS = ["Naive", "ElasticNet (+Lag)", "RandomForest (+Lag)", "XGBoost (+Lag)",
                  "LSTM (+Lag)", "Ensemble (ENet + LSTM)"]
SHORT_NAMES = {"Naive": "Naive", "ElasticNet": "ENet (weather)",
               "RandomForest": "RF (weather)", "XGBoost": "XGB (weather)",
               "ElasticNet (+Lag)": "ENet (+Lag)", "RandomForest (+Lag)": "RF (+Lag)",
               "XGBoost (+Lag)": "XGB (+Lag)", "LSTM (Weather-Only)": "LSTM (weather)",
               "LSTM (+Lag)": "LSTM (+Lag)", "Ensemble (ENet + LSTM)": "Ensemble"}
TABLE_DIR = ROOT / "output/thesis/tables"
REPORT_DIR = ROOT / "output/data/processed/diagnostics"


def forecast_metrics(predictions):
    rows = []
    for (model, horizon), group in predictions.groupby(["model", "H"]):
        error = group.y_pred_anomaly - group.y_true_anomaly
        raw_true = group.y_true_anomaly + group.trend
        raw_pred = group.y_pred_anomaly + group.trend
        rows.append({"model": model, "H": horizon, "n_obs": len(group),
                     "RMSE": np.sqrt(np.mean(error**2)), "MAE": np.mean(abs(error)),
                     "R2_OOS": 1 - np.sum(error**2) / np.sum(group.y_true_anomaly**2),
                     "R2_level": 1 - np.sum(error**2) / np.sum((raw_true - raw_true.mean())**2),
                     "r_level": raw_true.corr(raw_pred)})
    return pd.DataFrame(rows)


def inference_report(predictions):
    rows = []
    for horizon in sorted(predictions.H.unique(), reverse=True):
        for model in predictions.model.unique():
            if model == "Naive":
                continue
            references = ["Naive"]
            if model in ["ElasticNet (+Lag)", "RandomForest (+Lag)", "XGBoost (+Lag)"]:
                references.append(model.replace(" (+Lag)", ""))
            if model == "LSTM (+Lag)":
                references.append("LSTM (Linear Weather-Only)")
                references.append("ElasticNet (+Lag)")
            if model == "Ensemble (ENet + LSTM)":
                references.extend(["ElasticNet (+Lag)", "LSTM (+Lag)"])
            for reference in references:
                difference = annual_loss_differentials(predictions, model, horizon, reference)
                try:
                    statistic, p_value = diebold_mariano_test(difference.to_numpy(), np.zeros(len(difference)))
                    status = "ok"
                except ValueError as error:
                    statistic, p_value, status = np.nan, np.nan, str(error)
                rows.append({"model": model, "reference": reference, "H": horizon,
                             "n_years": len(difference), "hac_lags": 2,
                             "mean_loss_difference": difference.mean(), "DM": statistic,
                             "p_one_sided": p_value, "status": status})
    return pd.DataFrame(rows)


def regional_report(predictions, scope):
    rows = []
    for model, group in predictions[predictions.H.eq(1)].groupby("model"):
        annual = group.groupby("year")[["y_true_anomaly", "y_pred_anomaly", "trend"]].mean()
        error = annual.y_pred_anomaly - annual.y_true_anomaly
        raw_true = annual.y_true_anomaly + annual.trend
        raw_pred = annual.y_pred_anomaly + annual.trend
        for year, values in annual.iterrows():
            rows.append({"scope": scope, "model": model, "year": year,
                         "observed": values.y_true_anomaly, "predicted": values.y_pred_anomaly,
                         "trend": values.trend, "RMSE": np.sqrt(np.mean(error**2)),
                         "MAE": np.mean(abs(error)), "Bias": error.mean(),
                         "r_anomaly": np.nan if annual.y_pred_anomaly.std() == 0 else annual.y_true_anomaly.corr(annual.y_pred_anomaly),
                         "r_level": raw_true.corr(raw_pred)})
    return pd.DataFrame(rows)




def write_table(filename, caption, label, headers, rows, notes=""):
    """Write evidence using the established thesis typography and table layout."""
    deciles = filename == "tab_decile_error_performance.tex"
    regional = filename == "tab_yearly_macro_yields.tex"
    horizons = filename == "tab_horizon_performance.tex"
    shocks = filename == "tab_tail_risk_performance.tex"
    spacing = "1.08" if deciles else ("1.10" if regional else "1.12")
    padding = "3.0pt" if deciles else ("3.2pt" if regional else ("3.5pt" if horizons else "4.0pt"))
    cutoffs = {11: "Dec 31", 10: "Jan 31", 9: "Feb 28/29", 8: "Mar 31", 7: "Apr 30",
               6: "May 31", 5: "Jun 30", 4: "Jul 31", 3: "Aug 31", 2: "Sep 30", 1: r"$H^*(s)$"}
    if horizons:
        headers = ["H", "Cutoff Date", "ENet", "RF", "XGB", "LSTM", "ENet", "RF", "XGB", "LSTM", "Ens"]
    elif shocks:
        headers = ["H", "Cutoff Date", "Naive Trend", "ElasticNet", "XGBoost", "LSTM", "Ens"]
    columns = "c l" + " c"*(len(headers)-2) if horizons or shocks else ("l" if deciles else "c") + " c"*(len(headers)-1)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing", r"\footnotesize",
             r"\setlength{\tabcolsep}{" + padding + "}", r"\renewcommand{\arraystretch}{" + spacing + "}",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + columns + " @{}}", r"\toprule"]
    if horizons:
        lines.extend([r"& & \multicolumn{4}{c}{\textbf{Weather-Only ($R^2_{\mathrm{OOS}}$)}} & \multicolumn{5}{c}{\textbf{Integrated (+Lag, $R^2_{\mathrm{OOS}}$)}} \\",
                      r"\cmidrule(lr){3-6}\cmidrule(lr){7-11}"])
    lines.extend([" & ".join(r"\textbf{" + header + "}" for header in headers) + r" \\", r"\midrule"])
    panels = {0: r"\textbf{Panel A: Early Vegetative Horizon} ($H=5$, June 30)",
              10: r"\textbf{Panel B: Pod-Filling Horizon} ($H=3$, August 31)",
              20: r"\textbf{Panel C: Harvest Closure Horizon} ($H^*(s)$)"}
    for index, row in enumerate(rows):
        cells = list(map(str, row))
        if horizons:
            horizon = int(cells[0])
            if horizon in [11, 6]:
                if horizon == 6:
                    lines.append(r"\midrule")
                title = r"Pre-Sowing Horizons ($H=11\dots7$)" if horizon == 11 else r"Crop Campaign ($H=6\dots1$)"
                lines.append(r"\multicolumn{11}{l}{\textit{" + title + r"}} \\")
            scores = list(map(float, cells[1:]))
            cells[1:] = [r"\textbf{" + value + "}" if float(value) == max(scores) else value for value in cells[1:]]
            cells = ["$H=" + cells[0] + "$", cutoffs[horizon], *cells[1:]]
        elif shocks:
            if index in [0, 5]:
                if index:
                    lines.append(r"\midrule")
                title = "Panel A: 2012 Flash Drought" if index == 0 else "Panel B: 2016 Bumper Season"
                lines.append(r"\multicolumn{7}{l}{\textbf{" + title + r"}} \\")
            horizon = int(cells[1])
            values = list(map(float, cells[2:]))
            cells = ["$H=" + cells[1] + "$", cutoffs[horizon], *[r"\textbf{" + value + "}" if float(value) == min(values) else value for value in cells[2:]]]
        elif deciles:
            if index in panels:
                if index:
                    lines.append(r"\midrule")
                lines.append(r"\multicolumn{8}{l}{" + panels[index] + r"} \\")
            observed = float(cells[2])
            errors = [abs(float(value)-observed) for value in cells[3:]]
            cells[2] = "$" + cells[2] + "$"
            cells[3:] = [r"\textbf{" + value + "}" if abs(error-min(errors)) < 1e-9 else "$" + value + "$"
                         for value, error in zip(cells[3:], errors)]
        elif regional:
            if index == 15:
                lines.append(r"\midrule")
            if index < 15:
                observed = float(cells[1])
                errors = [abs(float(value)-observed) for value in cells[3:]]
                cells[3:] = [r"\textbf{" + value + "}" if abs(error-min(errors)) < 1e-9 else value
                             for value, error in zip(cells[3:], errors)]
            else:
                cells[0] = r"\textbf{" + cells[0] + "}"
                numeric = [(i, float(value)) for i, value in enumerate(cells) if i >= 2 and value != "---"]
                if numeric:
                    best = max(value for _, value in numeric) if index == 18 else min(abs(value) for _, value in numeric)
                    for i, value in numeric:
                        if abs((value if index == 18 else abs(value))-best) < 1e-9:
                            cells[i] = r"\textbf{" + cells[i] + "}"
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\vspace{0.2em}",
                  r"{\footnotesize\textit{Source: Author's calculation on USDA NASS and ECMWF ERA5-Land data, 2011--2025 out-of-sample forecasts.} " + notes + "}",
                  r"\end{table}", ""])
    (TABLE_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


def write_evidence_tables(predictions, metrics, inference, regional):
    indexed = metrics.set_index(["model", "H"])
    weather = ["ElasticNet", "RandomForest", "XGBoost", "LSTM (Weather-Only)"]
    integrated = ["ElasticNet (+Lag)", "RandomForest (+Lag)", "XGBoost (+Lag)", "LSTM (+Lag)", "Ensemble (ENet + LSTM)"]
    all_models = weather + integrated
    rows = []
    for horizon in ACTIVE_HORIZONS:
        scores = [indexed.loc[(model,horizon), "R2_OOS"] for model in all_models]
        rows.append([str(horizon), *[f"{value:+.2f}" for value in scores]])
    write_table("tab_horizon_performance.tex", r"Pooled out-of-sample $R^2$ across forecast horizons and information sets (2011--2025).",
                "tab:horizon_performance", ["H", "ENet W", "RF W", "XGB W", "LSTM W", "ENet L", "RF L", "XGB L", "LSTM L", "Ens L"], rows,
                r"W: weather-only; L: integrated lagged-yield specification; Ens: ensemble.")
    reg=regional[regional.scope.eq("harvest_closure")]
    observed=reg[reg.model.eq("Naive")].set_index("year").observed
    rows=[]
    for year in observed.sort_values().index:
        values=reg[reg.year.eq(year)].set_index("model").predicted
        rows.append([str(year),f"{observed[year]:+.2f}",*[f"{values[m]:+.2f}" for m in PRIMARY_MODELS]])
    for field in ["RMSE","MAE","Bias","r_anomaly"]:
        values=reg.groupby("model")[field].first()
        title = r"Pearson $r$ (anomaly)" if field == "r_anomaly" else field + " (bu/ac)"
        rows.append([title,"---",*["---" if not np.isfinite(values[m]) else f"{values[m]:.3f}" for m in PRIMARY_MODELS]])
    write_table("tab_yearly_macro_yields.tex", r"Regional mean yield anomalies (bu/ac) and forecast accuracy at harvest closure (2011--2025).",
                "tab:yearly_macro_yields", ["Year", "Observed", "Naive", "ENet", "RF", "XGB", "LSTM", "Ens"], rows,
                r"ENet: ElasticNet; RF: Random Forest; XGB: XGBoost; Ens: ensemble.")
    rows=[]
    for year in [2012,2016]:
        for horizon in [5,4,3,2,1]:
            values=[]
            for model in ["Naive", "ElasticNet (+Lag)", "XGBoost (+Lag)", "LSTM (+Lag)", "Ensemble (ENet + LSTM)"]:
                group=predictions[predictions.model.eq(model)&predictions.H.eq(horizon)&predictions.year.eq(year)]
                values.append(f"{np.sqrt(np.mean((group.y_pred_anomaly-group.y_true_anomaly)**2)):.2f}")
            rows.append([str(year),str(horizon),*values])
    write_table("tab_tail_risk_performance.tex", "County-level forecast RMSE (bu/ac) during the 2012 drought and 2016 bumper season.",
                "tab:tail_risk_performance", ["Year","H","Naive","ENet","XGB","LSTM","Ens"],rows,
                r"Ens: ensemble.")
    rows=[]
    for model in ["Naive","ElasticNet (+Lag)","XGBoost (+Lag)","LSTM (+Lag)","Ensemble (ENet + LSTM)"]:
        values=indexed.loc[(model,1)]
        rows.append([SHORT_NAMES[model],*[f"{values[field]:.3f}" for field in ["RMSE","MAE","R2_OOS","R2_level","r_level"]]])
    write_table("tab_raw_vs_anomaly_performance.tex", r"Forecast accuracy in anomaly and reconstructed yield space at $H^*(s)$ (2011--2025).",
                "tab:raw_vs_anomaly_performance", ["Model","RMSE","MAE",r"$R^2_{\rm OOS}$",r"$R^2_{\rm level}$",r"$r_{\rm level}$"],rows,
                "RMSE and MAE are in bu/ac.")
    ref=predictions[predictions.model.eq("Naive")&predictions.H.eq(1)][["year","county_fips","y_true_anomaly"]].copy()
    ref["decile"]=pd.qcut(ref.y_true_anomaly,10,labels=False)+1
    rows=[]
    for horizon in [5,3,1]:
        for decile,group in ref.groupby("decile"):
            selected=predictions[predictions.H.eq(horizon)].merge(group[["year","county_fips"]],on=["year","county_fips"],validate="many_to_one")
            means=selected.groupby("model").y_pred_anomaly.mean()
            name = "$D_{" + str(decile) + "}$"
            if decile == 1:
                name += " (Severe Shortfall)"
            elif decile == 10:
                name += " (Peak Yield)"
            interval = f"$[{group.y_true_anomaly.min():+.2f}, {group.y_true_anomaly.max():+.2f}]$"
            rows.append([name,interval,f"{group.y_true_anomaly.mean():+.2f}",*[f"{means[model]:+.2f}" for model in PRIMARY_MODELS[1:]]])
    write_table("tab_decile_error_performance.tex", "Mean forecast anomalies (bu/ac) within realized anomaly deciles at three cutoffs (2011--2025).",
                "tab:decile_error_performance",["Decile","Range (bu/ac)","Mean Obs.","ENet","RF","XGB","LSTM","Ens"],rows,
                r"Ens: ensemble.")


def residual_diagnostics(predictions):
    """Replicate the stated AR(1) and inverse-distance Moran diagnostics."""
    features = pd.read_parquet(ROOT / "output/data/processed/model_datasets/features_H01.parquet",
                               columns=["county_fips", "lat_norm", "lon_norm"])
    features["county_fips"] = features.county_fips.astype(str).str.zfill(5)
    locations = features.drop_duplicates().set_index("county_fips").sort_index()
    if locations.index.duplicated().any() or not np.isfinite(locations.to_numpy()).all():
        raise ValueError("County coordinates must be unique and finite")
    xy = locations.to_numpy()
    distances = np.sqrt(np.sum((xy[:, None, :] - xy[None, :, :])**2, axis=2))
    diagonal = np.eye(len(xy), dtype=bool)
    if (distances[~diagonal] <= 0).any():
        raise ValueError("County coordinates must be distinct")
    weights = np.zeros_like(distances)
    weights[~diagonal] = 1 / distances[~diagonal]
    weights /= weights.sum(axis=1, keepdims=True)
    rows=[]
    for (model,horizon), group in predictions.groupby(["model","H"]):
        group=group.copy()
        group["error"]=group.y_pred_anomaly-group.y_true_anomaly
        errors=group.pivot(index="year",columns="county_fips",values="error").reindex(index=TEST_YEARS,columns=locations.index).to_numpy()
        if not np.isfinite(errors).all():raise ValueError("Incomplete residual panel")
        temporal=errors-errors.mean(axis=0)
        denominator=np.sum(temporal**2,axis=0)
        if (denominator<=0).any():raise ValueError("Undefined county serial correlation")
        ar=np.sum(temporal[1:]*temporal[:-1],axis=0)/denominator
        spatial=errors-errors.mean(axis=1,keepdims=True)
        variance=np.sum(spatial**2,axis=1)
        if (variance<=0).any():raise ValueError("Undefined annual spatial correlation")
        moran=np.einsum('ti,ij,tj->t',spatial,weights,spatial)/variance
        rows.append({"model":model,"H":horizon,"temporal_AR1_mean":ar.mean(),
                     "morans_I_spatial_mean":moran.mean(),"n_years":len(errors),"n_counties":len(xy)})
    return pd.DataFrame(rows)


def write_lstm_summaries():
    """Refresh existing calendar-horizon exports; main thesis metrics use H* separately."""
    frame = pd.read_parquet(PREDS_LSTM_PARQUET)
    frame = frame[frame.year.isin(TEST_YEARS) & frame.frequency.eq(10)]
    rows = []
    reference = frame[frame.horizon.eq(1) & frame.head_variant.eq("linear") & ~frame.include_lag]
    naive_rmse = np.sqrt(np.mean(reference.actual_anomaly.to_numpy(dtype=float)**2))
    groups = [("secular_trend", False, 12, reference.assign(pred_anomaly=0.0))]
    groups.extend((head, lag, horizon, group) for (head, lag, horizon), group
                  in frame.groupby(["head_variant", "include_lag", "horizon"]))
    for head, lag, horizon, group in groups:
        observed = group.actual_anomaly.to_numpy(dtype=float)
        predicted = group.pred_anomaly.to_numpy(dtype=float)
        if len(group) != 2025 or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
            raise ValueError("Incomplete calendar-horizon LSTM summary")
        errors = predicted-observed
        rmse = np.sqrt(np.mean(errors**2))
        shock = group.year.to_numpy() == 2012
        normal = np.abs(observed) <= 1.5*observed.std()
        rows.append({"model": "Naive" if head == "secular_trend" else "LSTM", "variant": head,
                     "include_lag": lag, "frequency": 10, "H": horizon,
                     "RMSE_OOS_pooled": rmse, "R2_OOS_pooled": 1-np.sum(errors**2)/np.sum(observed**2),
                     "MAE_OOS_pooled": np.mean(abs(errors)), "Skill_Score": 1-rmse/naive_rmse,
                     "Hit_Rate": np.nan if head == "secular_trend" else np.mean(np.sign(observed) == np.sign(predicted)),
                     "RMSE_2012": np.sqrt(np.mean(errors[shock]**2)), "Bias_2012": errors[shock].mean(),
                     "RMSE_normal": np.sqrt(np.mean(errors[normal]**2))})
    summary = pd.DataFrame(rows)
    summary.to_csv(PREDS_LSTM_PARQUET.with_name("evaluation_summary_lstm.csv"), index=False)
    summary.to_parquet(PREDS_LSTM_PARQUET.with_name("evaluation_summary_lstm.parquet"), index=False)
    candidates = summary[summary.model.eq("LSTM")]
    integrated = candidates[candidates.include_lag].drop(columns="include_lag")
    weather = candidates[~candidates.include_lag].drop(columns="include_lag")
    columns = ["model", "variant", "H", "R2_OOS_pooled", "RMSE_OOS_pooled"]
    comparison = integrated[columns].merge(weather[columns], on=["model", "variant", "H"],
                                            suffixes=("_integrated", "_weather_only"), validate="one_to_one")
    comparison["delta_R2"] = comparison.R2_OOS_pooled_integrated-comparison.R2_OOS_pooled_weather_only
    comparison["delta_RMSE"] = comparison.RMSE_OOS_pooled_integrated-comparison.RMSE_OOS_pooled_weather_only
    comparison.to_csv(REPORT_DIR / "comparison_weather_vs_integrated_lstm.csv", index=False)


def main():
    raw=load_combined_predictions()
    predictions=apply_minnesota_phenological_rule(raw)
    metrics=forecast_metrics(predictions)
    inference=inference_report(predictions)
    regional=pd.concat([regional_report(raw,"calendar_h1"),regional_report(predictions,"harvest_closure")],ignore_index=True)
    residuals=residual_diagnostics(predictions)
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    TABLE_DIR.mkdir(parents=True,exist_ok=True)
    previous_path = REPORT_DIR / "forecast_metrics.csv"
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        changes = previous.merge(metrics, on=["model", "H"], suffixes=("_previous", "_current"),
                                 validate="one_to_one")
        for field in ["RMSE", "MAE", "R2_OOS", "R2_level"]:
            changes[field + "_change"] = changes[field + "_current"] - changes[field + "_previous"]
        if changes.RMSE_change.abs().max() > 1e-10:
            changes.to_csv(REPORT_DIR / "test_refresh_comparison.csv", index=False)
    for name,frame in [("forecast_metrics",metrics),("forecast_inference",inference),
                       ("regional_performance",regional),("forecast_residual_diagnostics",residuals)]:
        frame.to_csv(REPORT_DIR/f"{name}.csv",index=False)
    write_evidence_tables(predictions,metrics,inference,regional)
    write_lstm_summaries()
    print(metrics[metrics.H.eq(1)][["model","RMSE","R2_OOS"]].to_string(index=False))
    print("\nVerified harvest-closure regional metrics:")
    print(regional[regional.scope.eq("harvest_closure")].groupby("model")[["RMSE","r_anomaly","r_level"]].first().to_string())
    print("\nAnnual HAC inference saved with 15 years and two annual lags.")


if __name__ == "__main__":
    main()
