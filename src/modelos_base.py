"""
Modelos de referencia (fase 3)
==============================

Todos siguen la firma de `backtest.Modelo`. Los modelos estadísticos trabajan en
logaritmos y devuelven la previsión en pernoctaciones.

Tratamiento del COVID (marzo 2020 – marzo 2022, columna `covid`):
- Ingenuos: no usan esos meses porque la evaluación empieza en abril de 2023.
- SARIMA / SARIMAX: los meses COVID pasan a valor desconocido (NaN). El filtro de Kalman
  del modelo de espacio de estados los salta sin necesidad de inventarlos.
- ETS: no admite huecos, así que los meses COVID se rellenan con el mismo mes de 2019
  (escenario "sin pandemia"). Es una suposición, y el notebook la compara.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

INICIO_ENTRENO = "1999-01-01"


def _log_sin_covid(hist: pd.DataFrame, objetivo: str) -> pd.Series:
    y = np.log(hist[objetivo].where(hist[objetivo] > 0))
    return y.where(hist["covid"] == 0)


def _mismo_mes_hace(hist: pd.DataFrame, objetivo: str, fecha: pd.Timestamp) -> float:
    """Valor del mismo mes del año anterior conocido y sin COVID (retrocede de 12 en 12)."""
    f = fecha - pd.DateOffset(months=12)
    while f >= hist.index[0]:
        if f in hist.index and hist.at[f, "covid"] == 0 and pd.notna(hist.at[f, objetivo]):
            return hist.at[f, objetivo]
        f -= pd.DateOffset(months=12)
    return np.nan


# ---------------------------------------------------------------------------
# 1. Ingenuo estacional: "igual que el mismo mes del año pasado"
# ---------------------------------------------------------------------------
def ingenuo_estacional(hist, futuro, objetivo):
    pred = [_mismo_mes_hace(hist, objetivo, f) for f in futuro.index]
    return pd.DataFrame({"pred": pred}, index=futuro.index)


# ---------------------------------------------------------------------------
# 2. Ingenuo con tendencia reciente: el mismo mes del año pasado × el crecimiento
#    interanual de los últimos 3 meses conocidos
# ---------------------------------------------------------------------------
def ingenuo_tendencia(hist, futuro, objetivo, meses: int = 3):
    ult = hist.index[-meses:]
    ant = ult - pd.DateOffset(months=12)
    ok = hist.loc[ult, "covid"].eq(0).all() and hist.reindex(ant)["covid"].eq(0).all()
    crec = hist.loc[ult, objetivo].sum() / hist.reindex(ant)[objetivo].sum() if ok else 1.0
    base = ingenuo_estacional(hist, futuro, objetivo)
    return base.assign(pred=base.pred * crec)


# ---------------------------------------------------------------------------
# 3. ETS (Holt-Winters): nivel + tendencia amortiguada + estacionalidad, en logaritmos
# ---------------------------------------------------------------------------
def ets(hist, futuro, objetivo):
    y = np.log(hist[objetivo].where(hist[objetivo] > 0)).loc[INICIO_ENTRENO:]
    # COVID -> mismo mes de 2019 (escenario sin pandemia)
    covid = hist.loc[y.index, "covid"] == 1
    for f in y.index[covid]:
        y[f] = np.log(hist.at[pd.Timestamp(2019, f.month, 1), objetivo])
    y = y.interpolate()                                      # por si quedara algún hueco suelto
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                                 seasonal_periods=12, initialization_method="estimated").fit()
    fc = m.forecast(len(futuro))
    return pd.DataFrame({"pred": np.exp(fc.values)}, index=futuro.index)


# ---------------------------------------------------------------------------
# 4 y 5. SARIMA y SARIMAX con días de Semana Santa
#        Orden elegido por AIC sobre 2003–2019: (1,0,0)(1,1,1)12 con constante
# ---------------------------------------------------------------------------
ORDEN, ORDEN_EST = (1, 0, 0), (1, 1, 1, 12)


def _sarimax(hist, futuro, objetivo, exog_cols, alpha=0.2):
    y = _log_sin_covid(hist, objetivo).loc[INICIO_ENTRENO:]
    ex = hist.loc[y.index, exog_cols] if exog_cols else None
    ex_f = futuro[exog_cols] if exog_cols else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAX(y, exog=ex, order=ORDEN, seasonal_order=ORDEN_EST, trend="c").fit(disp=False)
        fc = m.get_forecast(len(futuro), exog=ex_f)
    ci = fc.conf_int(alpha=alpha)
    return pd.DataFrame(
        {"pred": np.exp(fc.predicted_mean.values),           # mediana de la previsión
         "inf": np.exp(ci.iloc[:, 0].values),
         "sup": np.exp(ci.iloc[:, 1].values)},
        index=futuro.index,
    )


def sarima(hist, futuro, objetivo):
    return _sarimax(hist, futuro, objetivo, exog_cols=[])


def sarimax_semana_santa(hist, futuro, objetivo):
    return _sarimax(hist, futuro, objetivo, exog_cols=["dias_semana_santa"])


# ---------------------------------------------------------------------------
# 6. Combinación: media geométrica de ETS y SARIMAX + Semana Santa
#    El intervalo es el del SARIMAX, recentrado en la previsión combinada.
# ---------------------------------------------------------------------------
def combinacion(hist, futuro, objetivo):
    a = ets(hist, futuro, objetivo)
    b = sarimax_semana_santa(hist, futuro, objetivo)
    pred = np.sqrt(a.pred * b.pred)
    factor = pred / b.pred
    return pd.DataFrame({"pred": pred, "inf": b["inf"] * factor, "sup": b["sup"] * factor}, index=futuro.index)


MODELOS = {
    "Ingenuo estacional": ingenuo_estacional,
    "Ingenuo + tendencia": ingenuo_tendencia,
    "ETS": ets,
    "SARIMA": sarima,
    "SARIMAX + Semana Santa": sarimax_semana_santa,
    "Combinación ETS + SARIMAX": combinacion,
}
