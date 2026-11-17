"""
LightGBM (fase 4)
=================

Previsión directa a varios horizontes. Para un mes origen T (último dato conocido) y un
horizonte h = 1 … 12, el mes a prever es M = T + h y su "mes base" es B = M − 12, que
siempre es conocido (B ≤ T).

El modelo NO predice las pernoctaciones, sino su **crecimiento interanual**:

    z = log(y_M) − log(y_B)          →    previsión  y_M = y_B · exp(ẑ)

La estacionalidad la aporta y_B; el modelo se concentra en el crecimiento, que es justo lo
que fallaba en los modelos clásicos (fase 3: sesgo negativo en los años de aceleración).

Todas las variables se calculan con datos hasta T. Para el aeropuerto se respeta además
el retraso de publicación de Eurostat: en T solo se conoce hasta T − 3.

Sigue la firma de `backtest.Modelo` y se entrena de nuevo en cada origen.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

H = 12
RETRASO_AEROPUERTO = 3      # meses que Eurostat publica después que el INE
INICIO = "2000-01-01"       # primer origen de entrenamiento (hacen falta 12 meses previos)

PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=4, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, deterministic=True, n_jobs=1, verbose=-1,
)

# Todas las variables construidas. La selección final (VARIABLES) se eligió en el periodo de
# validación 2010–2014, sin mirar los periodos de evaluación (ver notebook 04).
VARIABLES_TODAS = [
    "h", "mes_objetivo",
    # calendario: diferencia entre el mes a prever y su mes base
    "ss_dif", "finde_dif", "dias_dif",
    # momento reciente del objetivo (crecimiento interanual en log)
    "crec_1", "crec_3", "crec_6", "crec_12", "crec_base",
    # por tipo de cliente
    "crec_ext_3", "crec_esp_3",
    # oferta
    "crec_plazas_3",
    # aeropuerto, con retraso de publicación
    "crec_aero_intl_3", "crec_aero_nac_3",
]

VARIABLES = [
    "h", "mes_objetivo", "ss_dif", "finde_dif", "dias_dif",
    "crec_1", "crec_3", "crec_6", "crec_12", "crec_base",
    "crec_aero_intl_3", "crec_aero_nac_3",
]


def _prefijo(objetivo: str) -> str:
    return objetivo.split("_pernoctaciones")[0]                 # costa_blanca / benidorm


def _log_valido(s: pd.Series, covid: pd.Series) -> pd.Series:
    return np.log(s.where((s > 0) & (covid == 0)))


def variables_origen(d: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    """Variables que solo dependen del mes origen T (calculadas con datos hasta T)."""
    cv = d["covid"]
    ly = _log_valido(d[objetivo], cv)
    g = ly - ly.shift(12)                                       # crecimiento interanual mensual
    p = _prefijo(objetivo)

    def crec_3(col):
        if col not in d:
            return pd.Series(np.nan, index=d.index)
        l = _log_valido(d[col], cv)
        return (l - l.shift(12)).rolling(3).mean()

    y_ok = d[objetivo].where(cv == 0)
    suma12 = y_ok.rolling(12).sum()
    v = pd.DataFrame(index=d.index)
    v["crec_1"] = g
    v["crec_3"] = g.rolling(3).mean()
    v["crec_6"] = g.rolling(6).mean()
    v["crec_12"] = np.log(suma12) - np.log(suma12.shift(12))
    v["crec_ext_3"] = crec_3(f"{p}_pernoctaciones_ext")
    v["crec_esp_3"] = crec_3(f"{p}_pernoctaciones_esp")
    v["crec_plazas_3"] = crec_3(f"{p}_plazas_total")
    v["crec_aero_intl_3"] = crec_3("aeropuerto_pasajeros_intl").shift(RETRASO_AEROPUERTO)
    v["crec_aero_nac_3"] = crec_3("aeropuerto_pasajeros_nac").shift(RETRASO_AEROPUERTO)
    v["_g"] = g                                                 # auxiliar para crec_base
    v["_ly"] = ly                                               # auxiliar para el objetivo
    return v


def filas(v: pd.DataFrame, cal: pd.DataFrame, origenes: pd.DatetimeIndex) -> pd.DataFrame:
    """Una fila por (origen T, horizonte h) con las variables y el objetivo z (si se conoce)."""
    partes = []
    for h in range(1, H + 1):
        f = v.loc[origenes].copy()
        M = origenes + pd.DateOffset(months=h)
        B = M - pd.DateOffset(months=12)
        f["origen"], f["fecha"], f["h"] = origenes, M, h
        f["mes_objetivo"] = M.month
        cM, cB = cal.reindex(M), cal.reindex(B)
        f["ss_dif"] = cM["dias_semana_santa"].values - cB["dias_semana_santa"].values
        f["finde_dif"] = cM["dias_finde"].values - cB["dias_finde"].values
        f["dias_dif"] = cM["dias_mes"].values - cB["dias_mes"].values
        f["crec_base"] = v["_g"].reindex(B).values
        f["ly_base"] = v["_ly"].reindex(B).values
        f["z"] = v["_ly"].reindex(M).values - f["ly_base"].values
        partes.append(f.reset_index(drop=True))
    return pd.concat(partes, ignore_index=True)


def _calendario(hist: pd.DataFrame, futuro: pd.DataFrame) -> pd.DataFrame:
    cols = ["dias_semana_santa", "dias_finde", "dias_mes"]
    return pd.concat([hist[cols], futuro[cols]]).groupby(level=0).last()


def entrenamiento(hist: pd.DataFrame, objetivo: str, variables=VARIABLES) -> pd.DataFrame:
    """Filas de entrenamiento: solo las que tienen el mes objetivo ya observado (M ≤ T)."""
    v = variables_origen(hist, objetivo)
    cal = _calendario(hist, hist.iloc[:0])
    T = hist.index.max()
    origenes = hist.index[(hist.index >= INICIO) & (hist.index < T)]
    f = filas(v, cal, origenes)
    return f[(f.fecha <= T) & f.z.notna()]


def ajustar(tr: pd.DataFrame, variables=VARIABLES, objetivo_lgb="l2", alpha=None, params=None, vida_media=None):
    """Entrena LightGBM sobre z. `vida_media` (años) da más peso a las filas recientes."""
    params = dict(PARAMS, **(params or {}), objective=objetivo_lgb)
    if alpha is not None:
        params["alpha"] = alpha
    m = lgb.LGBMRegressor(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = None
        if vida_media:
            edad = (tr["fecha"].max() - tr["fecha"]).dt.days / 365.25
            w = 0.5 ** (edad / vida_media)
        m.fit(tr[variables], tr["z"], sample_weight=w)
    return m


def hacer_lgbm(variables=VARIABLES, intervalos=True, nivel=0.8, params=None, objetivo_lgb="l2", vida_media=None):
    """Crea un modelo con la firma de backtest. `variables` permite probar variantes."""
    a = (1 - nivel) / 2

    def modelo(hist, futuro, objetivo):
        tr = entrenamiento(hist, objetivo, variables)
        T = hist.index.max()
        v = variables_origen(hist, objetivo)
        cal = _calendario(hist, futuro)
        X = filas(v, cal, pd.DatetimeIndex([T]))
        X = X[X.fecha.isin(futuro.index)].set_index("fecha")
        base = np.exp(X["ly_base"])

        m = ajustar(tr, variables, objetivo_lgb, params=params, vida_media=vida_media)
        out = pd.DataFrame({"pred": base * np.exp(m.predict(X[variables]))}, index=X.index)
        if intervalos:
            lo = ajustar(tr, variables, "quantile", a, params, vida_media).predict(X[variables])
            hi = ajustar(tr, variables, "quantile", 1 - a, params, vida_media).predict(X[variables])
            out["inf"] = base * np.exp(np.minimum(lo, hi))
            out["sup"] = base * np.exp(np.maximum(lo, hi))
        return out.reindex(futuro.index)

    return modelo


lgbm = hacer_lgbm()


def lgbm_mas_sarimax(hist, futuro, objetivo):
    """Combinación final: media geométrica de LightGBM y SARIMAX + Semana Santa.
    Intervalo: el de LightGBM (cuantiles), recentrado en la previsión combinada."""
    from modelos_base import sarimax_semana_santa

    a = lgbm(hist, futuro, objetivo)
    b = sarimax_semana_santa(hist, futuro, objetivo)
    pred = np.sqrt(a.pred * b.pred)
    f = pred / a.pred
    return pd.DataFrame({"pred": pred, "inf": a["inf"] * f, "sup": a["sup"] * f}, index=futuro.index)
