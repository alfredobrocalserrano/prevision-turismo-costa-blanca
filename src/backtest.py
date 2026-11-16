"""
Validación de origen móvil (backtesting)
=======================================

Simula cómo habría funcionado un modelo si lo hubiéramos usado de verdad: en cada mes
"origen" T se entrena solo con datos hasta T, se predicen los meses T+1 … T+12 y se
compara con lo que pasó. Después se avanza un mes y se repite.

Un modelo es una función con esta firma:

    modelo(hist: pd.DataFrame, futuro: pd.DataFrame, objetivo: str) -> pd.DataFrame

    hist     filas del dataset hasta el origen T (incluido); es todo lo que se sabía en T
    futuro   filas T+1 … T+h solo con columnas conocidas de antemano (calendario)
    objetivo nombre de la columna a predecir

y devuelve un DataFrame indexado por las fechas de `futuro` con la columna `pred` y,
opcionalmente, `inf` y `sup` (límites del intervalo de predicción).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]

# Columnas que se conocen de antemano para cualquier mes futuro
CALENDARIO = ["anyo", "mes", "dias_mes", "dias_finde", "dias_semana_santa", "pascua_en_mes", "covid"]

# Periodo de evaluación: desde abril de 2023 el mes de "hace un año" ya no es COVID
EVAL_DESDE = "2023-04-01"

Modelo = Callable[[pd.DataFrame, pd.DataFrame, str], pd.DataFrame]


def cargar_dataset(ruta: Path | None = None) -> pd.DataFrame:
    ruta = ruta or RAIZ / "data" / "processed" / "dataset_mensual.csv"
    d = pd.read_csv(ruta, parse_dates=["fecha"]).set_index("fecha").asfreq("MS")
    return d


def backtest(
    d: pd.DataFrame,
    modelo: Modelo,
    objetivo: str = "costa_blanca_pernoctaciones_total",
    eval_desde: str = EVAL_DESDE,
    eval_hasta: str | None = None,
    horizonte: int = 12,
    nombre: str | None = None,
) -> pd.DataFrame:
    """Devuelve una fila por (origen, mes previsto) con real, pred y error."""
    eval_desde = pd.Timestamp(eval_desde)
    eval_hasta = pd.Timestamp(eval_hasta) if eval_hasta else d[objetivo].last_valid_index()
    primer_origen = eval_desde - pd.DateOffset(months=1)
    origenes = pd.date_range(primer_origen, eval_hasta - pd.DateOffset(months=1), freq="MS")

    filas = []
    for T in origenes:
        fechas = pd.date_range(T + pd.DateOffset(months=1), periods=horizonte, freq="MS")
        fechas = fechas[fechas <= eval_hasta]
        hist = d.loc[:T]
        futuro = d.loc[fechas, [c for c in CALENDARIO if c in d.columns]]
        p = modelo(hist, futuro, objetivo)
        r = p.reindex(fechas).assign(
            origen=T,
            h=np.arange(1, len(fechas) + 1),
            real=d.loc[fechas, objetivo].values,
        )
        filas.append(r)

    res = pd.concat(filas).rename_axis("fecha").reset_index()
    res["error_pct"] = (res["pred"] - res["real"]) / res["real"] * 100
    res["modelo"] = nombre or getattr(modelo, "__name__", "modelo")
    return res


def metricas(res: pd.DataFrame, por: str | list[str] = "modelo") -> pd.DataFrame:
    """MAPE, MAE y sesgo medio (%). Si hay intervalos, su cobertura real (%)."""
    r = res.assign(ape=res.error_pct.abs(), ae=(res.pred - res.real).abs())
    if "inf" in res:
        dentro = ((res.real >= res["inf"]) & (res.real <= res["sup"])).astype(float)
        r["dentro"] = dentro.where(res["inf"].notna())        # sin intervalo -> no cuenta
    g = r.groupby(por)
    out = pd.DataFrame({
        "MAPE": g.ape.mean(),
        "MAE": g.ae.mean(),
        "sesgo_pct": g.error_pct.mean(),
        "n": g.size(),
    })
    if "inf" in res:
        out["cobertura"] = g.dentro.mean() * 100
    return out


def mape_por_horizonte(res: pd.DataFrame) -> pd.DataFrame:
    return res.assign(ape=res.error_pct.abs()).pivot_table(index="h", columns="modelo", values="ape", aggfunc="mean")
