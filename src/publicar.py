"""
Publicación (fase 5)
====================

Genera `docs/datos.json`, que lee la página web (`docs/index.html`, GitHub Pages):

- histórico mensual de pernoctaciones en la Costa Blanca
- previsión de los 12 meses siguientes con el modelo final (LightGBM + SARIMAX) e intervalo del 80 %
- explicación de cada mes previsto: contribuciones SHAP de la parte LightGBM
- resultados del backtesting (fases 3 y 4) para la tabla de precisión y la "máquina del tiempo"

Uso:
    python src/descarga.py      # datos nuevos del INE y Eurostat
    python src/publicar.py      # previsión y datos de la web

La GitHub Action `.github/workflows/actualizar.yml` ejecuta los dos pasos cada mes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import modelo_lgbm as ml
from backtest import CALENDARIO, cargar_dataset, mape_por_horizonte, metricas
from descarga import calendario

RAIZ = Path(__file__).resolve().parents[1]
OBJ = "costa_blanca_pernoctaciones_total"
SALIDA = RAIZ / "docs" / "datos.json"

NOMBRES = {
    "crec_base": "Crecimiento del mes base",
    "crec_12": "Tendencia de los últimos 12 meses",
    "ss_dif": "Semana Santa",
    "crec_aero_intl_3": "Aeropuerto (internacional)",
    "crec_aero_nac_3": "Aeropuerto (nacional)",
    "crec_6": "Tendencia de los últimos 6 meses",
    "crec_3": "Tendencia de los últimos 3 meses",
    "finde_dif": "Fines de semana",
    "crec_1": "Último mes",
    "dias_dif": "Días del mes",
    "mes_objetivo": "Mes del año",
    "h": "Horizonte",
}


def _r(x, n=0):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def prevision(d: pd.DataFrame) -> tuple[list[dict], pd.Timestamp]:
    T = d[OBJ].last_valid_index()
    futuro = calendario((T + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                        (T + pd.DateOffset(months=12)).strftime("%Y-%m-%d")).set_index("fecha")
    futuro["covid"] = 0
    fut = futuro[[c for c in CALENDARIO if c in futuro]]
    hist = d.loc[:T]
    p = ml.lgbm_mas_sarimax(hist, fut, OBJ)

    # Explicación: SHAP de la parte LightGBM (crecimiento interanual, en puntos %)
    tr = ml.entrenamiento(hist, OBJ)
    m = ml.ajustar(tr)
    v = ml.variables_origen(hist, OBJ)
    cal = ml._calendario(hist, fut)
    X = ml.filas(v, cal, pd.DatetimeIndex([T])).set_index("fecha")
    contrib = m.predict(X[ml.VARIABLES], pred_contrib=True)
    shap = pd.DataFrame(contrib[:, :-1] * 100, columns=ml.VARIABLES, index=X.index)
    media = contrib[0, -1] * 100

    filas = []
    for f, r in p.iterrows():
        base = d.at[f - pd.DateOffset(months=12), OBJ]
        s = shap.loc[f].sort_values(key=abs, ascending=False)
        filas.append({
            "fecha": f.strftime("%Y-%m"),
            "pred": _r(r.pred), "inf": _r(r["inf"]), "sup": _r(r["sup"]),
            "anterior": _r(base),
            "var_pct": _r((r.pred / base - 1) * 100, 1),
            "dias_semana_santa": int(futuro.at[f, "dias_semana_santa"]),
            "explicacion": {
                "media_pct": _r(media, 2),
                "crec_lgbm_pct": _r(media + s.sum(), 2),
                "factores": [{"variable": k, "nombre": NOMBRES[k], "puntos": _r(val, 2)}
                             for k, val in s.items() if abs(val) >= 0.05][:6],
            },
        })
    return filas, T


def precision() -> dict:
    f3 = pd.read_csv(RAIZ / "outputs" / "backtest_fase3.csv", parse_dates=["fecha", "origen"])
    f4 = pd.read_csv(RAIZ / "outputs" / "backtest_fase4.csv", parse_dates=["fecha", "origen"])
    res = pd.concat([f3, f4])
    modelos = ["Ingenuo estacional", "ETS", "SARIMA", "SARIMAX + Semana Santa",
               "Combinación ETS + SARIMAX", "LightGBM", "LightGBM + SARIMAX"]
    t = metricas(res, ["periodo", "modelo"])
    tabla = [{"modelo": mo,
              **{per: _r(t.at[(per, mo), "MAPE"], 2) for per in ["2015–2019", "2023–2026"]},
              "final": mo == "LightGBM + SARIMAX"} for mo in modelos]
    hz = mape_por_horizonte(res[res.periodo == "2023–2026"])
    horizonte = [{"h": int(h), "final": _r(hz.at[h, "LightGBM + SARIMAX"], 2),
                  "clasico": _r(hz.at[h, "Combinación ETS + SARIMAX"], 2),
                  "ingenuo": _r(hz.at[h, "Ingenuo estacional"], 2)} for h in hz.index]

    # Máquina del tiempo: previsiones pasadas del modelo final, 2023–2026
    bt = f4[(f4.modelo == "LightGBM + SARIMAX") & (f4.periodo == "2023–2026")]
    pasadas = {}
    for o, g in bt.groupby("origen"):
        pasadas[o.strftime("%Y-%m")] = [
            {"fecha": f.strftime("%Y-%m"), "pred": _r(p), "inf": _r(i), "sup": _r(s), "real": _r(re)}
            for f, p, i, s, re in g.sort_values("fecha")[["fecha", "pred", "inf", "sup", "real"]].itertuples(index=False)
        ]
    final = res[(res.modelo == "LightGBM + SARIMAX")]
    cob = t.loc[[(per, "LightGBM + SARIMAX") for per in ["2015–2019", "2023–2026"]], "cobertura"]
    return {"tabla": tabla, "horizonte": horizonte, "pasadas": pasadas,
            "cobertura": {per: _r(v, 1) for (per, _), v in cob.items()},
            "n_previsiones": int(len(final))}


def main():
    d = cargar_dataset()
    prev, T = prevision(d)
    y = d[OBJ]
    historico = [{"fecha": f.strftime("%Y-%m"), "valor": _r(v), "covid": int(c)}
                 for f, v, c in zip(d.index, y, d["covid"]) if f <= T]
    ult12 = y.loc[:T].iloc[-12:].sum()
    datos = {
        "generado": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "ultimo_dato": T.strftime("%Y-%m"),
        "resumen": {
            "total_previsto": _r(sum(p["pred"] for p in prev)),
            "total_inf": _r(sum(p["inf"] for p in prev)),
            "total_sup": _r(sum(p["sup"] for p in prev)),
            "total_anterior": _r(ult12),
        },
        "historico": historico,
        "prevision": prev,
        "precision": precision(),
    }
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    r = datos["resumen"]
    print(f"Último dato: {datos['ultimo_dato']} · previsión 12 meses: {r['total_previsto']/1e6:.2f} M "
          f"({(r['total_previsto']/r['total_anterior']-1)*100:+.1f} %) · {SALIDA.relative_to(RAIZ)} "
          f"({SALIDA.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
