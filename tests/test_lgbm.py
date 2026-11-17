import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modelo_lgbm as ml  # noqa: E402
from descarga import calendario  # noqa: E402


def _dataset(n_anyos=8, semilla=0):
    rng = np.random.default_rng(semilla)
    f = pd.date_range("2005-01-01", periods=12 * n_anyos, freq="MS")
    base = 100 + 50 * np.sin(2 * np.pi * (f.month - 1) / 12)
    d = pd.DataFrame(index=f)
    obj = "costa_blanca_pernoctaciones_total"
    d[obj] = base * np.exp(rng.normal(0, 0.03, len(f)))
    d["costa_blanca_pernoctaciones_esp"] = d[obj] * 0.4
    d["costa_blanca_pernoctaciones_ext"] = d[obj] * 0.6
    d["costa_blanca_plazas_total"] = 1000.0
    d["aeropuerto_pasajeros_intl"] = d[obj] * 2 * np.exp(rng.normal(0, 0.05, len(f)))
    d["aeropuerto_pasajeros_nac"] = d[obj] * 0.5
    cal = calendario(f.min().strftime("%Y-%m-%d"), f.max().strftime("%Y-%m-%d")).set_index("fecha")
    d = d.join(cal)
    d["covid"] = 0
    return d.asfreq("MS"), obj


def test_variables_sin_fuga_de_futuro():
    d, obj = _dataset()
    T = pd.Timestamp("2010-06-01")
    v_corto = ml.variables_origen(d.loc[:T], obj).loc[T]
    v_largo = ml.variables_origen(d, obj).loc[T]
    # Las variables del mes T no cambian al añadir datos posteriores
    pd.testing.assert_series_equal(v_corto, v_largo, check_names=False)


def test_aeropuerto_respeta_retraso_de_publicacion():
    d, obj = _dataset()
    T = pd.Timestamp("2010-06-01")
    d2 = d.copy()
    d2.loc[T - pd.DateOffset(months=2):T, "aeropuerto_pasajeros_intl"] *= 10   # últimos 3 meses aún no publicados
    a = ml.variables_origen(d, obj).at[T, "crec_aero_intl_3"]
    b = ml.variables_origen(d2, obj).at[T, "crec_aero_intl_3"]
    assert a == b


def test_entrenamiento_solo_con_meses_observados_y_prediccion():
    d, obj = _dataset()
    hist = d.loc[:"2011-12-01"]
    tr = ml.entrenamiento(hist, obj)
    assert tr.fecha.max() <= hist.index.max()
    fut = d.loc["2012-01-01":"2012-12-01", ["anyo", "mes", "dias_mes", "dias_finde", "dias_semana_santa", "pascua_en_mes", "covid"]]
    p = ml.hacer_lgbm(intervalos=True)(hist, fut, obj)
    assert p.pred.notna().all() and (p["inf"] <= p["sup"]).all()
    # La previsión debe estar en el orden de magnitud del mismo mes del año anterior
    ratio = p.pred.values / hist[obj].iloc[-12:].values
    assert (ratio > 0.7).all() and (ratio < 1.4).all()
