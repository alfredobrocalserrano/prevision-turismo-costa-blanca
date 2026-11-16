import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backtest import backtest, metricas  # noqa: E402
from modelos_base import ingenuo_estacional  # noqa: E402


def _dataset(n_anyos=6):
    fechas = pd.date_range("2015-01-01", periods=12 * n_anyos, freq="MS")
    estacional = 100 + 50 * np.sin(2 * np.pi * (fechas.month - 1) / 12)
    return pd.DataFrame(
        {"y": estacional * (1.05 ** (np.arange(len(fechas)) / 12)), "covid": 0,
         "mes": fechas.month, "dias_semana_santa": 0},
        index=fechas,
    ).asfreq("MS")


def test_el_modelo_nunca_ve_el_futuro():
    d = _dataset()
    vistos = []

    def espia(hist, futuro, objetivo):
        vistos.append((hist.index.max(), futuro.index.min(), set(futuro.columns)))
        return pd.DataFrame({"pred": 1.0}, index=futuro.index)

    backtest(d, espia, "y", eval_desde="2019-01-01", horizonte=12)
    for ultimo_hist, primer_futuro, cols in vistos:
        assert ultimo_hist < primer_futuro            # sin fuga de datos
        assert "y" not in cols                         # el objetivo futuro no se pasa


def test_ingenuo_estacional_y_metricas():
    d = _dataset()
    res = backtest(d, ingenuo_estacional, "y", eval_desde="2019-01-01", horizonte=3, nombre="ing")
    # Serie con crecimiento del 5 % anual: el ingenuo se queda corto un 4,76 %
    assert np.allclose(res.error_pct, (1 / 1.05 - 1) * 100)
    m = metricas(res).loc["ing"]
    assert round(m.MAPE, 2) == 4.76 and m.sesgo_pct < 0
