import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from descarga import a_ancho, a_long, calendario, validar  # noqa: E402


def _serie(valores, desde="M01", anyo=2024):
    mes0 = int(desde.lstrip("M"))
    return {
        "Nombre": "prueba",
        "Data": [
            {"Anyo": anyo, "T3_Periodo": f"M{mes0 + i:02d}", "T3_TipoDato": "Definitivo", "Valor": v}
            for i, v in enumerate(valores)
        ],
    }


def test_totales_y_estancia():
    long = pd.concat(
        [
            a_long("EOT7681", _serie([100, None, 300])),  # pernoctaciones esp
            a_long("EOT7682", _serie([200, 50, 300])),  # pernoctaciones ext
            a_long("EOT7679", _serie([50, 10, 100])),  # viajeros esp
            a_long("EOT7680", _serie([50, 10, 100])),  # viajeros ext
        ]
    )
    ancho = a_ancho(long)
    assert ancho.costa_blanca_pernoctaciones_total.tolist()[0] == 300
    assert pd.isna(ancho.costa_blanca_pernoctaciones_total[1])  # falta un sumando -> vacío, no 50
    assert ancho.costa_blanca_estancia_media[2] == 3.0


def test_semana_santa():
    cal = calendario("2024-03-01", "2025-04-01").set_index("fecha")
    # Pascua 2024: 31 de marzo -> Ramos a Pascua en marzo, Lunes de Pascua en abril
    assert cal.loc["2024-03-01", "dias_semana_santa"] == 8
    assert cal.loc["2024-04-01", "dias_semana_santa"] == 1
    assert cal.loc["2024-03-01", "pascua_en_mes"] == 1
    # Pascua 2025: 20 de abril -> toda en abril
    assert cal.loc["2025-04-01", "dias_semana_santa"] == 9


def test_validar_detecta_huecos():
    long = a_long("EOT1700", _serie([1, 2, 3]))
    ancho = a_ancho(long).drop(index=1)  # quitamos febrero
    avisos = validar(long, ancho)
    assert any("Faltan 1 meses" in a for a in avisos)
