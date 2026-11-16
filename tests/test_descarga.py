import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from descarga import a_ancho, a_long, aeropuerto_a_tabla, calendario, validar  # noqa: E402


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


def test_eurostat_jsonstat():
    # Estructura mínima de una respuesta JSON-stat: 3 coberturas x 2 meses
    js = {
        "id": ["freq", "tra_cov", "time"],
        "size": [1, 3, 2],
        "dimension": {
            "tra_cov": {"category": {"index": {"TOTAL": 0, "NAT": 1, "INTL": 2}}},
            "time": {"category": {"index": {"2025-07": 0, "2025-08": 1}}},
        },
        # posición = cobertura * 2 + mes
        "value": {"0": 100, "1": 110, "2": 30, "3": 32, "4": 70, "5": 78},
        "status": {"1": "u"},
    }
    t = aeropuerto_a_tabla(js)
    assert t.aeropuerto_pasajeros_total.tolist() == [100, 110]
    assert t.aeropuerto_pasajeros_nac.tolist() == [30, 32]
    assert t.aeropuerto_pasajeros_intl.tolist() == [70, 78]
    assert t.aeropuerto_baja_fiabilidad.tolist() == [False, True]
