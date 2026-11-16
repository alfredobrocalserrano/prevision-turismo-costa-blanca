"""
Fase 1 · Descarga de datos
==========================

Descarga de la API JSON del INE (Encuesta de Ocupación Hotelera, EOH) las series
mensuales de la Costa Blanca, Benidorm y la provincia de Alicante, las valida y
las guarda en DuckDB y en CSV. Genera además una tabla de calendario
(fines de semana, Semana Santa, periodo COVID).

Uso:
    python src/descarga.py                 # descarga todo y guarda en data/
    python src/descarga.py buscar 2078 Torrevieja   # buscar más series en una tabla del INE

Salidas:
    data/turismo.duckdb       tablas eoh_long, eoh_mensual, calendario, descargas
    data/processed/eoh_mensual.csv
    data/processed/calendario.csv

La API del INE no necesita clave. Documentación:
https://www.ine.es/dyngs/DAB/index.htm?cid=1099
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests
from dateutil.easter import easter

API = "https://servicios.ine.es/wstempus/js/ES"
RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
PROCESSED = DATA / "processed"
RAW = DATA / "raw"
DB = DATA / "turismo.duckdb"

# ---------------------------------------------------------------------------
# Catálogo de series. Código INE -> (ámbito, variable, residencia)
# Tablas de origen: 2039 (zonas turísticas), 2013 (ocupación por zona),
# 2078 (puntos turísticos), 2076 (ocupación por punto), 2074 (provincias)
# ---------------------------------------------------------------------------
SERIES: dict[str, tuple[str, str, str]] = {
    # Costa Blanca (zona turística) · desde 1999
    "EOT7679": ("costa_blanca", "viajeros", "esp"),
    "EOT7680": ("costa_blanca", "viajeros", "ext"),
    "EOT7681": ("costa_blanca", "pernoctaciones", "esp"),
    "EOT7682": ("costa_blanca", "pernoctaciones", "ext"),
    "EOT558": ("costa_blanca", "establecimientos", "total"),
    "EOT559": ("costa_blanca", "plazas", "total"),
    "EOT560": ("costa_blanca", "ocupacion_plazas", "total"),
    "EOT11682": ("costa_blanca", "ocupacion_finde", "total"),  # desde 2005
    # Benidorm (punto turístico) · desde 2005
    "EOT2623": ("benidorm", "viajeros", "esp"),
    "EOT2624": ("benidorm", "viajeros", "ext"),
    "EOT2625": ("benidorm", "pernoctaciones", "esp"),
    "EOT2626": ("benidorm", "pernoctaciones", "ext"),
    "EOT3046": ("benidorm", "plazas", "total"),
    "EOT3118": ("benidorm", "ocupacion_plazas", "total"),
    # Provincia de Alicante · desde 1999
    "EOT1697": ("prov_alicante", "viajeros", "total"),
    "EOT1698": ("prov_alicante", "viajeros", "esp"),
    "EOT1699": ("prov_alicante", "viajeros", "ext"),
    "EOT1700": ("prov_alicante", "pernoctaciones", "total"),
    "EOT1701": ("prov_alicante", "pernoctaciones", "esp"),
    "EOT1702": ("prov_alicante", "pernoctaciones", "ext"),
}

# Meses con restricciones por la pandemia. Es una decisión de análisis:
# revísala en el EDA (fase 2) y documenta el criterio en el README.
COVID_INICIO, COVID_FIN = "2020-03-01", "2021-12-01"


# ---------------------------------------------------------------------------
# API del INE
# ---------------------------------------------------------------------------
def _get(url: str, intentos: int = 4, timeout: int = 30):
    for i in range(intentos):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == intentos - 1:
                raise RuntimeError(f"Fallo al descargar {url}: {e}") from e
            time.sleep(2 ** i)


def descargar_serie(codigo: str) -> dict:
    """Devuelve la serie completa con salida 'amigable' (tip=A)."""
    return _get(f"{API}/DATOS_SERIE/{codigo}?nult=1000&tip=A")


def buscar_series(tabla: int, patron: str) -> list[tuple[str, str]]:
    """Lista las series de una tabla del INE cuyo nombre contiene `patron`."""
    series = _get(f"{API}/SERIES_TABLA/{tabla}")
    p = patron.lower()
    return [(s["COD"], s["Nombre"]) for s in series if p in s["Nombre"].lower()]


# ---------------------------------------------------------------------------
# Transformación
# ---------------------------------------------------------------------------
def a_long(codigo: str, serie: dict) -> pd.DataFrame:
    ambito, variable, residencia = SERIES[codigo]
    filas = []
    for d in serie.get("Data", []):
        mes = int(str(d["T3_Periodo"]).lstrip("M"))
        filas.append(
            {
                "fecha": date(int(d["Anyo"]), mes, 1),
                "codigo": codigo,
                "ambito": ambito,
                "variable": variable,
                "residencia": residencia,
                "valor": d["Valor"],
                "tipo_dato": d.get("T3_TipoDato"),  # Definitivo / Provisional
                "nombre_ine": serie.get("Nombre", "").strip(),
            }
        )
    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df


def a_ancho(long: pd.DataFrame) -> pd.DataFrame:
    """Una fila por mes; columnas tipo costa_blanca_pernoctaciones_esp."""
    long = long.assign(col=long.ambito + "_" + long.variable + "_" + long.residencia)
    ancho = long.pivot_table(index="fecha", columns="col", values="valor", aggfunc="first")
    ancho = ancho.sort_index()

    # Totales que el INE no da directamente para zona y punto turístico
    for amb in ("costa_blanca", "benidorm"):
        for var in ("viajeros", "pernoctaciones"):
            e, x = f"{amb}_{var}_esp", f"{amb}_{var}_ext"
            if e in ancho and x in ancho:
                # min_count=2: si falta uno de los dos, el total queda vacío
                ancho[f"{amb}_{var}_total"] = ancho[[e, x]].sum(axis=1, min_count=2)
        p, v = f"{amb}_pernoctaciones_total", f"{amb}_viajeros_total"
        if p in ancho and v in ancho:
            ancho[f"{amb}_estancia_media"] = (ancho[p] / ancho[v]).round(3)

    # Mes provisional = al menos una serie del mes aún no es definitiva
    prov = long.groupby("fecha")["tipo_dato"].apply(lambda s: (s == "Provisional").any())
    ancho["provisional"] = prov.reindex(ancho.index).fillna(False).astype(bool)

    ancho.columns.name = None
    return ancho.reset_index()


def calendario(inicio: str, fin: str) -> pd.DataFrame:
    """Variables de calendario por mes."""
    meses = pd.date_range(inicio, fin, freq="MS")
    filas = []
    for m in meses:
        dias = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
        pascua = pd.Timestamp(easter(m.year))
        # Semana Santa amplia: Domingo de Ramos -> Lunes de Pascua (festivo en la C. Valenciana)
        ss = pd.date_range(pascua - timedelta(days=7), pascua + timedelta(days=1), freq="D")
        filas.append(
            {
                "fecha": m,
                "anyo": m.year,
                "mes": m.month,
                "dias_mes": len(dias),
                "dias_finde": int((dias.day_of_week >= 5).sum()),
                "dias_semana_santa": int(dias.isin(ss).sum()),
                "pascua_en_mes": int(pascua.month == m.month),
                "covid": int(COVID_INICIO <= m.strftime("%Y-%m-%d") <= COVID_FIN),
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def validar(long: pd.DataFrame, ancho: pd.DataFrame) -> list[str]:
    avisos = []

    # 1. Meses consecutivos sin huecos
    esperado = pd.date_range(ancho.fecha.min(), ancho.fecha.max(), freq="MS")
    faltan = esperado.difference(ancho.fecha)
    if len(faltan):
        avisos.append(f"Faltan {len(faltan)} meses en la tabla: {list(faltan.strftime('%Y-%m'))}")

    # 2. Valores vacíos por serie (esperados: may–jun 2020 en zonas y puntos)
    nulos = long[long.valor.isna()].groupby(["ambito", "variable", "residencia"]).fecha.apply(
        lambda s: ", ".join(s.dt.strftime("%Y-%m"))
    )
    for (amb, var, res), meses in nulos.items():
        avisos.append(f"Vacíos en {amb}/{var}/{res}: {meses}")

    # 3. Total provincial = residentes en España + extranjero
    for var in ("viajeros", "pernoctaciones"):
        t, e, x = (f"prov_alicante_{var}_{r}" for r in ("total", "esp", "ext"))
        if {t, e, x} <= set(ancho.columns):
            dif = (ancho[t] - ancho[e] - ancho[x]).abs()
            if (dif > 1).any():
                avisos.append(f"prov_alicante {var}: total ≠ esp + ext en {int((dif > 1).sum())} meses")

    # 4. Grados de ocupación entre 0 y 100
    for c in [c for c in ancho.columns if "ocupacion" in c]:
        malos = ancho[(ancho[c] < 0) | (ancho[c] > 100)]
        if len(malos):
            avisos.append(f"{c}: {len(malos)} valores fuera de 0–100")

    return avisos


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------
def guardar(long: pd.DataFrame, ancho: pd.DataFrame, cal: pd.DataFrame, origen: str) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with duckdb.connect(str(DB)) as con:
        con.register("t_long", long)
        con.register("t_ancho", ancho)
        con.register("t_cal", cal)
        con.execute("CREATE OR REPLACE TABLE eoh_long AS SELECT * REPLACE (fecha::DATE AS fecha) FROM t_long")
        con.execute("CREATE OR REPLACE TABLE eoh_mensual AS SELECT * REPLACE (fecha::DATE AS fecha) FROM t_ancho ORDER BY fecha")
        con.execute("CREATE OR REPLACE TABLE calendario AS SELECT * REPLACE (fecha::DATE AS fecha) FROM t_cal ORDER BY fecha")
        con.execute(
            "CREATE TABLE IF NOT EXISTS descargas "
            "(descargado_en VARCHAR, origen VARCHAR, ultimo_mes DATE, n_series INTEGER)"
        )
        con.execute(
            "INSERT INTO descargas VALUES (?, ?, ?, ?)",
            [ahora, origen, ancho.fecha.max().date(), long.codigo.nunique()],
        )

    ancho.to_csv(PROCESSED / "eoh_mensual.csv", index=False)
    cal.to_csv(PROCESSED / "calendario.csv", index=False)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def ejecutar(desde_json: Path | None = None) -> pd.DataFrame:
    partes = []
    for codigo in SERIES:
        if desde_json:  # modo sin conexión: lee JSON ya descargados
            serie = json.loads((desde_json / f"{codigo}.json").read_text(encoding="utf-8"))
        else:
            serie = descargar_serie(codigo)
            RAW.mkdir(parents=True, exist_ok=True)
            (RAW / f"{codigo}.json").write_text(json.dumps(serie, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.3)  # no saturar la API
        partes.append(a_long(codigo, serie))
        print(f"  ✓ {codigo:9s} {SERIES[codigo]}")

    long = pd.concat(partes, ignore_index=True)
    ancho = a_ancho(long)
    cal = calendario(ancho.fecha.min().strftime("%Y-%m-%d"), ancho.fecha.max().strftime("%Y-%m-%d"))
    ancho = ancho.merge(cal[["fecha", "covid"]], on="fecha", how="left")

    avisos = validar(long, ancho)
    guardar(long, ancho, cal, origen=str(desde_json) if desde_json else API)

    print(f"\n{len(SERIES)} series · {len(ancho)} meses "
          f"({ancho.fecha.min():%Y-%m} → {ancho.fecha.max():%Y-%m}) · guardado en {DB.relative_to(RAIZ)}")
    if avisos:
        print("\nAvisos de validación:")
        for a in avisos:
            print("  ·", a)
    return ancho


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("buscar", help="buscar series en una tabla del INE")
    b.add_argument("tabla", type=int)
    b.add_argument("patron")
    ap.add_argument("--desde-json", type=Path, help="carpeta con JSON ya descargados (sin conexión)")
    args = ap.parse_args(argv)

    if args.cmd == "buscar":
        for cod, nombre in buscar_series(args.tabla, args.patron):
            print(f"{cod:10s} {nombre}")
        return
    ejecutar(args.desde_json)


if __name__ == "__main__":
    sys.exit(main())
