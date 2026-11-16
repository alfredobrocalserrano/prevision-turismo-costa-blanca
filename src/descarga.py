"""
Descarga de datos
=================

1. INE · Encuesta de Ocupación Hotelera (EOH): series mensuales de la Costa Blanca,
   Benidorm y la provincia de Alicante.
2. Eurostat · pasajeros mensuales del aeropuerto de Alicante-Elche (tabla avia_paoa).
3. Calendario: fines de semana, Semana Santa y periodo COVID.

Valida los datos y los guarda en DuckDB y en CSV.

Uso:
    python src/descarga.py                 # descarga todo y guarda en data/
    python src/descarga.py buscar 2078 Torrevieja   # buscar más series en una tabla del INE

Salidas:
    data/turismo.duckdb       tablas eoh_long, eoh_mensual, aeropuerto, calendario,
                              dataset_mensual (todo junto, para modelar), descargas
    data/processed/*.csv      las mismas tablas en CSV

Ninguna de las dos API necesita clave:
    INE:      https://www.ine.es/dyngs/DAB/index.htm?cid=1099
    Eurostat: https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access
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

# Meses afectados por la pandemia. Criterio del EDA (notebooks/02_eda.ipynb, sección 7):
# de marzo de 2020 a marzo de 2022 (ómicron en el primer trimestre de 2022). Desde abril
# de 2022 los meses de temporada vuelven al 93–100 % de 2019.
COVID_INICIO, COVID_FIN = "2020-03-01", "2022-03-01"

# Eurostat · pasajeros por aeropuerto. ES_LEAL = Alicante-Elche Miguel Hernández.
# Totales, nacionales e internacionales, todo tipo de vuelo (regular y chárter).
EUROSTAT = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/avia_paoa"
    "?format=JSON&lang=EN&freq=M&unit=PAS&tra_meas=PAS_CRD&rep_airp=ES_LEAL"
    "&schedule=TOTAL&tra_cov=TOTAL&tra_cov=NAT&tra_cov=INTL"
)
EUROSTAT_COLS = {"TOTAL": "aeropuerto_pasajeros_total", "NAT": "aeropuerto_pasajeros_nac",
                 "INTL": "aeropuerto_pasajeros_intl"}
EUROSTAT_JSON = "eurostat_avia_paoa_ES_LEAL.json"


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


def descargar_aeropuerto() -> dict:
    """Respuesta JSON-stat de Eurostat con los pasajeros de Alicante-Elche."""
    return _get(EUROSTAT, timeout=60)


def aeropuerto_a_tabla(js: dict) -> pd.DataFrame:
    """Convierte la respuesta JSON-stat de Eurostat en una tabla mensual.

    En JSON-stat los valores vienen en un diccionario {posición: valor}, donde la
    posición se calcula a partir del índice de cada dimensión (orden de `id`,
    tamaños en `size`). Aquí solo varían 'tra_cov' y 'time'.
    """
    ids, size = js["id"], js["size"]
    strides = [1] * len(ids)
    for i in range(len(ids) - 2, -1, -1):
        strides[i] = strides[i + 1] * size[i + 1]
    pos = {d: i for i, d in enumerate(ids)}
    cov_idx = js["dimension"]["tra_cov"]["category"]["index"]
    time_idx = js["dimension"]["time"]["category"]["index"]
    estado = js.get("status", {})

    filas = []
    for mes, ti in time_idx.items():
        fila = {"fecha": pd.Timestamp(f"{mes}-01")}
        baja_fiabilidad = False
        for cov, ci in cov_idx.items():
            k = str(ci * strides[pos["tra_cov"]] + ti * strides[pos["time"]])
            fila[EUROSTAT_COLS[cov]] = js["value"].get(k)
            baja_fiabilidad |= estado.get(k) == "u"
        fila["aeropuerto_baja_fiabilidad"] = baja_fiabilidad
        filas.append(fila)
    df = pd.DataFrame(filas)
    return df.dropna(subset=list(EUROSTAT_COLS.values()), how="all").reset_index(drop=True)


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

    # 2. Valores vacíos (esperados: may–jun 2020 en zona y punto turístico, feb 2021 en Benidorm)
    vacios = long[long.valor.isna()]
    for amb, g in vacios.groupby("ambito"):
        meses = ", ".join(sorted(g.fecha.dt.strftime("%Y-%m").unique()))
        avisos.append(f"Meses vacíos en {amb} ({g.codigo.nunique()} series): {meses}")

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


def validar_aeropuerto(aero: pd.DataFrame) -> list[str]:
    avisos = []
    t, n, i = (EUROSTAT_COLS[k] for k in ("TOTAL", "NAT", "INTL"))
    dif = (aero[t] - aero[n] - aero[i]).abs() / aero[t]
    if (dif > 0.01).any():
        avisos.append(f"aeropuerto: total ≠ nacional + internacional (>1 %) en {int((dif > 0.01).sum())} meses")
    esperado = pd.date_range(aero.fecha.min(), aero.fecha.max(), freq="MS")
    faltan = esperado.difference(aero.fecha)
    if len(faltan):
        avisos.append(f"aeropuerto: faltan {len(faltan)} meses")
    if aero.aeropuerto_baja_fiabilidad.any():
        f = aero[aero.aeropuerto_baja_fiabilidad].fecha
        avisos.append(f"aeropuerto: Eurostat marca {len(f)} meses como poco fiables "
                      f"({f.min():%Y-%m} → {f.max():%Y-%m})")
    return avisos


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------
def guardar(tablas: dict[str, pd.DataFrame], origen: str, ultimo_mes, n_series: int) -> None:
    """Guarda cada tabla en DuckDB (fecha como DATE) y en data/processed/<tabla>.csv."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with duckdb.connect(str(DB)) as con:
        for nombre, df in tablas.items():
            con.register("t", df)
            con.execute(
                f"CREATE OR REPLACE TABLE {nombre} AS "
                "SELECT * REPLACE (fecha::DATE AS fecha) FROM t ORDER BY fecha"
            )
            con.unregister("t")
        con.execute(
            "CREATE TABLE IF NOT EXISTS descargas "
            "(descargado_en VARCHAR, origen VARCHAR, ultimo_mes DATE, n_series INTEGER)"
        )
        con.execute("INSERT INTO descargas VALUES (?, ?, ?, ?)", [ahora, origen, ultimo_mes, n_series])

    for nombre, df in tablas.items():
        if nombre != "eoh_long":
            df.to_csv(PROCESSED / f"{nombre}.csv", index=False)


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

    # Aeropuerto (Eurostat)
    if desde_json:
        js = json.loads((desde_json / EUROSTAT_JSON).read_text(encoding="utf-8"))
    else:
        js = descargar_aeropuerto()
        (RAW / EUROSTAT_JSON).write_text(json.dumps(js), encoding="utf-8")
    aero = aeropuerto_a_tabla(js)
    print(f"  ✓ Eurostat  aeropuerto Alicante-Elche ({aero.fecha.min():%Y-%m} → {aero.fecha.max():%Y-%m})")

    # Tabla para modelar: un mes por fila con INE + aeropuerto + calendario.
    # El aeropuerto va unos meses por detrás del INE: esos meses quedan vacíos.
    dataset = (
        ancho.drop(columns="covid")
        .merge(aero, on="fecha", how="left")
        .merge(cal, on="fecha", how="left")
    )

    avisos = validar(long, ancho) + validar_aeropuerto(aero)
    guardar(
        {"eoh_long": long, "eoh_mensual": ancho, "aeropuerto": aero,
         "calendario": cal, "dataset_mensual": dataset},
        origen=str(desde_json) if desde_json else f"{API} + Eurostat avia_paoa",
        ultimo_mes=ancho.fecha.max().date(),
        n_series=long.codigo.nunique() + len(EUROSTAT_COLS),
    )

    print(f"\n{len(SERIES)} series del INE · {len(ancho)} meses "
          f"({ancho.fecha.min():%Y-%m} → {ancho.fecha.max():%Y-%m})")
    print(f"Aeropuerto: {len(aero)} meses · tabla para modelar: data/processed/dataset_mensual.csv")
    print(f"Guardado en {DB.relative_to(RAIZ)}")
    if avisos:
        print("\nAvisos de validación:")
        for a in avisos:
            print("  ·", a)
    return dataset


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
