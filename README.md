# Previsión de pernoctaciones hoteleras en la Costa Blanca

¿Cuántas pernoctaciones hoteleras tendrán Benidorm y la Costa Blanca en los próximos
1–12 meses, y qué factores lo explican?

> Estado: **fase 1 de 5 (datos)** completada.

## Datos

Encuesta de Ocupación Hotelera (EOH) del INE, descargada con su
[API JSON](https://www.ine.es/dyngs/DAB/index.htm?cid=1099) (sin clave).

| Ámbito | Desde | Series |
|---|---|---|
| Costa Blanca (zona turística) | 1999-01 | viajeros y pernoctaciones (residentes en España / extranjero), establecimientos, plazas, ocupación por plazas y en fin de semana |
| Benidorm (punto turístico) | 2005-01 | viajeros y pernoctaciones (España / extranjero), plazas, ocupación |
| Provincia de Alicante | 1999-01 | viajeros y pernoctaciones (total, España, extranjero) |

Los códigos de cada serie están en `SERIES`, dentro de `src/descarga.py`.

**Huecos conocidos:** mayo y junio de 2020 (hoteles cerrados por el estado de alarma)
en zona y punto turístico, y febrero de 2021 en Benidorm. La provincia no tiene huecos.
Abril de 2020 vale 0, no está vacío.

**Datos provisionales:** los meses del año en curso son provisionales y el INE los revisa
al pasarlos a definitivos. Cada descarga queda registrada en la tabla `descargas`.

## Uso

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/descarga.py          # descarga, valida y guarda
pytest                          # tests
```

Para añadir otro municipio o zona, busca sus códigos y añádelos a `SERIES`:

```bash
python src/descarga.py buscar 2078 Torrevieja   # 2078 = puntos turísticos
python src/descarga.py buscar 2039 "Costa"      # 2039 = zonas turísticas
```

## Salidas

`data/turismo.duckdb`:

| Tabla | Contenido |
|---|---|
| `eoh_mensual` | una fila por mes, una columna por serie, más totales, estancia media y las marcas `provisional` y `covid` |
| `eoh_long` | formato largo: fecha, código INE, ámbito, variable, residencia, valor, tipo de dato |
| `calendario` | días del mes, días de fin de semana, días de Semana Santa (Domingo de Ramos a Lunes de Pascua), mes de Pascua, marca COVID |
| `descargas` | registro de cada descarga |

También en CSV: `data/processed/eoh_mensual.csv` y `data/processed/calendario.csv`.

## Estructura

```
├── data/          raw/ (JSON del INE) · processed/ (CSV) · turismo.duckdb
├── notebooks/     02_eda · 03_baselines · 04_lightgbm   (próximas fases)
├── src/           descarga.py
├── tests/
└── outputs/       métricas y previsiones para el portfolio
```

## Próximas fases

2. Análisis exploratorio: estacionalidad, residentes frente a extranjeros, efecto de Semana Santa, criterio COVID
3. Modelos base: ingenuo estacional, SARIMA / ETS
4. LightGBM con retardos, calendario y variables externas; validación temporal; SHAP
5. App de previsión y sección en el portfolio
