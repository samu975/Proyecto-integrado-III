"""Métricas de demanda del dataset Bike Sharing (preguntas P1 a P6 de AE3).

Cada función recibe el ``DataFrame`` preparado (o una versión ya filtrada por
año, estación o tipo de día) y devuelve la tabla equivalente a la de AE3.
Las unidades son: alquileres (conteo de viajes), alquileres por hora
(promedio por hora observada) y porcentajes (0 a 100).

Nota sobre filtros: el ``pct_del_total_alquileres`` se calcula sobre el total
del ``DataFrame`` recibido, así que al filtrar es el porcentaje dentro de la
selección.
"""
from __future__ import annotations

import pandas as pd


def resumen_demanda(df: pd.DataFrame, por: str | list[str]) -> pd.DataFrame:
    """Resumen de la demanda por grupo (una fila por grupo).

    Métricas, calculadas sobre ``alquileres_total`` de cada hora observada:

    - ``horas_observadas``: cantidad de filas (horas) del grupo [horas].
    - ``total_alquileres``: suma de alquileres del grupo [alquileres].
    - ``promedio_alquileres_por_hora``: media por hora observada [alquileres/hora], 2 decimales.
    - ``mediana_alquileres_por_hora``: mediana por hora observada [alquileres/hora], 1 decimal.
    - ``pct_del_total_alquileres``: total del grupo / total del ``df`` * 100 [%], 2 decimales.
    """
    g = df.groupby(por, observed=True)
    tabla = g.agg(
        horas_observadas=("alquileres_total", "size"),
        total_alquileres=("alquileres_total", "sum"),
        promedio_alquileres_por_hora=("alquileres_total", "mean"),
        mediana_alquileres_por_hora=("alquileres_total", "median"),
    ).reset_index()
    tabla["pct_del_total_alquileres"] = tabla["total_alquileres"] / df["alquileres_total"].sum() * 100
    return tabla.round({"promedio_alquileres_por_hora": 2, "mediana_alquileres_por_hora": 1, "pct_del_total_alquileres": 2})


def participacion(df: pd.DataFrame, por: str | list[str]) -> pd.DataFrame:
    """Participación (%) de usuarios casuales y registrados dentro de cada grupo.

    ``pct_casuales`` = casuales / alquileres_total * 100 [%]; ``pct_registrados``
    análogo. Ambos con 2 decimales, y suman 100 % en cada grupo.
    """
    g = df.groupby(por, observed=True)[["usuarios_casuales", "usuarios_registrados", "alquileres_total"]].sum().reset_index()
    g["pct_casuales"] = g["usuarios_casuales"] / g["alquileres_total"] * 100
    g["pct_registrados"] = g["usuarios_registrados"] / g["alquileres_total"] * 100
    return g.round({"pct_casuales": 2, "pct_registrados": 2})


# ------------------------------------------------------------------ P1
def p1_demanda_por_franja_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """P1: demanda por franja horaria.

    Métrica: resumen de demanda (total, promedio, mediana y % del total) por
    ``franja_horaria``. Unidades: alquileres, alquileres/hora y %.
    """
    return resumen_demanda(df, "franja_horaria")


def p1_demanda_por_hora_del_dia(df: pd.DataFrame) -> pd.DataFrame:
    """P1: demanda por hora del día (0 a 23).

    Métrica: resumen de demanda por (``hora``, ``franja_horaria``). Unidades:
    alquileres, alquileres/hora y %.
    """
    return resumen_demanda(df, ["hora", "franja_horaria"])


# ------------------------------------------------------------------ P2
def p2_demanda_por_tipo_de_dia(df: pd.DataFrame) -> pd.DataFrame:
    """P2: demanda por tipo de día (laboral, fin de semana, festivo).

    Métrica: resumen de demanda por ``tipo_dia``. Unidades: alquileres,
    alquileres/hora y %.
    """
    return resumen_demanda(df, "tipo_dia")


def p2_demanda_por_tipo_de_dia_y_franja(df: pd.DataFrame) -> pd.DataFrame:
    """P2: demanda por tipo de día y franja horaria.

    Métrica: resumen de demanda por (``tipo_dia``, ``franja_horaria``).
    Unidades: alquileres, alquileres/hora y %.
    """
    return resumen_demanda(df, ["tipo_dia", "franja_horaria"])


def p2_demanda_por_festivo(df: pd.DataFrame) -> pd.DataFrame:
    """P2: demanda en cada festivo del período.

    Considera solo las horas con ``nombre_festivo`` distinto de "No festivo".
    Columnas: ``fechas`` (días distintos), ``horas_observadas`` [horas],
    ``total_alquileres`` [alquileres], ``promedio_alquileres_por_hora``
    [alquileres/hora] y ``pct_casuales`` (casuales / total * 100 [%]).
    Ordenada de mayor a menor promedio por hora.
    """
    festivos = df[df["nombre_festivo"] != "No festivo"]
    tabla = (
        festivos.groupby(["nombre_festivo", "tipo_festivo"])
        .agg(
            fechas=("fecha", "nunique"),
            horas_observadas=("alquileres_total", "size"),
            total_alquileres=("alquileres_total", "sum"),
            promedio_alquileres_por_hora=("alquileres_total", "mean"),
            total_casuales=("usuarios_casuales", "sum"),
        )
        .reset_index()
    )
    tabla["pct_casuales"] = tabla["total_casuales"] / tabla["total_alquileres"] * 100
    return (
        tabla.drop(columns="total_casuales")
        .round({"promedio_alquileres_por_hora": 2, "pct_casuales": 2})
        .sort_values("promedio_alquileres_por_hora", ascending=False)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------ P3
def p3_demanda_por_estacion(df: pd.DataFrame) -> pd.DataFrame:
    """P3: demanda por estación del año.

    Métrica: resumen de demanda por ``estacion``. Unidades: alquileres,
    alquileres/hora y %.
    """
    return resumen_demanda(df, "estacion")


def p3_demanda_por_mes(df: pd.DataFrame) -> pd.DataFrame:
    """P3: demanda por mes calendario.

    Métrica: resumen de demanda por (``mes``, ``nombre_mes``). Unidades:
    alquileres, alquileres/hora y %.
    """
    return resumen_demanda(df, ["mes", "nombre_mes"])


# ------------------------------------------------------------------ P4
def p4_demanda_por_situacion_climatica(df: pd.DataFrame) -> pd.DataFrame:
    """P4: demanda por situación climática.

    Métrica: resumen de demanda por ``clima``. Unidades: alquileres,
    alquileres/hora y %.
    """
    return resumen_demanda(df, "clima")


def p4_demanda_por_rango_de_temperatura(df: pd.DataFrame) -> pd.DataFrame:
    """P4: demanda por rango de temperatura (°C).

    Métrica: resumen de demanda por ``rango_temperatura``. Unidades:
    alquileres, alquileres/hora y %.
    """
    return resumen_demanda(df, "rango_temperatura")


# ------------------------------------------------------------------ P5
def p5_participacion_casuales_y_registrados(df: pd.DataFrame) -> pd.DataFrame:
    """P5: participación (%) de casuales y registrados.

    Se calcula por tipo de día, franja horaria, estación, tipo de festivo y
    para todo el período. Columnas: ``dimension``, ``categoria``, totales de
    usuarios [alquileres] y ``pct_casuales`` / ``pct_registrados`` [%].
    """
    partes = []
    for dimension, columna in [
        ("Tipo de día", "tipo_dia"),
        ("Franja horaria", "franja_horaria"),
        ("Estación", "estacion"),
        ("Tipo de festivo", "tipo_festivo"),
    ]:
        parte = participacion(df, columna).rename(columns={columna: "categoria"})
        parte.insert(0, "dimension", dimension)
        partes.append(parte)
    total = participacion(df.assign(categoria="Todo el período"), "categoria")
    total.insert(0, "dimension", "Total")
    partes.append(total)
    p5 = pd.concat(partes, ignore_index=True)
    p5["categoria"] = p5["categoria"].astype(str)
    return p5


# ------------------------------------------------------------------ P6
def p6_variacion_2011_2012(df: pd.DataFrame) -> pd.DataFrame:
    """P6: variación de la demanda entre 2011 y 2012.

    Una fila por tipo de usuario (Casuales, Registrados, Total). Columnas:
    totales anuales [alquileres], ``variacion_total_pct`` = (2012 - 2011) /
    2011 * 100 [%], horas observadas por año [horas], promedios por hora
    [alquileres/hora] y ``variacion_promedio_pct`` [%] (compara por hora
    observada, porque 2012 tiene más horas registradas que 2011).
    """
    filas = []
    for etiqueta, columna in [
        ("Casuales", "usuarios_casuales"),
        ("Registrados", "usuarios_registrados"),
        ("Total", "alquileres_total"),
    ]:
        t11 = int(df.loc[df["anio"] == 2011, columna].sum())
        t12 = int(df.loc[df["anio"] == 2012, columna].sum())
        h11 = int((df["anio"] == 2011).sum())
        h12 = int((df["anio"] == 2012).sum())
        filas.append({
            "tipo_usuario": etiqueta,
            "total_2011": t11, "total_2012": t12,
            "variacion_total_pct": (t12 - t11) / t11 * 100,
            "horas_observadas_2011": h11, "horas_observadas_2012": h12,
            "promedio_por_hora_2011": t11 / h11, "promedio_por_hora_2012": t12 / h12,
            "variacion_promedio_pct": ((t12 / h12) - (t11 / h11)) / (t11 / h11) * 100,
        })
    return pd.DataFrame(filas).round({
        "variacion_total_pct": 2, "promedio_por_hora_2011": 2,
        "promedio_por_hora_2012": 2, "variacion_promedio_pct": 2,
    })


# Nombre de tabla (igual al CSV de AE3) -> función que la recalcula.
FUNCIONES = {
    "P1_demanda_por_franja_horaria": p1_demanda_por_franja_horaria,
    "P1_demanda_por_hora_del_dia": p1_demanda_por_hora_del_dia,
    "P2_demanda_por_tipo_de_dia": p2_demanda_por_tipo_de_dia,
    "P2_demanda_por_tipo_de_dia_y_franja": p2_demanda_por_tipo_de_dia_y_franja,
    "P2_demanda_por_festivo": p2_demanda_por_festivo,
    "P3_demanda_por_estacion": p3_demanda_por_estacion,
    "P3_demanda_por_mes": p3_demanda_por_mes,
    "P4_demanda_por_situacion_climatica": p4_demanda_por_situacion_climatica,
    "P4_demanda_por_rango_de_temperatura": p4_demanda_por_rango_de_temperatura,
    "P5_participacion_casuales_y_registrados": p5_participacion_casuales_y_registrados,
    "P6_variacion_2011_2012": p6_variacion_2011_2012,
}
