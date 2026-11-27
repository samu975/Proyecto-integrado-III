"""Carga del dataset preparado (AE3) y de sus tablas de resultados.

Las rutas se resuelven a partir de la ubicación de este archivo, nunca con
rutas absolutas, para que funcionen igual en el notebook, en Docker, en CI
y en Streamlit Cloud.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Órdenes de categorías copiados de src/clases_po/transformacion_bike.py (AE3).
# No se importan de ahí porque clases_po/__init__.py exige ``datasets`` y ``openai``.
ESTACIONES = ["Invierno", "Primavera", "Verano", "Otoño"]
CLIMAS = [
    "Despejado o pocas nubes",
    "Niebla con nubes",
    "Lluvia o nieve ligera",
    "Lluvia fuerte, tormenta o nieve con niebla",
]
ORDEN_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
ORDEN_FRANJAS = [
    "Madrugada (0-5 h)",
    "Pico de la mañana (6-9 h)",
    "Mediodía (10-15 h)",
    "Pico de la tarde (16-19 h)",
    "Noche (20-23 h)",
]
TIPOS_DIA = ["Laboral", "Fin de semana", "Festivo"]
RANGOS_TEMP = ["Frío (< 10 °C)", "Fresco (10 a < 20 °C)", "Templado (20 a < 30 °C)", "Caluroso (≥ 30 °C)"]

# Columna -> categorías ordenadas (mismo criterio que AE3).
ORDENES = {
    "franja_horaria": ORDEN_FRANJAS,
    "tipo_dia": TIPOS_DIA,
    "estacion": ESTACIONES,
    "clima": CLIMAS,
    "rango_temperatura": RANGOS_TEMP,
    "nombre_mes": MESES,
    "dia_semana": ORDEN_DIAS,
}

TABLAS = [
    "P1_demanda_por_franja_horaria",
    "P1_demanda_por_hora_del_dia",
    "P2_demanda_por_tipo_de_dia",
    "P2_demanda_por_tipo_de_dia_y_franja",
    "P2_demanda_por_festivo",
    "P3_demanda_por_estacion",
    "P3_demanda_por_mes",
    "P4_demanda_por_situacion_climatica",
    "P4_demanda_por_rango_de_temperatura",
    "P5_participacion_casuales_y_registrados",
    "P6_variacion_2011_2012",
]


def raiz_repo() -> Path:
    """Devuelve la raíz del repositorio (la carpeta que contiene ``src/analitica_bike``)."""
    return Path(__file__).resolve().parents[2]


def ruta_dataset() -> Path:
    """Ruta del CSV preparado en AE3 (``AE3/datos/bike_sharing_preparado.csv``)."""
    return raiz_repo() / "AE3" / "datos" / "bike_sharing_preparado.csv"


def ruta_tablas() -> Path:
    """Carpeta con las tablas de resultados de AE3."""
    return raiz_repo() / "AE3" / "resultados" / "tablas"


def restaurar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a categóricas ordenadas las columnas conocidas que existan en ``df``.

    Devuelve una copia; no modifica el ``DataFrame`` recibido.
    """
    salida = df.copy()
    for columna, orden in ORDENES.items():
        if columna in salida.columns:
            salida[columna] = pd.Categorical(salida[columna], categories=orden, ordered=True)
    return salida


def cargar_dataset() -> pd.DataFrame:
    """Lee el dataset preparado (17.379 filas) con fechas y categóricas ordenadas restauradas."""
    df = pd.read_csv(ruta_dataset(), parse_dates=["fecha", "fecha_hora"])
    return restaurar_categorias(df)


def cargar_tabla(nombre: str) -> pd.DataFrame:
    """Lee una tabla de resultados de AE3 por su nombre (sin ``.csv``), con categóricas restauradas."""
    if nombre not in TABLAS:
        raise KeyError(f"Tabla desconocida: {nombre}. Opciones: {TABLAS}")
    return restaurar_categorias(pd.read_csv(ruta_tablas() / f"{nombre}.csv"))


def cargar_tablas() -> dict[str, pd.DataFrame]:
    """Lee las 11 tablas de resultados de AE3 en un diccionario ``nombre -> DataFrame``."""
    return {nombre: cargar_tabla(nombre) for nombre in TABLAS}
