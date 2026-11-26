"""App web de la demanda de bicicletas compartidas (AE4).

Ejecutar en local (desde la raíz del repositorio):
    streamlit run AE4/app/app.py

En Streamlit Community Cloud: Main file path = AE4/app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]            # raíz del repositorio
CARPETA_PAQUETE = RAIZ / "src" / "analitica_bike"
sys.path.insert(0, str(CARPETA_PAQUETE))               # importa el módulo suelto, sin cargar clases_po
                                                       # (así la app no necesita datasets ni openai)
from dashboard_bike import DashboardBike               # noqa: E402

RUTA_DATOS = RAIZ / "AE3" / "datos" / "bike_sharing_preparado.csv"


@st.cache_resource(show_spinner="Cargando el dataset de Bike Sharing…")
def cargar() -> DashboardBike:
    return DashboardBike.desde_csv(RUTA_DATOS)


cargar().render()
