"""Análisis y visualización del dataset Bike Sharing (AE4).

Paquete liviano (solo ``pandas``, ``numpy`` y ``plotly``; ``streamlit`` solo se importa al
llamar a ``DashboardBike.render``) pensado para usarse tanto desde el notebook de AE4 como
desde la app de Streamlit. No depende del paquete ``clases_po``.
"""
from . import datos, metricas
from .dashboard_bike import DashboardBike

__all__ = ["datos", "metricas", "DashboardBike"]
