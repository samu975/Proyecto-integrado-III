"""Generación de reportes de DataFrames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class Reportes:
    """Imprime y guarda un resumen legible de un DataFrame."""

    def generar_reporte(
        self,
        df: pd.DataFrame,
        encabezado: dict[str, object] | None = None,
    ) -> str:
        """Genera el contenido textual del reporte.

        Args:
            df: DataFrame a resumir.
            encabezado: Datos opcionales (por ejemplo dataset, split o
                revisión) que se imprimen como líneas ``clave: valor`` justo
                después del título y antes de ``Filas``.
        """
        self._validar_dataframe(df)
        numericas = df.select_dtypes(include="number")
        lineas = [
            "REPORTE DEL DATAFRAME",
            "=" * 60,
            *[f"{clave}: {valor}" for clave, valor in (encabezado or {}).items()],
            f"Filas: {len(df)}",
            f"Columnas: {len(df.columns)}",
            f"Duplicados: {int(df.duplicated().sum())}",
            "",
            "Columnas y tipos:",
            *[f"- {columna}: {tipo}" for columna, tipo in df.dtypes.items()],
            "",
            "Valores nulos por columna:",
            *[f"- {columna}: {cantidad}" for columna, cantidad in df.isna().sum().items()],
        ]

        if not numericas.empty:
            lineas += [
                "",
                "Estadísticas numéricas:",
                numericas.describe().round(3).to_string(),
            ]

        return "\n".join(lineas) + "\n"

    def imprimir_reporte(
        self,
        df: pd.DataFrame,
        encabezado: dict[str, object] | None = None,
    ) -> None:
        """Imprime en la consola el reporte del DataFrame."""
        print(self.generar_reporte(df, encabezado), end="")

    def guardar_reporte(
        self,
        df: pd.DataFrame,
        ruta: str | Path,
        encabezado: dict[str, object] | None = None,
    ) -> Path:
        """Guarda el reporte en un archivo de texto y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generar_reporte(df, encabezado), encoding="utf-8")
        return path

    @staticmethod
    def _validar_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")