"""Enriquecimiento de las fechas festivas del dataset con un LLM (NVIDIA Nemotron).

El dataset marca ``holiday = 1`` en ciertas fechas, pero no dice **qué festivo es**
ni si es **federal** o **local de Washington D.C.** Esta clase le pide esa
información al LLM, la valida contra un catálogo oficial y la une al dataset por
fecha. No se inventan filas ni valores de demanda: solo se agregan dos columnas
descriptivas (``nombre_festivo`` y ``tipo_festivo``).

Patrón de caché (pedido en clase): si el archivo JSON existe, **no** se vuelve a
llamar al LLM. Si no existe y hay ``NVIDIA_API_KEY``, se llama a Nemotron, se
valida y se guarda el archivo. Si no existe y no hay clave (o la llamada falla)
se usa el catálogo de respaldo y **no** se escribe el archivo del LLM.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_nvidia import LLMNvidia

FEDERAL = "Federal"
LOCAL_DC = "Local (D.C.)"
NO_FESTIVO = "No festivo"
SIN_IDENTIFICAR = "Festivo sin identificar"

# Catálogo de respaldo: feriados federales de EE. UU. según la OPM (Office of
# Personnel Management), con fechas OBSERVADAS (si el feriado cae sábado se
# observa el viernes; si cae domingo, el lunes), más el Día de la Emancipación
# de Washington D.C. (16 de abril; 2011 cayó sábado y se observó el viernes 15).
# Fuente: https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
# (fecha ISO -> (nombre en español, tipo, nombre oficial en inglés)).
CATALOGO_OPM: dict[str, tuple[str, str, str]] = {
    # 2011 (el Año Nuevo del 1/1/2011, sábado, se observó el 31/12/2010: fuera del dataset)
    "2011-01-17": ("Día de Martin Luther King Jr.", FEDERAL, "Birthday of Martin Luther King, Jr."),
    "2011-02-21": ("Cumpleaños de Washington", FEDERAL, "Washington's Birthday"),
    "2011-04-15": ("Día de la Emancipación de D.C.", LOCAL_DC, "District of Columbia Emancipation Day"),
    "2011-05-30": ("Día de los Caídos", FEDERAL, "Memorial Day"),
    "2011-07-04": ("Día de la Independencia", FEDERAL, "Independence Day"),
    "2011-09-05": ("Día del Trabajo", FEDERAL, "Labor Day"),
    "2011-10-10": ("Día de Colón", FEDERAL, "Columbus Day"),
    "2011-11-11": ("Día de los Veteranos", FEDERAL, "Veterans Day"),
    "2011-11-24": ("Acción de Gracias", FEDERAL, "Thanksgiving Day"),
    "2011-12-26": ("Navidad", FEDERAL, "Christmas Day"),  # 25/12 fue domingo: observado el lunes
    # 2012
    "2012-01-02": ("Año Nuevo", FEDERAL, "New Year's Day"),  # 1/1 fue domingo: observado el lunes
    "2012-01-16": ("Día de Martin Luther King Jr.", FEDERAL, "Birthday of Martin Luther King, Jr."),
    "2012-02-20": ("Cumpleaños de Washington", FEDERAL, "Washington's Birthday"),
    "2012-04-16": ("Día de la Emancipación de D.C.", LOCAL_DC, "District of Columbia Emancipation Day"),  # 16/4 fue lunes
    "2012-05-28": ("Día de los Caídos", FEDERAL, "Memorial Day"),
    "2012-07-04": ("Día de la Independencia", FEDERAL, "Independence Day"),
    "2012-09-03": ("Día del Trabajo", FEDERAL, "Labor Day"),
    "2012-10-08": ("Día de Colón", FEDERAL, "Columbus Day"),
    "2012-11-12": ("Día de los Veteranos", FEDERAL, "Veterans Day"),  # 11/11 fue domingo: observado el lunes
    "2012-11-22": ("Acción de Gracias", FEDERAL, "Thanksgiving Day"),
    "2012-12-25": ("Navidad", FEDERAL, "Christmas Day"),
}

NOMBRES_PERMITIDOS = sorted({valor[0] for valor in CATALOGO_OPM.values()})
TIPOS_PERMITIDOS = [FEDERAL, LOCAL_DC]
DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


class EnriquecedorFestivos:
    """Clasifica las fechas festivas del dataset con un LLM y un catálogo de respaldo.

    Args:
        ruta_cache: Archivo JSON versionado con la respuesta del LLM. Si existe
            (y corresponde a las fechas del dataset) no se llama al LLM.
        llm: Cliente ``LLMNvidia``. Si es ``None`` se crea uno que lee
            ``NVIDIA_API_KEY`` del entorno.
        catalogo: Catálogo de respaldo (por defecto ``CATALOGO_OPM``).

    Después de ``obtener`` quedan disponibles ``fuente`` (``"LLM"`` o
    ``"respaldo"``), ``detalle_fuente`` (texto explicativo) y ``metadatos``.
    """

    def __init__(
        self,
        ruta_cache: str | Path,
        llm: LLMNvidia | None = None,
        catalogo: dict[str, tuple[str, str, str]] | None = None,
    ) -> None:
        self.ruta_cache = Path(ruta_cache)
        self.llm = llm if llm is not None else LLMNvidia()
        self.catalogo = catalogo if catalogo is not None else CATALOGO_OPM
        self.fuente: str | None = None
        self.detalle_fuente: str = ""
        self.metadatos: dict[str, Any] = {}

    # ------------------------------------------------------------------ entrada
    @staticmethod
    def fechas_festivas(
        df: pd.DataFrame,
        col_fecha: str = "fecha",
        col_festivo: str = "tipo_dia",
        valor_festivo: object = "Festivo",
    ) -> list[str]:
        """Fechas únicas (ISO, ordenadas) donde el dataset marca el día como festivo."""
        fechas = df.loc[df[col_festivo] == valor_festivo, col_fecha].dt.strftime("%Y-%m-%d").unique()
        return sorted(fechas.tolist())

    # ------------------------------------------------------------------ flujo principal
    def obtener(self, fechas: list[str]) -> pd.DataFrame:
        """Devuelve la tabla ``fecha, nombre_festivo, tipo_festivo`` y fija ``fuente``."""
        tabla = self._leer_cache(fechas)
        if tabla is not None:
            self.fuente = "LLM"
            self.detalle_fuente = (
                f"archivo versionado {self.ruta_cache.name} "
                f"(modelo {self.metadatos.get('modelo')}, generado el {self.metadatos.get('fecha_generacion')}); "
                "no se llamó al LLM"
            )
            return tabla

        prompt = self._construir_prompt(fechas)
        respaldo = self._tabla_respaldo(fechas)
        datos, fuente = self.llm.pedir_json(
            "Festivos",
            prompt,
            validar=lambda crudo: self._validar_respuesta(crudo, fechas),
            respaldo=respaldo,
        )
        self.fuente = fuente
        if fuente == "LLM":
            tabla = pd.DataFrame(datos)[["fecha", "nombre_festivo", "tipo_festivo"]]
            tabla = tabla.sort_values("fecha").reset_index(drop=True)
            self.metadatos = {
                "modelo": self.llm.modelo,
                "fecha_generacion": datetime.now().astimezone().isoformat(timespec="seconds"),
                "prompt": prompt,
            }
            self._guardar_cache(tabla)
            self.detalle_fuente = f"llamada nueva a {self.llm.modelo}; resultado guardado en {self.ruta_cache.name}"
            return tabla

        self.metadatos = {}
        motivo = "no hay NVIDIA_API_KEY" if not self.llm.activo else "la respuesta del LLM no fue válida"
        self.detalle_fuente = (
            f"catálogo de respaldo OPM ({motivo}); NO se escribió {self.ruta_cache.name}"
        )
        return respaldo

    # ------------------------------------------------------------------ validaciones
    @staticmethod
    def validar_fechas(fechas: list[str], tabla: pd.DataFrame) -> dict[str, list[str]]:
        """Compara el conjunto de fechas de la tabla con el de las fechas esperadas."""
        esperadas, obtenidas = set(fechas), set(tabla["fecha"])
        return {
            "faltantes": sorted(esperadas - obtenidas),
            "sobrantes": sorted(obtenidas - esperadas),
        }

    def comparar_con_catalogo(self, tabla: pd.DataFrame) -> pd.DataFrame:
        """Compara, fecha a fecha, nombre y tipo de la tabla contra el catálogo OPM."""
        filas = []
        for _, fila in tabla.iterrows():
            ref = self.catalogo.get(fila["fecha"])
            nombre_ref, tipo_ref = (ref[0], ref[1]) if ref else ("(fecha fuera del catálogo)", "-")
            filas.append(
                {
                    "fecha": fila["fecha"],
                    "nombre_festivo": fila["nombre_festivo"],
                    "nombre_opm": nombre_ref,
                    "tipo_festivo": fila["tipo_festivo"],
                    "tipo_opm": tipo_ref,
                    "coincide_nombre": fila["nombre_festivo"] == nombre_ref,
                    "coincide_tipo": fila["tipo_festivo"] == tipo_ref,
                }
            )
        return pd.DataFrame(filas)

    def comparar_catalogo_con_dataset(self, fechas: list[str]) -> pd.DataFrame:
        """Cruza las fechas ``holiday = 1`` del dataset con el catálogo OPM (diferencias de ambos lados)."""
        todas = sorted(set(fechas) | set(self.catalogo))
        filas = [
            {
                "fecha": f,
                "marcada_en_dataset": f in fechas,
                "en_catalogo_opm": f in self.catalogo,
                "festivo_opm": self.catalogo[f][0] if f in self.catalogo else "-",
            }
            for f in todas
        ]
        return pd.DataFrame(filas)

    # ------------------------------------------------------------------ unión
    @staticmethod
    def unir(df: pd.DataFrame, tabla: pd.DataFrame, col_fecha: str = "fecha", col_suma: str = "alquileres_total") -> pd.DataFrame:
        """Unión por fecha (left join) comprobando que no se multipliquen filas ni cambie la demanda.

        Las filas sin festivo reciben ``"No festivo"`` en ambas columnas: es una
        categoría real (ausencia de festivo), no un dato faltante, y así los
        agrupamientos no descartan esas filas.
        """
        filas_antes, suma_antes = len(df), int(df[col_suma].sum())
        auxiliar = tabla.copy()
        auxiliar[col_fecha] = pd.to_datetime(auxiliar["fecha"])
        if col_fecha != "fecha":
            auxiliar = auxiliar.drop(columns="fecha")
        resultado = df.merge(auxiliar, on=col_fecha, how="left", validate="many_to_one")
        resultado[["nombre_festivo", "tipo_festivo"]] = resultado[["nombre_festivo", "tipo_festivo"]].fillna(NO_FESTIVO)
        assert len(resultado) == filas_antes, "La unión cambió la cantidad de filas."
        assert int(resultado[col_suma].sum()) == suma_antes, "La unión cambió la suma de alquileres."
        return resultado

    # ------------------------------------------------------------------ internos
    def _tabla_respaldo(self, fechas: list[str]) -> pd.DataFrame:
        filas = []
        for f in fechas:
            ref = self.catalogo.get(f)
            filas.append(
                {
                    "fecha": f,
                    "nombre_festivo": ref[0] if ref else SIN_IDENTIFICAR,
                    "tipo_festivo": ref[1] if ref else SIN_IDENTIFICAR,
                }
            )
        return pd.DataFrame(filas)

    def _construir_prompt(self, fechas: list[str]) -> str:
        lista = "\n".join(
            f"- {f} ({DIAS_ES[date.fromisoformat(f).weekday()]})" for f in fechas
        )
        nombres = "\n".join(f"- {n}" for n in NOMBRES_PERMITIDOS)
        return (
            "Contexto: dataset de bicicletas compartidas (Capital Bikeshare, Washington D.C., 2011-2012). "
            "Las siguientes fechas están marcadas como día festivo. Son fechas OBSERVADAS: si el festivo "
            "cae en fin de semana, se observa el viernes o el lunes más cercano.\n\n"
            f"Fechas:\n{lista}\n\n"
            "Tarea: para CADA fecha indicá qué festivo es y si es federal de EE. UU. o local de "
            "Washington D.C. No agregues ni omitas fechas.\n"
            f"Valores permitidos de nombre_festivo (usá exactamente uno):\n{nombres}\n"
            f"Valores permitidos de tipo_festivo: \"{FEDERAL}\" o \"{LOCAL_DC}\".\n\n"
            "Respondé solo con un arreglo JSON de objetos con las claves "
            "\"fecha\" (AAAA-MM-DD), \"nombre_festivo\" y \"tipo_festivo\"."
        )

    @staticmethod
    def _validar_respuesta(crudo: Any, fechas: list[str]) -> list[dict[str, str]]:
        if isinstance(crudo, dict):
            crudo = next((v for v in crudo.values() if isinstance(v, list)), None)
        if not isinstance(crudo, list):
            raise ValueError("Se esperaba una lista JSON")
        limpio = []
        for item in crudo:
            fecha = str(item["fecha"]).strip()
            nombre = str(item["nombre_festivo"]).strip()
            tipo = str(item["tipo_festivo"]).strip()
            if nombre not in NOMBRES_PERMITIDOS:
                raise ValueError(f"nombre_festivo fuera del vocabulario: {nombre}")
            if tipo not in TIPOS_PERMITIDOS:
                raise ValueError(f"tipo_festivo fuera del vocabulario: {tipo}")
            limpio.append({"fecha": fecha, "nombre_festivo": nombre, "tipo_festivo": tipo})
        obtenidas = [item["fecha"] for item in limpio]
        if len(obtenidas) != len(set(obtenidas)):
            raise ValueError("Hay fechas repetidas")
        if set(obtenidas) != set(fechas):
            raise ValueError(
                f"Las fechas no coinciden: faltan {sorted(set(fechas) - set(obtenidas))}, "
                f"sobran {sorted(set(obtenidas) - set(fechas))}"
            )
        return limpio

    def _leer_cache(self, fechas: list[str]) -> pd.DataFrame | None:
        if not self.ruta_cache.exists():
            return None
        try:
            contenido = json.loads(self.ruta_cache.read_text(encoding="utf-8"))
            tabla = pd.DataFrame(self._validar_respuesta(contenido["festivos"], fechas))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            print(
                f"El archivo {self.ruta_cache.name} existe pero no es válido para estas fechas "
                f"({type(error).__name__}); se ignora."
            )
            return None
        self.metadatos = contenido.get("metadatos", {})
        return tabla.sort_values("fecha").reset_index(drop=True)

    def _guardar_cache(self, tabla: pd.DataFrame) -> None:
        self.ruta_cache.parent.mkdir(parents=True, exist_ok=True)
        contenido = {"metadatos": self.metadatos, "festivos": tabla.to_dict(orient="records")}
        self.ruta_cache.write_text(json.dumps(contenido, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
