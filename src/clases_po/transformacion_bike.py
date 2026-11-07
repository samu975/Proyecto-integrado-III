"""Transformación, validación y enriquecimiento del dataset Bike Sharing (por hora).

Cada paso es un método independiente para que el notebook pueda ejecutarlo,
mostrarlo y explicarlo por separado. Ningún método modifica el ``DataFrame``
recibido: todos devuelven una copia.

Fuente de los datos: UCI Bike Sharing Dataset (id 275), Fanaee-T y Gama, CC BY 4.0.
"""
from __future__ import annotations

import pandas as pd

ESTACIONES = {1: "Invierno", 2: "Primavera", 3: "Verano", 4: "Otoño"}
CLIMAS = {
    1: "Despejado o pocas nubes",
    2: "Niebla con nubes",
    3: "Lluvia o nieve ligera",
    4: "Lluvia fuerte, tormenta o nieve con niebla",
}
# En el dataset original 0 = domingo ... 6 = sábado (verificado contra la fecha).
DIAS_SEMANA = {
    0: "Domingo", 1: "Lunes", 2: "Martes", 3: "Miércoles", 4: "Jueves", 5: "Viernes", 6: "Sábado",
}
ORDEN_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
# Franjas definidas con el perfil horario observado (picos en la hora 8 y en las 17-18).
FRANJAS = [
    ("Madrugada (0-5 h)", 0, 5),
    ("Pico de la mañana (6-9 h)", 6, 9),
    ("Mediodía (10-15 h)", 10, 15),
    ("Pico de la tarde (16-19 h)", 16, 19),
    ("Noche (20-23 h)", 20, 23),
]
ORDEN_FRANJAS = [f[0] for f in FRANJAS]
TIPOS_DIA = ["Laboral", "Fin de semana", "Festivo"]
RANGOS_TEMP = ["Frío (< 10 °C)", "Fresco (10 a < 20 °C)", "Templado (20 a < 30 °C)", "Caluroso (≥ 30 °C)"]

# Desnormalización documentada por UCI.
TEMP_MIN, TEMP_MAX = -8, 39        # temp  = (t - t_min) / (t_max - t_min), en °C
ATEMP_MIN, ATEMP_MAX = -16, 50     # atemp = (t - t_min) / (t_max - t_min), en °C
HUM_MAX = 100                      # hum   = humedad / 100
VIENTO_MAX = 67                    # windspeed = viento / 67 (unidad original NO documentada)

RENOMBRAR_FINAL = {
    "hr": "hora",
    "mnth": "mes",
    "casual": "usuarios_casuales",
    "registered": "usuarios_registrados",
    "cnt": "alquileres_total",
}
COLUMNAS_FINALES = [
    "instant", "fecha", "fecha_hora", "anio", "mes", "nombre_mes", "dia_semana", "hora",
    "franja_horaria", "tipo_dia", "estacion", "clima", "temp_c", "sensacion_c", "rango_temperatura",
    "humedad_pct", "viento_original", "usuarios_casuales", "usuarios_registrados",
    "alquileres_total", "pct_casual",
]


def _categoria(serie: pd.Series, orden: list[str]) -> pd.Series:
    return pd.Categorical(serie, categories=orden, ordered=True)


class TransformacionBike:
    """Pasos de transformación y enriquecimiento del dataset de bicicletas compartidas."""

    # ------------------------------------------------------------------ tipos
    @staticmethod
    def convertir_tipos(df: pd.DataFrame) -> pd.DataFrame:
        """``dteday`` (texto) -> ``fecha`` (datetime) y crea ``fecha_hora`` = fecha + hora."""
        salida = df.copy()
        salida["dteday"] = pd.to_datetime(salida["dteday"], format="%Y-%m-%d", errors="raise")
        salida = salida.rename(columns={"dteday": "fecha"})
        salida["fecha_hora"] = salida["fecha"] + pd.to_timedelta(salida["hr"], unit="h")
        return salida

    # ------------------------------------------------------------------ controles
    @staticmethod
    def controles_calidad(df: pd.DataFrame) -> pd.DataFrame:
        """Ejecuta los controles de calidad sobre el dataset con fecha ya convertida.

        Devuelve una fila por control con los registros afectados. Un control sin
        problemas se informa igual, con 0 registros.
        """
        dia_esperado = (df["fecha"].dt.dayofweek + 1) % 7  # lunes=0 en pandas -> domingo=0 en el dataset
        laboral_esperado = (df["weekday"].isin([1, 2, 3, 4, 5]) & (df["holiday"] == 0)).astype(int)
        controles = [
            ("Valores nulos (celdas)", "Cualquier celda vacía en cualquier columna", int(df.isna().sum().sum())),
            ("Filas completamente duplicadas", "Dos filas idénticas en todas las columnas", int(df.duplicated().sum())),
            ("Identificador `instant` repetido", "El mismo `instant` en más de una fila", int(df["instant"].duplicated().sum())),
            ("Fecha y hora repetidas", "Más de una fila con la misma (fecha, hr)", int(df.duplicated(["fecha", "hr"]).sum())),
            ("cnt distinto de casual + registered", "cnt != casual + registered", int((df["cnt"] != df["casual"] + df["registered"]).sum())),
            ("Hora fuera de 0 a 23", "hr < 0 o hr > 23", int((~df["hr"].between(0, 23)).sum())),
            ("Día de la semana incoherente con la fecha", "weekday != día de la semana calculado desde fecha (0 = domingo)", int((df["weekday"] != dia_esperado).sum())),
            ("Día laboral incoherente", "workingday != (lunes a viernes y no festivo)", int((df["workingday"] != laboral_esperado).sum())),
            ("Año incoherente con la fecha", "yr != año de la fecha - 2011", int((df["yr"] != df["fecha"].dt.year - 2011).sum())),
            ("Mes incoherente con la fecha", "mnth != mes de la fecha", int((df["mnth"] != df["fecha"].dt.month).sum())),
            ("Estación fuera de 1 a 4", "season no está en {1, 2, 3, 4}", int((~df["season"].isin(ESTACIONES)).sum())),
            ("Clima fuera de 1 a 4", "weathersit no está en {1, 2, 3, 4}", int((~df["weathersit"].isin(CLIMAS)).sum())),
            ("temp fuera de [0, 1]", "temp < 0 o temp > 1", int((~df["temp"].between(0, 1)).sum())),
            ("atemp fuera de [0, 1]", "atemp < 0 o atemp > 1", int((~df["atemp"].between(0, 1)).sum())),
            ("hum fuera de [0, 1]", "hum < 0 o hum > 1", int((~df["hum"].between(0, 1)).sum())),
            ("windspeed fuera de [0, 1]", "windspeed < 0 o windspeed > 1", int((~df["windspeed"].between(0, 1)).sum())),
            ("Conteos negativos", "casual, registered o cnt < 0", int((df[["casual", "registered", "cnt"]] < 0).any(axis=1).sum())),
            ("Horas con cnt = 0", "cnt == 0 (afecta el cálculo de pct_casual)", int((df["cnt"] == 0).sum())),
            ("Humedad igual a 0 (valor improbable)", "hum == 0", int((df["hum"] == 0).sum())),
            ("Viento igual a 0", "windspeed == 0 (posible dato no registrado)", int((df["windspeed"] == 0).sum())),
        ]
        tabla = pd.DataFrame(controles, columns=["control", "regla", "registros_afectados"])
        informativos = {"Humedad igual a 0 (valor improbable)", "Viento igual a 0"}
        tabla["resultado"] = [
            ("Sin problemas" if n == 0 else ("Hallazgo informativo" if c in informativos else "Problema detectado"))
            for c, n in zip(tabla["control"], tabla["registros_afectados"])
        ]
        return tabla

    @staticmethod
    def horas_faltantes(df: pd.DataFrame) -> pd.DataFrame:
        """Horas del calendario (entre la primera y la última fecha) que no tienen fila.

        Devuelve un ``DataFrame`` con una fila por hora faltante (``fecha_hora``, ``fecha``, ``hora``).
        """
        calendario = pd.date_range(df["fecha"].min(), df["fecha"].max() + pd.Timedelta(hours=23), freq="h")
        faltantes = calendario.difference(pd.DatetimeIndex(df["fecha_hora"]))
        resultado = pd.DataFrame({"fecha_hora": faltantes})
        resultado["fecha"] = resultado["fecha_hora"].dt.normalize()
        resultado["hora"] = resultado["fecha_hora"].dt.hour
        return resultado

    @staticmethod
    def horas_faltantes_por_fecha(faltantes: pd.DataFrame) -> pd.DataFrame:
        """Resumen por fecha: cuántas horas faltan y cuáles."""
        agrupado = faltantes.groupby("fecha")["hora"]
        return pd.DataFrame(
            {
                "horas_faltantes": agrupado.size(),
                "horas": agrupado.apply(lambda h: ", ".join(str(x) for x in h)),
            }
        ).reset_index()

    # ------------------------------------------------------------------ decodificación
    @staticmethod
    def decodificar(df: pd.DataFrame) -> pd.DataFrame:
        """Reemplaza los códigos numéricos por etiquetas en español (categorías ordenadas)."""
        s = df.copy()
        s["estacion"] = _categoria(s["season"].map(ESTACIONES), list(ESTACIONES.values()))
        s["clima"] = _categoria(s["weathersit"].map(CLIMAS), list(CLIMAS.values()))
        s["dia_semana"] = _categoria(s["weekday"].map(DIAS_SEMANA), ORDEN_DIAS)
        s["anio"] = (s["yr"] + 2011).astype("int64")
        s["festivo"] = _categoria(s["holiday"].map({0: "No festivo", 1: "Festivo"}), ["No festivo", "Festivo"])
        s["dia_laboral"] = _categoria(s["workingday"].map({0: "No laboral", 1: "Laboral"}), ["No laboral", "Laboral"])
        return s

    @staticmethod
    def desnormalizar(df: pd.DataFrame) -> pd.DataFrame:
        """Vuelve temperatura, sensación térmica, humedad y viento a sus unidades originales."""
        s = df.copy()
        s["temp_c"] = (s["temp"] * (TEMP_MAX - TEMP_MIN) + TEMP_MIN).round(2)
        s["sensacion_c"] = (s["atemp"] * (ATEMP_MAX - ATEMP_MIN) + ATEMP_MIN).round(2)
        s["humedad_pct"] = (s["hum"] * HUM_MAX).round(1)
        s["viento_original"] = (s["windspeed"] * VIENTO_MAX).round(2)
        return s

    # ------------------------------------------------------------------ enriquecimiento
    @staticmethod
    def franja_de_hora(hora: int) -> str:
        for nombre, desde, hasta in FRANJAS:
            if desde <= hora <= hasta:
                return nombre
        raise ValueError(f"Hora fuera de rango: {hora}")

    @staticmethod
    def tipo_de_dia(df: pd.DataFrame) -> pd.Series:
        """Festivo si ``festivo`` = Festivo; fin de semana si es sábado o domingo; laboral en otro caso."""
        es_finde = df["dia_semana"].isin(["Sábado", "Domingo"])
        tipo = pd.Series("Laboral", index=df.index)
        tipo[es_finde] = "Fin de semana"
        tipo[df["festivo"] == "Festivo"] = "Festivo"
        return pd.Series(_categoria(tipo, TIPOS_DIA), index=df.index)

    def enriquecer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega mes, franja horaria, tipo de día, rango de temperatura y % de casuales."""
        s = df.copy()
        s["nombre_mes"] = _categoria(s["mnth"].map(lambda m: MESES[m - 1]), MESES)
        s["franja_horaria"] = _categoria(s["hr"].map(self.franja_de_hora), ORDEN_FRANJAS)
        s["tipo_dia"] = self.tipo_de_dia(s)
        s["rango_temperatura"] = pd.cut(
            s["temp_c"], bins=[-float("inf"), 10, 20, 30, float("inf")], labels=RANGOS_TEMP, right=False, ordered=True
        )
        s["pct_casual"] = (s["casual"] / s["cnt"] * 100).round(2)
        return s

    # ------------------------------------------------------------------ selección
    @staticmethod
    def seleccionar_columnas(df: pd.DataFrame) -> pd.DataFrame:
        """Conserva las columnas pertinentes y renombra las originales que se mantienen."""
        s = df.rename(columns=RENOMBRAR_FINAL)
        return s[COLUMNAS_FINALES].copy()

    # ------------------------------------------------------------------ antes / después
    @staticmethod
    def comparar_resumen(antes: pd.DataFrame, despues: pd.DataFrame) -> pd.DataFrame:
        """Filas, columnas, nulos y tipos de columnas, antes y después."""

        def tipos(df: pd.DataFrame) -> dict[str, int]:
            return {
                "numéricas": int(df.select_dtypes(include="number").shape[1]),
                "fecha": int(df.select_dtypes(include="datetime").shape[1]),
                "categóricas ordenadas": int(sum(isinstance(t, pd.CategoricalDtype) for t in df.dtypes)),
                "texto": int(df.select_dtypes(include=["object", "string"]).shape[1]),
            }

        ta, td = tipos(antes), tipos(despues)
        filas = [
            ("Filas", len(antes), len(despues)),
            ("Columnas", antes.shape[1], despues.shape[1]),
            ("Celdas nulas", int(antes.isna().sum().sum()), int(despues.isna().sum().sum())),
            *[(f"Columnas {k}", ta[k], td[k]) for k in ta],
        ]
        return pd.DataFrame(filas, columns=["métrica", "antes", "después"])

    @staticmethod
    def comparar_columnas(antes: pd.DataFrame, despues: pd.DataFrame) -> pd.DataFrame:
        """Tipo y nulos de cada columna en ambas versiones (``-`` si la columna no existe)."""
        nombres = list(dict.fromkeys([*antes.columns, *despues.columns]))
        filas = []
        for n in nombres:
            filas.append(
                {
                    "columna": n,
                    "tipo_antes": str(antes[n].dtype) if n in antes else "-",
                    "nulos_antes": int(antes[n].isna().sum()) if n in antes else "-",
                    "tipo_después": str(despues[n].dtype) if n in despues else "-",
                    "nulos_después": int(despues[n].isna().sum()) if n in despues else "-",
                }
            )
        return pd.DataFrame(filas)
