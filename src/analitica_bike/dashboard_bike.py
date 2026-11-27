"""Dashboard web de la demanda de bicicletas compartidas (Bike Sharing) con Streamlit + Plotly.

La clase :class:`DashboardBike` toma el dataset preparado en AE3 y:

1. calcula los indicadores reutilizando ``metricas.py`` (las mismas tablas de AE3);
2. arma cada visual como una figura de Plotly (un método por figura);
3. redacta la respuesta a cada pregunta (P1 a P6) con cifras calculadas, nunca escritas a mano;
4. compone todo en una app de Streamlit con filtros (``render()``).

Uso en Jupyter::

    dash = DashboardBike.desde_csv()
    dash.kpis()
    dash.figura_hora()
    dash.filtrar(anios=[2012]).kpis()

Uso en la app (``AE4/app/app.py``)::

    DashboardBike.desde_csv(RUTA_DATOS).render()

Qué se mide: **alquileres** (demanda). El dataset no tiene precios ni costos, así que no se
habla de ingresos ni de rentabilidad. Todo es descriptivo: asociaciones, no causas.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:  # dentro del paquete analitica_bike
    from . import datos as _datos
    from . import metricas as _m
except ImportError:  # la app agrega esta carpeta al path y lo importa como módulo suelto
    import datos as _datos
    import metricas as _m

ESTACIONES = _datos.ESTACIONES
TIPOS_DIA = _datos.TIPOS_DIA
ORDEN_DIAS = _datos.ORDEN_DIAS
PH = "promedio_alquileres_por_hora"

# Una categoría con menos horas observadas que este umbral se marca como "pocas horas":
# se muestra, pero no se interpreta.
MIN_HORAS = 100
# Una hora es "valle" si su promedio es menor a esta fracción del promedio de la hora pico.
UMBRAL_VALLE = 0.10
# Si el mes más alto supera al más bajo en menos de este %, los meses son parejos y no se recomienda diferenciarlos.
UMBRAL_MESES_PAREJOS = 10


# ---------------------------------------------------------------------------- formato
def numero(x: float, decimales: int = 0) -> str:
    """12.345,67 · punto de miles y coma decimal."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "s/d"
    return f"{x:,.{decimales}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x: float, decimales: int = 1) -> str:
    """El valor ya viene en porcentaje (0 a 100): 33,4 %."""
    return f"{numero(x, decimales)} %"


def lista(items) -> str:
    """a, b y c."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def plural(n, uno: str, varios: str | None = None) -> str:
    """1 hora / 3 horas (con el número ya formateado)."""
    return f"{numero(n)} {uno if int(n) == 1 else (varios or uno + 's')}"


def horas(n) -> str:
    return plural(n, "hora")


def fechas(n) -> str:
    return plural(n, "fecha")


def alquileres(n) -> str:
    return plural(n, "alquiler", "alquileres")


def horas_obs(n) -> str:
    """1 hora observada / 3 horas observadas."""
    return f"{numero(n)} " + ("hora observada" if int(n) == 1 else "horas observadas")


def signed(x: float, decimales: int = 1) -> str:
    """Variación con signo explícito: +64,9 %, −0,2 %."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "s/d"
    r = round(float(x), decimales)
    return ("+" if r > 0 else "−" if r < 0 else "") + pct(abs(r), decimales)


def _sin_rango(nombre) -> str:
    """'Madrugada (0-5 h)' -> 'Madrugada'."""
    return str(nombre).partition(" (")[0]


def _la(h) -> str:
    """la 1 h / las 17 h."""
    return f"la {int(h)} h" if int(h) == 1 else f"las {int(h)} h"


def rangos_horas(horas) -> str:
    """[2, 3, 4, 5] -> 'de las 2 h a las 5 h'; admite varios tramos."""
    horas = sorted(int(h) for h in horas)
    if not horas:
        return "sin horas"
    tramos, ini, ant = [], horas[0], horas[0]
    for h in horas[1:]:
        if h != ant + 1:
            tramos.append((ini, ant))
            ini = h
        ant = h
    tramos.append((ini, ant))
    return lista(_la(a) if a == b else f"de {_la(a)} a {_la(b)}" for a, b in tramos)


ETIQUETAS = {
    "hora": "Hora del día [h]", "franja_horaria": "Franja horaria", "tipo_dia": "Tipo de día",
    "estacion": "Estación", "mes": "Mes (número)", "nombre_mes": "Mes", "clima": "Situación climática",
    "rango_temperatura": "Rango de temperatura [°C]", "nombre_festivo": "Festivo", "tipo_festivo": "Tipo de festivo",
    "fechas": "Fechas [días]", "horas_observadas": "Horas observadas [horas]",
    "total_alquileres": "Total [alquileres]", PH: "Promedio [alquileres/hora]",
    "mediana_alquileres_por_hora": "Mediana [alquileres/hora]",
    "pct_del_total_alquileres": "% del total de la selección", "pct_casuales": "Casuales [% de los alquileres de la fila]",
    "pct_registrados": "Registrados [% de los alquileres de la fila]", "dimension": "Dimensión",
    "categoria": "Categoría", "usuarios_casuales": "Casuales [alquileres]",
    "usuarios_registrados": "Registrados [alquileres]", "alquileres_total": "Total [alquileres]",
    "tipo_usuario": "Tipo de usuario", "total_2011": "Total 2011 [alquileres]", "total_2012": "Total 2012 [alquileres]",
    "variacion_total_pct": "Variación del total [%]", "horas_observadas_2011": "Horas observadas 2011 [horas]",
    "horas_observadas_2012": "Horas observadas 2012 [horas]", "promedio_por_hora_2011": "Promedio 2011 [alquileres/hora]",
    "promedio_por_hora_2012": "Promedio 2012 [alquileres/hora]",
    "variacion_promedio_pct": "Variación del promedio por hora [%]", "dia_semana": "Día de la semana",
}
SIN_FORMATO = {"hora", "mes", "anio"}


class DashboardBike:
    """Dashboard de la demanda de bicicletas: figuras Plotly + app Streamlit.

    Parámetros
    ----------
    df_total : DataFrame preparado en AE3 (una fila por hora observada) con categóricas ordenadas.
    filtros : dict opcional con ``anios``, ``estaciones``, ``tipos_dia`` y ``horas`` (tupla desde-hasta).
              Una lista vacía o ausente significa "todas las opciones".
    """

    TITULO = "Demanda de bicicletas compartidas · Capital Bikeshare, Washington D.C. (2011-2012)"
    PREGUNTA_NEGOCIO = (
        "¿Cómo varía la cantidad de alquileres por hora según la franja horaria, el tipo de día, la estación "
        "y el clima, y qué diferencias hay entre usuarios casuales y registrados entre 2011 y 2012, para "
        "planificar la redistribución de bicicletas y el mantenimiento?"
    )
    PREGUNTAS = {
        "P1": "¿En qué horas y franjas se concentra la demanda?",
        "P2": "¿Cómo cambia la demanda según el tipo de día?",
        "P3": "¿Cómo varía la demanda entre estaciones y meses?",
        "P4": "¿Cómo se relaciona la demanda con el clima?",
        "P5": "¿En qué se diferencian los usuarios casuales y los registrados?",
        "P6": "¿Cuánto cambió la demanda entre 2011 y 2012?",
    }
    COLUMNAS_REQUERIDAS = [
        "fecha", "fecha_hora", "anio", "mes", "nombre_mes", "dia_semana", "hora", "franja_horaria", "tipo_dia",
        "nombre_festivo", "tipo_festivo", "estacion", "clima", "temp_c", "rango_temperatura",
        "usuarios_casuales", "usuarios_registrados", "alquileres_total",
    ]

    # Paleta: azul = valor normal, naranja = máximo, violeta = mínimo, gris = pocas horas
    AZUL, AZUL_CLARO = "#2a78d6", "#9ec5f0"
    MAXIMO, MINIMO, GRIS = "#eb6834", "#6250d6", "#b9b8b2"
    CASUAL, REGISTRADO = "#0f8a5f", "#2a78d6"   # oscuros: llevan texto blanco encima
    TEXTO, TEXTO_2, REJILLA = "#1f2430", "#5f6573", "#e6e8ee"
    COLOR_TIPO_DIA = {"Laboral": "#2a78d6", "Fin de semana": "#1baf7a", "Festivo": "#eb6834"}
    ESCALA = "Blues"
    DISCLAIMER = (
        "Se mide la **demanda en alquileres**, no ingresos ni rentabilidad: el dataset no tiene precios ni "
        "costos. Todo lo que se muestra es descriptivo: una diferencia entre grupos **se asocia** a otras "
        "variables, pero no demuestra una causa. Es una sola ciudad y solo 2011-2012."
    )

    def __init__(self, df_total: pd.DataFrame, filtros: dict | None = None):
        faltan = [c for c in self.COLUMNAS_REQUERIDAS if c not in df_total.columns]
        if faltan:
            raise ValueError(f"Al dataset le faltan columnas de AE3: {faltan}")
        self.df_total = df_total
        self.filtros = dict(filtros or {})
        self.df = self._aplicar(ignorar_anio=False)
        self.df_p6 = self._aplicar(ignorar_anio=True)  # P6 compara los dos años: no usa el filtro de año
        self.total = int(self.df["alquileres_total"].sum())
        self.total_dataset = int(df_total["alquileres_total"].sum())
        self.horas = len(self.df)
        self._cache: dict = {}

    # ================================================================== datos
    @classmethod
    def desde_csv(cls, ruta: str | Path | None = None, **kwargs) -> "DashboardBike":
        """Carga el CSV preparado en AE3 (por defecto ``AE3/datos/bike_sharing_preparado.csv``)."""
        if ruta is None:
            return cls(_datos.cargar_dataset(), **kwargs)
        df = pd.read_csv(ruta, parse_dates=["fecha", "fecha_hora"])
        return cls(_datos.restaurar_categorias(df), **kwargs)

    def _aplicar(self, ignorar_anio: bool) -> pd.DataFrame:
        d, f = self.df_total, self.filtros
        if f.get("anios") and not ignorar_anio:
            d = d[d["anio"].isin([int(a) for a in f["anios"]])]
        if f.get("estaciones"):
            d = d[d["estacion"].astype(str).isin(f["estaciones"])]
        if f.get("tipos_dia"):
            d = d[d["tipo_dia"].astype(str).isin(f["tipos_dia"])]
        if f.get("horas"):
            desde, hasta = f["horas"]
            d = d[d["hora"].between(int(desde), int(hasta))]
        return d.copy()

    def filtrar(self, anios=None, estaciones=None, tipos_dia=None, horas=None) -> "DashboardBike":
        """Devuelve un dashboard nuevo con la selección filtrada (el original no cambia)."""
        return DashboardBike(self.df_total, {"anios": anios, "estaciones": estaciones,
                                             "tipos_dia": tipos_dia, "horas": horas})

    @property
    def vacio(self) -> bool:
        return self.df.empty

    def filtros_activos(self, con_anio: bool = True) -> list[str]:
        """Descripción de los filtros que realmente restringen los datos (vacío = sin filtros)."""
        f, t, act = self.filtros, self.df_total, []
        anios = sorted(int(a) for a in t["anio"].unique())
        if con_anio and f.get("anios") and sorted(int(a) for a in f["anios"]) != anios:
            act.append("Año: " + lista(sorted(int(a) for a in f["anios"])))
        if f.get("estaciones") and set(f["estaciones"]) != set(ESTACIONES):
            act.append("Estación: " + lista([e for e in ESTACIONES if e in f["estaciones"]]))
        if f.get("tipos_dia") and set(f["tipos_dia"]) != set(TIPOS_DIA):
            act.append("Tipo de día: " + lista([e for e in TIPOS_DIA if e in f["tipos_dia"]]))
        if f.get("horas") and tuple(f["horas"]) != (int(t["hora"].min()), int(t["hora"].max())):
            act.append("Horas: " + rangos_horas(range(int(f["horas"][0]), int(f["horas"][1]) + 1)))
        return act

    def periodo(self, p6: bool = False) -> str:
        """Período de la selección; con ``p6=True``, el de los datos que usa P6 (sin filtro de año)."""
        fe = (self.df_p6 if p6 else self.df)["fecha"]
        return f"{fe.min():%d/%m/%Y} a {fe.max():%d/%m/%Y}"

    def _en_rango(self) -> str:
        """Aclaración para recomendaciones por hora cuando hay filtro de horas."""
        return ", dentro del rango de horas seleccionado" if any(a.startswith("Horas") for a in self.filtros_activos()) else ""

    def base_txt(self) -> str:
        """Base de todos los porcentajes de % del total: se dice siempre sobre cuánto se calculan."""
        return f"{numero(self.total)} alquileres de la selección"

    def tabla(self, nombre: str) -> pd.DataFrame:
        """Tabla de AE3 recalculada con la selección (sin filtros es igual al CSV de AE3)."""
        if nombre not in self._cache:
            self._cache[nombre] = _m.FUNCIONES[nombre](self.df)
        return self._cache[nombre]

    def tabla_p6(self) -> pd.DataFrame | None:
        """Tabla P6 (2011 vs 2012). Ignora el filtro de año; ``None`` si falta alguno de los dos años."""
        if "P6" not in self._cache:
            anios = set(self.df_p6["anio"].unique())
            ok = {2011, 2012} <= anios
            try:
                self._cache["P6"] = _m.p6_variacion_2011_2012(self.df_p6) if ok else None
            except ZeroDivisionError:  # un total de 2011 en cero: no hay variación porcentual
                self._cache["P6"] = None
        return self._cache["P6"]

    def horas_por(self, columna: str) -> pd.Series:
        """Horas observadas por categoría (sirve para marcar categorías con pocas horas)."""
        s = self.df.groupby(columna, observed=True).size()
        s.index = s.index.astype(str)
        return s

    # ================================================================== indicadores generales
    def kpis(self) -> dict:
        """Indicadores de la selección (sin filtros: los totales de AE3)."""
        d = self.df
        casuales = d["usuarios_casuales"].sum()
        return {
            "Horas observadas": self.horas,
            "Alquileres totales": self.total,
            "Promedio por hora": self.total / self.horas,
            "% de casuales": casuales / self.total * 100,
            "% del dataset completo": self.total / self.total_dataset * 100,
        }

    def horas_faltantes(self) -> dict:
        """Horas del calendario sin registro, calculadas a partir de las fechas del dataset."""
        t = self.df_total
        esperadas = ((t["fecha"].max() - t["fecha"].min()).days + 1) * 24
        return {"esperadas": int(esperadas), "observadas": len(t), "faltantes": int(esperadas - len(t))}

    # ================================================================== utilidades de contenido
    @staticmethod
    def _fila(t: pd.DataFrame, columna: str, valor: str):
        f = t[t[columna].astype(str) == valor]
        return f.iloc[0] if len(f) else None

    @staticmethod
    def _delta(valor: float, ref: float) -> float:
        return (valor / ref - 1) * 100 if ref else float("nan")

    @staticmethod
    def _de_a(a, b) -> str:
        """'de 5 a 7' o, si no cambió, '5 en ambos años'."""
        return f"{numero(a)} en ambos años" if a == b else f"de {numero(a)} a {numero(b)}"

    @staticmethod
    def _ext(t: pd.DataFrame, col: str, ok) -> tuple | None:
        """(máximo, mínimo) entre las filas con muestra suficiente (``ok``); ``None`` si no hay nada que comparar."""
        v = t[np.asarray(ok, dtype=bool)]
        if len(v) < 2 or v[col].max() == v[col].min():
            return None
        return v.loc[v[col].idxmax()], v.loc[v[col].idxmin()]

    @staticmethod
    def _top_bottom(t: pd.DataFrame, col: str, ok, max_k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Mayores y menores sin solaparse: k = min(max_k, mitad de las filas con muestra suficiente)."""
        v = t[np.asarray(ok, dtype=bool)].sort_values(col, ascending=False)
        k = min(max_k, len(v) // 2)
        return v.head(k), v.tail(k).sort_values(col)

    @staticmethod
    def _sin_comp(plur: str, uno: str, t: pd.DataFrame, col: str, ok, fmt=str, masc: bool = False) -> str:
        """Frase honesta cuando una dimensión no permite comparar (una sola categoría o pocas horas)."""
        ok = np.asarray(ok, dtype=bool)
        n, nv = len(t), int(ok.sum())
        if n == 1:
            nombre = fmt(t[col].iloc[0])
            if nv == 1:
                return f"Con el filtro actual queda {uno} ({nombre}), así que no hay comparación entre {plur}."
            return (f"Con el filtro actual queda {uno} ({nombre}) y tiene menos de {MIN_HORAS} horas observadas: no hay "
                    f"comparación entre {plur} y la cifra es indicativa.")
        if nv == 0:
            return (f"{'Ninguno de los' if masc else 'Ninguna de las'} {n} {plur} tiene al menos {MIN_HORAS} horas observadas: las cifras son indicativas y no se comparan.")
        if nv == 1:
            return (f"Solo {fmt(t[col][ok].iloc[0])} tiene al menos {MIN_HORAS} horas observadas, así que no hay comparación entre {plur}.")
        return (f"{'Los' if masc else 'Las'} {plur} con al menos {MIN_HORAS} horas observadas tienen el mismo valor: no hay diferencias para comparar.")

    @staticmethod
    def _nota_pocas(t: pd.DataFrame, col: str, ok, plur: str, fmt=str, masc: bool = False) -> str:
        """Aviso de las categorías con muestra chica que se dejan fuera de la interpretación."""
        m = ~np.asarray(ok, dtype=bool)
        k = int(m.sum())
        if k == 0 or k == len(t):
            return ""
        if k <= 3:
            det = lista(f"{fmt(c)} ({horas_obs(h)})" for c, h in zip(t.loc[m, col], t.loc[m, "horas_observadas"]))
            return f" No se interpretan {'los' if masc else 'las'} {plur} con menos de {MIN_HORAS} horas observadas: {det}."
        return f" No se interpretan {k} de {'los' if masc else 'las'} {len(t)} {plur}: tienen menos de {MIN_HORAS} horas observadas."

    @staticmethod
    def _entre(plur: str, ok, masc: bool = False) -> str:
        """'Entre las franjas con al menos 100 horas observadas, ' si se excluyó alguna categoría; si no, vacío."""
        ok = np.asarray(ok, dtype=bool)
        if ok.all() or not ok.any():
            return ""
        return f"Entre {'los' if masc else 'las'} {plur} con al menos {MIN_HORAS} horas observadas, "

    @staticmethod
    def _solo_ok(plur: str, ok, masc: bool = False, coma: bool = False) -> str:
        """' (entre las franjas con al menos 100 horas observadas)' (o ', entre las ...' con ``coma``) si se excluyó
        alguna categoría; si no, vacío."""
        e = DashboardBike._entre(plur, ok, masc)
        if not e:
            return ""
        e = e[0].lower() + e[1:-2]
        return f", {e}" if coma else f" ({e})"

    @staticmethod
    def _ini(prefijo: str, frase: str) -> str:
        """Une un prefijo ('Entre las X con ..., ') con una frase que pasa a empezar en minúscula."""
        return prefijo + frase[0].lower() + frase[1:] if prefijo else frase

    @staticmethod
    def _min1(x) -> str:
        """Primera letra en minúscula ('Laboral' -> 'laboral') para insertar nombres de categoría a mitad de frase."""
        x = str(x)
        return x[:1].lower() + x[1:]

    # ---- valores sin redondear para la presentación (las tablas de AE3 siguen redondeadas a 2 decimales)
    @staticmethod
    def _afinar(t: pd.DataFrame, total: float) -> pd.DataFrame:
        """Recalcula promedios y porcentajes desde las columnas de suma, para no redondear dos veces (2 y luego 1 decimal)."""
        t, c = t.copy(), set(t.columns)
        if {"total_alquileres", "horas_observadas", PH} <= c:
            t[PH] = t["total_alquileres"] / t["horas_observadas"]
        if {"total_alquileres", "pct_del_total_alquileres"} <= c and total:
            t["pct_del_total_alquileres"] = t["total_alquileres"] / total * 100
        if {"usuarios_casuales", "usuarios_registrados", "alquileres_total", "pct_casuales", "pct_registrados"} <= c:
            t["pct_casuales"] = t["usuarios_casuales"] / t["alquileres_total"] * 100
            t["pct_registrados"] = t["usuarios_registrados"] / t["alquileres_total"] * 100
        return t

    def _resumen(self, df: pd.DataFrame, por) -> pd.DataFrame:
        """``metricas.resumen_demanda`` con promedios y porcentajes sin redondear."""
        return self._afinar(_m.resumen_demanda(df, por), int(df["alquileres_total"].sum()))

    def tp(self, nombre: str) -> pd.DataFrame:
        """Tabla ``nombre`` para presentación: mismas filas que la de AE3, pero con valores sin redondear."""
        k = "x:" + nombre
        if k not in self._cache:
            t = self._afinar(self.tabla(nombre), self.total)
            if nombre == "P2_demanda_por_festivo" and len(t):
                cas = self.df.groupby([self.df["nombre_festivo"].astype(str), self.df["tipo_festivo"].astype(str)])["usuarios_casuales"].sum()
                t["pct_casuales"] = [cas[(str(n), str(f))] / a * 100
                                     for n, f, a in zip(t["nombre_festivo"], t["tipo_festivo"], t["total_alquileres"])]
            self._cache[k] = t
        return self._cache[k]

    def tp6(self) -> pd.DataFrame | None:
        """Tabla P6 con promedios y variaciones sin redondear (``None`` si no se puede comparar)."""
        if "x:P6" not in self._cache:
            t = self.tabla_p6()
            if t is not None:
                t = t.copy()
                for a in ("2011", "2012"):
                    t[f"promedio_por_hora_{a}"] = t[f"total_{a}"] / t[f"horas_observadas_{a}"]
                t["variacion_total_pct"] = (t["total_2012"] - t["total_2011"]) / t["total_2011"] * 100
                t["variacion_promedio_pct"] = (t["promedio_por_hora_2012"] / t["promedio_por_hora_2011"] - 1) * 100
            self._cache["x:P6"] = t
        return self._cache["x:P6"]

    @property
    def insuficiente(self) -> bool:
        """Chequeo único: la selección total no alcanza (menos de MIN_HORAS horas) para recomendar ni rankear."""
        return self.horas < MIN_HORAS

    def formatear(self, t: pd.DataFrame) -> pd.DataFrame:
        """Para mostrar: nombres con unidades y números con coma decimal."""
        out = t.copy()
        for c in out.columns:
            if c in SIN_FORMATO:
                out[c] = out[c].astype(str)
            elif pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].map(lambda v: numero(v, 2))
            elif pd.api.types.is_integer_dtype(out[c]):
                out[c] = out[c].map(lambda v: numero(v))
            else:
                out[c] = out[c].astype(str)
        return out.rename(columns=ETIQUETAS)

    # ================================================================== contenido por pregunta
    def contenido(self) -> dict:
        """Para cada pregunta: métrica, KPIs, textos y tablas, todo calculado con la selección."""
        if "contenido" not in self._cache:
            c = {p: getattr(self, f"_c_{p.lower()}")() for p in self.PREGUNTAS}
            if self.insuficiente:   # un solo chequeo: sin muestra suficiente no hay recomendaciones ni mayor/menor en ninguna pestaña
                txt = (f"La selección tiene solo {horas(self.horas)} observadas (menos de {MIN_HORAS}): no alcanzan para sacar "
                       "conclusiones, así que no se hacen recomendaciones ni rankings de mayor y menor.")
                for v in c.values():
                    if not v.get("vacio"):
                        v["conclusion"] = v["decision"] = txt
                        v["kpis"] = [("Horas observadas", numero(self.horas), None, "Horas con registro en la selección.")]
            self._cache["contenido"] = c
        return self._cache["contenido"]

    def respuestas(self) -> list[dict]:
        """Pregunta, respuesta con cifras, conclusión corta y decisión que apoya."""
        c = self.contenido()
        return [{"id": p, "pregunta": self.PREGUNTAS[p], "respuesta": c[p]["respuesta"],
                 "conclusion": c[p]["conclusion"], "decision": c[p]["decision"]} for p in self.PREGUNTAS]

    # ---------------------------------------------------------------- P1
    def _c_p1(self) -> dict:
        fr, hr = self.tp("P1_demanda_por_franja_horaria"), self.tp("P1_demanda_por_hora_del_dia")
        okf = (fr["horas_observadas"] >= MIN_HORAS).to_numpy()
        okh = (hr["horas_observadas"] >= MIN_HORAS).to_numpy()
        ef, eh = self._ext(fr, PH, okf), self._ext(hr, PH, okh)
        fmt_h = lambda h: f"{int(h)} h"
        sin_f = self._sin_comp("franjas", "una sola franja", fr, "franja_horaria", okf, _sin_rango)
        sin_h = self._sin_comp("horas del día", "una sola hora", hr, "hora", okh, fmt_h)
        hv = hr[okh]
        n_h = hr.groupby("franja_horaria", observed=True)["hora"].nunique()
        top, low = ef if ef else (None, None)
        ht, hm = eh if eh else (None, None)
        partes = []
        if ef:
            partes.append(self._ini(self._entre("franjas", okf),
                f"La franja con mayor promedio es {top['franja_horaria']}: {numero(top[PH], 1)} alquileres por hora, y "
                f"reúne el {pct(top['pct_del_total_alquileres'])} de los {self.base_txt()} en {horas_obs(top['horas_observadas'])}. "
                f"La de menor promedio es {low['franja_horaria']} ({numero(low[PH], 1)} por hora)."))
        else:
            partes.append(sin_f)
        partes[-1] += self._nota_pocas(fr, "franja_horaria", okf, "franjas", _sin_rango)
        if eh:
            s = (f"Por hora del día{self._solo_ok('horas del día', okh)}, el máximo es a {_la(ht['hora'])} "
                 f"({numero(ht[PH], 1)} por hora) y el mínimo a {_la(hm['hora'])} ({numero(hm[PH], 1)}).")
            if len(hv) >= 4:
                dos = hv.nlargest(2, PH)
                s += (f" Las dos horas más altas ({lista(f'{int(h)} h' for h in dos['hora'])}) "
                      f"suman el {pct(dos['pct_del_total_alquileres'].sum())} de los {self.base_txt()}.")
            partes.append(s)
        else:
            partes.append(sin_h)
        partes[-1] += self._nota_pocas(hr, "hora", okh, "horas del día", fmt_h)
        if len(n_h) >= 2 and n_h.min() != n_h.max():
            partes.append(f"Atención: el % del total depende de cuántas horas del día tiene cada franja (entre {int(n_h.min())} y "
                          f"{int(n_h.max())} horas); para comparar franjas se usa el promedio por hora, no el total.")
        respuesta = " ".join(partes)

        kpis = []
        if ef:
            kpis.append(("Franja con más demanda por hora", f"{numero(top[PH], 1)} alq/h", str(top["franja_horaria"]),
                         f"Promedio de alquileres por hora de la franja {top['franja_horaria']}."))
        elif len(fr) == 1:
            r = fr.iloc[0]
            kpis.append((f"Promedio por hora · {r['franja_horaria']}", f"{numero(r[PH], 1)} alq/h", horas_obs(r["horas_observadas"]),
                         "Única franja de la selección: no hay comparación entre franjas."))
        if eh:
            kpis.append(("Hora pico", f"{int(ht['hora'])} h", f"{numero(ht[PH], 1)} alq/h",
                         "Hora del día con mayor promedio de alquileres por hora."))
            kpis.append(("Hora de menor demanda", f"{int(hm['hora'])} h", f"{numero(hm[PH], 1)} alq/h",
                         "Hora del día con menor promedio de alquileres por hora."))
        if not kpis:
            kpis.append(("Horas observadas", numero(self.horas), None, "Horas con registro en la selección."))

        dec = []
        if eh:
            k3 = min(3, max(len(hv) // 2, 1))   # sin solaparse con el valle cuando el rango de horas es angosto
            top3 = sorted(int(h) for h in hv.nlargest(k3, PH)["hora"])
            valle = sorted(int(h) for h in hv.loc[hv[PH] < UMBRAL_VALLE * ht[PH], "hora"])
            dec.append(f"Redistribución: dejar las bicicletas dispuestas antes de las horas de mayor demanda "
                       f"({lista(f'{h} h' for h in top3)}){self._en_rango()}.")
            if valle:
                dec.append(f"Mantenimiento: programarlo en las horas de valle ({rangos_horas(valle)}), donde el promedio es "
                           f"menor al {numero(UMBRAL_VALLE * 100)} % del de la hora pico.")
            else:
                dec.append(f"Mantenimiento: ninguna hora baja del {numero(UMBRAL_VALLE * 100)} % del promedio de la hora pico, "
                           "así que no hay una ventana clara de baja demanda.")
        else:
            dec.append(f"{sin_h} Por eso no se recomienda cuándo reforzar ni cuándo hacer mantenimiento con esta selección.")
        if ef:
            dec.append("Dotación: concentrar al personal de campo en las franjas de mayor promedio por hora.")

        if ef and eh:
            conclusion = (f"La demanda se concentra en {top['franja_horaria']} y alcanza su máximo a {_la(ht['hora'])} "
                          f"({numero(ht[PH], 1)} alquileres por hora); el valle está a {_la(hm['hora'])}"
                          f"{self._solo_ok('franjas y horas del día', np.concatenate([okf, okh]))}.")
        else:
            conclusion = " ".join([
                (f"La franja de mayor promedio es {top['franja_horaria']} ({numero(top[PH], 1)} alquileres por hora)"
                 f"{self._solo_ok('franjas', okf, coma=True)}.") if ef else sin_f,
                (f"El máximo por hora del día es a {_la(ht['hora'])} ({numero(ht[PH], 1)}) y el mínimo a {_la(hm['hora'])}"
                 f"{self._solo_ok('horas del día', okh)}.") if eh else sin_h])
        return {
            "pregunta": self.PREGUNTAS["P1"],
            "metrica": (
                "**Promedio de alquileres por hora** = suma de `alquileres_total` del grupo ÷ horas observadas del grupo "
                "[alquileres/hora]. **% del total** = alquileres del grupo ÷ alquileres de la selección × 100 [%]. "
                "Los grupos son la franja horaria (5) y la hora del día (0 a 23). "
                f"Los grupos con menos de {MIN_HORAS} horas observadas se marcan en gris y no se interpretan."),
            "kpis": kpis,
            "respuesta": respuesta,
            "conclusion": conclusion,
            "decision": " ".join(dec),
            "info": ("Alquileres por estación de bicicletas (origen y destino) para saber en qué lugares redistribuir; "
                     "disponibilidad de bicicletas por hora para detectar faltantes reales (la demanda observada no incluye "
                     "los viajes que no se hicieron por falta de bicicletas); y la causa de las horas sin registro."),
            "limitaciones": (
                f"Es un promedio {self._alcance_p1()}. No hay datos por estación de "
                f"bicicletas, así que no se sabe dónde redistribuir. Las {numero(self.horas_faltantes()['faltantes'])} horas sin "
                "registro del calendario no entran en ningún promedio. Las franjas se definieron mirando el mismo perfil que describen."),
            "tablas": {"P1 · Demanda por franja horaria": self.tabla("P1_demanda_por_franja_horaria"),
                       "P1 · Demanda por hora del día": self.tabla("P1_demanda_por_hora_del_dia")},
        }

    def _alcance_p1(self) -> str:
        """Texto de la limitación de P1: depende de los años y estaciones que deja el filtro."""
        anios = sorted(int(a) for a in self.df["anio"].unique())
        txt_anio = "de los dos años" if len(anios) > 1 else f"de {anios[0]} (un solo año)"
        estaciones = [e for e in ESTACIONES if e in set(self.df["estacion"].astype(str))]
        if len(estaciones) == len(ESTACIONES):
            txt_est = "de todas las estaciones del año juntas"
        elif len(estaciones) == 1:
            txt_est = f"de una sola estación del año ({estaciones[0]})"
        else:
            txt_est = "de las estaciones del año seleccionadas, juntas"
        return f"{txt_anio} y {txt_est}"

    # ---------------------------------------------------------------- P2
    def _c_p2(self) -> dict:
        td = self.tp("P2_demanda_por_tipo_de_dia")
        fe = self.tp("P2_demanda_por_festivo")
        okd = (td["horas_observadas"] >= MIN_HORAS).to_numpy()
        lab = self._fila(td, "tipo_dia", "Laboral")
        ref = lab[PH] if lab is not None and lab["horas_observadas"] >= MIN_HORAS else float("nan")
        partes, kpis = [], []
        for i, (_, r) in enumerate(td.iterrows()):
            chica = not okd[i]
            aviso = f" [solo {horas_obs(r['horas_observadas'])}: no se interpreta]" if chica else ""
            partes.append(f"{self._min1(r['tipo_dia'])}, {numero(r[PH], 1)} alquileres por hora ({pct(r['pct_del_total_alquileres'])} de los "
                          f"{self.base_txt()}, en {horas(r['horas_observadas'])}){aviso}")
            if chica:
                dl = f"* solo {horas_obs(r['horas_observadas'])}: indicativo"
            elif r["tipo_dia"] == "Laboral" or np.isnan(ref):
                dl = None
            else:
                dl = f"{signed(self._delta(r[PH], ref), 1)} vs. laboral"
            kpis.append((f"Promedio · {r['tipo_dia']}", f"{numero(r[PH], 1)} alq/h{'*' if chica else ''}", dl,
                         f"Alquileres por hora en promedio en días de tipo {r['tipo_dia']} ({horas(r['horas_observadas'])})."
                         + (f" Menos de {MIN_HORAS} horas observadas: indicativo, no se compara." if chica else "")))
        picos = self.tabla_pico_por_tipo_dia()
        sin_perfil = [str(t) for t in td["tipo_dia"] if str(t) not in picos]
        txt_pico = "; ".join(
            [f"{self._min1(t)}, máximo a {_la(h)} ({numero(v, 1)} por hora)" for t, (h, v) in picos.items()]
            + [f"{self._min1(t)}, sin perfil interpretable (ninguna hora tiene al menos {MIN_HORAS} horas observadas)" for t in sin_perfil])
        nota_perfil = (f" Las horas con menos de {MIN_HORAS} horas observadas se dibujan punteadas (indicativo) y no se usan para "
                       "afirmaciones ni recomendaciones." if sin_perfil or self._perfil_con_pocas() else "")
        resp = (
            "Promedios por tipo de día: " + "; ".join(partes) + ". "
            "El % del total de cada tipo de día depende de cuántas horas tiene (hay más días laborales que festivos), por eso se "
            f"compara con el promedio por hora. Perfil horario: {txt_pico}.{nota_perfil} "
        )
        lim_fe = ""
        if len(fe):
            okf = (fe["horas_observadas"] >= MIN_HORAS).to_numpy()
            fmin, fmax = int(fe["fechas"].min()), int(fe["fechas"].max())
            rf = f"solo {fechas(fmin)}" if fmin == fmax else f"solo entre {fmin} y {fmax} fechas"
            ef = self._ext(fe, PH, okf)
            if ef:
                top, bajos = self._top_bottom(fe, PH, okf)
                k = len(top)
                resp += self._ini(self._entre("festivos", okf, masc=True),
                                  f"{'El festivo' if k == 1 else 'Los festivos'} de mayor promedio {'es' if k == 1 else 'son'} "
                                  f"{lista(top['nombre_festivo'])} y {'el' if k == 1 else 'los'} de menor {lista(bajos['nombre_festivo'])}; "
                                  f"cada festivo tiene {rf}, por lo que el orden es indicativo.")
                meses_top = sorted({m for n in top["nombre_festivo"] for m in self.df.loc[self.df["nombre_festivo"] == n, "mes"].unique()})
                nombre_meses = [_datos.MESES[int(m) - 1].lower() for m in meses_top]
                lim_fe = (f"{'El festivo' if k == 1 else f'Los {k} festivos'} de mayor promedio {'cae' if k == 1 else 'caen'} en "
                          f"{lista(nombre_meses)}: la tabla de festivos mezcla el efecto del festivo con el de la estación y el clima, "
                          "así que no se puede atribuir la diferencia al festivo en sí. ")
            else:
                resp += (self._sin_comp("festivos", "un solo festivo", fe, "nombre_festivo", okf, masc=True)
                         + f" Cada festivo tiene {rf}: el gráfico los ordena por promedio solo como referencia, "
                         "y ese orden no se interpreta ni se usa para recomendar.")
            resp += self._nota_pocas(fe, "nombre_festivo", okf, "festivos", masc=True)
            lim_fe += f"Cada festivo tiene {rf}. "
        n_fest, n_lab = self._horas_tipo("Festivo"), self._horas_tipo("Laboral")
        lim_h = (f"Hay pocas horas de festivo ({numero(n_fest)}) frente a las de días laborales ({numero(n_lab)})."
                 if n_fest and n_lab else "")

        dec = []
        if picos:
            dec.append("Redistribución: según el perfil horario de la selección, reforzar antes del máximo de cada tipo de día con "
                       f"muestra suficiente ({'; '.join(f'{self._min1(t)}: {h} h' for t, (h, _) in picos.items())}){self._en_rango()}"
                       + (f"; para {lista(self._min1(t) for t in sin_perfil)} hay menos de {MIN_HORAS} horas observadas por hora y "
                          "no se recomienda." if sin_perfil else "."))
        if len(picos) > 1:
            dec.append("Mantenimiento: revisar el perfil horario de cada tipo de día para ubicar las ventanas de baja demanda.")
        d5 = self._dim5("Tipo de día")
        h5 = self.horas_por("tipo_dia")
        e5 = self._ext(d5, "pct_casuales", np.array([h5.get(c, 0) >= MIN_HORAS for c in d5["categoria"]]))
        if e5:
            hi5, lo5 = e5
            dec.append(f"Promociones: los casuales pesan más en {str(hi5['categoria']).lower()} ({pct(hi5['pct_casuales'])}) que en "
                       f"{str(lo5['categoria']).lower()} ({pct(lo5['pct_casuales'])}) (P5): ahí se pueden dirigir ofertas a ese segmento.")
        if not dec:
            dec.append("Con el filtro actual no hay datos suficientes para recomendar por tipo de día.")
        if len(picos) > 1:
            conclusion = "Perfil horario por tipo de día, solo con muestra suficiente: " + txt_pico + "."
        elif picos:
            conclusion = "Perfil horario de la selección: " + txt_pico + ". No hay comparación entre tipos de día con el filtro actual."
        else:
            conclusion = (f"Con el filtro actual no hay perfil horario interpretable: ninguna hora tiene al menos {MIN_HORAS} horas "
                          "observadas. No hay comparación entre tipos de día.")
        return {
            "pregunta": self.PREGUNTAS["P2"],
            "metrica": (
                "**Promedio de alquileres por hora** de cada tipo de día = alquileres del grupo ÷ horas observadas del grupo "
                "[alquileres/hora]; **% del total** = alquileres del grupo ÷ alquileres de la selección × 100 [%]. "
                "Tipos de día: laboral, fin de semana y festivo (el nombre del festivo viene del enriquecimiento con LLM). "
                f"Los grupos con menos de {MIN_HORAS} horas observadas se marcan con * y no se interpretan."),
            "kpis": kpis[:3],
            "respuesta": resp,
            "conclusion": conclusion,
            "decision": " ".join(dec),
            "info": ("Calendario completo de eventos locales (conciertos, desfiles, competencias) y datos por estación de bicicletas "
                     "para saber dónde se concentra la demanda de fin de semana; más años para tener más fechas por festivo."),
            "limitaciones": (
                lim_fe
                + "El nombre y el tipo de festivo dependen de una clasificación hecha con un LLM (validada contra el calendario federal). "
                + lim_h),
            "tablas": {"P2 · Por tipo de día": self.tabla("P2_demanda_por_tipo_de_dia"),
                       "P2 · Por tipo de día y franja": self.tabla("P2_demanda_por_tipo_de_dia_y_franja"),
                       "P2 · Por festivo": self.tabla("P2_demanda_por_festivo")},
        }

    def _horas_tipo(self, tipo: str) -> int:
        return int((self.df["tipo_dia"].astype(str) == tipo).sum())

    def _perfil_hora_tipo(self) -> pd.DataFrame:
        """Promedio por (tipo de día, hora) sin redondear, con sus horas observadas."""
        if "perfil" not in self._cache:
            self._cache["perfil"] = self._resumen(self.df, ["tipo_dia", "hora"])
        return self._cache["perfil"]

    def _perfil_con_pocas(self) -> bool:
        return bool((self._perfil_hora_tipo()["horas_observadas"] < MIN_HORAS).any())

    def tabla_pico_por_tipo_dia(self) -> dict:
        """Hora de mayor promedio de cada tipo de día, solo entre las horas con al menos MIN_HORAS observadas:
        {tipo: (hora, promedio)}. Un tipo sin ninguna hora con muestra suficiente no aparece."""
        t = self._perfil_hora_tipo()
        out = {}
        for tipo, g in t.groupby("tipo_dia", observed=True):
            g = g[g["horas_observadas"] >= MIN_HORAS]
            if len(g):
                r = g.loc[g[PH].idxmax()]
                out[str(tipo)] = (int(r["hora"]), float(r[PH]))
        return out

    # ---------------------------------------------------------------- P3
    def _c_p3(self) -> dict:
        es, ms = self.tp("P3_demanda_por_estacion"), self.tp("P3_demanda_por_mes")
        oke = (es["horas_observadas"] >= MIN_HORAS).to_numpy()
        okm = (ms["horas_observadas"] >= MIN_HORAS).to_numpy()
        low = lambda s: str(s).lower()
        ee, em = self._ext(es, PH, oke), self._ext(ms, PH, okm)
        sin_e = self._sin_comp("estaciones", "una sola estación", es, "estacion", oke)
        sin_m = self._sin_comp("meses", "un solo mes", ms, "nombre_mes", okm, low, masc=True)
        e_hi, e_lo = ee if ee else (None, None)
        m_hi, m_lo = em if em else (None, None)
        partes = []
        if ee:
            partes.append(self._ini(self._entre("estaciones", oke),
                f"La estación con mayor promedio es {e_hi['estacion']} ({numero(e_hi[PH], 1)} alquileres por hora; "
                f"{pct(e_hi['pct_del_total_alquileres'])} de los {self.base_txt()}) y la de menor es {e_lo['estacion']} "
                f"({numero(e_lo[PH], 1)}): la más alta es {numero(e_hi[PH] / e_lo[PH], 1)} veces la más baja."))
        else:
            partes.append(sin_e)
        partes[-1] += self._nota_pocas(es, "estacion", oke, "estaciones")
        txt_m, parejos = "", False
        if em:
            top, bot = self._top_bottom(ms, PH, okm)
            f_top = lista(f"{low(m)} ({numero(v, 1)})" for m, v in zip(top["nombre_mes"], top[PH]))
            f_bot = lista(f"{low(m)} ({numero(v, 1)})" for m, v in zip(bot["nombre_mes"], bot[PH]))
            spread = (m_hi[PH] / m_lo[PH] - 1) * 100   # diferencia entre el mes más alto y el más bajo
            parejos = bool(spread < UMBRAL_MESES_PAREJOS)
            txt_m = (f"{'el mes de mayor promedio es' if len(top) == 1 else 'los meses de mayor promedio son'} {f_top} "
                     f"y {'el de menor' if len(bot) == 1 else 'los de menor'} {f_bot}")
            if parejos:
                txt_m = (f"los meses son parejos: el promedio por hora va de {numero(m_lo[PH], 1)} ({low(m_lo['nombre_mes'])}) a "
                         f"{numero(m_hi[PH], 1)} ({low(m_hi['nombre_mes'])}), solo {pct(spread)} de diferencia")
            partes.append(f"Por mes{self._solo_ok('meses', okm, masc=True)}, {txt_m}.")
        else:
            partes.append(sin_m)
        partes[-1] += self._nota_pocas(ms, "nombre_mes", okm, "meses", low, masc=True)
        mv = ms[okm]
        if len(mv) >= 2:
            m_tot = mv.loc[mv["total_alquileres"].idxmax()]
            s = (f"En totales, {low(m_tot['nombre_mes'])} es el mes que más suma ({numero(m_tot['total_alquileres'])} alquileres; "
                 f"{pct(m_tot['pct_del_total_alquileres'])} de los {self.base_txt()})")
            if ms["horas_observadas"].min() != ms["horas_observadas"].max():
                s += (f", pero los totales mensuales dependen de las horas observadas de cada mes ({numero(ms['horas_observadas'].min())} a "
                      f"{numero(ms['horas_observadas'].max())} horas), por eso el ranking usa el promedio por hora")
            partes.append(s + ".")
        anios = sorted(int(a) for a in self.df["anio"].unique())
        dos_anios = {2011, 2012} <= set(anios)
        txt_mezcla = ""
        if dos_anios:
            pe = self._resumen(self.df, ["estacion", "anio"])
            pp = pe.pivot(index="estacion", columns="anio", values=PH)
            ph_ = pe.pivot(index="estacion", columns="anio", values="horas_observadas")
            ok_e = (ph_[2011].fillna(0) >= MIN_HORAS) & (ph_[2012].fillna(0) >= MIN_HORAS)
            txt_mezcla = "; ".join(f"{numero(pp.loc[e, 2011], 1)} en 2011 y {numero(pp.loc[e, 2012], 1)} en 2012 en {e}"
                                   for e in pp.index[ok_e.to_numpy()])
        if dos_anios:
            lim_anio = ("El cambio de 2011 a 2012 (P6) se mezcla con la estacionalidad."
                        + (f" Por estación, el promedio por hora fue {txt_mezcla}." if txt_mezcla else "") + " ")
            metrica_anio = "Se agrupan los dos años juntos."
        else:
            lim_anio = (f"La selección incluye solo {lista(anios)}: no se puede separar la estacionalidad del cambio entre años (P6). ")
            metrica_anio = f"Se usa solo {lista(anios)}."

        kpis = []
        if ee:
            kpis.append(("Estación con más demanda por hora", f"{numero(e_hi[PH], 1)} alq/h", str(e_hi["estacion"]),
                         "Promedio de alquileres por hora de la estación más alta."))
        elif len(es) == 1:
            r = es.iloc[0]
            kpis.append((f"Promedio por hora · {r['estacion']}", f"{numero(r[PH], 1)} alq/h", horas_obs(r["horas_observadas"]),
                         "Única estación de la selección: no hay comparación entre estaciones."))
        if em:
            kpis.append(("Mes con más demanda por hora", f"{numero(m_hi[PH], 1)} alq/h", str(m_hi["nombre_mes"]),
                         "Promedio de alquileres por hora del mes más alto."))
            kpis.append(("Mes con menos demanda por hora", f"{numero(m_lo[PH], 1)} alq/h", str(m_lo["nombre_mes"]),
                         "Promedio de alquileres por hora del mes más bajo."))
        if not kpis:
            kpis.append(("Horas observadas", numero(self.horas), None, "Horas con registro en la selección."))

        dec = []
        if em and parejos:
            dec.append(f"Los meses son parejos (el más alto supera al más bajo solo en {pct(spread)}, menos del "
                       f"{numero(UMBRAL_MESES_PAREJOS)} %): no se recomienda diferenciar la flota ni el mantenimiento mayor por mes.")
        elif em:
            dec.append(f"Flota: dimensionar para los meses de mayor promedio ({lista(low(m) for m in top['nombre_mes'])}). "
                       f"Mantenimiento mayor (revisiones largas, reparaciones): concentrarlo en los meses de menor promedio "
                       f"({lista(low(m) for m in bot['nombre_mes'])}).")
        else:
            dec.append("Con el filtro actual no hay comparación entre meses, así que no se recomienda ajustar la flota ni el "
                       "mantenimiento mayor por mes.")
        if ee:
            dec.append("Personal: planificar turnos por temporada, según las diferencias entre estaciones de la selección.")

        if ee and em:
            conclusion = (f"La demanda es estacional{self._solo_ok('estaciones', oke)}: {e_hi['estacion']} promedia "
                          f"{numero(e_hi[PH], 1)} alquileres por hora y {e_lo['estacion']} {numero(e_lo[PH], 1)}; {txt_m}.")
        else:
            conclusion = " ".join([
                f"Por estación{self._solo_ok('estaciones', oke)}, {e_hi['estacion']} promedia {numero(e_hi[PH], 1)} alquileres "
                f"por hora y {e_lo['estacion']} {numero(e_lo[PH], 1)}." if ee else sin_e,
                f"Por mes{self._solo_ok('meses', okm, masc=True)}, {txt_m}." if em else sin_m])
        return {
            "pregunta": self.PREGUNTAS["P3"],
            "metrica": (
                "**Promedio de alquileres por hora** de la estación o del mes = alquileres del grupo ÷ horas observadas del grupo "
                "[alquileres/hora]; **% del total** = alquileres del grupo ÷ alquileres de la selección × 100 [%]. "
                f"{metrica_anio} Los grupos con menos de {MIN_HORAS} horas observadas se marcan en gris y no se interpretan."),
            "kpis": kpis,
            "respuesta": " ".join(partes),
            "conclusion": conclusion,
            "decision": " ".join(dec),
            "info": ("Datos de más años para separar estacionalidad de crecimiento; cantidad de bicicletas y de estaciones disponibles "
                     "en cada mes (la oferta puede haber cambiado); calendario de eventos de la ciudad."),
            "limitaciones": (
                lim_anio
                + "Algunos meses mezclan estaciones (los meses de cambio de estación). El dataset tiene solo dos años y una ciudad, y el "
                  "clima, que también cambia con la estación, no está controlado."),
            "tablas": {"P3 · Por estación": self.tabla("P3_demanda_por_estacion"), "P3 · Por mes": self.tabla("P3_demanda_por_mes")},
        }

    # ---------------------------------------------------------------- P4
    def _c_p4(self) -> dict:
        cl, tm = self.tp("P4_demanda_por_situacion_climatica"), self.tp("P4_demanda_por_rango_de_temperatura")
        okc = (cl["horas_observadas"] >= MIN_HORAS).to_numpy()
        okt = (tm["horas_observadas"] >= MIN_HORAS).to_numpy()
        cv, tv = cl[okc], tm[okt]
        ref = cv.iloc[0] if len(cv) else None  # primera categoría con muestra suficiente (despejado si alcanza)
        partes, partes_v, kpis, pocas = [], [], [], []
        for i, (_, r) in enumerate(cl.iterrows()):
            chica = not okc[i]
            if chica:
                pocas.append((str(r["clima"]), r["horas_observadas"]))
            d = ("" if chica or ref is None or r["clima"] == ref["clima"]
                 else f", {signed(self._delta(r[PH], ref[PH]), 1)} vs. {str(ref['clima']).lower()}")
            aviso = (f" [solo {horas_obs(r['horas_observadas'])} y {alquileres(r['total_alquileres'])}: no permite concluir]"
                     if chica else "")
            txt = (f"{self._min1(r['clima'])}, {numero(r[PH], 1)} alquileres por hora ({horas(r['horas_observadas'])}, "
                   f"{pct(r['pct_del_total_alquileres'])} de los {self.base_txt()}{d}){aviso}")
            partes.append(txt)
            if not chica:
                partes_v.append(txt)
        for _, r in cv.head(2).iterrows():
            dl = None if r["clima"] == ref["clima"] else f"{signed(self._delta(r[PH], ref[PH]), 1)} vs. {str(ref['clima']).lower()}"
            kpis.append((str(r["clima"]), f"{numero(r[PH], 1)} alq/h", dl, f"Promedio de alquileres por hora ({horas(r['horas_observadas'])})."))
        sufijo = f" (solo entre los rangos con al menos {MIN_HORAS} horas observadas)" if len(tv) < len(tm) else ""
        crece = bool(len(tv) >= 2 and np.all(np.diff(tv[PH].to_numpy()) > 0))
        baja = bool(ref is not None and len(cv) >= 2 and (cv[PH].iloc[1:] < ref[PH]).all())
        if len(tv) >= 2:
            s_t = ("El promedio crece en cada rango más cálido" if crece
                   else "El promedio no crece de forma continua entre rangos") + sufijo + "."
            frio, calor = tv.iloc[0], tv.iloc[-1]
            kpis.append((f"{calor['rango_temperatura']}", f"{numero(calor[PH], 1)} alq/h",
                         f"{numero(calor[PH] / frio[PH], 1)} veces {frio['rango_temperatura']}",
                         f"Promedio de alquileres por hora en el rango {calor['rango_temperatura']} ({horas(calor['horas_observadas'])})."))
        else:
            s_t = (f"No hay dos rangos de temperatura con al menos {MIN_HORAS} horas observadas, así que no se compara entre rangos.")
            if len(tv) == 1:
                r = tv.iloc[0]
                kpis.append((f"{r['rango_temperatura']}", f"{numero(r[PH], 1)} alq/h", horas_obs(r["horas_observadas"]),
                             "Único rango de temperatura con muestra suficiente."))
        if not kpis:
            kpis.append(("Horas observadas", numero(self.horas), None, "Horas con registro en la selección."))
        s_c = ("" if len(cv) >= 2 else
               f" No hay dos situaciones climáticas con al menos {MIN_HORAS} horas observadas, así que no se comparan entre sí.")
        txt_t = "; ".join(
            (f"{r['rango_temperatura']}*: {numero(r[PH], 1)} (solo {horas_obs(r['horas_observadas'])}: no se interpreta)"
             if not okt[i] else f"{r['rango_temperatura']}: {numero(r[PH], 1)} ({horas(r['horas_observadas'])})")
            for i, (_, r) in enumerate(tm.iterrows()))
        for i, (_, r) in enumerate(tm.iterrows()):
            if not okt[i]:
                pocas.append((str(r["rango_temperatura"]), r["horas_observadas"]))
        resp = ("Por situación climática: " + "; ".join(partes) + "." + s_c + " "
                f"Por rango de temperatura (alquileres por hora): {txt_t}. " + s_t + " "
                "Los porcentajes del total reflejan cuántas horas tiene cada condición, "
                "no cuánto rinde cada hora; el promedio por hora permite la comparación.")
        if baja and crece:
            conclusion = "La demanda por hora baja con clima adverso y sube con el calor: " + "; ".join(partes_v[:2]) + "."
        else:
            conclusion = ((f"Por clima{self._solo_ok('situaciones climáticas', okc)}: " + "; ".join(partes_v[:2]) + "."
                           if len(cv) >= 2 else s_c.strip()) + " " + s_t)
        dec = []
        if baja:
            dec.append("Mantenimiento: aprovechar los días con clima adverso (menor promedio por hora en la selección"
                       f"{self._solo_ok('situaciones climáticas', okc, coma=True)}) para revisiones.")
        if crece:
            dec.append("Flota y personal: reforzar la oferta en los rangos de temperatura más cálidos, donde el promedio es mayor"
                       f"{self._solo_ok('rangos de temperatura', okt, masc=True, coma=True)}.")
        if not dec:
            dec.append("Con la selección actual los datos no muestran un patrón del clima o la temperatura que permita recomendar "
                       "una acción.")
        dec.append("Con un pronóstico del tiempo se puede anticipar la redistribución del día siguiente.")
        return {
            "pregunta": self.PREGUNTAS["P4"],
            "metrica": (
                "**Promedio de alquileres por hora** por situación climática y por rango de temperatura (`temp_c`, °C) = alquileres "
                "del grupo ÷ horas observadas del grupo [alquileres/hora]; **% del total** = alquileres del grupo ÷ alquileres de la "
                f"selección × 100 [%]. Los grupos con menos de {MIN_HORAS} horas observadas se marcan en gris con * y no se interpretan."),
            "kpis": kpis[:3],
            "respuesta": resp,
            "conclusion": conclusion,
            "decision": " ".join(dec),
            "info": ("Pronóstico y observación del clima a nivel de hora y de zona de la ciudad; datos de humedad y viento con unidad "
                     "documentada (la del viento no lo está); más años con eventos extremos para poder decir algo de lluvia fuerte o tormenta."),
            "limitaciones": (
                "La temperatura está muy ligada a la estación y a la hora del día, así que esta relación es una asociación, no una causa, "
                "y no está ajustada por esos factores. "
                + (f"Con menos de {MIN_HORAS} horas observadas no se sacan conclusiones de: "
                   f"{lista(f'{c} ({horas_obs(h)})' for c, h in pocas)}. " if pocas else "")
                + "El clima viene normalizado y se llevó a unidades originales con las fórmulas de UCI."),
            "tablas": {"P4 · Por situación climática": self.tabla("P4_demanda_por_situacion_climatica"),
                       "P4 · Por rango de temperatura": self.tabla("P4_demanda_por_rango_de_temperatura")},
        }

    # ---------------------------------------------------------------- P5
    DIMENSIONES = {"Tipo de día": "tipo_dia", "Franja horaria": "franja_horaria",
                   "Estación": "estacion", "Tipo de festivo": "tipo_festivo"}
    MIN_SEGMENTO = 14   # % mínimo de un segmento para llevar su etiqueta adentro de la barra

    def _dim5(self, dimension: str) -> pd.DataFrame:
        p5 = self.tp("P5_participacion_casuales_y_registrados")
        return p5[p5["dimension"] == dimension].reset_index(drop=True)

    def _c_p5(self) -> dict:
        p5 = self.tp("P5_participacion_casuales_y_registrados")
        tot = p5[p5["dimension"] == "Total"].iloc[0]
        d_td, d_fr = self._dim5("Tipo de día"), self._dim5("Franja horaria")
        d_es, d_fe = self._dim5("Estación"), self._dim5("Tipo de festivo")
        h_td, h_fr, h_es, h_fe = (self.horas_por(c) for c in ("tipo_dia", "franja_horaria", "estacion", "tipo_festivo"))
        mask = lambda d, h: np.array([h.get(c, 0) >= MIN_HORAS for c in d["categoria"]])
        ok_td, ok_fr, ok_es = mask(d_td, h_td), mask(d_fr, h_fr), mask(d_es, h_es)
        chicos = [c for c in d_fe["categoria"] if h_fe.get(c, 0) < MIN_HORAS and c != "No festivo"]

        def item(r, h):
            n = h.get(r["categoria"], 0)
            s = f"{self._min1(r['categoria'])} {pct(r['pct_casuales'])} (de {alquileres(r['alquileres_total'])})"
            return s + (f" [solo {horas_obs(n)}: no se interpreta]" if n < MIN_HORAS else "")

        def extremos(d, e, ok_d, nombre, plur, uno):
            if not e:
                return self._sin_comp(plur, uno, d, 'categoria', ok_d, _sin_rango)
            hi, lo = e
            return (f"{nombre}{self._solo_ok(plur, ok_d)}: más casuales en {hi['categoria']} ({pct(hi['pct_casuales'])} de "
                    f"{alquileres(hi['alquileres_total'])}) y menos en {lo['categoria']} ({pct(lo['pct_casuales'])} de "
                    f"{alquileres(lo['alquileres_total'])}).")

        e_td = self._ext(d_td, "pct_casuales", ok_td)
        e_fr = self._ext(d_fr, "pct_casuales", ok_fr)
        e_es = self._ext(d_es, "pct_casuales", ok_es)
        txt_td = "; ".join(item(r, h_td) for _, r in d_td.iterrows())
        fe_txt = "; ".join(item(r, h_fe) for _, r in d_fe.iterrows())
        resp = (
            f"En la selección, los casuales son el {pct(tot['pct_casuales'])} y los registrados el {pct(tot['pct_registrados'])} de los "
            f"{self.base_txt()}. Cada porcentaje se calcula sobre los alquileres de su propia categoría (se indican entre paréntesis), "
            f"no sobre el total. Por tipo de día, % de casuales: {txt_td}. "
            f"{extremos(d_fr, e_fr, ok_fr, 'Por franja horaria', 'franjas', 'una sola franja')} "
            f"{extremos(d_es, e_es, ok_es, 'Por estación', 'estaciones', 'una sola estación')} "
            f"Por tipo de festivo: {fe_txt}."
        )
        lab, fin = self._fila(d_td, "categoria", "Laboral"), self._fila(d_td, "categoria", "Fin de semana")
        kpis = [
            ("% de casuales", pct(tot["pct_casuales"], 1), f"de {alquileres(tot['alquileres_total'])}",
             "Casuales ÷ alquileres totales de la selección × 100."),
            ("% de casuales · fin de semana", pct(fin["pct_casuales"], 1) if fin is not None else "s/d",
             f"de {alquileres(fin['alquileres_total'])}" if fin is not None else None,
             "Casuales ÷ alquileres de las horas de fin de semana × 100."),
            ("% de casuales · laboral", pct(lab["pct_casuales"], 1) if lab is not None else "s/d",
             f"de {alquileres(lab['alquileres_total'])}" if lab is not None else None,
             "Casuales ÷ alquileres de las horas de días laborales × 100."),
        ]
        pares = [e for e in (e_td, e_fr, e_es) if e]
        his = [hi for hi, _ in pares]
        los = [lo for _, lo in pares]
        if tot["pct_registrados"] >= 50:
            conclusion = f"Los registrados dominan ({pct(tot['pct_registrados'], 1)} de los {self.base_txt()})."
        else:
            conclusion = f"Los casuales superan a los registrados ({pct(tot['pct_casuales'], 1)} de los {self.base_txt()})."
        if his:
            conclusion += " Los casuales pesan más en " + lista(f"{h['categoria']} ({pct(h['pct_casuales'], 1)})" for h in his) + "."
            decision = (
                "Promociones y campañas a casuales: concentrarlas donde su participación es mayor ("
                + lista(f"{h['categoria']}: {pct(h['pct_casuales'])} de casuales" for h in his) + "). Operación: donde los registrados pesan más ("
                + lista(f"{lo['categoria']}: {pct(lo['pct_registrados'])} de registrados" for lo in los)
                + "), priorizar una disponibilidad confiable.")
        else:
            conclusion += " Con el filtro actual no hay comparación suficiente entre categorías para decir dónde pesan más los casuales."
            decision = ("Con el filtro actual no hay comparaciones suficientes entre categorías para recomendar dónde concentrar "
                        "campañas a casuales ni dónde priorizar la disponibilidad.")
        return {
            "pregunta": self.PREGUNTAS["P5"],
            "metrica": (
                "**% de casuales** = `usuarios_casuales` de la categoría ÷ `alquileres_total` de la categoría × 100 [%]; "
                "**% de registrados** = 100 − % de casuales [%]. La base de cada porcentaje es el total de alquileres de esa categoría, "
                "no el total de la selección. Se mide en alquileres, no en personas. Dimensiones: tipo de día, franja horaria, "
                "estación y tipo de festivo."),
            "kpis": kpis,
            "respuesta": resp,
            "conclusion": conclusion,
            "decision": decision,
            "info": ("Tipo de viaje y motivo (trabajo, paseo), duración y recorrido; cantidad de personas distintas (el dataset cuenta "
                     "alquileres, no usuarios); qué define a un usuario casual frente a uno registrado en la fuente."),
            "limitaciones": (
                "La participación se mide en alquileres y no en personas. No hay motivo del viaje, así que decir 'desplazamiento' o "
                "'recreación' es una hipótesis, no un dato. "
                + (f"Con menos de {MIN_HORAS} horas observadas y sin interpretar: "
                   f"{lista(f'{c} ({horas_obs(h_fe[c])})' for c in chicos)}. " if chicos else "")
                + "Las categorías de tipo de día y franja se solapan (un fin de semana contiene todas las franjas), así que no son independientes."),
            "tablas": {"P5 · Participación de casuales y registrados": self.tabla("P5_participacion_casuales_y_registrados")},
        }

    # ---------------------------------------------------------------- P6
    def _c_p6(self) -> dict:
        p6 = self.tp6()
        if p6 is None:
            vacio = ("La selección no permite comparar 2011 con 2012: falta alguno de los dos años con los demás filtros "
                     "(o un total de 2011 es cero).")
            return {"pregunta": self.PREGUNTAS["P6"], "metrica": "", "kpis": [], "respuesta": vacio, "conclusion": vacio,
                    "decision": vacio, "info": "", "limitaciones": "", "tablas": {}, "vacio": True}
        t = p6.set_index("tipo_usuario")
        T, C, R = t.loc["Total"], t.loc["Casuales"], t.loc["Registrados"]
        part11, part12 = C["total_2011"] / T["total_2011"] * 100, C["total_2012"] / T["total_2012"] * 100
        h11, h12 = int(T["horas_observadas_2011"]), int(T["horas_observadas_2012"])
        dif_h = h12 - h11
        pocas6 = min(h11, h12) < MIN_HORAS
        pe = self._resumen(self.df_p6, ["estacion", "anio"])
        pp = pe.pivot(index="estacion", columns="anio", values=PH)
        ph_ = pe.pivot(index="estacion", columns="anio", values="horas_observadas")
        ok_e = ((ph_[2011].fillna(0) >= MIN_HORAS) & (ph_[2012].fillna(0) >= MIN_HORAS)).to_numpy()
        var_e = ((pp[2012] / pp[2011] - 1) * 100)[ok_e]
        if len(var_e) == 0:
            txt_e = (f"ninguna estación tiene al menos {MIN_HORAS} horas observadas en cada año, así que no se compara la variación por estación")
        elif len(var_e) == 1:
            txt_e = (f"en {var_e.index[0]} el promedio por hora varió {signed(var_e.iloc[0], 1)} (una sola estación con datos "
                     "suficientes: no hay comparación entre estaciones)")
        elif var_e.max() == var_e.min():
            txt_e = f"la variación del promedio por hora es la misma en todas las estaciones ({signed(var_e.max(), 1)})"
        else:
            ent = "entre las estaciones con al menos %d horas observadas en cada año, " % MIN_HORAS if not ok_e.all() else ""
            txt_e = (f"{ent}la variación del promedio por hora va de {signed(var_e.min(), 1)} ({var_e.idxmin()}) a "
                     f"{signed(var_e.max(), 1)} ({var_e.idxmax()}) según la estación")
        if dif_h == 0:
            txt_h = f"Los dos años tienen las mismas horas observadas ({numero(h11)}). "
        else:
            txt_h = (f"La variación del promedio difiere de la del total porque 2012 tiene {horas_obs(abs(dif_h))} "
                     f"{'más' if dif_h > 0 else 'menos'} que 2011 ({numero(h12)} frente a {numero(h11)}); por eso se compara por hora. ")
        aviso = (f" Atención: hay {horas_obs(h11)} en 2011 y {horas_obs(h12)} en 2012; con menos de {MIN_HORAS} horas en alguno de los "
                 "años las variaciones son indicativas y no se interpretan." if pocas6 else "")
        resp = (
            f"El total de alquileres pasó de {numero(T['total_2011'])} en 2011 a {numero(T['total_2012'])} en 2012 "
            f"({signed(T['variacion_total_pct'])}); los casuales variaron {signed(C['variacion_total_pct'])} ({self._de_a(C['total_2011'], C['total_2012'])}) "
            f"y los registrados {signed(R['variacion_total_pct'])} ({self._de_a(R['total_2011'], R['total_2012'])}). "
            f"El promedio por hora pasó de {numero(T['promedio_por_hora_2011'], 1)} a {numero(T['promedio_por_hora_2012'], 1)} "
            f"({signed(T['variacion_promedio_pct'])}). {txt_h}"
            f"Los casuales pasaron de ser el {pct(part11)} de los {numero(T['total_2011'])} alquileres de 2011 al {pct(part12)} de los "
            f"{numero(T['total_2012'])} de 2012.{aviso}"
        )
        if pocas6:
            conclusion = (f"Con {horas_obs(h11)} en 2011 y {horas_obs(h12)} en 2012 (menos de {MIN_HORAS} en alguno de los años), "
                          "las variaciones son indicativas y no se interpretan.")
            decision = ("Con menos de 100 horas en alguno de los años, la comparación de 2011 con 2012 es indicativa: no se recomienda "
                        "ajustar flota ni personal solo con ella.")
        else:
            conclusion = (f"El promedio por hora varió {signed(T['variacion_promedio_pct'], 1)} de 2011 a 2012 "
                          f"({signed(C['variacion_promedio_pct'], 1)} casuales, {signed(R['variacion_promedio_pct'], 1)} registrados).")
            mayor = ("registrados" if R["variacion_promedio_pct"] > C["variacion_promedio_pct"]
                     else "casuales" if C["variacion_promedio_pct"] > R["variacion_promedio_pct"] else None)
            decision = (
                "Planificación de flota y personal: no basarse solo en el comportamiento de 2011; el promedio por hora de 2012 "
                f"({numero(T['promedio_por_hora_2012'], 1)} alquileres por hora) varió {signed(T['variacion_promedio_pct'], 1)} respecto de 2011 "
                f"({numero(T['promedio_por_hora_2011'], 1)}). Por tipo de usuario, el promedio por hora varió "
                f"{signed(C['variacion_promedio_pct'], 1)} en casuales y {signed(R['variacion_promedio_pct'], 1)} en registrados"
                + (f": la variación fue mayor en {mayor}." if mayor else "."))
        return {
            "pregunta": self.PREGUNTAS["P6"],
            "metrica": (
                "**Variación del total** = (total 2012 − total 2011) ÷ total 2011 × 100 [%], con totales en alquileres. "
                "**Variación del promedio por hora** = (promedio 2012 − promedio 2011) ÷ promedio 2011 × 100 [%], con promedios en "
                "alquileres/hora (corrige que los dos años no tienen las mismas horas observadas). "
                "Ignora el filtro de año (usa 2011 y 2012 juntos); sí respeta los demás filtros."),
            "kpis": [
                ("Variación del total", signed(T["variacion_total_pct"], 1), f"{numero(T['total_2011'])} → {numero(T['total_2012'])} alquileres",
                 "(total 2012 − total 2011) ÷ total 2011 × 100."),
                ("Variación del promedio por hora", signed(T["variacion_promedio_pct"], 1),
                 f"{numero(T['promedio_por_hora_2011'], 1)} → {numero(T['promedio_por_hora_2012'], 1)} alq/h",
                 "Compara por hora observada, no por total."),
                ("Horas observadas", f"{numero(h11)} → {numero(h12)}", "2011 → 2012",
                 "Horas con registro en cada año dentro de la selección."),
            ],
            "respuesta": resp,
            "conclusion": conclusion,
            "decision": decision,
            "info": ("Cantidad de bicicletas y de estaciones por año (¿creció la oferta?), número de socios, precios y campañas, y datos "
                     "de 2013 en adelante para saber si la demanda sigue cambiando o se estabiliza."),
            "limitaciones": (
                "El dataset no permite saber a qué se asocia el crecimiento; harían falta datos de cantidad de estaciones, bicicletas y socios por año. "
                f"Son solo dos años de una sola ciudad. El cambio se mezcla con la estacionalidad: {txt_e}. "
                "El análisis es descriptivo, no causal."),
            "tablas": {"P6 · Variación 2011-2012": self.tabla_p6()},
        }

    # ================================================================== estilo de las figuras
    # Anchos pensados para columnas de ~440 px: el título y el subtítulo se parten en líneas cortas.
    MAX_TITULO, MAX_SUB = 58, 70
    Y_PH = "Promedio [alquileres/hora]"

    @staticmethod
    def _partir(partes: list[str], maximo: int) -> list[str]:
        """Junta las partes con ' · ' sin pasar del máximo de caracteres por línea."""
        lineas: list[str] = []
        for p in partes:
            if lineas and len(lineas[-1]) + 3 + len(p) <= maximo:
                lineas[-1] += f" · {p}"
            else:
                lineas.append(p)
        return lineas

    @staticmethod
    def _franja(nombre: str, ancho: int = 40) -> str:
        """'Pico de la tarde (16-19 h)' -> varias líneas cortas, sin rotar."""
        txt, _, rango = str(nombre).partition(" (")
        lineas = textwrap.wrap(txt, ancho) or [txt]
        if rango:
            lineas.append("(" + rango)
        return "<br>".join(lineas)

    @staticmethod
    def _aviso_pocas(pocas) -> str | None:
        """Nota de las figuras cuando alguna categoría tiene menos de MIN_HORAS horas."""
        return f"* gris = menos de {numero(MIN_HORAS)} horas: no se interpreta" if np.any(pocas) else None

    def _sub(self, con_pct: bool = False, nota: str | list | None = None) -> list[str]:
        partes = [f"Período: {self.periodo()}", horas_obs(self.horas)]
        if self.filtros_activos():
            partes.append("selección filtrada")
        if con_pct:
            partes.append(f"% sobre {self.base_txt()}")
        if nota:
            partes.extend([nota] if isinstance(nota, str) else [n for n in nota if n])
        return self._partir(partes, self.MAX_SUB)

    def _sub_p6(self, extra: list[str] | None = None) -> list[str]:
        """Subtítulo de las figuras de P6: período y horas de los datos que de verdad usa (sin filtro de año)."""
        partes = [f"Período: {self.periodo(p6=True)}", horas_obs(len(self.df_p6))]
        if self.filtros_activos(con_anio=False):
            partes.append("selección filtrada (sin el filtro de año)")
        partes += (extra or []) + ["el filtro de año no se aplica"]
        return partes

    def _estilo(self, fig: go.Figure, titulo: str, x: str | None, y: str | None, alto: int = 380,
                leyenda: bool = False, con_pct: bool = False, sub: list[str] | str | None = None,
                nota: str | list | None = None, extra_arriba: int = 0, ancho_leyenda: int = 0) -> go.Figure:
        if sub is None:
            lineas = self._sub(con_pct, nota)
        elif isinstance(sub, str):
            lineas = [sub]
        else:
            lineas = self._partir(sub, self.MAX_SUB)
        tit = textwrap.wrap(titulo, self.MAX_TITULO) or [titulo]
        arriba = 30 + 20 * len(tit) + 14 * len(lineas) + extra_arriba
        abajo = 20 + (62 if leyenda else 0) + (28 if x else 0)
        alto_plot = max(alto - arriba - abajo, 120)
        fig.update_layout(
            title={"text": "<br>".join(tit) + "<br>" + "<br>".join(f"<sup>{l}</sup>" for l in lineas), "x": 0, "xanchor": "left",
                   "y": 1, "yanchor": "top", "yref": "container", "pad": {"t": 14, "l": 4},
                   "font": {"size": 15, "color": self.TEXTO}},
            height=alto, margin={"l": 10, "r": 26, "t": arriba, "b": abajo},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": self.TEXTO_2},
            separators=",.", showlegend=leyenda,
            legend={"orientation": "h", "yanchor": "top", "y": -(66 / alto_plot), "xanchor": "left", "x": 0, "title": None},
            hoverlabel={"bgcolor": "white", "font_size": 12},
        )
        fig.update_xaxes(showgrid=False, linecolor=self.REJILLA, ticks="", title_text=x, title_standoff=8, automargin=True)
        fig.update_yaxes(gridcolor=self.REJILLA, zeroline=False, linecolor=self.REJILLA, title_text=y, title_standoff=22, automargin=True)
        return fig

    def _colores(self, valores, minimo: bool = True, pocas=None) -> list[str]:
        """Azul para todos, naranja para el máximo, violeta para el mínimo y gris para categorías con pocas horas."""
        v = np.asarray(valores, dtype=float)
        ok = np.ones(len(v), dtype=bool) if pocas is None else ~np.asarray(pocas, dtype=bool)
        col = [self.AZUL] * len(v)
        if ok.sum() >= 2 and np.ptp(v[ok]) > 0:   # sin comparación posible no se resalta ningún máximo
            imax = int(np.argmax(np.where(ok, v, -np.inf)))
            col[imax] = self.MAXIMO
            if minimo:
                col[int(np.argmin(np.where(ok, v, np.inf)))] = self.MINIMO
        return [self.GRIS if not o else c for o, c in zip(ok, col)]

    def _vacia(self, titulo: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos para esta selección", showarrow=False, font={"size": 14})
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return self._estilo(fig, titulo, None, None, alto=260)

    def _barras_h(self, etiquetas, valores, textos, hover, titulo, x, colores, alto=None, ref=None,
                  ref_texto="", con_pct=True, nota=None) -> go.Figure:
        """Barras horizontales; la primera etiqueta queda arriba."""
        fig = go.Figure(go.Bar(
            x=valores, y=etiquetas, orientation="h", marker={"color": colores}, text=textos, textposition="outside",
            textfont={"size": 12, "color": self.TEXTO}, cliponaxis=False, hovertemplate=hover, width=0.62))
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(range=[0, float(np.nanmax(valores)) * 1.35], showgrid=True, gridcolor=self.REJILLA)
        extra = 0
        if ref is not None:
            fig.add_vline(x=ref, line_dash="dot", line_color=self.TEXTO_2, line_width=1)
            fig.add_annotation(x=ref, xref="x", y=1, yref="paper", yanchor="bottom", xanchor="left", showarrow=False,
                               text=ref_texto, font={"size": 12, "color": self.TEXTO}, bgcolor="white", borderpad=2)
            extra = 22
        return self._estilo(fig, titulo, x, None, alto=(alto or 62 * len(etiquetas) + 120) + extra, con_pct=con_pct,
                            nota=nota, extra_arriba=extra)

    # ================================================================== P1
    def _hover_base(self) -> str:
        return ("<b>%{y}</b><br>Promedio: %{x:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                "<br>Total: %{customdata[1]:,} alquileres<br>%{customdata[2]:.1f} % de los " + self.base_txt() + "<extra></extra>")

    def figura_franja(self) -> go.Figure:
        t = self.tp("P1_demanda_por_franja_horaria")
        if t.empty:
            return self._vacia("P1 · Demanda por franja horaria")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        txt = [f"{numero(p, 1)}<br>solo {numero(h)} h: sin conclusión" if poca else f"{numero(p, 1)}<br>{pct(s, 1)}"
               for p, s, h, poca in zip(t[PH], t["pct_del_total_alquileres"], t["horas_observadas"], pocas)]
        etq = [self._franja(f) + (" *" if p else "") for f, p in zip(t["franja_horaria"], pocas)]
        fig = self._barras_h(etq, t[PH], txt, self._hover_base(),
                             "P1 · Alquileres por hora según la franja horaria", self.Y_PH, self._colores(t[PH], pocas=pocas),
                             nota=["etiqueta: promedio por hora y % del total", self._aviso_pocas(pocas)])
        fig.data[0].customdata = np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1)
        return fig

    def figura_hora(self) -> go.Figure:
        t = self.tp("P1_demanda_por_hora_del_dia")
        if t.empty:
            return self._vacia("P1 · Demanda por hora del día")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        fig = go.Figure(go.Bar(
            x=t["hora"], y=t[PH], marker={"color": self._colores(t[PH], pocas=pocas)},
            customdata=np.stack([t["horas_observadas"], t["pct_del_total_alquileres"], t["total_alquileres"]], axis=-1),
            hovertemplate=("%{x} h<br>Promedio: %{y:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                           "<br>Total: %{customdata[2]:,} alquileres<br>%{customdata[1]:.1f} % de los " + self.base_txt() + "<extra></extra>")))
        fig.update_xaxes(range=[-0.5, 23.5], tickmode="linear", tick0=0, dtick=2, tickangle=0)
        return self._estilo(fig, "P1 · Alquileres por hora del día", "Hora del día [h, 0 a 23]", self.Y_PH, con_pct=True,
                            nota=["naranja = máximo, violeta = mínimo", self._aviso_pocas(pocas)], alto=400)

    def figura_mapa_calor(self) -> go.Figure:
        """Día de la semana × hora: promedio de alquileres por hora (dónde se concentra la demanda)."""
        t = self._resumen(self.df, ["dia_semana", "hora"])
        if t.empty:
            return self._vacia("Mapa de calor día de la semana × hora")
        z = t.pivot(index="dia_semana", columns="hora", values=PH).reindex([d for d in ORDEN_DIAS if d in set(t["dia_semana"].astype(str))])
        fig = go.Figure(go.Heatmap(
            z=z.values, x=[int(h) for h in z.columns], y=list(z.index.astype(str)), colorscale=self.ESCALA, xgap=1, ygap=1,
            hovertemplate="<b>%{y}</b> · %{x} h<br>Promedio: %{z:,.1f} alquileres/hora<extra></extra>",
            colorbar={"title": {"text": "Alquileres/hora", "side": "right"}, "thickness": 14, "len": 0.9}))
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(range=[-0.5, 23.5], tickmode="linear", tick0=0, dtick=1, tickangle=0)
        return self._estilo(fig, "Dónde se concentra la demanda: día de la semana × hora", "Hora del día [h, 0 a 23]",
                            "Día de la semana", alto=400, nota="promedio de alquileres por hora")

    # ================================================================== P2
    def figura_tipo_dia(self) -> go.Figure:
        t = self.tp("P2_demanda_por_tipo_de_dia")
        if t.empty:
            return self._vacia("P2 · Demanda por tipo de día")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        txt = [f"{numero(p, 1)}<br>solo {numero(h)} h: indicativo" if poca else f"{numero(p, 1)}<br>{pct(s, 1)}"
               for p, s, h, poca in zip(t[PH], t["pct_del_total_alquileres"], t["horas_observadas"], pocas)]
        etq = [f"{x}{' *' if p else ''}" for x, p in zip(t["tipo_dia"].astype(str), pocas)]
        fig = self._barras_h(etq, t[PH], txt, self._hover_base(),
                             "P2 · Alquileres por hora según el tipo de día", self.Y_PH,
                             self._colores(t[PH], minimo=False, pocas=pocas),
                             alto=62 * len(t) + 130, nota=["etiqueta: promedio por hora y % del total", self._aviso_pocas(pocas)])
        fig.data[0].customdata = np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1)
        return fig

    def figura_perfil_tipo_dia(self) -> go.Figure:
        t = self._perfil_hora_tipo()
        if t.empty:
            return self._vacia("P2 · Perfil horario por tipo de día")
        fig, hay_pocas = go.Figure(), False
        for tipo, g in t.groupby("tipo_dia", observed=True):
            g = g.sort_values("hora")
            color = self.COLOR_TIPO_DIA.get(str(tipo), self.AZUL)
            ok = (g["horas_observadas"] >= MIN_HORAS).to_numpy()
            hover = (f"<b>{tipo}</b> · %{{x}} h<br>Promedio: %{{y:,.1f}} alquileres/hora<br>Horas observadas: %{{customdata:,}}"
                     "<extra></extra>")
            if ok.any():   # tramo con muestra suficiente: línea continua
                fig.add_trace(go.Scatter(
                    x=g["hora"], y=g[PH].where(ok), mode="lines+markers", name=str(tipo), legendgroup=str(tipo),
                    line={"color": color, "width": 2}, customdata=g["horas_observadas"], hovertemplate=hover))
            if not ok.all():   # horas con menos de MIN_HORAS: punteada y tenue, solo indicativa
                hay_pocas = True
                fig.add_trace(go.Scatter(
                    x=g["hora"], y=g[PH].where(~ok) if ok.any() else g[PH], mode="lines+markers" if not ok.any() else "markers",
                    name=f"{tipo} *", legendgroup=str(tipo), showlegend=not ok.any(), opacity=0.55,
                    line={"color": color, "width": 1.5, "dash": "dot"}, marker={"size": 5, "symbol": "circle-open"},
                    customdata=g["horas_observadas"], hovertemplate=hover))
        fig.update_xaxes(range=[-0.5, 23.5], tickmode="linear", tick0=0, dtick=2, tickangle=0)
        return self._estilo(fig, "P2 · Perfil horario según el tipo de día", "Hora del día [h, 0 a 23]", self.Y_PH,
                            leyenda=True, alto=420,
                            nota=f"* menos de {MIN_HORAS} h por hora: indicativo" if hay_pocas else None)

    def figura_tipo_dia_franja(self) -> go.Figure:
        t = self.tp("P2_demanda_por_tipo_de_dia_y_franja")
        if t.empty:
            return self._vacia("P2 · Tipo de día × franja horaria")
        z = t.pivot(index="tipo_dia", columns="franja_horaria", values=PH)
        n = t.pivot(index="tipo_dia", columns="franja_horaria", values="horas_observadas")
        idx, cols = [x for x in TIPOS_DIA if x in z.index], [c for c in _datos.ORDEN_FRANJAS if c in z.columns]
        z, n = z.reindex(index=idx, columns=cols), n.reindex(index=idx, columns=cols)
        pocas = (n.fillna(0) < MIN_HORAS).to_numpy()
        fig = go.Figure(go.Heatmap(
            z=z.values, x=[self._franja(c, 11) for c in z.columns], y=list(z.index.astype(str)), colorscale=self.ESCALA, xgap=2, ygap=2,
            customdata=n.fillna(0).to_numpy(),
            text=[[f"{numero(v, 1)}{'*' if poca else ''}" for v, poca in zip(fila, pf)] for fila, pf in zip(z.values, pocas)],
            texttemplate="%{text}", textfont={"size": 12},
            hovertemplate="<b>%{y}</b> · %{x}<br>Promedio: %{z:,.1f} alquileres/hora<br>Horas observadas: %{customdata:,}<extra></extra>",
            colorbar={"title": {"text": "Alquileres/hora", "side": "right"}, "thickness": 14, "len": 0.95,
                      "lenmode": "fraction", "nticks": 5}))
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(tickangle=0, tickfont={"size": 11})
        return self._estilo(fig, "P2 · Tipo de día × franja horaria", None, None, alto=420,
                            nota=["promedio de alquileres por hora",
                                  f"* menos de {numero(MIN_HORAS)} h en la celda: indicativo" if pocas.any() else None])

    def figura_festivos(self) -> go.Figure:
        t = self.tp("P2_demanda_por_festivo")
        if t.empty:
            return self._vacia("P2 · Demanda por festivo")
        td = self._fila(self.tp("P2_demanda_por_tipo_de_dia"), "tipo_dia", "Laboral")
        etq = [f"{n} ({int(f)} {'fecha' if f == 1 else 'fechas'})" for n, f in zip(t["nombre_festivo"], t["fechas"])]
        hover = ("<b>%{y}</b><br>Promedio: %{x:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                 "<br>Total: %{customdata[1]:,} alquileres<br>Casuales: %{customdata[2]:.1f} % de los alquileres del festivo<extra></extra>")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        etq = [f"{e}{' *' if p else ''}" for e, p in zip(etq, pocas)]
        fig = self._barras_h(etq, t[PH], [numero(v, 1) for v in t[PH]], hover,
                             "P2 · Demanda por festivo", "Promedio [alquileres/hora]", self._colores(t[PH], pocas=pocas),
                             alto=30 * len(t) + 140, ref=None if td is None else td[PH],
                             ref_texto="" if td is None else f"día laboral: {numero(td[PH], 1)}", con_pct=False,
                             nota=["indicativo: pocas fechas por festivo", self._aviso_pocas(pocas)])
        fig.data[0].customdata = np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_casuales"]], axis=-1)
        return fig

    # ================================================================== P3
    def figura_estacion(self) -> go.Figure:
        t = self.tp("P3_demanda_por_estacion")
        if t.empty:
            return self._vacia("P3 · Demanda por estación")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        txt = [f"{numero(p, 1)}<br>solo {numero(h)} h" if poca else f"{numero(p, 1)}<br>{pct(s, 1)}"
               for p, s, h, poca in zip(t[PH], t["pct_del_total_alquileres"], t["horas_observadas"], pocas)]
        fig = go.Figure(go.Bar(
            x=[f"{e}{' *' if p else ''}" for e, p in zip(t["estacion"].astype(str), pocas)], y=t[PH],
            marker={"color": self._colores(t[PH], pocas=pocas)}, text=txt, textposition="outside",
            textfont={"size": 12, "color": self.TEXTO}, cliponaxis=False, width=0.55,
            customdata=np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1),
            hovertemplate=("<b>%{x}</b><br>Promedio: %{y:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                           "<br>Total: %{customdata[1]:,} alquileres<br>%{customdata[2]:.1f} % de los " + self.base_txt() + "<extra></extra>")))
        fig.update_yaxes(range=[0, float(t[PH].max()) * 1.3])
        fig.update_layout(uniformtext={"minsize": 12, "mode": "show"})
        return self._estilo(fig, "P3 · Alquileres por hora según la estación", "Estación del año", self.Y_PH, con_pct=True,
                            nota=["etiqueta: promedio y % del total", self._aviso_pocas(pocas)], alto=400)

    def figura_meses_ranking(self) -> go.Figure:
        t = self.tp("P3_demanda_por_mes")
        if t.empty:
            return self._vacia("P3 · Ranking de meses")
        r = t.sort_values(PH, ascending=False).reset_index(drop=True)
        pocas = (r["horas_observadas"] < MIN_HORAS).to_numpy()
        fig = self._barras_h([f"{m}{' *' if p else ''}" for m, p in zip(r["nombre_mes"].astype(str), pocas)], r[PH],
                             [numero(v, 1) for v in r[PH]], self._hover_base(),
                             "P3 · Meses de mayor a menor demanda por hora", self.Y_PH, self._colores(r[PH], pocas=pocas),
                             alto=30 * len(r) + 140, con_pct=True,
                             nota=["naranja = máximo, violeta = mínimo", self._aviso_pocas(pocas)])
        fig.data[0].customdata = np.stack([r["horas_observadas"], r["total_alquileres"], r["pct_del_total_alquileres"]], axis=-1)
        return fig

    def figura_mes_linea(self) -> go.Figure:
        t = self.tp("P3_demanda_por_mes").sort_values("mes")
        if t.empty:
            return self._vacia("P3 · Demanda mensual")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        fig = go.Figure(go.Scatter(
            x=t["nombre_mes"].astype(str), y=t[PH], mode="lines+markers", line={"color": self.AZUL, "width": 2},
            marker={"size": 9, "color": self._colores(t[PH], pocas=pocas)},
            customdata=np.stack([t["horas_observadas"], t["total_alquileres"]], axis=-1),
            hovertemplate=("<b>%{x}</b><br>Promedio: %{y:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                           "<br>Total: %{customdata[1]:,} alquileres<extra></extra>")))
        fig.update_yaxes(rangemode="tozero")
        return self._estilo(fig, "P3 · Alquileres por hora a lo largo del año", "Mes del año", self.Y_PH,
                            nota=["promedio por mes", self._aviso_pocas(pocas)])

    # ================================================================== P4
    def figura_clima(self) -> go.Figure:
        t = self.tp("P4_demanda_por_situacion_climatica")
        if t.empty:
            return self._vacia("P4 · Demanda por situación climática")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        etq = ["<br>".join(textwrap.wrap(str(c), 22)) + (" *" if p else "") for c, p in zip(t["clima"], pocas)]
        txt = [(f"{numero(p, 1)}<br>solo {numero(h)} h: sin conclusión" if poca else f"{numero(p, 1)}<br>{numero(h)} h")
               for p, h, poca in zip(t[PH], t["horas_observadas"], pocas)]
        nota = self._aviso_pocas(pocas) or "etiqueta: promedio y horas"
        fig = self._barras_h(etq, t[PH], txt, self._hover_base(), "P4 · Alquileres por hora según el clima",
                             self.Y_PH, self._colores(t[PH], minimo=False, pocas=pocas), alto=76 * len(t) + 130,
                             con_pct=False, nota=nota)
        fig.data[0].customdata = np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1)
        return fig

    def figura_temperatura(self) -> go.Figure:
        t = self.tp("P4_demanda_por_rango_de_temperatura")
        if t.empty:
            return self._vacia("P4 · Demanda por temperatura")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        txt = [f"{numero(p, 1)}<br>{numero(h)} h{' *' if poca else ''}" for p, h, poca in zip(t[PH], t["horas_observadas"], pocas)]
        fig = go.Figure(go.Bar(
            x=[self._franja(r, 11) + (" *" if poca else "") for r, poca in zip(t["rango_temperatura"], pocas)], y=t[PH], marker={"color": self._colores(t[PH], minimo=False, pocas=pocas)},
            text=txt, textposition="outside", textfont={"size": 12, "color": self.TEXTO}, cliponaxis=False, width=0.55,
            customdata=np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1),
            hovertemplate=("<b>%{x}</b><br>Promedio: %{y:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                           "<br>Total: %{customdata[1]:,} alquileres<br>%{customdata[2]:.1f} % de los " + self.base_txt() + "<extra></extra>")))
        fig.update_yaxes(range=[0, float(t[PH].max()) * 1.3])
        fig.update_xaxes(tickangle=0)
        fig.update_layout(uniformtext={"minsize": 12, "mode": "show"})
        return self._estilo(fig, "P4 · Alquileres por hora según la temperatura", "Rango de temperatura [°C]", self.Y_PH,
                            con_pct=False, nota=["etiqueta: promedio y horas observadas", self._aviso_pocas(pocas)], alto=400)

    # ================================================================== P5
    def figura_participacion(self, dimension: str) -> go.Figure:
        """Barras 100 % apiladas: casuales y registrados dentro de cada categoría de una dimensión."""
        d = self._dim5(dimension)
        if d.empty:
            return self._vacia(f"P5 · Casuales y registrados por {dimension.lower()}")
        col = self.DIMENSIONES[dimension]
        horas = self.horas_por(col)
        pocas = [horas.get(c, 0) < MIN_HORAS for c in d["categoria"]]
        etq = [f"{c}{' *' if p else ''}" for c, p in zip(d["categoria"], pocas)]
        vis = [self._franja(e) if col == "franja_horaria" else e for e in etq]
        base = [f"de {alquileres(a)}" for a in d["alquileres_total"]]
        # los segmentos angostos no caben con texto adentro: su cifra va afuera de la barra, en oscuro y sin rotar
        txt = lambda serie: [pct(v, 1) if v >= self.MIN_SEGMENTO else "" for v in serie]
        fuente = {"size": 12, "color": "white"}
        fig = go.Figure([
            go.Bar(name="Casuales", y=etq, x=d["pct_casuales"], orientation="h", marker_color=self.CASUAL,
                   text=txt(d["pct_casuales"]), textposition="inside", insidetextanchor="middle", textangle=0,
                   insidetextfont=fuente, customdata=d["alquileres_total"],
                   hovertemplate="<b>%{y}</b><br>Casuales: %{x:.1f} % de los %{customdata:,} alquileres de la categoría<extra></extra>"),
            go.Bar(name="Registrados", y=etq, x=d["pct_registrados"], orientation="h", marker_color=self.REGISTRADO,
                   text=txt(d["pct_registrados"]), textposition="inside", insidetextanchor="middle", textangle=0,
                   insidetextfont=fuente, customdata=d["alquileres_total"],
                   hovertemplate="<b>%{y}</b><br>Registrados: %{x:.1f} % de los %{customdata:,} alquileres de la categoría<extra></extra>"),
        ])
        fig.update_layout(barmode="stack")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)",
                         ticktext=[f"{e}<br>{b}" for e, b in zip(vis, base)], tickvals=etq)
        izq = [(e, v) for e, v in zip(etq, d["pct_casuales"]) if v < self.MIN_SEGMENTO]
        der = [(e, v) for e, v in zip(etq, d["pct_registrados"]) if v < self.MIN_SEGMENTO]
        for lista_, color, lado in [(izq, self.CASUAL, "izq"), (der, self.REGISTRADO, "der")]:
            for e, v in lista_:
                fig.add_annotation(
                    x=-1.5 if lado == "izq" else 101.5, xref="x", y=e, yref="y", text=f"<b>{pct(v, 1)}</b>", showarrow=False,
                    xanchor="right" if lado == "izq" else "left", font={"size": 12, "color": color})
        fig.update_xaxes(range=[-17 if izq else 0, 117 if der else 100], tickvals=[0, 25, 50, 75, 100], ticksuffix=" %")
        fig.update_layout(legend_traceorder="normal")
        sub = [f"Período: {self.periodo()}", "cada % sobre los alquileres de su categoría (bajo cada nombre)"]
        if any(pocas):
            sub.append("* pocas horas: no se interpreta")
        return self._estilo(fig, f"P5 · Casuales y registrados según {dimension.lower()}",
                            "Participación [% de los alquileres de la categoría]", None,
                            alto=72 * len(d) + 180, leyenda=True, sub=sub)

    def figura_participacion_todas(self) -> dict[str, go.Figure]:
        return {d: self.figura_participacion(d) for d in self.DIMENSIONES}

    # ================================================================== P6
    def _fig_p6(self, columna_a: str, columna_b: str, titulo: str, y: str, variacion: str, decimales: int,
                rotar: bool = False) -> go.Figure:
        p6 = self.tp6()
        if p6 is None:
            return self._vacia(titulo)
        h11, h12 = int(p6["horas_observadas_2011"].iloc[0]), int(p6["horas_observadas_2012"].iloc[0])
        pocas = min(h11, h12) < MIN_HORAS
        etq = [f"{u}<br>({signed(v, 1)}{'*' if pocas else ''})" for u, v in zip(p6["tipo_usuario"], p6[variacion])]
        fuente = {"size": 12, "color": self.TEXTO}
        ang = -90 if rotar else 0
        fig = go.Figure([
            go.Bar(name="2011", x=etq, y=p6[columna_a], marker_color=self.AZUL_CLARO, text=[numero(v, decimales) for v in p6[columna_a]],
                   textposition="outside", textangle=ang, textfont=fuente, cliponaxis=False),
            go.Bar(name="2012", x=etq, y=p6[columna_b], marker_color=self.AZUL, text=[numero(v, decimales) for v in p6[columna_b]],
                   textposition="outside", textangle=ang, textfont=fuente, cliponaxis=False),
        ])
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05, uniformtext={"minsize": 11, "mode": "show"})
        mx = float(max(p6[columna_a].max(), p6[columna_b].max()))
        fig.update_yaxes(range=[0, mx * (1.4 if rotar else 1.2)], tickformat=",d" if decimales == 0 else ",.0f")
        sub = self._sub_p6(["variación 2011 → 2012 entre paréntesis"]
                           + ([f"* menos de {numero(MIN_HORAS)} horas en un año: indicativo"] if pocas else []))
        return self._estilo(fig, titulo, "Tipo de usuario", y, leyenda=True, sub=sub, alto=430)

    def figura_variacion_totales(self) -> go.Figure:
        return self._fig_p6("total_2011", "total_2012", "P6 · Alquileres totales por año", "Total [alquileres]",
                            "variacion_total_pct", 0, rotar=True)

    def figura_variacion_promedio(self) -> go.Figure:
        return self._fig_p6("promedio_por_hora_2011", "promedio_por_hora_2012", "P6 · Alquileres por hora por año",
                            self.Y_PH, "variacion_promedio_pct", 1)

    def figura_mes_por_anio(self) -> go.Figure:
        t = self._resumen(self.df_p6, ["anio", "mes", "nombre_mes"])
        if t.empty or self.tabla_p6() is None:
            return self._vacia("P6 · Evolución mensual por año")
        fig = go.Figure()
        for anio, color in [(2011, self.AZUL_CLARO), (2012, self.AZUL)]:
            g = t[t["anio"] == anio].sort_values("mes")
            fig.add_trace(go.Scatter(
                x=g["nombre_mes"].astype(str), y=g[PH], mode="lines+markers", name=str(anio), line={"color": color, "width": 2},
                customdata=g["horas_observadas"],
                hovertemplate=f"<b>{anio} · %{{x}}</b><br>Promedio: %{{y:,.1f}} alquileres/hora<br>Horas observadas: %{{customdata:,}}<extra></extra>"))
        fig.update_yaxes(rangemode="tozero")
        return self._estilo(fig, "P6 · Alquileres por hora mes a mes, 2011 frente a 2012", "Mes del año", self.Y_PH,
                            leyenda=True, sub=self._sub_p6())

    # ================================================================== contexto (población completa)
    def figura_contexto_serie(self) -> go.Figure:
        t = self.df_total
        m = t.groupby(t["fecha"].dt.to_period("M")).agg(total=("alquileres_total", "sum"), horas=("alquileres_total", "size")).reset_index()
        m["fecha"] = m["fecha"].dt.to_timestamp()
        fig = go.Figure(go.Bar(x=m["fecha"], y=m["total"], marker_color=self.AZUL, customdata=m["horas"],
                               hovertemplate="%{x|%m/%Y}<br>Total: %{y:,} alquileres<br>Horas observadas: %{customdata:,}<extra></extra>"))
        fig.update_xaxes(tickformat="%m/%Y")
        fig.update_yaxes(tickformat=",d")
        sub = [f"Período: {t['fecha'].min():%d/%m/%Y} a {t['fecha'].max():%d/%m/%Y}", "dataset completo",
               f"{numero(len(t))} horas, {numero(self.total_dataset)} alquileres"]
        return self._estilo(fig, "Contexto · Alquileres por mes (suma)", "Mes", "Total del mes [alquileres]", sub=sub)

    def figura_contexto_histograma(self) -> go.Figure:
        t = self.df_total
        fig = go.Figure(go.Histogram(x=t["alquileres_total"], nbinsx=40, marker_color=self.AZUL,
                                     hovertemplate="%{x} alquileres en la hora<br>%{y:,} horas<extra></extra>"))
        fig.update_xaxes(tickformat=",d")
        fig.update_yaxes(tickformat=",d")
        sub = [f"{numero(len(t))} horas observadas", f"mediana {numero(float(t['alquileres_total'].median()), 0)}",
               f"promedio {numero(float(t['alquileres_total'].mean()), 1)} alquileres por hora"]
        return self._estilo(fig, "Contexto · Cuántos alquileres hay en una hora típica", "Alquileres en la hora [alquileres]",
                            "Cantidad de horas [horas]", sub=sub)

    # ================================================================== figuras por pregunta
    def figuras(self, pid: str) -> list[list[tuple[str, go.Figure]]]:
        """Filas de figuras de cada pregunta: [[(clave, figura), ...], ...]."""
        if pid == "P1":
            return [[("franja", self.figura_franja()), ("hora", self.figura_hora())], [("calor", self.figura_mapa_calor())]]
        if pid == "P2":
            return [[("tipo_dia", self.figura_tipo_dia()), ("tipo_dia_franja", self.figura_tipo_dia_franja())],
                    [("perfil", self.figura_perfil_tipo_dia()), ("festivos", self.figura_festivos())]]
        if pid == "P3":
            return [[("estacion", self.figura_estacion()), ("ranking", self.figura_meses_ranking())], [("linea", self.figura_mes_linea())]]
        if pid == "P4":
            return [[("clima", self.figura_clima()), ("temperatura", self.figura_temperatura())]]
        if pid == "P5":
            f = self.figura_participacion_todas()
            ks = list(f)
            return [[(ks[0], f[ks[0]]), (ks[1], f[ks[1]])], [(ks[2], f[ks[2]]), (ks[3], f[ks[3]])]]
        if pid == "P6":
            return [[("totales", self.figura_variacion_totales()), ("promedio", self.figura_variacion_promedio())],
                    [("mensual", self.figura_mes_por_anio())]]
        raise KeyError(pid)

    def todas_las_figuras(self) -> dict[str, go.Figure]:
        """Todas las figuras de la app (para probar que se construyen sin error)."""
        out = {"contexto_serie": self.figura_contexto_serie(), "contexto_histograma": self.figura_contexto_histograma()}
        for pid in self.PREGUNTAS:
            for fila in self.figuras(pid):
                for clave, fig in fila:
                    out[f"{pid}_{clave}"] = fig
        return out

    # ================================================================== Streamlit
    def render(self) -> None:
        """Arma la app completa. Se llama desde ``AE4/app/app.py`` con ``streamlit run``."""
        import streamlit as st

        st.set_page_config(page_title="Bike Sharing · Demanda por hora", page_icon="🚲", layout="wide")
        base = self.df_total
        with st.sidebar:
            st.header("Filtros")
            anios = sorted(int(a) for a in base["anio"].unique())
            sel_anio = st.multiselect("Año", anios, default=anios, key="f_anio")
            sel_est = st.multiselect("Estación del año", ESTACIONES, default=ESTACIONES, key="f_estacion")
            sel_td = st.multiselect("Tipo de día", TIPOS_DIA, default=TIPOS_DIA, key="f_tipo_dia")
            hmin, hmax = int(base["hora"].min()), int(base["hora"].max())
            sel_h = st.slider("Rango de horas del día [h]", hmin, hmax, (hmin, hmax), key="f_horas")
            st.caption("Un filtro vacío equivale a todas las opciones. P6 compara 2011 con 2012, así que ignora el filtro de año.")
            st.divider()
            st.caption("Datos: UCI Bike Sharing Dataset (Fanaee-T y Gama, 2013), Capital Bikeshare, Washington D.C., "
                       "preparado y enriquecido en AE3.")

        d = self.filtrar(sel_anio, sel_est, sel_td, sel_h)
        st.title(self.TITULO)
        st.markdown(f"**Pregunta general.** {self.PREGUNTA_NEGOCIO}")
        st.info(self.DISCLAIMER)
        if d.vacio:
            st.warning("No hay horas con esos filtros. Amplía la selección en la barra lateral.")
            return
        d._kpis_y_aviso(st)

        ids = list(self.PREGUNTAS)
        tabs = st.tabs(["Contexto"] + [f"{p} · {t}" for p, t in zip(ids, ["Horas", "Tipo de día", "Estación y mes", "Clima",
                                                                            "Casuales y registrados", "2011 vs 2012"])] + ["Conclusiones"])
        with tabs[0]:
            d._tab_contexto(st)
        for i, pid in enumerate(ids, start=1):
            with tabs[i]:
                d._tab_pregunta(st, pid)
        with tabs[-1]:
            d._tab_conclusiones(st)

    def _kpis_y_aviso(self, st) -> None:
        k = self.kpis()
        act = self.filtros_activos()
        if act:
            st.warning(
                f"**Las cifras de las pestañas P1 a P6 se calculan sobre la selección**, no sobre todo el dataset. Filtros activos: "
                f"{'; '.join(act)}. La selección tiene {horas(k['Horas observadas'])} y {alquileres(k['Alquileres totales'])} "
                f"({pct(k['% del dataset completo'])} de los {numero(self.total_dataset)} alquileres del dataset). "
                "Los porcentajes del total se calculan sobre los alquileres de la selección.")
        else:
            st.caption("Sin filtros activos: las cifras coinciden con las tablas de AE3 (`AE3/resultados/tablas`). "
                       f"Los porcentajes del total se calculan sobre los {numero(self.total)} alquileres del dataset.")
        if self.horas < MIN_HORAS:
            st.warning(f"La selección tiene solo {horas(self.horas)} observadas (menos de {MIN_HORAS}): las cifras son indicativas y no "
                       "alcanzan para sacar conclusiones.")
        c = st.columns(4)
        c[0].metric("Horas observadas", numero(k["Horas observadas"]), help="Cada fila del dataset es una hora de servicio con registro.")
        c[1].metric("Alquileres totales", numero(k["Alquileres totales"]), help="Suma de alquileres_total en la selección.")
        c[2].metric("Promedio por hora", f"{numero(k['Promedio por hora'], 1)} alq/h", help="Alquileres totales ÷ horas observadas.")
        c[3].metric("% de casuales", pct(k["% de casuales"], 1), f"de {numero(k['Alquileres totales'])} alquileres",
                    delta_color="off", delta_arrow="off", help="Alquileres de usuarios casuales ÷ alquileres totales de la selección × 100.")

    def _grafico(self, st, contenedor, fig, key: str) -> None:
        contenedor.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)

    def _tab_contexto(self, st) -> None:
        t = self.df_total
        hf = self.horas_faltantes()
        sinfiltro = self.total_dataset
        st.caption("Esta pestaña describe el dataset completo (la población); los filtros de la barra lateral se aplican en P1 a P6.")
        st.header("Contexto: de dónde salen los datos y qué se quiere saber")
        st.subheader("Necesidad")
        st.markdown(
            "**Escenario hipotético** (no hay una empresa real detrás de este trabajo): el equipo de operaciones de un sistema público de "
            "bicicletas compartidas, parecido a Capital Bikeshare, debe **redistribuir las bicicletas** y **programar su mantenimiento**. "
            "No cuenta con una caracterización organizada de cuándo se concentra la demanda ni de cómo cambia con el tipo de día, la "
            "estación, el clima y el tipo de usuario. Es una necesidad propuesta, no un problema observado: no se afirma que hoy falten "
            "bicicletas en alguna estación (eso exigiría datos por estación).")
        st.markdown("**Preguntas específicas**")
        st.markdown("\n".join(f"- **{p}.** {q}" for p, q in self.PREGUNTAS.items()))

        st.subheader("Identificación del dataset")
        ids = pd.DataFrame({
            "Dato": ["Nombre", "Fuente", "Autores", "Operador", "Ciudad y período", "Unidad de análisis", "Licencia"],
            "Valor": [
                "Bike Sharing Dataset (archivo por hora)",
                "UCI Machine Learning Repository, id 275 (copia en Hugging Face: t22000t/bike-sharing-tabular)",
                "Fanaee-T y Gama (2013)",
                "Capital Bikeshare",
                f"Washington D.C., del {t['fecha'].min():%d/%m/%Y} al {t['fecha'].max():%d/%m/%Y}",
                f"Una fila = una hora de servicio: {numero(len(t))} horas observadas, {t.shape[1]} columnas en el dataset preparado",
                "CC BY 4.0",
            ]})
        st.dataframe(ids, hide_index=True, width="stretch")

        st.markdown(
            f"**Horas faltantes.** Dos años completos son {numero(hf['esperadas'])} horas; el dataset tiene {numero(hf['observadas'])}, es decir "
            f"faltan {numero(hf['faltantes'])} ({pct(hf['faltantes'] / hf['esperadas'] * 100, 1)}). La fuente no explica la causa (pueden ser horas sin "
            "alquileres o sin registro), así que **no se imputaron**: imputar sesgaría los promedios. Por eso se usa siempre el promedio "
            "por hora observada, no el total, para comparar grupos con distinta cantidad de horas.")

        c1, c2 = st.columns(2)
        self._grafico(st, c1, self.figura_contexto_serie(), "ctx_serie")
        self._grafico(st, c2, self.figura_contexto_histograma(), "ctx_hist")

        st.subheader("Transformaciones y enriquecimiento (AE3)")
        n_fest = int(t.loc[t["tipo_festivo"].astype(str) != "No festivo", "fecha"].nunique())
        st.markdown(
            "- **Transformaciones:** `dteday` pasa a fecha y se crea `fecha_hora`; controles de calidad (nulos, duplicados, coherencia de "
            "fecha, hora, día de la semana, año y mes, rangos y conteos); las horas faltantes se informan y no se imputan; se decodifican "
            "estación, clima, día de la semana, año, festivo y día laboral a etiquetas en español como categorías ordenadas; el clima se lleva "
            "a unidades originales (`temp_c`, `sensacion_c`, `humedad_pct`; la unidad del viento no está documentada).\n"
            "- **Variables derivadas:** `franja_horaria`, `tipo_dia` (laboral, fin de semana o festivo), `rango_temperatura`, `nombre_mes` "
            "y `pct_casual`.\n"
            f"- **Enriquecimiento con un LLM (NVIDIA Nemotron):** el dataset marca los festivos pero no dice cuál es. Un LLM clasificó las "
            f"{numero(n_fest)} fechas festivas en `nombre_festivo` y `tipo_festivo` (federal o local de D.C.); la respuesta se validó contra el "
            "calendario federal de la OPM y se guardó versionada, de modo que no hace falta volver a llamar al modelo. No agrega filas ni demanda.")
        dic = pd.DataFrame({
            "Variable": ["fecha_hora", "hora", "dia_semana", "franja_horaria", "tipo_dia", "nombre_festivo / tipo_festivo", "estacion",
                         "clima", "temp_c", "rango_temperatura", "usuarios_casuales / usuarios_registrados", "alquileres_total", "pct_casual"],
            "Qué es": ["Fecha y hora de la observación", "Hora del día (0 a 23)", "Día de la semana", "Franja del día (derivada)",
                       "Laboral, fin de semana o festivo (derivada)", "Festivo identificado con el LLM", "Estación del año",
                       "Situación climática en 4 niveles", "Temperatura", "Temperatura agrupada (derivada)",
                       "Alquileres por tipo de usuario", "Alquileres de la hora (casuales + registrados)", "Casuales ÷ alquileres_total × 100"],
            "Unidad": ["fecha y hora", "h", "texto", "texto", "texto", "texto", "texto", "texto", "°C", "texto", "alquileres", "alquileres", "%"]})
        with st.expander("Variables que usa la app"):
            st.dataframe(dic, hide_index=True, width="stretch")

    def _tab_pregunta(self, st, pid: str) -> None:
        c = self.contenido()[pid]
        st.header(f"{pid}. {c['pregunta']}")
        act = self.filtros_activos()
        if pid == "P6":
            act6 = self.filtros_activos(con_anio=False)
            h6, t6 = len(self.df_p6), int(self.df_p6["alquileres_total"].sum())
            st.caption(f"Esta pregunta compara 2011 con 2012, así que usa ambos años ({self.periodo(p6=True)}) y no aplica el filtro de año. "
                       + (f"Demás filtros: {'; '.join(act6)}. " if act6 else "Sin otros filtros. ")
                       + f"Cifras de {horas(h6)} y {alquileres(t6)}.")
        elif act:
            st.caption(f"Cifras de la selección ({'; '.join(act)}): {horas(self.horas)} y {self.base_txt()}.")
        else:
            st.caption(f"Sin filtros: cifras del dataset completo, iguales a las tablas de AE3 ({self.base_txt().replace(' de la selección', '')}).")
        if c.get("vacio") or not c["kpis"]:
            st.warning(c["respuesta"])
            return
        st.markdown(f"**Métrica.** {c['metrica']}")
        cols = st.columns(len(c["kpis"]))
        for col, (et, val, delta, ayuda) in zip(cols, c["kpis"]):
            col.metric(et, val, delta, delta_color="off", delta_arrow="off", help=ayuda)
        if pid == "P2":
            st.caption("El mapa de calor día de la semana × hora está en la pestaña P1.")
        for i, fila in enumerate(self.figuras(pid)):
            cols = st.columns(len(fila))
            for col, (clave, fig) in zip(cols, fila):
                self._grafico(st, col, fig, f"{pid}_{clave}")
        with st.expander("Ver las tablas de datos"):
            for titulo, t in c["tablas"].items():
                st.markdown(f"**{titulo}**")
                st.dataframe(self.formatear(t), hide_index=True, width="stretch")
        st.subheader("Qué muestran los datos")
        st.markdown(c["respuesta"])
        st.subheader("Qué decisión apoya")
        st.markdown(c["decision"])
        st.subheader("Qué información adicional se necesitaría")
        st.markdown(c["info"])
        st.subheader("Limitaciones")
        st.markdown(c["limitaciones"])

    def _tab_conclusiones(self, st) -> None:
        st.header("Conclusiones y recomendaciones")
        act = self.filtros_activos()
        if act:
            st.warning(f"Las cifras refieren a la selección ({'; '.join(act)}): {horas(self.horas)} y {self.base_txt()}. "
                       "Las conclusiones y recomendaciones se calculan con esa selección, no con el dataset completo "
                       "(P6 usa 2011 y 2012 juntos).")
        st.subheader("Conclusiones")
        for r in self.respuestas():
            st.markdown(f"- **{r['id']}.** {r['conclusion']}")
        st.subheader("Recomendaciones para operaciones")
        for r in self.respuestas():
            st.markdown(f"- **{r['id']}.** {r['decision']}")
        hf = self.horas_faltantes()
        st.subheader("Limitaciones generales")
        st.markdown(
            "- Se mide **demanda en alquileres**, no ingresos ni rentabilidad: no hay precios ni costos en el dataset.\n"
            "- Todo es descriptivo: las diferencias **se asocian** a otras variables, pero no prueban una causa. Estación del año, hora, "
            "temperatura y clima están entrelazados y no se controlaron.\n"
            "- Una sola ciudad y solo 2011 y 2012: los patrones pueden no representar el sistema actual ni otros sistemas.\n"
            "- El crecimiento entre años se mezcla con la estacionalidad (P3 y P6).\n"
            f"- Hay {numero(hf['faltantes'])} horas del calendario sin registro (no imputadas) y categorías con muy pocas horas "
            f"(menos de {numero(MIN_HORAS)}), que se muestran pero no se interpretan.\n"
            "- No hay datos por estación de bicicletas ni por viaje: no se puede decir *dónde* redistribuir, solo *cuándo*.\n"
            "- Los festivos dependen de una clasificación hecha con un LLM y cada uno tiene solo una o dos fechas.")
        st.subheader("Siguientes pasos")
        st.markdown(
            "Conseguir datos por estación y por viaje (origen y destino), más años y la oferta de bicicletas por período para separar "
            "demanda de capacidad; con eso se podría pasar de describir *cuándo* hay más demanda a recomendar *dónde* redistribuir.")
