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


def rangos_horas(horas) -> str:
    """[2, 3, 4, 5] -> 'de las 2 h a las 5 h'; admite varios tramos."""
    horas = sorted(int(h) for h in horas)
    if not horas:
        return "ninguna hora"
    tramos, ini, ant = [], horas[0], horas[0]
    for h in horas[1:]:
        if h != ant + 1:
            tramos.append((ini, ant))
            ini = h
        ant = h
    tramos.append((ini, ant))
    return lista(f"las {a} h" if a == b else f"de las {a} h a las {b} h" for a, b in tramos)


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

    def filtros_activos(self) -> list[str]:
        """Descripción de los filtros que realmente restringen los datos (vacío = sin filtros)."""
        f, t, act = self.filtros, self.df_total, []
        anios = sorted(int(a) for a in t["anio"].unique())
        if f.get("anios") and sorted(int(a) for a in f["anios"]) != anios:
            act.append("Año: " + lista(sorted(int(a) for a in f["anios"])))
        if f.get("estaciones") and set(f["estaciones"]) != set(ESTACIONES):
            act.append("Estación: " + lista([e for e in ESTACIONES if e in f["estaciones"]]))
        if f.get("tipos_dia") and set(f["tipos_dia"]) != set(TIPOS_DIA):
            act.append("Tipo de día: " + lista([e for e in TIPOS_DIA if e in f["tipos_dia"]]))
        if f.get("horas") and tuple(f["horas"]) != (int(t["hora"].min()), int(t["hora"].max())):
            act.append(f"Horas: de las {int(f['horas'][0])} h a las {int(f['horas'][1])} h")
        return act

    def periodo(self) -> str:
        fe = self.df["fecha"]
        return f"{fe.min():%d/%m/%Y} a {fe.max():%d/%m/%Y}"

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
            self._cache["P6"] = _m.p6_variacion_2011_2012(self.df_p6) if ok else None
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
            self._cache["contenido"] = {p: getattr(self, f"_c_{p.lower()}")() for p in self.PREGUNTAS}
        return self._cache["contenido"]

    def respuestas(self) -> list[dict]:
        """Pregunta, respuesta con cifras, conclusión corta y decisión que apoya."""
        c = self.contenido()
        return [{"id": p, "pregunta": self.PREGUNTAS[p], "respuesta": c[p]["respuesta"],
                 "conclusion": c[p]["conclusion"], "decision": c[p]["decision"]} for p in self.PREGUNTAS]

    # ---------------------------------------------------------------- P1
    def _c_p1(self) -> dict:
        fr, hr = self.tabla("P1_demanda_por_franja_horaria"), self.tabla("P1_demanda_por_hora_del_dia")
        top, low = fr.loc[fr[PH].idxmax()], fr.loc[fr[PH].idxmin()]
        ht, hm = hr.loc[hr[PH].idxmax()], hr.loc[hr[PH].idxmin()]
        top3 = sorted(int(h) for h in hr.nlargest(3, PH)["hora"])
        valle = sorted(int(h) for h in hr.loc[hr[PH] < UMBRAL_VALLE * ht[PH], "hora"])
        n_h = hr.groupby("franja_horaria", observed=True)["hora"].nunique()
        dos = hr.nlargest(2, PH)
        respuesta = (
            f"La franja con mayor promedio es {top['franja_horaria']}: {numero(top[PH], 1)} alquileres por hora, y "
            f"reúne el {pct(top['pct_del_total_alquileres'])} de los {self.base_txt()} en {numero(top['horas_observadas'])} "
            f"horas observadas. La de menor promedio es {low['franja_horaria']} ({numero(low[PH], 1)} por hora). "
            f"Por hora del día, el máximo es a las {int(ht['hora'])} h ({numero(ht[PH], 1)} por hora) y el mínimo a las "
            f"{int(hm['hora'])} h ({numero(hm[PH], 1)}). Las dos horas más altas ({lista(f'{int(h)} h' for h in dos['hora'])}) "
            f"suman el {pct(dos['pct_del_total_alquileres'].sum())} de los {self.base_txt()}. "
            f"Atención: el % del total depende de cuántas horas del día tiene cada franja (entre {int(n_h.min())} y "
            f"{int(n_h.max())} horas); para comparar franjas se usa el promedio por hora, no el total."
        )
        return {
            "pregunta": self.PREGUNTAS["P1"],
            "metrica": (
                "**Promedio de alquileres por hora** = suma de `alquileres_total` del grupo ÷ horas observadas del grupo "
                "[alquileres/hora]. **% del total** = alquileres del grupo ÷ alquileres de la selección × 100 [%]. "
                "Los grupos son la franja horaria (5) y la hora del día (0 a 23)."),
            "kpis": [
                ("Franja con más demanda por hora", f"{numero(top[PH], 1)} alq/h", str(top["franja_horaria"]),
                 f"Promedio de alquileres por hora de la franja {top['franja_horaria']}."),
                ("Hora pico", f"{int(ht['hora'])} h", f"{numero(ht[PH], 1)} alq/h",
                 "Hora del día con mayor promedio de alquileres por hora."),
                ("Hora de menor demanda", f"{int(hm['hora'])} h", f"{numero(hm[PH], 1)} alq/h",
                 "Hora del día con menor promedio de alquileres por hora."),
            ],
            "respuesta": respuesta,
            "conclusion": (f"La demanda se concentra en {top['franja_horaria']} y alcanza su máximo a las {int(ht['hora'])} h "
                           f"({numero(ht[PH], 1)} alquileres por hora); el valle está a las {int(hm['hora'])} h."),
            "decision": (
                f"Redistribución: dejar las bicicletas dispuestas antes de las horas de mayor demanda "
                f"({lista(f'{h} h' for h in top3)}). Mantenimiento: programarlo en las horas de valle "
                f"({rangos_horas(valle)}), donde el promedio es menor al {numero(UMBRAL_VALLE * 100)} % del de la hora pico. "
                "Dotación: concentrar al personal de campo en las franjas de mayor promedio por hora."),
            "info": ("Alquileres por estación de bicicletas (origen y destino) para saber en qué lugares redistribuir; "
                     "disponibilidad de bicicletas por hora para detectar faltantes reales (la demanda observada no incluye "
                     "los viajes que no se hicieron por falta de bicicletas); y la causa de las horas sin registro."),
            "limitaciones": (
                f"Es un promedio {self._alcance_p1()}. No hay datos por estación de "
                f"bicicletas, así que no se sabe dónde redistribuir. Las {numero(self.horas_faltantes()['faltantes'])} horas sin "
                "registro del calendario no entran en ningún promedio. Las franjas se definieron mirando el mismo perfil que describen."),
            "tablas": {"P1 · Demanda por franja horaria": fr, "P1 · Demanda por hora del día": hr},
        }

    def _alcance_p1(self) -> str:
        """Texto de la limitación de P1: depende de los años y estaciones que deja el filtro."""
        anios = sorted(int(a) for a in self.df["anio"].unique())
        txt_anio = "de los dos años" if len(anios) > 1 else f"de {anios[0]} (un solo año)"
        n_est = self.df["estacion"].nunique()
        txt_est = ("de todas las estaciones del año juntas" if n_est == len(ESTACIONES)
                   else "de las estaciones del año seleccionadas, juntas")
        return f"{txt_anio} y {txt_est}"

    # ---------------------------------------------------------------- P2
    def _c_p2(self) -> dict:
        td = self.tabla("P2_demanda_por_tipo_de_dia")
        tdf = self.tabla("P2_demanda_por_tipo_de_dia_y_franja")
        fe = self.tabla("P2_demanda_por_festivo")
        lab = self._fila(td, "tipo_dia", "Laboral")
        ref = lab[PH] if lab is not None else float("nan")
        partes, kpis = [], []
        for _, r in td.iterrows():
            partes.append(f"{r['tipo_dia']}: {numero(r[PH], 1)} alquileres por hora ({pct(r['pct_del_total_alquileres'])} de los "
                          f"{self.base_txt()}, en {numero(r['horas_observadas'])} horas)")
            dl = None if r["tipo_dia"] == "Laboral" or np.isnan(ref) else f"{numero(self._delta(r[PH], ref), 1)} % vs. laboral"
            kpis.append((f"Promedio · {r['tipo_dia']}", f"{numero(r[PH], 1)} alq/h", dl,
                         f"Alquileres por hora en promedio en días de tipo {r['tipo_dia']} ({numero(r['horas_observadas'])} horas)."))
        picos = self.tabla_pico_por_tipo_dia()
        txt_pico = "; ".join(f"{t}: máximo a las {int(h)} h ({numero(v, 1)} por hora)" for t, (h, v) in picos.items())
        top3 = fe.head(3)
        bajos = fe.tail(3)
        meses_top = sorted({m for n in top3["nombre_festivo"] for m in self.df.loc[self.df["nombre_festivo"] == n, "mes"].unique()})
        nombre_meses = [_datos.MESES[int(m) - 1] for m in meses_top]
        fmin, fmax = int(fe["fechas"].min()), int(fe["fechas"].max())
        resp = (
            "Promedios por tipo de día: " + "; ".join(partes) + ". "
            f"El % del total de cada tipo de día depende de cuántas horas tiene (hay más días laborales que festivos), por eso se "
            f"compara con el promedio por hora. Perfil horario: {txt_pico}. "
        )
        if len(fe):
            resp += (f"Entre festivos, los de mayor promedio son {lista(top3['nombre_festivo'])} y los de menor {lista(bajos['nombre_festivo'])}; "
                     f"cada festivo tiene solo entre {fmin} y {fmax} fechas, por lo que el orden es indicativo.")
        return {
            "pregunta": self.PREGUNTAS["P2"],
            "metrica": (
                "**Promedio de alquileres por hora** de cada tipo de día = alquileres del grupo ÷ horas observadas del grupo "
                "[alquileres/hora]; **% del total** = alquileres del grupo ÷ alquileres de la selección × 100 [%]. "
                "Tipos de día: laboral, fin de semana y festivo (el nombre del festivo viene del enriquecimiento con LLM)."),
            "kpis": kpis[:3],
            "respuesta": resp,
            "conclusion": ("El perfil horario cambia según el tipo de día: " + txt_pico + "."),
            "decision": (
                "Redistribución: en días laborales reforzar antes de los picos de desplazamiento; en fines de semana y festivos, "
                "reforzar el mediodía y la tarde. Mantenimiento: las ventanas de baja demanda cambian con el tipo de día (ver el perfil "
                "horario). Promociones: los casuales pesan más en fines de semana y festivos (P5), buen momento para ofertas a "
                "ese segmento."),
            "info": ("Calendario completo de eventos locales (conciertos, desfiles, competencias) y datos por estación de bicicletas "
                     "para saber dónde se concentra la demanda de fin de semana; más años para tener más fechas por festivo."),
            "limitaciones": (
                f"Los {numero(len(top3))} festivos de mayor promedio caen en {lista(nombre_meses)}: la tabla de festivos mezcla el efecto del "
                "festivo con el de la estación y el clima, así que no se puede atribuir la diferencia al festivo en sí. "
                f"Cada festivo tiene solo entre {fmin} y {fmax} fechas. "
                f"El nombre y el tipo de festivo dependen de una clasificación hecha con un LLM (validada contra el calendario federal). "
                f"Hay pocas horas de festivo ({numero(self._horas_tipo('Festivo'))}) frente a las de días laborales ({numero(self._horas_tipo('Laboral'))})."),
            "tablas": {"P2 · Por tipo de día": td, "P2 · Por tipo de día y franja": tdf, "P2 · Por festivo": fe},
        }

    def _horas_tipo(self, tipo: str) -> int:
        return int((self.df["tipo_dia"].astype(str) == tipo).sum())

    def tabla_pico_por_tipo_dia(self) -> dict:
        """Hora de mayor promedio de cada tipo de día: {tipo: (hora, promedio)}."""
        t = _m.resumen_demanda(self.df, ["tipo_dia", "hora"])
        out = {}
        for tipo, g in t.groupby("tipo_dia", observed=True):
            r = g.loc[g[PH].idxmax()]
            out[str(tipo)] = (int(r["hora"]), float(r[PH]))
        return out

    # ---------------------------------------------------------------- P3
    def _c_p3(self) -> dict:
        es, ms = self.tabla("P3_demanda_por_estacion"), self.tabla("P3_demanda_por_mes")
        e_hi, e_lo = es.loc[es[PH].idxmax()], es.loc[es[PH].idxmin()]
        m_hi, m_lo = ms.loc[ms[PH].idxmax()], ms.loc[ms[PH].idxmin()]
        m_tot = ms.loc[ms["total_alquileres"].idxmax()]
        razon = e_hi[PH] / e_lo[PH]
        top3 = ms.nlargest(3, PH)["nombre_mes"].astype(str)
        bot3 = ms.nsmallest(3, PH)["nombre_mes"].astype(str)
        # el crecimiento entre años mezclado con la estacionalidad
        pe = _m.resumen_demanda(self.df_p6, ["estacion", "anio"])
        txt_mezcla = ""
        if {2011, 2012} <= set(pe["anio"]):
            p = pe.pivot(index="estacion", columns="anio", values=PH)
            txt_mezcla = "; ".join(f"{e}: {numero(p.loc[e, 2011], 1)} en 2011 y {numero(p.loc[e, 2012], 1)} en 2012" for e in p.index)
        resp = (
            f"La estación con mayor promedio es {e_hi['estacion']} ({numero(e_hi[PH], 1)} alquileres por hora; {pct(e_hi['pct_del_total_alquileres'])} "
            f"de los {self.base_txt()}) y la de menor es {e_lo['estacion']} ({numero(e_lo[PH], 1)}): la más alta es {numero(razon, 1)} veces la más baja. "
            f"Por mes, los promedios más altos son {lista(f'{m} ({numero(v, 1)})' for m, v in zip(ms.nlargest(3, PH)['nombre_mes'], ms.nlargest(3, PH)[PH]))} "
            f"y los más bajos {lista(f'{m} ({numero(v, 1)})' for m, v in zip(ms.nsmallest(3, PH)['nombre_mes'], ms.nsmallest(3, PH)[PH]))}. "
            f"En totales, {m_tot['nombre_mes']} es el mes que más suma ({numero(m_tot['total_alquileres'])} alquileres; "
            f"{pct(m_tot['pct_del_total_alquileres'])} de los {self.base_txt()}), pero los totales mensuales dependen de las horas observadas de cada mes "
            f"({numero(ms['horas_observadas'].min())} a {numero(ms['horas_observadas'].max())} horas), por eso el ranking usa el promedio por hora."
        )
        return {
            "pregunta": self.PREGUNTAS["P3"],
            "metrica": (
                "**Promedio de alquileres por hora** de la estación o del mes = alquileres del grupo ÷ horas observadas del grupo "
                "[alquileres/hora]; **% del total** = alquileres del grupo ÷ alquileres de la selección × 100 [%]. "
                "Se agrupan los dos años juntos."),
            "kpis": [
                ("Estación con más demanda por hora", f"{numero(e_hi[PH], 1)} alq/h", str(e_hi["estacion"]),
                 "Promedio de alquileres por hora de la estación más alta."),
                ("Mes con más demanda por hora", f"{numero(m_hi[PH], 1)} alq/h", str(m_hi["nombre_mes"]),
                 "Promedio de alquileres por hora del mes más alto."),
                ("Mes con menos demanda por hora", f"{numero(m_lo[PH], 1)} alq/h", str(m_lo["nombre_mes"]),
                 "Promedio de alquileres por hora del mes más bajo."),
            ],
            "respuesta": resp,
            "conclusion": (f"La demanda es estacional: {e_hi['estacion']} promedia {numero(e_hi[PH], 1)} alquileres por hora y "
                           f"{e_lo['estacion']} {numero(e_lo[PH], 1)}; los meses más altos son {lista(top3)} y los más bajos {lista(bot3)}."),
            "decision": (
                f"Flota: dimensionar para los meses de mayor promedio ({lista(top3)}). Mantenimiento mayor (revisiones largas, "
                f"reparaciones): concentrarlo en los meses de menor promedio ({lista(bot3)}). Personal: planificar turnos por temporada."),
            "info": ("Datos de más años para separar estacionalidad de crecimiento; cantidad de bicicletas y de estaciones disponibles "
                     "en cada mes (la oferta puede haber cambiado); calendario de eventos de la ciudad."),
            "limitaciones": (
                "El crecimiento de 2011 a 2012 (P6) se mezcla con la estacionalidad: "
                + (f"por estación, el promedio por hora fue {txt_mezcla}. " if txt_mezcla else "")
                + "Algunos meses mezclan estaciones (los meses de cambio de estación). Son solo dos años y una ciudad, y el clima, que "
                  "también cambia con la estación, no está controlado."),
            "tablas": {"P3 · Por estación": es, "P3 · Por mes": ms},
        }

    # ---------------------------------------------------------------- P4
    def _c_p4(self) -> dict:
        cl, tm = self.tabla("P4_demanda_por_situacion_climatica"), self.tabla("P4_demanda_por_rango_de_temperatura")
        ref = cl.iloc[0] if len(cl) else None  # primera categoría presente (despejado si está en la selección)
        partes, kpis = [], []
        pocas = []
        for _, r in cl.iterrows():
            chica = r["horas_observadas"] < MIN_HORAS
            if chica:
                pocas.append(r)
            d = "" if r["clima"] == ref["clima"] else f", {numero(self._delta(r[PH], ref[PH]), 1)} % vs. {str(ref['clima']).lower()}"
            aviso = (f" [solo {numero(r['horas_observadas'])} horas y {numero(r['total_alquileres'])} alquileres: no permite concluir]"
                     if chica else "")
            partes.append(f"{r['clima']}: {numero(r[PH], 1)} alquileres por hora ({numero(r['horas_observadas'])} horas, "
                          f"{pct(r['pct_del_total_alquileres'])} de los {self.base_txt()}{d}){aviso}")
        validas = cl[cl["horas_observadas"] >= MIN_HORAS]
        for _, r in validas.head(2).iterrows():
            dl = None if r["clima"] == ref["clima"] else f"{numero(self._delta(r[PH], ref[PH]), 1)} % vs. {str(ref['clima']).lower()}"
            kpis.append((str(r["clima"]), f"{numero(r[PH], 1)} alq/h", dl, f"Promedio de alquileres por hora ({numero(r['horas_observadas'])} horas)."))
        crece = bool(np.all(np.diff(tm[PH].to_numpy()) > 0)) if len(tm) > 1 else False
        frio, calor = tm.iloc[0], tm.iloc[-1]
        kpis.append((f"{calor['rango_temperatura']}", f"{numero(calor[PH], 1)} alq/h",
                     f"{numero(calor[PH] / frio[PH], 1)} veces {str(frio['rango_temperatura'])}" if frio['rango_temperatura'] != calor['rango_temperatura'] else None,
                     f"Promedio de alquileres por hora en el rango {calor['rango_temperatura']} ({numero(calor['horas_observadas'])} horas)."))
        txt_t = "; ".join(f"{r['rango_temperatura']}: {numero(r[PH], 1)} ({numero(r['horas_observadas'])} horas)" for _, r in tm.iterrows())
        resp = ("Por situación climática: " + "; ".join(partes) + ". "
                f"Por rango de temperatura (alquileres por hora): {txt_t}. "
                + ("El promedio crece en cada rango más cálido. " if crece else "El promedio no crece de forma continua entre rangos. ")
                + "Los porcentajes del total reflejan cuántas horas tiene cada condición (el clima despejado es el más frecuente), "
                  "no cuánto rinde cada hora; el promedio por hora permite la comparación.")
        return {
            "pregunta": self.PREGUNTAS["P4"],
            "metrica": (
                "**Promedio de alquileres por hora** por situación climática y por rango de temperatura (`temp_c`, °C) = alquileres "
                "del grupo ÷ horas observadas del grupo [alquileres/hora]; **% del total** = alquileres del grupo ÷ alquileres de la "
                f"selección × 100 [%]. Los grupos con menos de {MIN_HORAS} horas observadas se marcan en gris y no se interpretan."),
            "kpis": kpis[:3],
            "respuesta": resp,
            "conclusion": ("La demanda por hora baja con clima adverso y sube con el calor: " + "; ".join(partes[:2]) + "."),
            "decision": (
                "Mantenimiento: aprovechar los días de frío o lluvia (menor uso esperado) para revisiones; Flota y personal: reforzar "
                "la oferta en días cálidos y despejados. Con un pronóstico del tiempo se puede anticipar la redistribución del día siguiente."),
            "info": ("Pronóstico y observación del clima a nivel de hora y de zona de la ciudad; datos de humedad y viento con unidad "
                     "documentada (la del viento no lo está); más años con eventos extremos para poder decir algo de lluvia fuerte o tormenta."),
            "limitaciones": (
                "La temperatura está muy ligada a la estación y a la hora del día, así que esta relación es una asociación, no una causa, "
                "y no está ajustada por esos factores. "
                + (f"{lista(str(r['clima']) for r in pocas)}: solo {lista(numero(r['horas_observadas']) for r in pocas)} horas observadas, "
                   "no se sacan conclusiones de esas categorías. " if pocas else "")
                + "El clima viene normalizado y se llevó a unidades originales con las fórmulas de UCI."),
            "tablas": {"P4 · Por situación climática": cl, "P4 · Por rango de temperatura": tm},
        }

    # ---------------------------------------------------------------- P5
    DIMENSIONES = {"Tipo de día": "tipo_dia", "Franja horaria": "franja_horaria",
                   "Estación": "estacion", "Tipo de festivo": "tipo_festivo"}

    def _dim5(self, dimension: str) -> pd.DataFrame:
        p5 = self.tabla("P5_participacion_casuales_y_registrados")
        return p5[p5["dimension"] == dimension].reset_index(drop=True)

    def _c_p5(self) -> dict:
        p5 = self.tabla("P5_participacion_casuales_y_registrados")
        tot = p5[p5["dimension"] == "Total"].iloc[0]
        d_td, d_fr = self._dim5("Tipo de día"), self._dim5("Franja horaria")
        d_es, d_fe = self._dim5("Estación"), self._dim5("Tipo de festivo")
        horas_fe = self.horas_por("tipo_festivo")
        chicos = [c for c in d_fe["categoria"] if horas_fe.get(c, 0) < MIN_HORAS and c != "No festivo"]

        def extremos(d, nombre):
            hi, lo = d.loc[d["pct_casuales"].idxmax()], d.loc[d["pct_casuales"].idxmin()]
            return (f"{nombre}: más casuales en {hi['categoria']} ({pct(hi['pct_casuales'])} de {numero(hi['alquileres_total'])} alquileres) "
                    f"y menos en {lo['categoria']} ({pct(lo['pct_casuales'])} de {numero(lo['alquileres_total'])} alquileres)")

        txt_td = "; ".join(f"{r['categoria']} {pct(r['pct_casuales'])} (de {numero(r['alquileres_total'])})" for _, r in d_td.iterrows())
        fe_txt = "; ".join(f"{r['categoria']} {pct(r['pct_casuales'])} (de {numero(r['alquileres_total'])})" for _, r in d_fe.iterrows())
        resp = (
            f"En la selección, los casuales son el {pct(tot['pct_casuales'])} y los registrados el {pct(tot['pct_registrados'])} de los "
            f"{self.base_txt()}. Cada porcentaje se calcula sobre los alquileres de su propia categoría (se indican entre paréntesis), "
            f"no sobre el total. Por tipo de día, % de casuales: {txt_td}. {extremos(d_fr, 'Por franja horaria')}. "
            f"{extremos(d_es, 'Por estación')}. Por tipo de festivo: {fe_txt}"
            + (f" ({lista(chicos)} con solo {lista(numero(horas_fe[c]) for c in chicos)} horas observadas: no se interpreta)." if chicos else ".")
        )
        lab, fin = self._fila(d_td, "categoria", "Laboral"), self._fila(d_td, "categoria", "Fin de semana")
        hi = d_fr.loc[d_fr["pct_casuales"].idxmax()]
        kpis = [
            ("% de casuales", pct(tot["pct_casuales"], 1), f"de {numero(tot['alquileres_total'])} alquileres",
             "Casuales ÷ alquileres totales de la selección × 100."),
            ("% de casuales · fin de semana", pct(fin["pct_casuales"], 1) if fin is not None else "s/d",
             f"de {numero(fin['alquileres_total'])} alquileres" if fin is not None else None,
             "Casuales ÷ alquileres de las horas de fin de semana × 100."),
            ("% de casuales · laboral", pct(lab["pct_casuales"], 1) if lab is not None else "s/d",
             f"de {numero(lab['alquileres_total'])} alquileres" if lab is not None else None,
             "Casuales ÷ alquileres de las horas de días laborales × 100."),
        ]
        return {
            "pregunta": self.PREGUNTAS["P5"],
            "metrica": (
                "**% de casuales** = `usuarios_casuales` de la categoría ÷ `alquileres_total` de la categoría × 100 [%]; "
                "**% de registrados** = 100 − % de casuales [%]. La base de cada porcentaje es el total de alquileres de esa categoría, "
                "no el total de la selección. Se mide en alquileres, no en personas. Dimensiones: tipo de día, franja horaria, "
                "estación y tipo de festivo."),
            "kpis": kpis,
            "respuesta": resp,
            "conclusion": (f"Los registrados dominan ({pct(tot['pct_registrados'], 1)} de los {self.base_txt()}); los casuales pesan más en "
                           f"fines de semana y festivos y en {hi['categoria']} ({pct(hi['pct_casuales'], 1)})."),
            "decision": (
                "Promociones y campañas a casuales: concentrarlas en fines de semana, mediodías y estaciones templadas, donde pesan más. "
                "Operación: los registrados mueven las horas pico de días laborales, por lo que ahí se prioriza la disponibilidad "
                "confiable; los casuales mueven mediodías y fines de semana."),
            "info": ("Tipo de viaje y motivo (trabajo, paseo), duración y recorrido; cantidad de personas distintas (el dataset cuenta "
                     "alquileres, no usuarios); qué define a un usuario casual frente a uno registrado en la fuente."),
            "limitaciones": (
                "La participación se mide en alquileres y no en personas. No hay motivo del viaje, así que decir 'desplazamiento' o "
                "'recreación' es una hipótesis, no un dato. "
                + (f"{lista(chicos)} tiene solo {lista(numero(horas_fe[c]) for c in chicos)} horas observadas (pocas fechas). " if chicos else "")
                + "Las categorías de tipo de día y franja se solapan (un fin de semana contiene todas las franjas), así que no son independientes."),
            "tablas": {"P5 · Participación de casuales y registrados": p5},
        }

    # ---------------------------------------------------------------- P6
    def _c_p6(self) -> dict:
        p6 = self.tabla_p6()
        if p6 is None:
            vacio = "La selección no deja horas de ambos años, así que no se puede comparar 2011 con 2012."
            return {"pregunta": self.PREGUNTAS["P6"], "metrica": "", "kpis": [], "respuesta": vacio, "conclusion": vacio,
                    "decision": vacio, "info": "", "limitaciones": "", "tablas": {}}
        t = p6.set_index("tipo_usuario")
        T, C, R = t.loc["Total"], t.loc["Casuales"], t.loc["Registrados"]
        part11, part12 = C["total_2011"] / T["total_2011"] * 100, C["total_2012"] / T["total_2012"] * 100
        dif_h = int(T["horas_observadas_2012"] - T["horas_observadas_2011"])
        pe = _m.resumen_demanda(self.df_p6, ["estacion", "anio"]).pivot(index="estacion", columns="anio", values=PH)
        var_e = (pe[2012] / pe[2011] - 1) * 100
        resp = (
            f"El total de alquileres pasó de {numero(T['total_2011'])} en 2011 a {numero(T['total_2012'])} en 2012 "
            f"({pct(T['variacion_total_pct'])}); los casuales variaron {pct(C['variacion_total_pct'])} (de {numero(C['total_2011'])} a "
            f"{numero(C['total_2012'])}) y los registrados {pct(R['variacion_total_pct'])} (de {numero(R['total_2011'])} a {numero(R['total_2012'])}). "
            f"El promedio por hora pasó de {numero(T['promedio_por_hora_2011'], 1)} a {numero(T['promedio_por_hora_2012'], 1)} "
            f"({pct(T['variacion_promedio_pct'])}). La variación del promedio difiere de la del total porque 2012 tiene "
            f"{numero(abs(dif_h))} horas observadas {'más' if dif_h >= 0 else 'menos'} que 2011 ({numero(T['horas_observadas_2012'])} frente a "
            f"{numero(T['horas_observadas_2011'])}); por eso se compara por hora. Los casuales pasaron de ser el {pct(part11)} de los "
            f"{numero(T['total_2011'])} alquileres de 2011 al {pct(part12)} de los {numero(T['total_2012'])} de 2012."
        )
        return {
            "pregunta": self.PREGUNTAS["P6"],
            "metrica": (
                "**Variación del total** = (total 2012 − total 2011) ÷ total 2011 × 100 [%], con totales en alquileres. "
                "**Variación del promedio por hora** = (promedio 2012 − promedio 2011) ÷ promedio 2011 × 100 [%], con promedios en "
                "alquileres/hora (corrige que los dos años no tienen las mismas horas observadas). "
                "Ignora el filtro de año; sí respeta los demás filtros."),
            "kpis": [
                ("Variación del total", pct(T["variacion_total_pct"], 1), f"{numero(T['total_2011'])} → {numero(T['total_2012'])} alquileres",
                 "(total 2012 − total 2011) ÷ total 2011 × 100."),
                ("Variación del promedio por hora", pct(T["variacion_promedio_pct"], 1),
                 f"{numero(T['promedio_por_hora_2011'], 1)} → {numero(T['promedio_por_hora_2012'], 1)} alq/h",
                 "Compara por hora observada, no por total."),
                ("Horas observadas", f"{numero(T['horas_observadas_2011'])} → {numero(T['horas_observadas_2012'])}", "2011 → 2012",
                 "Horas con registro en cada año dentro de la selección."),
            ],
            "respuesta": resp,
            "conclusion": (f"La demanda por hora creció {pct(T['variacion_promedio_pct'], 1)} de 2011 a 2012 "
                           f"({pct(C['variacion_promedio_pct'], 1)} casuales, {pct(R['variacion_promedio_pct'], 1)} registrados)."),
            "decision": (
                "Planificación de flota y personal: no basarse solo en el comportamiento de 2011, la demanda base es mayor en 2012. "
                "Como los registrados crecieron más, el crecimiento se concentra en los viajes de rutina (horas pico de días laborales): "
                "reforzar la disponibilidad ahí."),
            "info": ("Cantidad de bicicletas y de estaciones por año (¿creció la oferta?), número de socios, precios y campañas, y datos "
                     "de 2013 en adelante para saber si el crecimiento continúa o se estabiliza."),
            "limitaciones": (
                "El dataset no explica el crecimiento: puede deberse a la expansión del servicio, a más socios o a otros factores. "
                "Son solo dos años de una sola ciudad. El crecimiento se mezcla con la estacionalidad: la variación del promedio por hora "
                f"va de {pct(var_e.min(), 1)} ({var_e.idxmin()}) a {pct(var_e.max(), 1)} ({var_e.idxmax()}) según la estación. "
                "El análisis es descriptivo, no causal."),
            "tablas": {"P6 · Variación 2011-2012": p6},
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

    def _sub(self, con_pct: bool = False, nota: str | None = None) -> list[str]:
        partes = [f"Período: {self.periodo()}", f"{numero(self.horas)} horas observadas"]
        if self.filtros_activos():
            partes.append("selección filtrada")
        if con_pct:
            partes.append(f"% sobre {self.base_txt()}")
        if nota:
            partes.append(nota)
        return self._partir(partes, self.MAX_SUB)

    def _estilo(self, fig: go.Figure, titulo: str, x: str | None, y: str | None, alto: int = 380,
                leyenda: bool = False, con_pct: bool = False, sub: list[str] | str | None = None,
                nota: str | None = None, extra_arriba: int = 0, ancho_leyenda: int = 0) -> go.Figure:
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
        fig.update_yaxes(gridcolor=self.REJILLA, zeroline=False, linecolor=self.REJILLA, title_text=y, title_standoff=8, automargin=True)
        return fig

    def _colores(self, valores, minimo: bool = True, pocas=None) -> list[str]:
        """Azul para todos, naranja para el máximo, violeta para el mínimo y gris para categorías con pocas horas."""
        v = np.asarray(valores, dtype=float)
        ok = np.ones(len(v), dtype=bool) if pocas is None else ~np.asarray(pocas, dtype=bool)
        col = [self.AZUL] * len(v)
        if ok.any():
            imax = int(np.argmax(np.where(ok, v, -np.inf)))
            col[imax] = self.MAXIMO
            if minimo and ok.sum() > 1:
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
        t = self.tabla("P1_demanda_por_franja_horaria")
        if t.empty:
            return self._vacia("P1 · Demanda por franja horaria")
        txt = [f"{numero(p, 1)}<br>{pct(s, 1)}" for p, s in zip(t[PH], t["pct_del_total_alquileres"])]
        fig = self._barras_h([self._franja(f) for f in t["franja_horaria"]], t[PH], txt, self._hover_base(),
                             "P1 · Alquileres por hora según la franja horaria", self.Y_PH, self._colores(t[PH]),
                             nota="etiqueta: promedio por hora y % del total")
        fig.data[0].customdata = np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1)
        return fig

    def figura_hora(self) -> go.Figure:
        t = self.tabla("P1_demanda_por_hora_del_dia")
        if t.empty:
            return self._vacia("P1 · Demanda por hora del día")
        fig = go.Figure(go.Bar(
            x=t["hora"], y=t[PH], marker={"color": self._colores(t[PH])},
            customdata=np.stack([t["horas_observadas"], t["pct_del_total_alquileres"], t["total_alquileres"]], axis=-1),
            hovertemplate=("%{x} h<br>Promedio: %{y:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                           "<br>Total: %{customdata[2]:,} alquileres<br>%{customdata[1]:.1f} % de los " + self.base_txt() + "<extra></extra>")))
        fig.update_xaxes(range=[-0.5, 23.5], tickmode="linear", tick0=0, dtick=2, tickangle=0)
        return self._estilo(fig, "P1 · Alquileres por hora del día", "Hora del día [h, 0 a 23]", self.Y_PH, con_pct=True,
                            nota="naranja = máximo, violeta = mínimo", alto=400)

    def figura_mapa_calor(self) -> go.Figure:
        """Día de la semana × hora: promedio de alquileres por hora (dónde se concentra la demanda)."""
        t = _m.resumen_demanda(self.df, ["dia_semana", "hora"])
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
        t = self.tabla("P2_demanda_por_tipo_de_dia")
        if t.empty:
            return self._vacia("P2 · Demanda por tipo de día")
        txt = [f"{numero(p, 1)}<br>{pct(s, 1)}" for p, s in zip(t[PH], t["pct_del_total_alquileres"])]
        fig = self._barras_h(t["tipo_dia"].astype(str), t[PH], txt, self._hover_base(),
                             "P2 · Alquileres por hora según el tipo de día", self.Y_PH, self._colores(t[PH], minimo=False),
                             alto=62 * len(t) + 130, nota="etiqueta: promedio por hora y % del total")
        fig.data[0].customdata = np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1)
        return fig

    def figura_perfil_tipo_dia(self) -> go.Figure:
        t = _m.resumen_demanda(self.df, ["tipo_dia", "hora"])
        if t.empty:
            return self._vacia("P2 · Perfil horario por tipo de día")
        fig = go.Figure()
        for tipo, g in t.groupby("tipo_dia", observed=True):
            g = g.sort_values("hora")
            fig.add_trace(go.Scatter(
                x=g["hora"], y=g[PH], mode="lines+markers", name=str(tipo), line={"color": self.COLOR_TIPO_DIA.get(str(tipo), self.AZUL), "width": 2},
                customdata=g["horas_observadas"],
                hovertemplate=f"<b>{tipo}</b> · %{{x}} h<br>Promedio: %{{y:,.1f}} alquileres/hora<br>Horas observadas: %{{customdata:,}}<extra></extra>"))
        fig.update_xaxes(range=[-0.5, 23.5], tickmode="linear", tick0=0, dtick=2, tickangle=0)
        return self._estilo(fig, "P2 · Perfil horario según el tipo de día", "Hora del día [h, 0 a 23]", self.Y_PH,
                            leyenda=True, alto=420)

    def figura_tipo_dia_franja(self) -> go.Figure:
        t = self.tabla("P2_demanda_por_tipo_de_dia_y_franja")
        if t.empty:
            return self._vacia("P2 · Tipo de día × franja horaria")
        z = t.pivot(index="tipo_dia", columns="franja_horaria", values=PH)
        z = z.reindex(index=[x for x in TIPOS_DIA if x in z.index], columns=[c for c in _datos.ORDEN_FRANJAS if c in z.columns])
        fig = go.Figure(go.Heatmap(
            z=z.values, x=[self._franja(c, 11) for c in z.columns], y=list(z.index.astype(str)), colorscale=self.ESCALA, xgap=2, ygap=2,
            text=[[numero(v, 1) for v in fila] for fila in z.values], texttemplate="%{text}", textfont={"size": 12},
            hovertemplate="<b>%{y}</b> · %{x}<br>Promedio: %{z:,.1f} alquileres/hora<extra></extra>",
            colorbar={"title": {"text": "Alquileres/hora", "side": "right"}, "thickness": 14, "len": 0.95,
                      "lenmode": "fraction", "nticks": 5}))
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(tickangle=0, tickfont={"size": 11})
        return self._estilo(fig, "P2 · Tipo de día × franja horaria", None, None, alto=420, nota="promedio de alquileres por hora")

    def figura_festivos(self) -> go.Figure:
        t = self.tabla("P2_demanda_por_festivo")
        if t.empty:
            return self._vacia("P2 · Demanda por festivo")
        td = self._fila(self.tabla("P2_demanda_por_tipo_de_dia"), "tipo_dia", "Laboral")
        etq = [f"{n} ({int(f)} {'fecha' if f == 1 else 'fechas'})" for n, f in zip(t["nombre_festivo"], t["fechas"])]
        hover = ("<b>%{y}</b><br>Promedio: %{x:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                 "<br>Total: %{customdata[1]:,} alquileres<br>Casuales: %{customdata[2]:.1f} % de los alquileres del festivo<extra></extra>")
        fig = self._barras_h(etq, t[PH], [numero(v, 1) for v in t[PH]], hover,
                             "P2 · Demanda por festivo", "Promedio [alquileres/hora]", self._colores(t[PH]),
                             alto=30 * len(t) + 140, ref=None if td is None else td[PH],
                             ref_texto="" if td is None else f"día laboral: {numero(td[PH], 1)}", con_pct=False,
                             nota="indicativo: pocas fechas por festivo")
        fig.data[0].customdata = np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_casuales"]], axis=-1)
        return fig

    # ================================================================== P3
    def figura_estacion(self) -> go.Figure:
        t = self.tabla("P3_demanda_por_estacion")
        if t.empty:
            return self._vacia("P3 · Demanda por estación")
        txt = [f"{numero(p, 1)}<br>{pct(s, 1)}" for p, s in zip(t[PH], t["pct_del_total_alquileres"])]
        fig = go.Figure(go.Bar(
            x=t["estacion"].astype(str), y=t[PH], marker={"color": self._colores(t[PH])}, text=txt, textposition="outside",
            textfont={"size": 12, "color": self.TEXTO}, cliponaxis=False, width=0.55,
            customdata=np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1),
            hovertemplate=("<b>%{x}</b><br>Promedio: %{y:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                           "<br>Total: %{customdata[1]:,} alquileres<br>%{customdata[2]:.1f} % de los " + self.base_txt() + "<extra></extra>")))
        fig.update_yaxes(range=[0, float(t[PH].max()) * 1.3])
        fig.update_layout(uniformtext={"minsize": 12, "mode": "show"})
        return self._estilo(fig, "P3 · Alquileres por hora según la estación", "Estación del año", self.Y_PH, con_pct=True,
                            nota="etiqueta: promedio y % del total", alto=400)

    def figura_meses_ranking(self) -> go.Figure:
        t = self.tabla("P3_demanda_por_mes")
        if t.empty:
            return self._vacia("P3 · Ranking de meses")
        r = t.sort_values(PH, ascending=False).reset_index(drop=True)
        fig = self._barras_h(r["nombre_mes"].astype(str), r[PH], [numero(v, 1) for v in r[PH]], self._hover_base(),
                             "P3 · Meses de mayor a menor demanda por hora", self.Y_PH, self._colores(r[PH]),
                             alto=30 * len(r) + 140, con_pct=True, nota="naranja = máximo, violeta = mínimo")
        fig.data[0].customdata = np.stack([r["horas_observadas"], r["total_alquileres"], r["pct_del_total_alquileres"]], axis=-1)
        return fig

    def figura_mes_linea(self) -> go.Figure:
        t = self.tabla("P3_demanda_por_mes").sort_values("mes")
        if t.empty:
            return self._vacia("P3 · Demanda mensual")
        fig = go.Figure(go.Scatter(
            x=t["nombre_mes"].astype(str), y=t[PH], mode="lines+markers", line={"color": self.AZUL, "width": 2},
            marker={"size": 9, "color": self._colores(t[PH])},
            customdata=np.stack([t["horas_observadas"], t["total_alquileres"]], axis=-1),
            hovertemplate=("<b>%{x}</b><br>Promedio: %{y:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                           "<br>Total: %{customdata[1]:,} alquileres<extra></extra>")))
        fig.update_yaxes(rangemode="tozero")
        return self._estilo(fig, "P3 · Alquileres por hora a lo largo del año", "Mes del año", self.Y_PH,
                            nota="promedio por mes")

    # ================================================================== P4
    def figura_clima(self) -> go.Figure:
        t = self.tabla("P4_demanda_por_situacion_climatica")
        if t.empty:
            return self._vacia("P4 · Demanda por situación climática")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        etq = ["<br>".join(textwrap.wrap(str(c), 22)) + (" *" if p else "") for c, p in zip(t["clima"], pocas)]
        txt = [(f"{numero(p, 1)}<br>solo {numero(h)} h: sin conclusión" if poca else f"{numero(p, 1)}<br>{numero(h)} h")
               for p, h, poca in zip(t[PH], t["horas_observadas"], pocas)]
        nota = f"* gris = menos de {numero(MIN_HORAS)} horas: no se interpreta" if pocas.any() else "etiqueta: promedio y horas"
        fig = self._barras_h(etq, t[PH], txt, self._hover_base(), "P4 · Alquileres por hora según el clima",
                             self.Y_PH, self._colores(t[PH], minimo=False, pocas=pocas), alto=76 * len(t) + 130,
                             con_pct=True, nota=nota)
        fig.data[0].customdata = np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1)
        return fig

    def figura_temperatura(self) -> go.Figure:
        t = self.tabla("P4_demanda_por_rango_de_temperatura")
        if t.empty:
            return self._vacia("P4 · Demanda por temperatura")
        pocas = (t["horas_observadas"] < MIN_HORAS).to_numpy()
        txt = [f"{numero(p, 1)}<br>{numero(h)} h" for p, h in zip(t[PH], t["horas_observadas"])]
        fig = go.Figure(go.Bar(
            x=[self._franja(r, 11) for r in t["rango_temperatura"]], y=t[PH], marker={"color": self._colores(t[PH], minimo=False, pocas=pocas)},
            text=txt, textposition="outside", textfont={"size": 12, "color": self.TEXTO}, cliponaxis=False, width=0.55,
            customdata=np.stack([t["horas_observadas"], t["total_alquileres"], t["pct_del_total_alquileres"]], axis=-1),
            hovertemplate=("<b>%{x}</b><br>Promedio: %{y:,.1f} alquileres/hora<br>Horas observadas: %{customdata[0]:,}"
                           "<br>Total: %{customdata[1]:,} alquileres<br>%{customdata[2]:.1f} % de los " + self.base_txt() + "<extra></extra>")))
        fig.update_yaxes(range=[0, float(t[PH].max()) * 1.3])
        fig.update_xaxes(tickangle=0)
        fig.update_layout(uniformtext={"minsize": 12, "mode": "show"})
        return self._estilo(fig, "P4 · Alquileres por hora según la temperatura", "Rango de temperatura [°C]", self.Y_PH,
                            con_pct=True, nota="etiqueta: promedio y horas observadas", alto=400)

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
        base = [f"de {numero(a)} alquileres" for a in d["alquileres_total"]]
        txt = lambda serie: [pct(v, 1) if v >= 14 else "" for v in serie]   # los segmentos angostos no llevan etiqueta
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
        fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix=" %")
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
        p6 = self.tabla_p6()
        if p6 is None:
            return self._vacia(titulo)
        etq = [f"{u}<br>({'+' if v >= 0 else ''}{numero(v, 1)} %)" for u, v in zip(p6["tipo_usuario"], p6[variacion])]
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
        sub = [f"Período: {self.periodo()}", f"{numero(self.horas)} horas observadas"]
        if self.filtros_activos():
            sub.append("selección filtrada")
        sub += ["variación 2011 → 2012 entre paréntesis", "el filtro de año no se aplica"]
        return self._estilo(fig, titulo, "Tipo de usuario", y, leyenda=True, sub=sub, alto=430)

    def figura_variacion_totales(self) -> go.Figure:
        return self._fig_p6("total_2011", "total_2012", "P6 · Alquileres totales por año", "Total [alquileres]",
                            "variacion_total_pct", 0, rotar=True)

    def figura_variacion_promedio(self) -> go.Figure:
        return self._fig_p6("promedio_por_hora_2011", "promedio_por_hora_2012", "P6 · Alquileres por hora por año",
                            self.Y_PH, "variacion_promedio_pct", 1)

    def figura_mes_por_anio(self) -> go.Figure:
        t = _m.resumen_demanda(self.df_p6, ["anio", "mes", "nombre_mes"])
        if t.empty:
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
                            leyenda=True, nota="el filtro de año no se aplica")

    # ================================================================== contexto (población completa)
    def figura_contexto_serie(self) -> go.Figure:
        t = self.df_total
        m = t.groupby(t["fecha"].dt.to_period("M")).agg(total=("alquileres_total", "sum"), horas=("alquileres_total", "size")).reset_index()
        m["fecha"] = m["fecha"].dt.to_timestamp()
        fig = go.Figure(go.Bar(x=m["fecha"], y=m["total"], marker_color=self.AZUL, customdata=m["horas"],
                               hovertemplate="%{x|%m/%Y}<br>Total: %{y:,} alquileres<br>Horas observadas: %{customdata:,}<extra></extra>"))
        fig.update_xaxes(tickformat="%m/%Y")
        sub = [f"Período: {t['fecha'].min():%d/%m/%Y} a {t['fecha'].max():%d/%m/%Y}", "dataset completo",
               f"{numero(len(t))} horas, {numero(self.total_dataset)} alquileres"]
        return self._estilo(fig, "Contexto · Alquileres por mes (suma)", "Mes", "Total del mes [alquileres]", sub=sub)

    def figura_contexto_histograma(self) -> go.Figure:
        t = self.df_total
        fig = go.Figure(go.Histogram(x=t["alquileres_total"], nbinsx=40, marker_color=self.AZUL,
                                     hovertemplate="%{x} alquileres en la hora<br>%{y:,} horas<extra></extra>"))
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
                    [("perfil", self.figura_perfil_tipo_dia()), ("festivos", self.figura_festivos())],
                    [("calor", self.figura_mapa_calor())]]
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
                f"{'; '.join(act)}. La selección tiene {numero(k['Horas observadas'])} horas y {numero(k['Alquileres totales'])} alquileres "
                f"({pct(k['% del dataset completo'])} de los {numero(self.total_dataset)} alquileres del dataset). "
                "Los porcentajes del total se calculan sobre los alquileres de la selección.")
        else:
            st.caption("Sin filtros activos: las cifras coinciden con las tablas de AE3 (`AE3/resultados/tablas`). "
                       f"Los porcentajes del total se calculan sobre los {numero(self.total)} alquileres del dataset.")
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
        if act:
            st.caption(f"Cifras de la selección ({'; '.join(act)}): {numero(self.horas)} horas y {self.base_txt()}.")
        else:
            st.caption(f"Sin filtros: cifras del dataset completo, iguales a las tablas de AE3 ({self.base_txt().replace(' de la selección', '')}).")
        if not c["kpis"]:
            st.warning(c["respuesta"])
            return
        st.markdown(f"**Métrica.** {c['metrica']}")
        cols = st.columns(len(c["kpis"]))
        for col, (et, val, delta, ayuda) in zip(cols, c["kpis"]):
            col.metric(et, val, delta, delta_color="off", delta_arrow="off", help=ayuda)
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
            st.warning(f"Las cifras refieren a la selección ({'; '.join(act)}): {numero(self.horas)} horas y {self.base_txt()}.")
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
