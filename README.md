# Proyecto Integrador III

Repositorio del curso Proyecto Integrador III (IU Digital). Contiene el mecanismo
para **obtener datos de Hugging Face, transformarlos y resumirlos en un `.txt`**,
que se reutilizará en las siguientes actividades.

**Integrantes:** Braian Rico · Samuel Rosero · Diego Cuervo

## EA1 – Carga y creación del dataset

Se trabaja con el dataset [`fauxNeuz/RealFakeNews`](https://huggingface.co/datasets/fauxNeuz/RealFakeNews)
(split `train`): noticias etiquetadas como `REAL` o `FAKE`. Se cargan las primeras
**10.000 filas** en modo *streaming* para no descargar el dataset completo.

### Flujo

```
HFDatasetLoader ──► DataFrame ──► 5 transformaciones ──► Reportes ──► EA1/resultados/fake-news.txt
 (clase de carga)                  (en el notebook)      (clase de resumen)
```

El notebook coordina el flujo y hace las transformaciones; las clases contienen la
lógica de carga y de generación del resumen.

### Estructura

```
src/
└── clases_po/                    # Código compartido entre evidencias
    ├── hf_loader.py              # HFDatasetLoader: Hugging Face -> DataFrame
    └── reportes.py               # Reportes: DataFrame -> resumen .txt
EA1/
├── notebooks/EA1_fake_news.ipynb # Notebook que orquesta el flujo
├── datos/fake-news.csv           # Datos crudos descargados (generado)
└── resultados/fake-news.txt      # Resumen del DataFrame transformado (generado)
EA2/                              # Entrega EA2 (ver sección EA2)
.github/workflows/                # Bot que ejecuta los notebooks en cada push a main
Dockerfile, docker-compose.yml
```

### Clases

**`HFDatasetLoader`** (`src/clases_po/hf_loader.py`): recibe el identificador del
dataset y el split, se conecta a Hugging Face y devuelve un DataFrame de pandas.
Opcionalmente limita las filas (`max_rows`) y guarda una copia en disco (`csv`,
`json`, `parquet` o `xlsx`).

```python
loader = HFDatasetLoader(
    dataset_name="fauxNeuz/RealFakeNews",
    split="train",
    max_rows=10_000,
    streaming=True,
    filename="../datos/fake-news.csv",
)
df = loader.load()
```

**`Reportes`** (`src/clases_po/reportes.py`): recibe un DataFrame y genera un
resumen con filas, columnas, duplicados, tipos, valores nulos y estadísticas de las
columnas numéricas.

```python
reportes = Reportes()
reportes.guardar_reporte(df_transformado, "../resultados/fake-news.txt")
```

### Transformaciones (en el notebook)

| # | Transformación | Operación | Resultado |
|---|---|---|---|
| 1 | Eliminar duplicados | `drop_duplicates()` | 10.000 → 9.900 filas |
| 2 | Renombrar columnas | `text` → `texto`, `label` → `etiqueta` | Nombres en español |
| 3 | Columna calculada | `cantidad_palabras` | Largo de cada noticia |
| 4 | Filtrar filas | `cantidad_palabras >= 5` | 9.900 → 9.872 filas |
| 5 | Columna binaria | `es_falsa` (`FAKE` = 1) | 40,16 % de noticias falsas |

El `.txt` se genera a partir del DataFrame **transformado** (9.872 filas, 4 columnas).

### Cómo se cubre la rúbrica

| Criterio | Dónde se evidencia |
|---|---|
| Código versionado en GitHub | Historial de commits y PRs de este repositorio |
| Clase para generar un DataFrame | `src/clases_po/hf_loader.py` |
| Clase para resumir y exportar | `src/clases_po/reportes.py` + `EA1/resultados/fake-news.txt` |
| Transformaciones del DataFrame | 5 transformaciones en `EA1/notebooks/EA1_fake_news.ipynb` |
| Funcionamiento y evidencias | Notebook guardado con salidas + ejecución automática en GitHub Actions |

## EA2 – Ideación, diseño y documentación de la necesidad

**Propósito:** definir una necesidad analítica, elegir un dataset de Hugging Face que
la apoye, cargarlo en un DataFrame, revisarlo de forma descriptiva y generar un
resumen `.txt` automáticamente. En esta entrega **no se transforma** el dataset.

**Necesidad (escenario hipotético):** el equipo de operaciones de un sistema público
de bicicletas compartidas necesita entender cómo varía la demanda por hora para
planificar la redistribución de bicicletas y el mantenimiento.

**Pregunta descriptiva:** ¿Cómo varía la cantidad de alquileres por hora según la
franja horaria, el tipo de día, la estación y las condiciones del clima, y qué
diferencias hay entre usuarios casuales y registrados durante 2011 y 2012?

**Unidad de análisis:** una fila = una hora de servicio del sistema.

### Dataset

| Dato | Valor |
|---|---|
| Nombre | Bike Sharing Demand - Hourly (Poisson), copia del UCI Bike Sharing Dataset (archivo por hora) |
| Identificador en Hugging Face | [`t22000t/bike-sharing-tabular`](https://huggingface.co/datasets/t22000t/bike-sharing-tabular) |
| Split | `train` |
| Revisión (commit) | `6b645e0cedbc5208fcca11ac94544b088478bf61` |
| Tamaño | 17.379 filas x 17 columnas (carga completa) |
| Fuente original | [UCI Bike Sharing Dataset (id 275)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Fanaee-T y Gama |
| Licencia | CC BY 4.0 |
| Fecha de consulta | 2026-10-25 |

### Estructura de `EA2/`

La guía sugiere una carpeta `AE2/`; acá se llama `EA2/` (evidencia de aprendizaje),
igual que `EA1/`.

```
EA2/
├── documento/                       # Documento Word de la entrega
├── notebooks/EA2_bike_sharing.ipynb # Notebook (el bot lo guarda con salidas)
└── resultados/resumen_dataframe.txt # Resumen generado por el notebook (lo sube el bot)
```

El notebook usa `HFDatasetLoader` (con el nuevo parámetro opcional `revision`, que
fija la versión del dataset) y `Reportes` (con el nuevo parámetro opcional
`encabezado`, que agrega dataset, split, revisión y muestra al resumen). Sin esos
parámetros el comportamiento es el mismo que en EA1.

### Cómo ejecutar el notebook de EA2

Con Docker (recomendado), sin abrir el navegador:

```bash
docker compose build
docker compose run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -w /app/EA2/notebooks notebook \
  jupyter nbconvert --to notebook --execute --inplace EA2_bike_sharing.ipynb
```

O con `docker compose up --build` y abriendo `EA2/notebooks/EA2_bike_sharing.ipynb`
en `http://localhost:8888`.

Sin Docker (Python 3.12):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt pyarrow openpyxl
cd EA2/notebooks
jupyter notebook EA2_bike_sharing.ipynb
```

Cada ejecución crea la carpeta `EA2/resultados/` si falta y regenera
`resumen_dataframe.txt`.

## Cómo ejecutarlo (EA1)

### Con Docker (recomendado)

Abrir Jupyter en el navegador:

```bash
docker compose up --build
```

Entrar a `http://localhost:8888` con el token que aparece en la consola y abrir
`EA1/notebooks/EA1_fake_news.ipynb`.

Ejecutar el notebook completo sin abrir el navegador:

```bash
docker compose run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -w /app/EA1/notebooks notebook \
  jupyter nbconvert --to notebook --execute --inplace EA1_fake_news.ipynb
```

`--user` evita que los archivos generados queden a nombre de `root`.

### Sin Docker

Requiere Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt pyarrow openpyxl
cd EA1/notebooks
jupyter notebook EA1_fake_news.ipynb
```

## Ejecución automática (GitHub Actions)

El workflow `.github/workflows/ejecutar-notebooks.yaml` corre en cada push a `main`
y procesa todos los notebooks de la lista `NOTEBOOKS` (hoy EA1 y EA2):

1. Instala las dependencias en una máquina limpia.
2. Borra las carpetas de resultados (`EA1/datos/`, `EA1/resultados/`,
   `EA2/resultados/`) para regenerarlas desde cero.
3. Ejecuta cada notebook completo desde su carpeta.
4. Commitea los notebooks con sus salidas y los archivos generados como
   `github-actions[bot]`. El mensaje indica qué commit lo disparó y resume cada
   reporte, y lleva `[skip ci]` para no volver a dispararse.

Así, los resultados del repositorio siempre corresponden al código de `main`. Por
eso **los archivos generados no se suben a mano**: en los PR se sube solo el código
y el notebook, y el bot genera el resto al integrar.

Para sumar una actividad nueva basta con agregar una línea a `NOTEBOOKS` con el
notebook, sus carpetas de resultados y su reporte `.txt`. Si los datos no cambian, el
commit del bot solo actualiza los tiempos de ejecución que cada notebook guarda en
sus celdas.

## Convenciones del equipo

- Trabajo en ramas (`feature/...`, `ci/...`, `docs/...`) y PR a `main` con revisión de
  otro integrante; se integra con *merge commit*.
- Commits con [Conventional Commits](https://www.conventionalcommits.org/es/):
  el tipo en inglés (`feat:`, `fix:`, `ci:`, `docs:`, `chore:`) y la descripción en
  español.
