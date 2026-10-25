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
.github/workflows/                # Bot que ejecuta el notebook en cada push a main
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

## Cómo ejecutarlo

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

El workflow `.github/workflows/ejecutar-ea1.yaml` corre en cada push a `main`:

1. Instala las dependencias en una máquina limpia.
2. Borra `EA1/datos/` y `EA1/resultados/` para regenerarlos desde cero.
3. Ejecuta el notebook completo.
4. Commitea el CSV, el reporte y el notebook con sus salidas como
   `github-actions[bot]`. El mensaje lleva `[skip ci]` para no volver a dispararse.

Así, los resultados del repositorio siempre corresponden al código de `main`.
Si los datos no cambian, el commit del bot solo actualiza los tiempos de ejecución
que el notebook guarda en cada celda.

## Convenciones del equipo

- Trabajo en ramas (`feature/...`, `ci/...`, `docs/...`) y PR a `main` con revisión de
  otro integrante; se integra con *merge commit*.
- Commits con [Conventional Commits](https://www.conventionalcommits.org/es/):
  el tipo en inglés (`feat:`, `fix:`, `ci:`, `docs:`, `chore:`) y la descripción en
  español.
