# Clasificación de TDAH desde conectividad funcional dinámica

Código y datos para los experimentos de clasificación TDAH vs. control a partir de
secuencias de conectividad funcional dinámica derivadas de rs-fMRI del repositorio
ADHD-200.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jpospinalo/tdha-revision/blob/main/tdha_experimentos.ipynb)

## Inicio rápido

```bash
git clone https://github.com/jpospinalo/tdha-revision.git
cd tdha-revision/src
pip install -r ../requirements.txt

python verify_setup.py                                       # comprueba el entorno
python run_experiment.py --list-roi-sets
python run_experiment.py --site NYU --roi-set 12 --dry-run   # valida sin entrenar
python run_experiment.py --site NYU --roi-set 12             # una corrida completa
python compile_results.py --site NYU --model lstm --stats

python run_queue.py --sites NYU --roi-sets 12 18 39 116      # varias encadenadas
```

Desde Colab, el notebook `tdha_experimentos.ipynb` hace todo lo anterior.

## Estructura

```
├── tdha_experimentos.ipynb   notebook de Colab: una corrida de principio a fin
├── requirements.txt
├── data/
│   ├── bold/                 señales BOLD por sitio (38 MB en total)
│   │   ├── NYU.joblib        177 sujetos · 116 ROIs · 172 TR
│   │   ├── Peking.joblib     183 · 116 · 232
│   │   ├── NeuroIMAGE.joblib  39 · 116 · 257
│   │   └── OHSU.joblib        66 · 116 ·  74
│   └── atlas/
│       ├── aal116.csv        índice, id AAL y nombre de los 116 ROIs
│       └── roi_sets.json     subconjuntos 12, 18, 39 y 116
├── src/
│   ├── data.py               carga de señales y construcción de secuencias
│   ├── run_experiment.py     una corrida
│   ├── run_queue.py          varias corridas encadenadas en un solo proceso
│   ├── compile_results.py    compilación y estadística
│   ├── verify_setup.py       comprobación del repositorio y del entorno
│   └── kerasmodels/          registro de arquitecturas
├── results/runs/             una carpeta por corrida
└── docs/                     auditoría, revisión metodológica y eficiencia
```

## Por qué se versionan las señales y no los tensores

El repositorio guarda **solo las señales BOLD**. Los tensores de conectividad —más de
460 MB— se derivan en cada corrida a partir de ellas, en unos pocos segundos.

Esto tiene tres consecuencias:

**El enventanado es un parámetro, no un archivo.** `--window` y `--step` se cambian
desde la línea de comandos, así que el análisis de sensibilidad al tamaño de ventana
se hace sin regenerar nada.

**Los datos no pueden quedar desincronizados.** En la versión anterior del proyecto,
un tensor llamado `X39` tenía 26 ventanas porque se había generado con paso 4, mientras
el resto usaba paso 2 y el artículo los comparaba entre sí. Con los tensores derivados
eso es imposible: los parámetros quedan registrados en el `config.json` de cada corrida
y forman parte del identificador.

**El repositorio se clona en segundos** y no necesita Git LFS.

Las reconstrucciones se verificaron contra los tensores del proyecto original: NYU con
12, 18 y 39 ROIs, y Peking, NeuroIMAGE y OHSU con 18, coinciden hasta 10⁻⁶, que es el
redondeo de `float32`.

## Diseño experimental

**Una corrida por carpeta.** El nombre incluye un hash de la configuración completa
(`NYU_rois12_w70s2_lstm_2136273e`). Varias personas pueden correr en paralelo y hacer
push al mismo repositorio sin conflictos, porque nadie escribe en un archivo
compartido. Si alguien repite una configuración idéntica, el script lo advierte en
lugar de sobrescribir.

**Comparaciones pareadas.** Con la misma `--seed` y las mismas etiquetas, todas las
configuraciones usan exactamente las mismas particiones. Eso permite usar ANOVA de
medidas repetidas y contrastes pareados, con bastante más potencia que sus equivalentes
para muestras independientes.

**Sin fuga en la selección de época.** Dentro de cada pliegue se aparta un 15 % del
entrenamiento para el early stopping. El pliegue de validación externo solo se usa en
la evaluación final, nunca para decidir nada.

**Trazabilidad.** Cada `config.json` guarda el hash de las señales, los parámetros de
enventanado, el commit de git, si el árbol estaba limpio, el usuario y las versiones de
Python, TensorFlow y GPU. `compile_results.py` usa esa información para negarse a
agregar corridas que no sean comparables.

### Reproducibilidad: qué se garantiza y qué no

| | Garantizado |
|---|---|
| Particiones de validación cruzada | Sí, idénticas en cualquier máquina con la misma semilla |
| Tensores derivados de las señales | Sí, hasta el redondeo de `float32` |
| Protocolo, hiperparámetros, datos | Sí, registrados en `config.json` |
| Valores exactos de las métricas | **No** por defecto |

Los kernels de cuDNN para redes recurrentes no son deterministas: acumulan en punto
flotante en orden variable. Dos corridas con la misma semilla difieren en los decimales,
y entre GPUs distintas la diferencia es mayor. `--deterministic` fuerza operaciones
deterministas, a costa de perder el camino rápido de cuDNN.

## Datos

`data/bold/{sitio}.joblib` contiene un diccionario con:

| Clave | Contenido |
|---|---|
| `subjects` | identificadores, longitud n |
| `bold` | `(n, 116, T)` float32, serie temporal media por ROI del atlas AAL116 |
| `labels` | `(n,)` int — 0 control, 1 TDAH |
| `roi_names` | 116 nombres, en el orden del eje 1 de `bold` |

| Sitio | Sujetos | Control / TDAH | TR | Ventanas con 70/2 |
|---|---|---|---|---|
| NYU | 177 | 87 / 90 | 172 | 52 |
| Peking | 183 | 109 / 74 | 232 | 82 |
| NeuroIMAGE | 39 | 22 / 17 | 257 | 94 |
| OHSU | 66 | 38 / 28 | 74 | 3 |

Advertencias que el script también emite en tiempo de ejecución:

- **OHSU** produce 3 ventanas por sujeto: sus adquisiciones son de 74 TR y la ventana
  de 70 apenas cabe. Los parámetros de enventanado de NYU no son trasladables ahí.
- **NeuroIMAGE** tiene 39 sujetos; con 10 pliegues la validación queda en ~4 sujetos.
  Conviene `--n-splits 5`.
- **Peking** está desbalanceado; usar `--class-weight` y mirar AUC y especificidad.

Los datos provienen del repositorio público ADHD-200, preprocesados con el pipeline
ATHENA del Neuro Bureau.

## Salidas de cada corrida

| Archivo | Contenido | Para qué |
|---|---|---|
| `config.json` | configuración, hashes, commit, entorno | reproducibilidad |
| `metrics_train.csv`, `metrics_val.csv` | métricas por pliegue | tablas de desempeño |
| `history.csv` | pérdida y accuracy por época | curvas de convergencia |
| `predictions_val.csv` | probabilidad por sujeto | matrices de confusión, ROC |
| `folds.csv` | sujetos de cada partición: `fit`, `inner_val`, `outer_val` | auditoría de fuga |

Con `--random-subset` los archivos llevan el sufijo `_setNN` y se añade
`random_subsets_summary.csv`.

## Añadir una arquitectura

Crear un módulo en `src/kerasmodels/` y registrarlo:

```python
# kerasmodels/mi_modelo.py
from . import register

@register("mi_modelo")
def build(n_windows, n_features, units=64):
    import keras
    from keras import layers
    inp = layers.Input(shape=(n_windows, n_features))
    ...
    return keras.Model(inp, out, name="mi_modelo")
```

Añadirlo a las importaciones del final de `kerasmodels/__init__.py`. Queda disponible
como `--model mi_modelo`.

El contrato es: recibir `(lote, n_windows, n_features)` y devolver un `keras.Model`
**sin compilar** con salida sigmoide de dimensión 1. La compilación la hace
`run_experiment.py`, de modo que la arquitectura queda desacoplada de los
hiperparámetros de entrenamiento.

## Añadir un subconjunto de ROIs

Editar `data/atlas/roi_sets.json` añadiendo una entrada con los índices base 0 sobre
el atlas AAL116, los nombres y una descripción. Los índices se validan al cargar.

## Convenciones de trabajo

1. Usar `--seed 42` en todo lo que se vaya a comparar.
2. Hacer commit del código antes de correr; el script avisa si el árbol está sucio.
3. Nunca editar a mano un CSV de `results/`. Si un resultado está mal, se vuelve a
   correr y se versiona la corrida nueva.
4. Ejecutar `--dry-run` antes de una corrida larga.

## Documentación adicional

`docs/` contiene la auditoría de las cifras del manuscrito frente a los resultados
versionados, y la revisión metodológica del pipeline anterior.
