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

# --representation y --window-seconds/--step-seconds explícitos: sin ellos, el script
# cae en el enventanado legado (70/2 TR) por compatibilidad hacia atrás, no en los
# 120 s recomendados — ver "La ventana física por defecto" más abajo.
python run_experiment.py --site NYU --roi-set 12 --representation ordered \
    --window-seconds 120 --step-seconds 12 --dry-run   # valida sin entrenar
python run_experiment.py --site NYU --roi-set 12 --representation ordered \
    --window-seconds 120 --step-seconds 12             # una corrida completa
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
├── results/runs/             subcarpeta por ROI_SET (12/18/39/116), una carpeta por
│                              corrida dentro de cada una
└── docs/                     arquitectura, metodología, validación, límites y eficiencia
                               del pipeline actual; además, auditoría histórica de un
                               manuscrito anterior (docs/auditoria-metricas.md)
```

## Por qué se versionan las señales y no los tensores

El repositorio guarda **solo las señales BOLD**. Los tensores de conectividad —más de
460 MB— se derivan en cada corrida a partir de ellas, en unos pocos segundos.

Esto tiene tres consecuencias:

**El enventanado es un parámetro, no un archivo.** La ventana se define en tiempo físico
con `--window-seconds` y `--step-seconds` (o en TR con `--window`/`--step`), así que el
análisis de sensibilidad al tamaño de ventana se hace sin regenerar nada.

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

**Una corrida por carpeta, agrupadas por ROI_SET.** El nombre incluye un hash de la
configuración completa (`NYU_rois12_w70s2_lstm_2136273e`) y vive dentro de
`results/runs/<roi_set>/` (`12`, `18`, `39` o `116`, según `--roi-set`), para no mezclar
en una sola carpeta corridas de tamaños de ROI distintos. Varias personas pueden correr
en paralelo y hacer push al mismo repositorio sin conflictos, porque nadie escribe en un
archivo compartido. Repetir una configuración ya completada en la misma carpeta detiene
la ejecución (`ESTA_CONFIGURACION_YA_SE_EJECUTO`) en vez de sobrescribirla: `--tag`
distingue la repetición como una carpeta nueva sin cambiar `config_hash`, y
`--overwrite` reemplaza deliberadamente la carpeta existente. Una corrida incompleta
(sin `metrics_val.csv`, por ejemplo tras una desconexión de Colab a mitad) sí se rehace
sola, con un aviso impreso. `compile_results.py` sigue reconociendo corridas guardadas
con el layout plano anterior (`results/runs/<run_id>/`, sin subcarpeta de ROI).

**La carpeta de una corrida no debe renombrarse manualmente.** Su nombre es parte de su
identidad operativa y debe coincidir exactamente con `run_id` en `config.json`; si no
coincide, `validate_run_artifacts()` la rechaza (`collect(strict=True)` lanza `ValueError`).

**Comparaciones pareadas.** Con la misma `--seed` y las mismas etiquetas, todas las
configuraciones usan exactamente las mismas particiones. Eso permite contrastes
pareados con bastante más potencia que sus equivalentes para muestras independientes.
El veredicto de significancia usa un t-test remuestreado con corrección de
Nadeau-Bengio (corrige por que los pliegues de una k-fold repetida no son
observaciones independientes) más corrección de Holm entre contrastes. `--stats`
también imprime un ANOVA de medidas repetidas y un t-test pareado ingenuo, pero
ambos quedan etiquetados como exploratorios/de referencia, no como el veredicto.
Las corridas se emparejan por un merge explícito de `(repeat, fold)`, no por
posición tras ordenar: si a una corrida le faltan columnas, tiene claves
`repeat`/`fold` duplicadas, o sus pliegues no coinciden exactamente con los de la
otra corrida, `--stats` falla con un error explícito en vez de comparar mal en
silencio.

**Sin fuga en la selección de época.** Dentro de cada pliegue se aparta un 15 % del
entrenamiento para el early stopping. El pliegue de validación externo solo se usa en
la evaluación final, nunca para decidir nada.

**Monitor de early stopping configurable.** `--early-stopping-monitor` elige qué serie
de validación interna observa `EarlyStopping` para detener el entrenamiento y qué punto
restaurar: `val_loss` (por defecto) es la pérdida total de Keras — BCE más las
penalizaciones L2 de la arquitectura; `val_bce` es solo la entropía cruzada binaria
predictiva, registrada como métrica separada (`bce`/`val_bce` en `history.csv`) sin
participar del objetivo de optimización. `--early-stopping-min-delta` (por defecto
`1e-5`) fija la mejora mínima exigida. `best_epoch`/`best_monitor_value` se leen de la
propia instancia de `EarlyStopping` tras `fit()`, no del mínimo global de la serie (con
`min_delta > 0` o `--start-from-epoch > 0` no son lo mismo), y quedan corroborados por
una reevaluación posterior independiente del modelo ya restaurado (`restored_monitor_value`),
que debe coincidir con `best_monitor_value`: si no coincide (o no es finita), la corrida
aborta con `RuntimeError` antes de tocar el pliegue externo, en vez de guardar metadatos
inconsistentes con los pesos realmente restaurados.
El monitor y su `min_delta` quedan en la identidad de la configuración: dos corridas
idénticas salvo el monitor tienen `config_hash`/`run_id` distintos, pero comparten
`early_stopping_ab_hash` (la identidad completa sin el monitor) — eso es lo que exige
`compile_results.py --stats --stats-by early_stopping_monitor` antes de compararlas
de forma pareada, junto con `config_schema_version >= 4`. Cambiar el monitor no cambia
el objetivo de entrenamiento (sigue siendo `binary_crossentropy` + L2); solo cambia qué
época se selecciona.

**Trazabilidad.** Cada `config.json` guarda el hash de las señales BOLD (`bold_hash`), el
hash del atlas (`atlas_hash`), hashes del código de datos y del runner
(`data_code_hash`, `runner_code_hash`), los parámetros de enventanado, el commit de
git, si el árbol estaba limpio, el usuario y las versiones de Python, TensorFlow y GPU.
`compile_results.py` usa parte de esa información para avisar cuando las corridas
seleccionadas no son comparables: semilla, `n_splits`/`n_repeats`, huella de
particiones, `bold_hash` distinto entre corridas del mismo sitio, árbol sucio, o
`config_hash`/`subset_suffix` duplicados. Ese último aviso también salta al comparar
corridas hechas a propósito con `--tag` para repetir la misma configuración (`--tag` no
cambia `config_hash`, solo el nombre de carpeta): no es un error de por sí, sino la señal
de que `compile_results.py` las está tratando como una sola configuración repetida y no
como configuraciones independientes que deban promediarse juntas. **No** compara `atlas_hash`,
`data_code_hash`, `runner_code_hash` ni versiones de software entre corridas —esos
campos quedan guardados para inspección manual, pero `check_comparability()` no los
usa como criterio de aborto. Con `--strict-comparability` los avisos que sí genera
abortan la ejecución en vez de solo imprimirse.

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

| Sitio | Sujetos | Control / TDAH | TR (s) | Puntos | Escaneo (s) | Ventana por defecto |
|---|---|---|---|---|---|---|
| NYU | 177 | 87 / 90 | 2.00 | 172 | 344 | 120 s → 19 ventanas |
| Peking | 183 | 109 / 74 | 2.00 | 232 | 464 | 120 s → 20 ventanas |
| NeuroIMAGE | 39 | 22 / 17 | 1.96 | 257 | 504 | 120 s → 20 ventanas |
| OHSU | 66 | 38 / 28 | 2.50 | 74 | 185 | estática |

La ventana física por defecto es 120 s (supera el piso de ~111 s de la conectividad
dinámica para el filtrado a 0.009 Hz de ATHENA). Advertencias que el script también emite
en tiempo de ejecución:

- **OHSU** dura 185 s: demasiado corto para una ventana válida, así que su default es la
  representación **estática**. Bajar `--n-splits` a 5.
- **NeuroIMAGE** tiene 39 sujetos; con 10 pliegues la validación queda en ~4 sujetos.
  Conviene `--n-splits 5`.
- **Peking** está desbalanceado; usar `--class-weight` y mirar AUC y especificidad.

Los datos provienen del repositorio público ADHD-200, preprocesados con el pipeline
ATHENA del Neuro Bureau.

## Salidas de cada corrida

| Archivo | Contenido | Para qué |
|---|---|---|
| `config.json` | configuración, hashes, commit, entorno | reproducibilidad (fuente de verdad) |
| `resumen.md` | vista legible del config y las métricas titulares, derivada de `config.json` | hojear muchas corridas de un vistazo |
| `metrics_train.csv`, `metrics_val.csv` | una fila por fold externo (`N_SPLITS x N_REPEATS` filas) | diagnóstico de dispersión entre pliegues |
| `history.csv` | una fila por época de cada fold | curvas de convergencia |
| `predictions_val.csv` | una fila por sujeto evaluado OOF en cada repetición | matrices de confusión, ROC, métricas OOF |
| `folds.csv` | una fila por sujeto, fold y partición (`fit`, `inner_val`, `outer_val`) | auditoría de fuga |

`metrics_val.csv` da una fila por pliegue externo: con `N_SPLITS=10`/`N_REPEATS=5` son
50 estimaciones ruidosas del mismo modelo. La estimación de referencia de una corrida no
es su media, es el **resumen OOF por repetición**: agrupar las predicciones de
`predictions_val.csv` por `repeat` y calcular cada métrica sobre la muestra completa
reconstruida, dando `N_REPEATS` valores en vez de `N_SPLITS x N_REPEATS`. Lo calcula
`compile_results.oof_metrics_per_repetition()`, lo compila `compile_results.py` en la
tabla agregada (`oof_auc_mean`, `oof_f1_macro_mean`, …), y el notebook lo muestra primero
en la sección de resultados, con la tabla por pliegue como diagnóstico secundario.

Con `--random-subset` los archivos llevan el sufijo `_setNN` y se añade
`random_subsets_summary.csv`.

## Modelos y representaciones

Arquitecturas registradas (`--model`): `lstm`, `gru`, `cnn1d`, `transformer`, `deepsets`
y `brainnetcnn`. Las tres últimas son útiles cuando el orden temporal no aporta señal:
`deepsets` es invariante al orden por construcción, `transformer` admite
`--model-arg positional=false` (modelo de conjuntos), y `brainnetcnn` opera sobre la
matriz de conectividad con filtros topológicos (edge-to-edge / edge-to-node).

`brainnetcnn` sirve con cualquier representación de una sola matriz por sujeto
(`static`, `partial`, `shrunk`, `mean`) y con `ordered`/`permuted`/sus variantes
`_scaled` (trata cada ventana como un canal fijo, sin modelar su orden temporal). **No**
sirve con `mean_std` ni `hybrid`: ambas duplican o multiplican las características por
conexión y el resultado ya no corresponde al triángulo superior de una matriz cuadrada.
Tampoco sirve con `tangent`: sus coeficientes no son pesos de conexión interpretables
topológicamente, y se rechaza explícitamente.

Representaciones (`--representation`): `ordered` (secuencia dinámica), `static` (una matriz
Pearson sobre toda la serie), `partial` (correlación parcial regularizada Ledoit-Wolf, una
matriz por sujeto), `shrunk` (correlación completa regularizada Ledoit-Wolf, misma
pregunta que `static` pero más estable con series cortas o muchos ROIs — ver
`docs/methodology.md`), `tangent` (proyección en espacio tangente vía nilearn, referencia
ajustada solo con `fit` de cada pliegue; no admite `--fisher-z` ni `brainnetcnn`; requiere
`pip install nilearn`), `hybrid` (estática + media/desviación/cambio de las ventanas),
`mean` / `mean_std` (resúmenes invariantes al orden), `permuted` (ventanas barajadas,
control para saber si el orden discrimina) y `ordered_scaled` / `permuted_scaled`
(mismas ventanas, reescaladas por conexión dentro de cada pliegue, sin fuga — control
para separar el efecto de reescalar del efecto de `tangent`). Si `ordered` y `permuted`
rinden igual, conviene preferir los modelos y representaciones invariantes al orden.

Eficiencia: `--mixed-precision` acelera los modelos grandes (39/116 ROIs, transformer,
brainnetcnn) en GPU; `run_queue.py --in-process` corre un lote sin reiniciar TensorFlow
por corrida; el early stopping usa `--patience 25` por defecto. Ver `docs/performance.md`.

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
   correr con `--tag` (carpeta nueva) o `--overwrite` (reemplaza la existente) y se
   versiona la corrida resultante.
4. Ejecutar `--dry-run` antes de una corrida larga.

## Documentación adicional

`docs/` describe el pipeline actual — no es solo para quien lo lea, también es el
contexto que un asistente de IA debería cargar antes de tocar el código, así que se
mantiene sincronizado con `src/` en cada cambio:

- `architecture.md` — módulos, responsabilidades, flujo de datos.
- `guia-experimentacion-colaborativa.md` — cómo usar el notebook para correr, validar,
  descargar y subir una corrida en Colab; qué corregir si algo falla a mitad de camino.
- `methodology.md` — qué representaciones de conectividad existen, por qué, y cómo se
  evalúa (incluye las métricas OOF por repetición y la corrección de Nadeau-Bengio en
  `compile_results.py`).
- `validation.md` — qué se verificó y con qué alcance.
- `limitations.md` — qué falta o no está probado (representaciones, compatibilidad de
  modelos, movimiento no filtrado por ATHENA).
- `performance.md` — optimizaciones computacionales, sin efecto en las métricas.
- `auditoria-metricas.md` — auditoría histórica de las cifras de un manuscrito anterior
  frente a los resultados versionados; es un documento de un punto en el tiempo, no se
  actualiza con cambios posteriores del pipeline.
