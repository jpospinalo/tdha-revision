# Guía de experimentación colaborativa (notebook)

Cómo usar `tdha_experimentos.ipynb` para correr, validar, descargar y subir una corrida
en Colab, y qué hacer cuando algo falla a mitad de camino. Para el detalle de cada
representación, arquitectura o parámetro, ver `methodology.md`; esta guía cubre el flujo
operativo del notebook, no el contenido metodológico.

## Flujo de una corrida

1. **Preparar el entorno** (sección 1): clona o actualiza el repositorio, instala lo que
   falte, deja el directorio de trabajo en `src/`.
2. **Configurar** (sección 2, la única celda que se edita): cinco bloques —A) datos,
   representación, modelo; B) enventanado; C) entrenamiento y validación; D) control
   anatómico; E) identidad y controles operativos. Todos los valores se escriben a mano en
   la celda, incluidos los que valen `None` (por ejemplo `OVERLAP`, `WINDOW_TR`,
   `RANDOM_SUBSET` o `TAG`): ahí `None` es una elección explícita con un significado
   documentado (p. ej. "sin solape fijo", "no usar ventana por TR", "usar todos los ROIs"),
   no un valor sin llenar que caiga en un default oculto del script.
3. **Comprobaciones previas** (sección 3): `verify_setup.py` revisa el repositorio y los
   datos; `preflight()` (al final de la celda del constructor de argumentos) valida esta
   configuración concreta contra las particiones reales, sin entrenar.
4. **Prueba de humo** (sección 4): si `EJECUTAR_PRUEBA_HUMO` es `True`, `prueba_humo()`
   entrena 2 pliegues, 1 repetición, 3 épocas, con `START_FROM_EPOCH` forzado a 0, en
   `/tmp` — nunca en la carpeta de resultados. `START_FROM_EPOCH` de la configuración
   formal (p. ej. un warm-up largo) no se usa en esta prueba: con solo 3 épocas de humo,
   heredarlo tal cual la haría fallar por `--start-from-epoch` mayor que `--epochs`, sin
   que eso diga nada sobre si la configuración formal es correcta. Confirma que el camino
   completo funciona antes de gastar tiempo de GPU en
   la corrida larga.
5. **La corrida** (sección 5): `ejecutar_corrida()` lanza la configuración de la celda 2
   tal cual — no admite argumentos sueltos ni overrides de última hora. Imprime `RUN_ID`
   y el `early_stopping_ab_hash` de la configuración.
6. **Resultados** (sección 6): el resumen OOF por repetición es la estimación de
   referencia (agrupa las predicciones de cada repetición sobre la muestra completa); la
   tabla por pliegue y los histogramas son diagnóstico de dispersión, no el resultado
   principal.
7. **Validar antes de exportar**: corre la misma validación semántica que
   `compile_results.py --strict` sobre la carpeta de la corrida. Si encuentra un
   problema, la celda falla con la lista completa y no se puede descargar ni subir nada.
8. **Descargar** (opcional) o **subir a GitHub** (sección 7): ambas llaman a
   `exigir_corrida_validada()`, que no solo exige `CORRIDA_VALIDADA = True` sino que
   revalida contra disco que `RUN_ID_VALIDADO` coincide con el `RUN_ID` de la corrida
   actual — si volvió a ejecutar la celda 5 después de validar, o si `RUN_ID` cambió por
   cualquier motivo, la validación quedó vieja y hay que repetirla antes de exportar.
9. **Diagnóstico de orden** (opcional, sección intermedia): solo se ejecuta si
   `EJECUTAR_DIAGNOSTICO_ORDEN = True` (por defecto `False`, porque entrena tres variantes
   completas y no aporta nada fuera de representaciones sensibles al orden temporal).
   Rechaza representaciones estáticas y `MODELO == "brainnetcnn"` con un error explícito
   en vez de correr un diagnóstico que no tiene sentido para ellos.
10. **Compilar** (sección 8, no hace falta correrlo en cada corrida): reúne todas las
    carpetas de `results/runs/`, arma la tabla agregada y avisa —o aborta, con
    `--strict-comparability`— si detecta corridas no comparables.

## Variables del notebook, argumento de la CLI y campo de `config.json`

`construir_argv()` (celda 10) traduce cada variable de la celda de configuración en un
argumento de `run_experiment.py`; el runner registra el valor efectivo en `config.json`,
casi siempre bajo el mismo nombre. Esta tabla es la referencia rápida para ir de una
variable del notebook al campo donde comprobar lo que realmente se ejecutó.

| Variable del notebook | Argumento CLI | Campo en `config.json` |
|---|---|---|
| `SITIO` | `--site` | `site` |
| `ROI_SET` | `--roi-set` | `roi_set` |
| `MODELO` | `--model` | `model` |
| `HIPERPARAMS` | `--model-arg clave=valor ...` | `arch` (dict resuelto) |
| `REPRESENTACION` | `--representation` | `representation` |
| `REPRESENTATION_SEED` | `--representation-seed` | `representation_seed` |
| `FISHER_Z` | `--fisher-z` | `fisher_z` |
| `CONSTANT_POLICY` | `--constant-policy` | `constant_policy` |
| `WINDOW_SECONDS` | `--window-seconds` | `windowing.window_seconds` |
| `STEP_SECONDS` | `--step-seconds` | `windowing.step_seconds` |
| `OVERLAP` | `--overlap` | `windowing.requested_overlap` / `windowing.effective_overlap` |
| `WINDOW_TR` | `--window` | `window`, `windowing.window_tr` |
| `STEP_TR` | `--step` | `step`, `windowing.step_tr` |
| `TR_SECONDS` | `--tr-seconds` | `windowing.tr_seconds` |
| `WINDOW_SHAPE` | `--window-shape` | `windowing.shape` |
| `GAUSSIAN_SIGMA` | `--gaussian-sigma` | `windowing.gaussian_sigma` |
| `LR` | `--lr` | `lr` |
| `BATCH_SIZE` | `--batch-size` | `batch_size` |
| `EPOCHS` | `--epochs` | `epochs` |
| `PATIENCE` | `--patience` | `patience` |
| `CLIPNORM` | `--clipnorm` | `clipnorm` |
| `INNER_VAL_FRAC` | `--inner-val-frac` | `inner_val_frac` |
| `START_FROM_EPOCH` | `--start-from-epoch` | `start_from_epoch` |
| `EARLY_STOPPING_MONITOR` | `--early-stopping-monitor` | `early_stopping_monitor` |
| `EARLY_STOPPING_MIN_DELTA` | `--early-stopping-min-delta` | `early_stopping_min_delta` |
| `N_SPLITS` | `--n-splits` | `n_splits` |
| `N_REPEATS` | `--n-repeats` | `n_repeats` |
| `CLASS_WEIGHT` | `--class-weight` | `class_weight` |
| `SEED` | `--seed` | `seed` |
| `DETERMINISTIC` | `--deterministic` | `deterministic` |
| `MIXED_PRECISION` | `--mixed-precision` | `mixed_precision` |
| `RANDOM_SUBSET` | `--random-subset` | `random_subset` |
| `N_RANDOM_SETS` | `--n-random-sets` | `n_random_sets` |
| `EXCLUDE_ROI_SET` | `--exclude-roi-set` | `exclude_roi_set` |
| `TAG` | `--tag` | no queda como campo — solo distingue el nombre de la carpeta |
| `OVERWRITE` | `--overwrite` | no queda registrado — solo controla si se reemplaza la carpeta |
| `NOMBRE` | (`git config user.name`, no es un flag de `run_experiment.py`) | `git.user` |

`config_hash`, `run_id`, `split_fingerprint` y `early_stopping_ab_hash` no vienen de
ninguna variable de la celda: se derivan de la combinación completa de las anteriores.

## Valores recomendados por sitio

Copiar directamente al bloque B de la celda de configuración (ventana física, 90-85%
de solape según el sitio):

| Sitio | `WINDOW_SECONDS` | `STEP_SECONDS` | `N_SPLITS` | `CLASS_WEIGHT` |
|---|---|---|---|---|
| NYU | 120 | 12 | 10 | `False` |
| Peking | 120 | 18 | 10 | `True` (desbalanceado) |
| NeuroIMAGE | 120 | 20 | 5 (39 sujetos) | `False` |
| OHSU | — (`REPRESENTACION="static"`, sin ventana) | — | 5 (66 sujetos) | `False` |

## Ejemplos completos, listos para copiar

Cada bloque reemplaza el bloque A-E entero de la celda de configuración. Están
etiquetados según su estatus:

- **[sintáctico]**: demuestra la forma de la configuración, no una elección metodológica.
- **[aprobado]**: coincide con una corrida ya versionada o con una convención acordada
  del proyecto (ver `methodology.md`/`limitations.md`).
- **[exploratorio]**: una variación válida para probar, pero sin resultados que la
  respalden todavía — no citarla como recomendación.

### [aprobado] LSTM sobre NYU, representación `ordered` — configuración base del artículo

```python
SITIO, ROI_SET, MODELO, HIPERPARAMS = "NYU", "12", "lstm", {}
REPRESENTACION, REPRESENTATION_SEED = "ordered", None
FISHER_Z, CONSTANT_POLICY = False, "zero"
WINDOW_SECONDS, STEP_SECONDS, OVERLAP = 120, 12, None
WINDOW_TR, STEP_TR, TR_SECONDS = None, None, None
WINDOW_SHAPE, GAUSSIAN_SIGMA = "rectangular", None
LR, BATCH_SIZE, EPOCHS, PATIENCE = 1e-4, 8, 150, 25
CLIPNORM, INNER_VAL_FRAC, START_FROM_EPOCH = None, 0.15, 0
EARLY_STOPPING_MONITOR, EARLY_STOPPING_MIN_DELTA = "val_loss", 1e-5
N_SPLITS, N_REPEATS, CLASS_WEIGHT, SEED = 10, 5, False, 42
DETERMINISTIC, MIXED_PRECISION = False, False
RANDOM_SUBSET, N_RANDOM_SETS, EXCLUDE_ROI_SET = None, 20, None
NOMBRE, CORREO = "Juan", "juan@ejemplo.com"
TAG, OVERWRITE, EJECUTAR_PRUEBA_HUMO = None, False, True
```

### [aprobado] BrainNetCNN sobre OHSU, representación `static` — sitio sin ventana válida

```python
SITIO, ROI_SET, MODELO = "OHSU", "12", "brainnetcnn"
HIPERPARAMS = {}
REPRESENTACION, REPRESENTATION_SEED = "static", None
FISHER_Z, CONSTANT_POLICY = False, "zero"
# Bloque B se ignora por completo con representación static: no editarlo.
WINDOW_SECONDS, STEP_SECONDS, OVERLAP = 120, 12, None
WINDOW_TR, STEP_TR, TR_SECONDS = None, None, None
WINDOW_SHAPE, GAUSSIAN_SIGMA = "rectangular", None
LR, BATCH_SIZE, EPOCHS, PATIENCE = 1e-4, 8, 150, 25
CLIPNORM, INNER_VAL_FRAC, START_FROM_EPOCH = None, 0.15, 0
EARLY_STOPPING_MONITOR, EARLY_STOPPING_MIN_DELTA = "val_loss", 1e-5
N_SPLITS, N_REPEATS, CLASS_WEIGHT, SEED = 5, 5, False, 42   # 5 folds: solo 66 sujetos
DETERMINISTIC, MIXED_PRECISION = False, False
RANDOM_SUBSET, N_RANDOM_SETS, EXCLUDE_ROI_SET = None, 20, None
NOMBRE, CORREO = "Juan", "juan@ejemplo.com"
TAG, OVERWRITE, EJECUTAR_PRUEBA_HUMO = None, False, True
```

### [exploratorio] Ventana gaussiana con sigma explícito sobre Peking

```python
SITIO, ROI_SET, MODELO, HIPERPARAMS = "Peking", "12", "deepsets", {"pooling": "mean"}
REPRESENTACION, REPRESENTATION_SEED = "ordered", None
FISHER_Z, CONSTANT_POLICY = False, "zero"
WINDOW_SECONDS, STEP_SECONDS, OVERLAP = 120, 18, None
WINDOW_TR, STEP_TR, TR_SECONDS = None, None, None
WINDOW_SHAPE, GAUSSIAN_SIGMA = "gaussian", 10          # sin resultados previos con sigma=10
LR, BATCH_SIZE, EPOCHS, PATIENCE = 1e-4, 8, 150, 25
CLIPNORM, INNER_VAL_FRAC, START_FROM_EPOCH = None, 0.15, 0
EARLY_STOPPING_MONITOR, EARLY_STOPPING_MIN_DELTA = "val_loss", 1e-5
N_SPLITS, N_REPEATS, CLASS_WEIGHT, SEED = 10, 5, True, 42   # Peking: desbalanceado
DETERMINISTIC, MIXED_PRECISION = False, False
RANDOM_SUBSET, N_RANDOM_SETS, EXCLUDE_ROI_SET = None, 20, None
NOMBRE, CORREO = "Juan", "juan@ejemplo.com"
TAG, OVERWRITE, EJECUTAR_PRUEBA_HUMO = None, False, True
```

### [sintáctico] Control anatómico con `RANDOM_SUBSET`

```python
SITIO, ROI_SET, MODELO, HIPERPARAMS = "NYU", "116", "deepsets", {}
REPRESENTACION, REPRESENTATION_SEED = "ordered", None
FISHER_Z, CONSTANT_POLICY = False, "zero"
WINDOW_SECONDS, STEP_SECONDS, OVERLAP = 120, 12, None
WINDOW_TR, STEP_TR, TR_SECONDS = None, None, None
WINDOW_SHAPE, GAUSSIAN_SIGMA = "rectangular", None
LR, BATCH_SIZE, EPOCHS, PATIENCE = 1e-4, 8, 150, 25
CLIPNORM, INNER_VAL_FRAC, START_FROM_EPOCH = None, 0.15, 0
EARLY_STOPPING_MONITOR, EARLY_STOPPING_MIN_DELTA = "val_loss", 1e-5
N_SPLITS, N_REPEATS, CLASS_WEIGHT, SEED = 10, 5, False, 42
DETERMINISTIC, MIXED_PRECISION = False, False
RANDOM_SUBSET, N_RANDOM_SETS, EXCLUDE_ROI_SET = 20, 20, None   # 20 de los 116 ROIs
NOMBRE, CORREO = "Juan", "juan@ejemplo.com"
TAG, OVERWRITE, EJECUTAR_PRUEBA_HUMO = None, False, True
```

## Ejemplo real: BrainNetCNN, el A/B `val_loss`/`val_bce`

Esta es la configuración de la corrida `val_loss` ya versionada
(`NYU_rois12_w60s6_brainnetcnn_a88f2eb7`) y la que reproduce su brazo `val_bce`. Los
valores de `config_hash`/`split_fingerprint`/`early_stopping_ab_hash` de abajo se
verificaron corriendo estos comandos con los datos reales del repositorio — no son
hipotéticos.

```python
SITIO, ROI_SET, MODELO = "NYU", "12", "brainnetcnn"
HIPERPARAMS = {"e2e": 4, "e2n": 8, "dense": 8, "dropout": 0.7, "leaky": 0.33,
               "l2_reg": 0.05, "inter_dropout": 0.6}
REPRESENTACION, REPRESENTATION_SEED = "ordered", None
FISHER_Z, CONSTANT_POLICY = False, "zero"
WINDOW_SECONDS, STEP_SECONDS, OVERLAP = 120, 12, None
WINDOW_TR, STEP_TR, TR_SECONDS = None, None, None
WINDOW_SHAPE, GAUSSIAN_SIGMA = "rectangular", None
LR, BATCH_SIZE, EPOCHS, PATIENCE = 1e-4, 32, 300, 25
CLIPNORM, INNER_VAL_FRAC, START_FROM_EPOCH = None, 0.15, 0
EARLY_STOPPING_MONITOR, EARLY_STOPPING_MIN_DELTA = "val_loss", 1e-5   # cambiar a "val_bce" para el otro brazo
N_SPLITS, N_REPEATS, CLASS_WEIGHT, SEED = 10, 5, False, 42
DETERMINISTIC, MIXED_PRECISION = False, False
RANDOM_SUBSET, N_RANDOM_SETS, EXCLUDE_ROI_SET = None, 20, None
NOMBRE, CORREO = "Juan", "juan@ejemplo.com"
TAG, OVERWRITE, EJECUTAR_PRUEBA_HUMO = None, False, True
```

| Campo | Brazo `val_loss` (ya corrido) | Brazo `val_bce` |
|---|---|---|
| `config_hash` | `a88f2eb7` | `7714baf4` |
| `run_id` | `NYU_rois12_w60s6_brainnetcnn_a88f2eb7` | `NYU_rois12_w60s6_brainnetcnn_7714baf4` |
| `split_fingerprint` | `aafd45ca73662139` | `aafd45ca73662139` (igual) |
| `early_stopping_ab_hash` | `2a4b6f1211c78ffc` | `2a4b6f1211c78ffc` (igual) |

`config_hash` y `run_id` cambian porque incluyen el monitor; `split_fingerprint` y
`early_stopping_ab_hash` se mantienen porque no dependen de él. Si al correr el brazo
`val_bce` alguno de los dos últimos sale distinto, algo más en la configuración cambió
(otra versión del código, otra semilla, otra ventana) y la comparación deja de ser
válida — ver la sección siguiente.

## Fold, repetición y OOF

- **Fold externo**: una de las `N_SPLITS` particiones de una repetición de la validación
  cruzada. Cada fold aparta un subconjunto de sujetos (`outer_val`) que el modelo nunca
  ve durante el ajuste ni la selección de época.
- **Repetición**: una pasada completa de `N_SPLITS` folds con una partición distinta
  (`RepeatedStratifiedKFold`). `N_REPEATS = 5` da cinco repeticiones.
- **Folds externos totales**: `N_SPLITS × N_REPEATS` (50 por defecto). `metrics_val.csv`
  y `metrics_train.csv` tienen una fila por fold externo — 50 estimaciones ruidosas del
  mismo modelo, no 50 repeticiones.
- **OOF (out-of-fold) por repetición**: para cada repetición, se juntan las predicciones
  de sus `N_SPLITS` folds — que cubren la muestra completa exactamente una vez— y se
  calcula cada métrica sobre esa muestra reconstruida. Da `N_REPEATS` valores (5 por
  defecto), menos ruidosos que promediar 50 estimaciones por fold. Es la estimación de
  referencia de una corrida; la tabla por fold es diagnóstico de dispersión.

## Criterios para declarar una corrida válida

La celda de validación (después de la sección de resultados) llama a
`compile_results.validate_run_artifacts(RUTA)` y exige que no reporte ningún problema.
Entre otras cosas, comprueba: que `config.json` traiga `config_schema_version`,
`n_splits`, `n_repeats` y `n_subjects` válidos (enteros en rango, no texto ni
booleanos); que `metrics_train.csv`/`metrics_val.csv` tengan exactamente
`N_SPLITS × N_REPEATS` filas con claves `(fold, repeat)` únicas y coincidentes entre
ambos archivos; que `history.csv` tenga una serie completa de épocas por fold y que el
valor registrado en `best_epoch` coincida con `best_monitor_value`; que
`restored_monitor_value` esté cerca de `best_monitor_value`; que `predictions_val.csv`
tenga `y_prob` en `[0,1]` y cubra cada sujeto una vez por repetición; y que cada fold de
`folds.csv` tenga sus tres particiones `fit`/`inner_val`/`outer_val` presentes y sin
solaparse entre sí. Si algo falla, la celda lanza `RuntimeError` con la lista completa
de problemas — no hay forma de continuar a la descarga o el push sin resolverlos.

## Descarga y subida

**Descarga** (opcional): comprime `results/runs/<ROI_SET>/<RUN_ID>` en
`/content/<RUN_ID>.zip` y lo ofrece para bajar desde el navegador; si la descarga
automática de Colab falla, el zip queda en esa ruta para bajarlo manualmente desde el
panel de archivos.

**Subida**: hace `git add results/runs/<ROI_SET>/<RUN_ID>`, commit, `git pull --no-rebase` y
`git push`. El token de GitHub nunca se escribe en el notebook: se lee del panel de
secretos de Colab (`GITHUB_TOKEN`) o, si no está configurado, se pide con
`getpass.getpass()` (no queda visible en pantalla ni en el historial de celdas). Si el
push falla, el mensaje de error se imprime con el token ya reemplazado por `***`.

## El A/B de `early_stopping_ab_hash`: qué debe coincidir

Comparar `val_loss` contra `val_bce` como monitor de `EarlyStopping` solo es válido si el
resto de la configuración es idéntica. `early_stopping_ab_hash` es la identidad completa
de la corrida sin el monitor — cubre sitio, ROIs, representación, ventana, arquitectura e
hiperparámetros, `lr`/`batch_size`/`epochs`/`patience`/`clipnorm`/`inner_val_frac`,
`class_weight`, semilla, `split_fingerprint`, y los hashes de datos/código/atlas—, así
que si coincide entre dos corridas, coinciden en todo eso; si no coincide, algo cambió y
la comparación no es válida aunque ambas corridas hayan terminado sin errores. La celda
de la corrida formal lo imprime apenas termina; compárelo a mano contra el de la corrida
ya hecha antes de lanzar el segundo brazo. `compile_results.py --stats --stats-by
early_stopping_monitor` hace esta misma verificación antes de comparar, y rechaza el par
si no coincide.

## `TAG` y `OVERWRITE`: cómo repetir una configuración

Cada corrida escribe en una carpeta cuyo nombre deriva de la configuración completa. Si
esa carpeta ya tiene una corrida terminada (`metrics_val.csv` presente),
`run_experiment.py` se detiene con `ESTA_CONFIGURACION_YA_SE_EJECUTO` en vez de
sobrescribirla. Hay dos formas explícitas de repetirla:

- `TAG = "algo"`: la corrida queda en una carpeta distinta (el tag entra en el nombre),
  sin tocar la anterior. No cambia `config_hash`: a efectos de agregación en
  `compile_results.py` sigue siendo la misma configuración metodológica, así que dos
  corridas con distinto `TAG` y el resto igual se agrupan juntas si se promedian.
- `OVERWRITE = True`: reemplaza la carpeta existente. Solo para cuando la corrida
  anterior estaba mal y se quiere descartar; no lo deje activo entre sesiones.

Una corrida **incompleta** (existe `config.json` pero no `metrics_val.csv` — por ejemplo,
Colab se desconectó a mitad) no necesita ni `TAG` ni `OVERWRITE`: se rehace sola, con un
aviso impreso, la próxima vez que se ejecute esa configuración.

**La carpeta de una corrida no debe renombrarse manualmente.** Su nombre es parte de su
identidad operativa y debe coincidir exactamente con `run_id` en `config.json`. Para
esquema 4, `validate_run_artifacts()` lo comprueba: una carpeta renombrada a mano queda
invisible para `compile_results.py --stats` aunque sus artefactos sigan siendo válidos.

## Trabajar en paralelo sin chocar

Cada corrida vive en su propia carpeta dentro de `results/runs/<ROI_SET>/`, así que dos personas
corriendo configuraciones distintas pueden hacer `git add`/`commit`/`push` sin conflicto
de fusión — están tocando archivos distintos. La celda de subida ya hace `git pull
--no-rebase origin main` antes del push por esta razón.

Si dos personas corren exactamente la **misma** configuración sin coordinarse, la segunda
en subir se va a encontrar con `ESTA_CONFIGURACION_YA_SE_EJECUTO` al intentar repetirla
localmente, o con una carpeta duplicada si ambas ya habían corrido antes de sincronizar.
`compile_results.py` detecta `config_hash` duplicados y avisa (o aborta, con
`--strict-comparability`) en vez de promediarlos como si fueran corridas independientes.

## Qué hacer si algo falla

**`ESTA_CONFIGURACION_YA_SE_EJECUTO`**: la configuración ya tiene una corrida completa en
esa carpeta. Use `TAG` para una carpeta nueva o `OVERWRITE` para reemplazarla a
propósito; no edite el CSV a mano.

**Corrida incompleta al reejecutar** (aviso, no error): existe `config.json` pero no
`metrics_val.csv` — típicamente porque Colab se desconectó a mitad de la corrida
anterior. `run_experiment.py` la detecta sola y la rehace desde cero; no hace falta
`TAG` ni `OVERWRITE` ni limpiar la carpeta a mano.

**Árbol de git sucio**: el aviso `el árbol de git tiene cambios sin confirmar` no detiene
la corrida — queda identificada por los hashes de código igual, pero nadie más puede
reconstruirla exactamente hasta que se haga commit. Antes de una corrida que se vaya a
subir, conviene confirmar los cambios primero.

**El `early_stopping_ab_hash` no coincide con el brazo complementario**: revise que
ambas corridas compartan sitio, ROIs, representación, ventana (segundos, no TR, o
viceversa), arquitectura e hiperparámetros, `lr`/`batch_size`/`epochs`/`patience`,
semilla y `TAG`/`OVERWRITE` aparte (`TAG`/`OVERWRITE` no entran en el hash, así que no
hace falta que coincidan). Un commit diferente no invalida por sí solo el A/B: el hash
cambia si cambia el contenido de alguno de los elementos que forman la identidad —por
ejemplo `run_experiment.py`, `data.py`, BOLD o atlas—, no por el número de commit en sí.
Cambios limitados al notebook, al compilador o a la documentación pueden conservar el
mismo hash aunque el commit sea distinto. La comprobación definitiva es comparar
`early_stopping_ab_hash`, no el commit.

**La celda de validación falla antes de descargar/subir**: lee la lista de problemas
impresa — nombra el archivo y el campo exactos. No hay forma de saltarse esta celda
desde el notebook; si el problema es real, hay que volver a correr la configuración.

**El push falla**: el mensaje de error ya viene sin el token (se reemplaza por `***`
antes de imprimirse). La causa más común es que alguien más subió resultados entre el
`git pull` de la celda de subida y el intento de push — reejecute esa celda, que vuelve a
traer lo nuevo antes de reintentar.

**GPU no disponible**: la celda 1 lo avisa (`SIN GPU`) pero no detiene el notebook; la
corrida sigue en CPU, mucho más lenta. Para una prueba de humo está bien; para la corrida
formal conviene reasignar entorno de ejecución con GPU antes de continuar.

**Colab se desconecta a mitad de la corrida formal**: reejecutar el notebook desde la
sección 1. La carpeta de la corrida queda incompleta (ver arriba) y se rehace sola sin
intervención manual.
