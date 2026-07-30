# Registro breve de experimentación

Última actualización: 2026-07-29  
Alcance actual: NYU, Peking, NeuroIMAGE y OHSU; conjuntos de 12, 18, 39 y 116 ROIs; BrainNetCNN.

## Para qué sirve este archivo

Este documento evita repetir experimentos y orienta las siguientes corridas. No sustituye
`config.json`, `resumen.md` ni los CSV de cada corrida: esos artefactos siguen siendo la
fuente de verdad. Algunas corridas resumidas aquí todavía pueden estar almacenadas como ZIP
fuera de este checkout; el `run_id` o el `config_hash` permite identificarlas.

## Convención de métricas

La tabla comparativa principal de este documento usa métricas **OOF por repetición**:

1. dentro de cada repetición se reúnen las predicciones externas de todos los folds;
2. se calcula cada métrica sobre todos los sujetos de esa repetición;
3. se informa la media y la desviación estándar entre las cinco repeticiones.

Esto es distinto de lo que informa `resumen.md` de cada corrida, que promedia las métricas
calculadas fold a fold: con folds pequeños ese promedio es más ruidoso, así que puede mostrar
una media y una desviación algo distintas de las de este documento. Ninguna de las dos
salidas está mal — miden lo mismo con dos agregaciones distintas —, pero para comparar
conjuntos de ROIs o sitios entre sí se prioriza siempre el OOF por repetición.

## Resultados actuales

`AUC / balanced accuracy / accuracy OOF` (porcentaje, media entre repeticiones):

| Sitio | 12 ROIs | 18 ROIs | 39 ROIs, baseline | 116 ROIs |
|---|---:|---:|---:|---:|
| NYU | 59.05 / 57.45 / 57.40 | 60.00 / 56.70 / 56.72 | 52.83 / 52.27 / 52.20 | 53.93 / 53.09 / 53.11 |
| NeuroIMAGE | 47.38 / 44.33 / 45.64 | 61.98 / 57.65 / 57.95 | 50.75 / 53.72 / 54.87 | 51.93 / 50.32 / 51.79 |
| OHSU | 54.92 / 53.57 / 54.55 | 52.22 / 51.07 / 52.42 | 50.13 / 46.95 / 47.58 | 55.83 / 54.15 / 54.24 |
| Peking | 56.37 / 54.11 / 55.52 | 56.75 / 53.98 / 55.41 | 62.22 / 60.31 / 61.20 | 60.33 / 58.09 / 59.02 |

- Las cuatro corridas activas de Peking usan `class_weight=True`; ya no hay una diferencia de
  `class_weight` entre columnas de esta fila (ver la sección "Peking: corridas activas y su
  procedencia" más abajo).
- Las corridas OHSU actuales usan `ordered`, producen solo seis ventanas y están registradas
  como exploratorias mientras se decide el baseline estático.
- El conjunto de 39 tiene una composición anatómica diferente; no es una ampliación del de
  12 o 18 (ver `data/atlas/roi_sets.json`). El de 116 es el atlas AAL116 completo.
- Las corridas realizadas con `git.clean=false` son descriptivas y tienen hashes de código,
  pero su procedencia es menos fuerte que la de una corrida ejecutada con árbol limpio. No
  son corridas inválidas. Tres de ellas (`2b729a8c`, `299719fe`, `bc841110`) tienen además
  una nota de procedencia del equipo que las acepta para análisis formal sin exigir réplica
  — ver "Nota de procedencia" más abajo.

**Conclusión (prudente):** el conjunto de 12 ROIs es la opción más parsimoniosa y sigue
siendo competitivo, pero los resultados no demuestran que sea universalmente el conjunto más
informativo. El conjunto con la estimación puntual más alta cambia según el sitio: 18 ROIs en
NYU y NeuroIMAGE, 116 en OHSU y 39 en Peking. Es decir, 12 ROIs no es el máximo en ningún
sitio, aunque queda cerca del máximo en NYU (59.05 frente a 60.00) y en Peking (56.37 frente
a 56.75).

Estas son comparaciones de **estimaciones puntuales**, no evidencia de superioridad: no se ha
ejecutado ninguna comparación pareada con intervalos entre tamaños, y las diferencias de unas
décimas están muy por debajo de la precisión disponible. El análisis formal de esta pregunta
está especificado en el plan de análisis estadístico de comparación de 12/18/39/116 ROIs y no
debe anticiparse desde esta tabla. No se declara superioridad estadística ni generalización
entre sitios.

### Comparabilidad y procedencia por sitio

`class_weight` y `git.clean` de las cuatro corridas activas (12/18/39/116 ROIs) de cada sitio:

- NYU 12/18/39/116: `class_weight=False`, configuraciones comparables y árbol limpio en las
  cuatro.
- NeuroIMAGE 12/18/39/116: `class_weight=False`; 18, 39 y 116 con árbol limpio, 12
  (`2b729a8c`) con `git.clean=false`.
- OHSU 12/18/39/116: `class_weight=False`; 12, 18 y 116 con árbol limpio, 39 (`299719fe`) con
  `git.clean=false`.
- Peking 12/18/39/116: `class_weight=True`; 18, 39 y 116 con árbol limpio, 12 (`bc841110`) con
  `git.clean=false` (detalle completo en la sección siguiente).

Las cuatro corridas de 116 ROIs se ejecutaron con árbol limpio y no añaden salvedades de
procedencia nuevas. Dentro de cada sitio comparten `split_fingerprint`, `bold_hash`,
`data_code_hash`, `runner_code_hash`, arquitectura y `folds.csv` con las de 12, 18 y 39, así
que las cuatro configuraciones de un mismo sitio son comparables de forma pareada sujeto a
sujeto.

### Nota de procedencia: `git.clean=false` en `2b729a8c`, `299719fe` y `bc841110`

`git.clean=false` en NeuroIMAGE–12 `2b729a8c`, OHSU–39 `299719fe` y Peking–12 `bc841110` se
debió exclusivamente a archivos nuevos o modificados bajo `results/` o documentación en el
momento de esas corridas. No había cambios sin confirmar en `run_experiment.py`, `data.py`,
BrainNetCNN, las representaciones ni el protocolo de entrenamiento — así lo confirmó el
equipo posteriormente.

Evidencia disponible que respalda esa declaración, verificada contra los `config.json`
reales:

- las tres corridas pasan `validate_run_artifacts()` (esquema 4, artefactos completos y
  estructuralmente válidos);
- dentro de cada sitio, `runner_code_hash`, `data_code_hash`, `bold_hash` y
  `split_fingerprint` son idénticos entre la corrida `git.clean=false` y sus dos
  comparadoras de árbol limpio — mismo código de ejecución, mismo código de datos, misma
  señal BOLD, mismas particiones;
- `arch_json`, `lr`, `batch_size`, `epochs`, `patience`, `seed`, `n_splits`, `n_repeats`,
  `early_stopping_monitor` y `class_weight` también coinciden dentro de cada sitio.

Limitación reconocida: `git.clean=false` por sí solo no guarda la lista histórica de qué
archivo estaba modificado en el momento de la corrida; la declaración anterior depende de la
memoria/registro del equipo, no de un artefacto verificable automáticamente con el estado
actual del repositorio.

Decisión: estas tres corridas se aceptan para análisis y comparación formal con esta
salvedad de trazabilidad documentada, sin exigir una réplica de árbol limpio. No se
reescribe `config.json` de ninguna de las tres, y no se repiten únicamente para cambiar
`git.clean=false` a `true`, buscar una semilla favorable, ni para crear una copia con
metadatos alterados. Una repetición futura solo estaría justificada por evidencia de
cambios ejecutables no declarados, un cambio de protocolo científico, o una réplica
independiente por una razón experimental nueva — ninguna de esas condiciones aplica hoy.

### NeuroIMAGE–39 y OHSU–39: corridas nuevas

- NeuroIMAGE–39 `dc028168`: `class_weight=False`, `git.clean=true`.
- OHSU–39 `299719fe`: `class_weight=False`, `git.clean=false` — cubierta por la nota de
  procedencia anterior; no se repite.

Métricas completas:

| Corrida | Accuracy | AUC | F1-macro | Balanced accuracy |
|---|---:|---:|---:|---:|
| NeuroIMAGE–39 `dc028168` | 54.87% | 50.75% | 53.52% | 53.72% |
| OHSU–39 `299719fe` | 47.58% | 50.13% | 46.68% | 46.95% |

### 116 ROIs (AAL116 completo): baseline diagnóstico en los cuatro sitios

Corridas activas, con su identificador real — no deben renombrarse:

| Sitio | `run_id` | Accuracy | AUC | F1-macro | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| NYU | `NYU_rois116_w60s6_brainnetcnn_control_baseline_v13_160b89cd` | 53.11% | 53.93% | 53.04% | 53.09% |
| NeuroIMAGE | `NeuroIMAGE_rois116_w61s6_brainnetcnn_control_baseline_v13_669d72bd` | 51.79% | 51.93% | 50.04% | 50.32% |
| OHSU | `OHSU_rois116_w48s5_brainnetcnn_control_baseline_v13_f82f17b4` | 54.24% | 55.83% | 53.66% | 54.15% |
| Peking | `Peking_rois116_w60s6_brainnetcnn_240732d1` | 59.02% | 60.33% | 57.86% | 58.09% |

Notas:

- Las cuatro se ejecutaron con `git.clean=true`. `class_weight` sigue la convención de su
  sitio: `False` en NYU, NeuroIMAGE y OHSU, `True` en Peking.
- La corrida de Peking, `Peking_rois116_w60s6_brainnetcnn_240732d1`, **no lleva la etiqueta
  `control_baseline_v13`** que sí llevan las otras tres. Es el identificador real y no se
  renombra; su configuración se valida por `config.json`, nunca por el nombre. Su carpeta
  contiene además `peking_dummy.txt`, que no forma parte del contrato de siete artefactos, se
  ignora y no invalida la corrida.
- Estas cifras son el baseline diagnóstico que se pedía antes de considerar cualquier barrido
  de hiperparámetros sobre 116 ROIs. Ese barrido sigue sin iniciarse y no se recomienda
  todavía.

### Peking: corridas activas y su procedencia

- Peking–12 `bc841110`: `class_weight=True`, `git.clean=false` — cubierta por la nota de
  procedencia de más arriba; sigue siendo la única referencia actual de 12 ROIs para este
  sitio y no se repite.
- Peking–18 `0bf7fa0e`: `class_weight=True`, `git.clean=true`, baseline vigente. Sustituye a
  la corrida histórica `b8e8a44d` (`class_weight=False`), que no forma parte de esta versión
  del compendio.
- Peking–39 `396e34d2`: `class_weight=True`, `git.clean=true`, baseline vigente.

Métricas completas de las dos corridas nuevas:

| Corrida | Accuracy | AUC | F1-macro | Balanced accuracy |
|---|---:|---:|---:|---:|
| Peking–18 `0bf7fa0e` | 55.41% | 56.75% | 53.89% | 53.98% |
| Peking–39 `396e34d2` | 61.20% | 62.22% | 60.10% | 60.31% |

- Con `class_weight=True`, Peking–18 elevó sensibilidad respecto a la ejecución histórica sin
  pesos, pero no mejoró AUC ni balanced accuracy.
- Peking–39 es el mejor modelo individual actual de Peking, aunque conserva una brecha alta
  entre entrenamiento y validación.
- No se declara generalización externa ni superioridad estadística a partir de estas cifras.

### Ensamble exploratorio Peking 18+39

Análisis post hoc con `analyze_ensemble.py` (pesos iguales, umbral fijo 0.5, sin optimizar
nada), combinando `0bf7fa0e` y `396e34d2`:

```text
Peking 18+39, pesos iguales:
accuracy 61.09%
AUC 62.80%
F1-macro 59.99%
balanced accuracy 60.18%
```

Frente al modelo Peking–39 individual, el ensamble solo mejora AUC en aproximadamente 0.58
puntos porcentuales y no mejora accuracy ni balanced accuracy. Se registra como análisis
exploratorio, no como nueva referencia: no se promueve todavía.

### Nota sobre `atlas_hash`

Las corridas anteriores y posteriores a la corrección textual de `roi_sets.json` pueden tener
`atlas_hash` y `config_hash` distintos aunque compartan exactamente los mismos índices ROI.
Para establecer equivalencia científica se comprobaron `roi_indices_hash`, `bold_hash`,
particiones, arquitectura e hiperparámetros. No se modifica retroactivamente ningún hash.

## Próximas corridas (todo el compendio)

Documentadas aquí, sin ejecutarlas salvo que se indique lo contrario.

**`NeuroIMAGE–12` `2b729a8c`, `OHSU–39` `299719fe` y `Peking–12` `bc841110` NO están en esta
lista.** Su `git.clean=false` ya está explicado y aceptado por la nota de procedencia de
más arriba; no se repiten solo para cambiar `git.clean=false` a `true`, y no hay ninguna
instrucción pendiente de replicarlas en esta versión.

Tareas metodológicas todavía abiertas — decisiones que corresponden al equipo, no
correcciones de artefactos:

1. Resolver una única política de folds para NeuroIMAGE y OHSU antes de comparaciones
   definitivas; no comparar `5×5` con `10×5` como si la única diferencia fueran los ROIs.
2. Evaluar OHSU `static` con 12 y 18 ROIs usando exactamente el mismo protocolo.
3. Evaluar el ensamble NYU 12+18.
4. Ejecutar el análisis estadístico pareado de 12/18/39/116 ROIs conforme a su plan
   congelado; no anticipar conclusiones desde la tabla de estimaciones puntuales de arriba.

Ya completadas:

- ~~Peking–18 baseline con `class_weight=True`~~ — completada (`0bf7fa0e`); no es necesario
  repetirla en esta fase.
- ~~Ensamble Peking 18+39~~ — evaluado exploratoriamente (ver arriba); no se promueve a
  referencia y no es necesario repetirlo en esta fase.
- ~~Ejecutar 116 ROIs como baseline diagnóstico~~ — completada en los **cuatro** sitios (ver
  "116 ROIs (AAL116 completo)" más arriba), no solo en Peking y NYU como se había planeado.
  Sigue vigente la indicación de no iniciar un barrido de hiperparámetros sobre 116.

Configuración usada en Peking–18 `0bf7fa0e` (ya ejecutada; la única diferencia frente a la
corrida histórica `b8e8a44d` fue `class_weight=False → True` y la etiqueta de la nueva
corrida):

```text
site=Peking
roi_set=18
model=brainnetcnn
representation=ordered
window=60 TR / 120 s
step=6 TR / 12 s
e2e=4
e2n=8
dense=8
dropout=0.7
leaky=0.33
l2_reg=0.05
inter_dropout=0.6
lr=1e-4
batch_size=32
epochs=300
patience=25
inner_val_frac=0.15
start_from_epoch=0
early_stopping_monitor=val_loss
early_stopping_min_delta=1e-5
n_splits=10
n_repeats=5
class_weight=True
seed=42
mixed_precision=False
```

## Historial: NYU, 12 ROIs (referencia previa a esta actualización)

Las secciones siguientes documentan el trabajo de ablación y ajuste hecho exclusivamente
sobre NYU con 12 ROIs, antes de ampliar el compendio a los demás sitios y conjuntos de ROIs
listados arriba. Se conservan como registro histórico; no se reescriben.

Las cifras de esta sección también son métricas **OOF por repetición** (ver la convención
arriba). Diferencias de unas décimas no deben interpretarse como mejora sin una comparación
pareada.

## Referencia vigente

Configuración que debe mantenerse fija al evaluar un cambio:

- NYU, 12 ROIs, BrainNetCNN y representación `ordered`.
- Ventana 120 s (60 TR), paso 12 s (6 TR), rectangular y sin Fisher z.
- `e2e=4`, `e2n=8`, `dense=8`, `dropout=0.7`, `leaky=0.33`,
  `l2_reg=0.05`, `inter_dropout=0.6`.
- `lr=1e-4`, batch 32, máximo 300 épocas, paciencia 25 y monitor `val_loss`.
- Validación 10 × 5, `inner_val_frac=0.15`, semilla 42 y sin `class_weight`.

Corrida de referencia formal de la versión 13:
`NYU_rois12_w60s6_brainnetcnn_control_baseline_v13_3e220e5c`.

Resultado OOF: accuracy 57.40 ± 2.79 %, AUC 59.05 ± 2.73 %, F1-macro
57.31 ± 2.77 % y balanced accuracy 57.45 ± 2.78 %.

La referencia histórica `a88f2eb7`, ejecutada antes del cambio operativo de layout de
resultados, produjo accuracy OOF 56.95 % y AUC OOF 58.98 %. La concordancia es buena y la
comparación pareada no detectó diferencias; `3e220e5c` queda como control formal porque
comparte `runner_code_hash` con los nuevos experimentos de v13.

## Qué se probó

| Cambio frente al control | Evidencia | Accuracy/AUC OOF | Decisión actual |
|---|---|---:|---|
| Máximo 150 épocas | `f7ada452` | 54.80 / 57.75 | Inferior; conservar 300 épocas. |
| Paso 24 s, 80 % de solapamiento, 150 épocas | `8e831252` | 52.77 / 52.64 | Inferior al control de 150 épocas; no repetir. |
| `ordered_scaled`, 150 épocas | `6645839e` | 53.90 / 55.33 | No mejoró frente a `ordered` con 150 épocas. |
| Batch 16, 300 épocas | `a9781609` | 56.50 / 59.31 | Resultado mixto, esencialmente empatado y más costoso; no es prioridad. |
| `l2_reg=0.01` | `ee291ab8` | 53.22 / 56.46 | Inferior; mantener 0.05. |
| `inter_dropout=0.3` | `ed0eed10` | 55.71 / 58.58 | Sin mejora consistente; mantener 0.6. |
| Monitor `val_bce`, sin warm-up | `7714baf4` | 54.01 / 55.56 | Inferior; mantener `val_loss`. |
| `val_bce`, warm-up 150, paciencia 25 | `80329a25` | 56.27 / 58.40 | No supera la referencia. |
| `val_bce`, warm-up 150, paciencia 75 | `410c2892` | 56.72 / 58.82 | No supera la referencia y aumenta el costo. |
| Batch 16 y máximo 500 épocas | `4c133d74` | 55.03 / 57.76 | No mejoró al batch 16/300; no repetir. |

También se evaluaron previamente Fisher z, conectividad `shrunk`, `static` y `mean`. No se
observó una ventaja consistente sobre Pearson `ordered`. No deben repetirse exactamente
las mismas configuraciones salvo que exista una hipótesis nueva o sea necesario reconstruir
una comparación formal con la versión vigente del código.

Interpretación prudente: estas decisiones se refieren a NYU, 12 ROIs y el protocolo actual.
"No priorizar" no significa que una técnica sea inútil en otros sitios, atlas o modelos.

## Análisis de errores ya realizado

En la repetición limpia del control, doce sujetos quedaron mal clasificados en sus cinco
predicciones OOF:

- Controles: `NYU-10004`, `NYU-10093`, `NYU-10110`, `NYU-3518345`,
  `NYU-3650634`, `NYU-4562206`.
- TDAH: `NYU-10050`, `NYU-10107`, `NYU-10118`, `NYU-10129`,
  `NYU-3174224`, `NYU-3653737`.

Son candidatos para inspeccionar movimiento, fenotipo, calidad de señal y posibles casos
atípicos. **No deben eliminarse por haber sido difíciles de clasificar**: hacerlo después de
ver las predicciones sesgaría la estimación de desempeño.

## Qué conviene probar ahora

Prioridad sugerida, siempre cambiando un solo factor respecto a la referencia:

1. **Ventana gaussiana:** misma ventana 120 s/paso 12 s, con
   `WINDOW_SHAPE="gaussian"` y dos corridas separadas con `GAUSSIAN_SIGMA=20.0` y
   `GAUSSIAN_SIGMA=30.0` (sigma se expresa en TR). No usar `None` en esta prueba: el valor
   automático `window/6` concentra demasiado los pesos para el ancho de banda de Athena.
2. **Ventana algo más larga:** `WINDOW_SECONDS=140`, `STEP_SECONDS=14`,
   `WINDOW_TR=STEP_TR=None` y forma rectangular. Solo si la gaussiana no ayuda; compara
   estabilidad de la correlación frente al número de ventanas.
3. **Capacidad de BrainNetCNN, barrido mínimo:** antes de correr, recuperar los
   `config.json` de las primeras corridas sin etiquetas claras. Probar únicamente anchos no
   cubiertos, uno por vez; por ejemplo una corrida con `e2e=8` y otra, separada, con
   `e2n=16`, manteniendo todos los demás valores de la referencia. Evitar una cuadrícula
   grande.
4. Repetir una configuración nueva únicamente si supera de forma coherente a la referencia;
   la réplica debe conservar el mismo `config_hash` y usar otro `TAG`.

No son experimentos de optimización: `DETERMINISTIC`, `MIXED_PRECISION` y el tipo de
máquina. Son controles de ejecución. Tampoco se debe buscar una semilla favorable, activar
`class_weight` en NYU sin una hipótesis específica, elegir el mejor fold ni ajustar decisiones
con `outer_val`.

## Regla para aceptar una mejora

- Misma semilla, particiones, sujetos y versión de datos que la referencia.
- Un solo cambio metodológico por comparación.
- `preflight`, prueba de humo y validación final sin fallos.
- Comparar principalmente AUC, balanced accuracy y F1-macro OOF por repetición.
- Usar la comparación pareada/corregida del proyecto; no decidir por el mejor fold ni por
  una diferencia marginal en una sola métrica.
- Confirmar una mejora prometedora con una repetición antes de convertirla en referencia.

## Cómo actualizar este registro

Al terminar una corrida aprobada, añadir una fila con: cambio único, `run_id` o
`config_hash`, métricas OOF principales y decisión (`prometedora`, `no mejora`,
`inconclusa`). Si cambian varios factores, marcarla como **inconclusa/confundida** y no
atribuir el resultado a uno de ellos.
