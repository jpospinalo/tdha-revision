# `analysis/roi_comparison/`

Análisis comparativo del desempeño de BrainNetCNN entre los cuatro tamaños de
conjunto de ROIs (12, 18, 39, 116) en los cuatro sitios (NYU, Peking,
NeuroIMAGE, OHSU), sobre las 16 corridas ya existentes en `results/runs/`.

Implementa, sin rediseñarlo, el plan estadístico congelado en
`analysis_plan.md` (versión 5.6) y la resolución de sus decisiones D1–D5
pendientes de equipo (ver más abajo). `results/` es la fuente de datos y
permanece de solo lectura.

> **Nota sobre `analysis_plan.md`:** es copia exacta, byte por byte, del
> texto final de la versión 5.6 del plan aprobado por el equipo. Su SHA-256
> canónico esperado es `199857a46006a082d97f6a055ffdaaa075fd25be87bbb4147e806aae28367163`;
> `run_statistical_analysis.py` lo verifica contra ese valor antes de iniciar
> el bootstrap y se detiene con código de salida distinto de cero si el
> archivo falta o no coincide. El hash también queda registrado en
> `outputs/analysis_manifest.json` (`plan_sha256`).
>
> La resolución D1–D5 (más abajo) complementa el plan congelado: no lo
> modifica ni lo reinterpreta, solo cierra las decisiones que el plan dejó
> explícitamente para el equipo.
>
> [`REVISION_PLAN_ANALISIS_ESTADISTICO.md`](REVISION_PLAN_ANALISIS_ESTADISTICO.md)
> registra la discusión posterior sobre el plan: confirma que D2 quedó cerrado
> como estimación sin margen y deja constancia de cuatro aperturas evaluadas
> (síntesis entre sitios, metaanálisis, QC por pliegue y reapertura de D2) que
> se descartaron, con su razón técnica. Es un registro de decisiones, **no** el
> plan ni una versión alternativa de este.

## Cómo reproducir desde cero

Sin modificar nada bajo `results/`:

```bash
python analysis/roi_comparison/scripts/build_analysis_dataset.py \
  --repo-root . \
  --config analysis/roi_comparison/config/analysis_config.json \
  --manifest analysis/roi_comparison/config/run_manifest.csv \
  --output-dir analysis/roi_comparison/outputs

python analysis/roi_comparison/scripts/run_statistical_analysis.py \
  --repo-root . \
  --config analysis/roi_comparison/config/analysis_config.json \
  --manifest analysis/roi_comparison/config/run_manifest.csv \
  --input-dir analysis/roi_comparison/outputs/data \
  --output-dir analysis/roi_comparison/outputs
```

Ambos scripts se niegan a sobrescribir salidas existentes salvo que se pase
`--overwrite`. `build_analysis_dataset.py` acepta además `--validate-only`
para ejecutar solo la auditoría de comparabilidad, sin construir el dataset.

Las decisiones científicas congeladas de `analysis_config.json` se validan
una sola vez, con `validate_analysis_config()` (definida en
`build_analysis_dataset.py` y reutilizada por `run_statistical_analysis.py`;
no hay dos validadores independientes). Una discrepancia detiene ambos
scripts con el campo, el valor recibido y el valor esperado; no se corrige
el JSON automáticamente ni se acepta un valor alternativo por línea de
comandos.

Ambos scripts escriben de forma segura: calculan todo en memoria, lo
serializan primero a un directorio de staging dentro de `outputs/`, y solo
promueven los archivos a sus nombres finales (`os.replace`) después de que
todas las serializaciones de ese lote terminaron sin errores. No quedan
tablas científicas parciales si algo falla a mitad de camino.

Git mejora la trazabilidad cuando está disponible (commit y `git status`
antes/después quedan registrados en el manifiesto), pero **no es necesario
para verificar la integridad de las entradas**: esa verificación se hace
siempre por SHA-256 (`results/README.md`, los siete artefactos de las 16
corridas, y `requirements.txt`/`tdha_experimentos.ipynb`/`src/`), comparando
un inventario capturado antes y después de la ejecución. Esto permite
ejecutar correctamente el análisis desde un ZIP descargado en Colab, sin
repositorio Git presente; en ese caso el manifiesto registra explícitamente
`git_provenance_status: "unavailable"` y dejar los campos de Git en `null`
en vez de inventar una lista de cambios vacía.

Pruebas:

```bash
python -m unittest discover -s analysis/roi_comparison/tests -v
```

Notebook (`roi_comparison.ipynb`): abrir y ejecutar "Restart and run all" en
CPU. Reproduce exactamente las mismas tablas y figuras que los scripts,
porque los llama directamente; no reimplementa ninguna fórmula.

## Entradas

Las 16 corridas listadas en `config/run_manifest.csv` (4 sitios × 4 tamaños
de ROI), leyendo de cada una `config.json`, `folds.csv`,
`predictions_val.csv` y, solo para comprobaciones estructurales,
`metrics_val.csv`. La configuración metodológica (métrica primaria, margen,
semilla del bootstrap, etc.) vive en `config/analysis_config.json` y no es
modificable desde la línea de comandos.

## Salidas

- `outputs/data/subject_scores.csv` — probabilidad por sujeto, sitio, tamaño
  de ROI y repetición (1.860 filas: 465 sujetos × 4 tamaños).
- `outputs/data/metrics_by_repeat.csv` — las seis métricas por repetición
  (80 filas: 4 sitios × 4 tamaños × 5 repeticiones).
- `outputs/tables/comparability_audit.csv` — auditoría de las 16 corridas
  (artefactos, hashes científicos —incluido `config_hash` y el SHA-256 de
  `folds.csv`—, pareamiento, `git.clean`, reconciliación con el README).
- `outputs/tables/descriptive_performance.csv`,
  `primary_12_vs_116.csv`, `precision_diagnostics.csv`,
  `secondary_pairwise_comparisons.csv`, `secondary_metric_intervals.csv`,
  `error_analysis_summary.csv` — resultados del análisis estadístico.
- `outputs/data/error_analysis_long.csv`, `subject_error_profiles.csv` —
  análisis de errores por sujeto (12 vs. 116).
- `outputs/figures/` — perfiles por ROI y forest plot del contraste
  principal, en SVG y PNG; más tres figuras adicionales de contrastes
  secundarios frente a 116 (ver "Figuras de contrastes secundarios frente a
  116" más abajo).
- `outputs/analysis_manifest.json` — hashes (incluidos los 16 `config_hash`),
  versiones, parámetros del bootstrap, tiempos observados, resumen de la
  resolución D1–D5, estado de Git antes/después cuando está disponible, e
  inventario de hashes de insumos antes/después (base de `results_read_only`,
  que nunca se infiere de la disponibilidad de Git).

## Métrica primaria

AUC OOF media de las cinco repeticiones (`mean_repeat_oof_auc`): cada
repetición agrupa las predicciones *out-of-fold* de sus 10 folds y se calcula
`sklearn.metrics.roc_auc_score` una vez sobre ese conjunto agrupado; el
resultado final es la media aritmética de las cinco repeticiones. No se
promedian probabilidades entre repeticiones antes de calcular AUC (eso
correspondería a un ensamble, un estimando distinto y no usado aquí — ver D5
más abajo).

## Estado de preinscripción (plan 5.6, secciones 2 y 14)

Este análisis **no es una preinscripción prospectiva ciega a los resultados**.
Durante la revisión de factibilidad que precedió al plan 5.6 se calcularon
varianzas de las diferencias por pliegue y quedaron visibles las diferencias
medias observadas entre 12 y 116 ROIs en los cuatro sitios; el plan se cerró
después de esa exposición. En consecuencia: los resultados de este análisis
se presentan como estimación con apoyo exploratorio, nunca como confirmación
definitiva; cualquier afirmación confirmatoria requeriría una cohorte o
conjunto de datos externo que no haya participado en estas decisiones; y el
margen de no inferioridad (D2) no se define ni se ajusta a partir de estos
resultados ya conocidos. Ver el plan, secciones 1 y 2, para el detalle
completo de esta limitación.

## Resolución D1–D5 (decisiones que el plan 5.6 dejó para el equipo)

- **D1:** AUC primaria como se describió arriba; balanced accuracy,
  F1-macro, sensibilidad y especificidad como secundarias; accuracy solo
  como auditoría (nunca en las tablas científicas).
- **D2 — sin margen de no inferioridad:** `noninferiority_margin` y
  `noninferiority_margin_rationale` son `null` en `analysis_config.json`. El
  análisis es de estimación pura: reporta diferencias puntuales e intervalos
  bootstrap bilaterales del 95%, y deliberadamente **no** produce un
  dictamen binario de no inferioridad. La ausencia de un margen no es una
  omisión: es la decisión del equipo, justificada porque (1) no hay un
  margen de pérdida de AUC científicamente justificable con independencia de
  estos resultados, (2) el equipo ya conocía diferencias y varianzas al
  cerrar el análisis, por lo que elegir un margen ahora sería una prueba
  retrospectiva, y (3) la precisión observada (ver
  `precision_diagnostics.csv`) es contexto sobre lo que estos datos pueden
  resolver, no una razón para fijar o rechazar un margen.
- **D3 — sin efecto combinado, sin heterogeneidad afirmada:** cada sitio se
  presenta por separado. El análisis no estima ni contrasta un efecto común
  entre sitios, y tampoco afirma ni niega heterogeneidad estadística: el
  diseño (cuatro sitios, bootstrap condicional a una sola partición
  existente) no permite distinguir heterogeneidad real de variabilidad de
  muestreo. La narrativa exacta se deriva en tiempo de ejecución de
  `primary_12_vs_116.csv` (nunca se escribe a mano): si los cuatro intervalos
  bilaterales del contraste 12−116 comparten una región común
  (`max(ci_low) <= min(ci_high)`), se usa una redacción que señala esa
  región común y advierte que no permite concluir heterogeneidad ni un
  efecto común; si no la comparten, se describe el solapamiento de cada uno
  de los seis pares de sitios, con la misma advertencia de cierre. Ver
  `generate_d3_narrative()` en `scripts/run_statistical_analysis.py`.
- **D4 — bootstrap condicional:** el remuestreo es pareado (mismos índices
  de sujeto reutilizados para los cuatro tamaños de ROI, las cinco
  repeticiones y todas las métricas de un sitio) y estratificado por clase.
  Está condicionado a las predicciones, los entrenamientos y las cinco
  particiones de validación cruzada ya existentes: no captura la
  variabilidad de reentrenar el modelo o de repartir los datos de nuevo.
- **D5 — estimando:** desempeño medio del *pipeline* de validación cruzada
  (cinco repeticiones de 10-fold), no el de un ensamble de probabilidades
  promediadas entre repeticiones, ni el de un único modelo final desplegado.
- **Repeticiones de validación cruzada:** no se usan las cinco repeticiones
  de validación cruzada como cinco observaciones independientes para
  pruebas Friedman/Wilcoxon. Las repeticiones reutilizan sujetos y
  entrenamientos solapados; la inferencia oficial se mantiene en estimación
  con bootstrap pareado por sujeto.

## Figuras de contrastes secundarios frente a 116

`outputs/figures/` incluye, además de las dos figuras del §10 de las
instrucciones de implementación (`paired_roi_profiles`,
`primary_contrast_forest`), tres figuras adicionales que **no** forman parte
del conjunto original especificado — se agregaron después, a pedido, para
visualizar contrastes que ya estaban calculados pero no graficados:

- `secondary_contrast_18_vs_116_forest.{svg,png}` — contraste 18−116 por sitio.
- `secondary_contrast_39_vs_116_forest.{svg,png}` — contraste 39−116 por sitio.
- `contrasts_vs_116_forest.{svg,png}` — los tres contrastes frente a 116
  (12, 18 y 39) lado a lado, en la misma escala horizontal.

Estas figuras no recalculan nada: leen directamente `primary_12_vs_116.csv`
y las filas de AUC de `secondary_pairwise_comparisons.csv` para los
contrastes `18-116` y `39-116`, que ya existían como dos de los cinco
contrastes secundarios definidos en el plan (§8). Importante para su
interpretación:

- **12−116 es el único contraste primario y preespecificado.** 18−116 y
  39−116 son secundarios y exploratorios: el plan pide reportarlos "sin
  declaraciones de significancia" y sin corrección por comparaciones
  múltiples (§8), precisamente porque se leen varios contrastes sin ajustar.
  Un intervalo que no cruza cero entre varios sin ajustar no es, por sí
  solo, evidencia fuerte — es exactamente el patrón esperable por azar al
  mirar suficientes comparaciones sin corrección.
- Se aplican las mismas reglas D2/D3/D5 que al contraste principal: sin
  margen, sin efecto combinado entre sitios, sin ensamble de probabilidades.
- Se hereda también el estado de preinscripción de la sección anterior: son
  resultados posteriores a la revisión de factibilidad, nunca confirmatorios.

## Cómo interpretar el máximo puntual

En `descriptive_performance.csv`, el tamaño de ROI con la media de AUC más
alta en un sitio dado es la estimación puntual más alta, no necesariamente
la configuración "óptima": los intervalos bootstrap en
`secondary_metric_intervals.csv` y `primary_12_vs_116.csv` casi siempre se
solapan con los de otros tamaños del mismo sitio, así que ese máximo no
implica una diferencia estadísticamente distinguible frente a las demás
configuraciones. Referirse siempre a la estimación puntual junto con su
intervalo, nunca al máximo aislado.

## Rendimiento

El bootstrap (10.000 iteraciones × 4 sitios) toma, en esta implementación
sobre CPU con un núcleo y `scikit-learn`, del orden de 15 a 20 minutos en
total. Ese costo es esperado y no justifica optimizar, paralelizar ni
sustituir `sklearn.metrics.roc_auc_score` — ver
`outputs/analysis_manifest.json` (`timing_seconds`) para los tiempos
exactos observados en la última ejecución productiva real.

Cada sitio imprime un mensaje de progreso cada 1.000 iteraciones (10%):
`sitio · iteraciones completadas/10000 · porcentaje`. La condición de
impresión no llama al generador aleatorio, no cambia el orden de los
bucles ni los índices bootstrap y no guarda los remuestreos crudos: el
resultado es numéricamente idéntico con o sin los mensajes (verificado en
`tests/test_analysis.py`).
