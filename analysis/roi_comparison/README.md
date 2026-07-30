# `analysis/roi_comparison/`

Análisis comparativo del desempeño de BrainNetCNN entre los cuatro tamaños de
conjunto de ROIs (12, 18, 39, 116) en los cuatro sitios (NYU, Peking,
NeuroIMAGE, OHSU), sobre las 16 corridas ya existentes en `results/runs/`.

Implementa, sin rediseñarlo, el plan estadístico congelado en
`analysis_plan.md` (versión 5.6) y la resolución de sus decisiones D1–D5
pendientes de equipo (ver más abajo). `results/` es la fuente de datos y
permanece de solo lectura.

> **Nota sobre `analysis_plan.md`:** este archivo debe ser una copia exacta
> del texto final de la versión 5.6 del plan aprobado por el equipo. Al
> momento de esta implementación no se contaba con ese texto en un archivo
> propio (solo se había revisado por partes, en rondas sucesivas, en el canal
> de comunicación con el equipo), así que el archivo todavía no existe en
> este directorio. En cuanto el equipo provea el texto definitivo, colocarlo
> tal cual en `analysis/roi_comparison/analysis_plan.md` y volver a ejecutar
> `run_statistical_analysis.py` para que su SHA-256 quede registrado en
> `outputs/analysis_manifest.json` (el campo `plan_sha256` queda en `null`
> mientras tanto).

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
  (artefactos, hashes científicos, pareamiento, reconciliación con el
  README).
- `outputs/tables/descriptive_performance.csv`,
  `primary_12_vs_116.csv`, `precision_diagnostics.csv`,
  `secondary_pairwise_comparisons.csv`, `secondary_metric_intervals.csv`,
  `error_analysis_summary.csv` — resultados del análisis estadístico.
- `outputs/data/error_analysis_long.csv`, `subject_error_profiles.csv` —
  análisis de errores por sujeto (12 vs. 116).
- `outputs/figures/` — perfiles por ROI y forest plot del contraste
  principal, en SVG y PNG.
- `outputs/analysis_manifest.json` — hashes, versiones, parámetros del
  bootstrap, tiempos observados y resumen de la resolución D1–D5.

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
total (~230 s por sitio observados en esta ejecución). Ese costo es esperado
y no justifica optimizar, paralelizar ni sustituir
`sklearn.metrics.roc_auc_score` — ver `outputs/analysis_manifest.json` para
los tiempos exactos observados.
