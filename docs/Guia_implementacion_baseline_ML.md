# Guía de implementación: baseline de aprendizaje automático clásico

**Fecha:** 3 de agosto de 2026
**Estado:** implementado, ejecutado y con contraste estadístico calculado. `src/run_baseline_ml.py`
corrió las dieciséis combinaciones sitio × grupo de ROIs con datos reales, sus salidas están en
`results/runs/`, y el contraste bootstrap frente al comparador BrainNetCNN pareado ya está en
`analysis/roi_comparison/outputs/baseline/` (ver §6). Falta la integración en el manuscrito y la
revisión del equipo — sigue vigente esa parte de §6.
**Contexto que hace esto viable:** `docs/PLAN_RESPUESTA_REVISORES.md` §9.1 ya excluyó este
experimento con una nota de contingencia textual: *«Si el editor solicita explícitamente un
baseline lineal, se preparará un protocolo específico con penalización preespecificada y
estandarización ajustada exclusivamente dentro de cada fold... La penalización fija evita una
búsqueda adicional de ese hiperparámetro.»* La enmienda del 3 de agosto de 2026 (v3.4) activó
exactamente esa contingencia; el texto vigente está en §9.1, §8.4 y la nota de cabecera de ese
documento.

---

## 0. Alcance final: dieciséis corridas, no ocho

La primera versión de esta guía proponía ocho corridas —cuatro sitios × grupos de 12 y 116 ROIs—,
pensando el baseline solo como comparador de la comparación primaria. Es un alcance incompleto: dejaba
sin comparador lineal la sensibilidad de dimensionalidad de §3.3.1, que sí evalúa 18 y 39 ROIs.
Correr los cuatro grupos no cuesta más —la regresión logística es igual de barata en cualquier
tamaño— así que el diseño final cubre los cuatro sitios × los cuatro grupos: **dieciséis corridas**.

Esto obligó a resolver un problema real que la primera versión no había verificado: **solo existe
una corrida BrainNetCNN `static` para 12 ROIs** (las cuatro de §8.1 del plan). Para 18, 39 y 116 no
hay comparador `static` en el repositorio, y generarlo exigiría corridas nuevas de BrainNetCNN —
reabrir la campaña de diez corridas ya cerrada, que la enmienda explícitamente no hace. La solución,
verificada y en producción en `find_paired_run()`:

| Grupo de ROIs | Comparador que localiza el script | Confusión |
|---|---|---|
| 12 | Corrida BrainNetCNN `static` (búsqueda directa en `results/runs/`) | Ninguna: un solo factor cambia, la arquitectura |
| 18, 39, 116 | Corrida BrainNetCNN `ordered`, vía `analysis/roi_comparison/config/run_manifest.csv` (la misma referencia primaria de la Tabla 6) | Representación y arquitectura cambian a la vez — mismo aviso que la dimensión «signal representation» de §2.6 del manuscrito |

Cada corrida registra en su propio `config.json` cuál comparador usó
(`comparator_representation: "static"` u `"ordered"`) y si hay confusión declarada
(`representation_confound: true/false`), así que la distinción no depende de recordarla: queda en
el artefacto.

---

## 1. Por qué el repositorio ya tenía la mayor parte de lo necesario

Verifiqué cuatro piezas reutilizables directamente, con datos reales del repositorio:

**La generación de particiones es una función pura, importable.** `build_split_plan(labels, args)`
en `src/run_experiment.py` solo necesita las etiquetas y un `argparse.Namespace` con `n_splits`,
`n_repeats`, `seed`, `inner_val_frac`. Con los mismos cuatro valores que ya usan las corridas de
BrainNetCNN —`10`, `5`, `42`, `0.15`— produce **exactamente** las mismas particiones. Lo comprobé
contra OHSU, 12 ROIs: `split_fingerprint` calculado y el registrado en el `config.json` de la
corrida BrainNetCNN pareada coinciden carácter por carácter (`a59dca47e72dc24d`).

**Las características ya existen como función de librería.** `data.build_flat_static_connectivity`
devuelve el triángulo superior de Pearson vectorizado. Con 12 ROIs da 66 columnas y con 116 da
6.670; el primero lo confirmé contra `n_features` en un `config.json` real.

**Los pesos de clase son la misma función.** `compute_class_weights(y_fit)` calcula los pesos
inversos a la frecuencia, solo sobre `fit`. Peking usa `class_weight=True` y los otros tres sitios
`False`, verificado en sus `config.json` reales.

**No hay dependencia de GPU ni de Colab.** Las dieciséis corridas completas —incluida la carga de
BOLD, el cálculo de conectividad estática y el ajuste de 160 modelos (10 pliegues × 16
combinaciones)— corrieron en CPU, sin TensorFlow activo más que para importar `run_experiment.py`.

---

## 2. `src/run_baseline_ml.py` — implementado

El archivo real está en el repositorio. Resumen de lo que hace, en el orden en que ocurre:

1. **Guardarraíl primero, antes de tocar los datos de entrenamiento.** `find_paired_run(site,
   roi_set)` busca la corrida `static` pareada; si no existe, cae a la corrida `ordered` de
   `run_manifest.csv`. Si ninguna existe, el script se detiene con `SystemExit` sin escribir nada.
2. Carga BOLD, resuelve los índices de ROI, construye las particiones y calcula
   `split_fingerprint`, `bold_hash`, `roi_indices_hash`.
3. Compara los tres hashes contra el comparador encontrado. Si alguno no coincide, se detiene con
   `SystemExit` y no escribe nada — no es una advertencia, es una compuerta real.
4. Construye `X` con `build_flat_static_connectivity`, ajusta una `LogisticRegression(penalty="l2",
   C=1.0, ...)` por pliegue, con `StandardScaler` ajustado exclusivamente sobre `fit`.
5. Evalúa con las mismas fórmulas que `evaluate()` de `run_experiment.py` (reimplementadas, porque
   `evaluate()` original llama al modelo como objeto Keras y una `LogisticRegression` expone
   `predict_proba`, no `__call__`), para que las métricas sean comparables punto por punto.
6. Escribe `config.json`, `folds.csv`, `predictions_val.csv`, `metrics_val.csv`,
   `metrics_train.csv` y `resumen.md` en `results/runs/{roi_set}/{run_id}/`, con la misma
   convención de nombres que las corridas de BrainNetCNN.

Uso:

```bash
cd src
for site in NYU Peking NeuroIMAGE OHSU; do
  for roi in 12 18 39 116; do
    python run_baseline_ml.py --site "$site" --roi-set "$roi"
  done
done
```

`--skip-guardrail` existe solo para pruebas de humo con datos sintéticos; no debe usarse en
corridas formales, y el script lo dice en su propio `--help`.

---

## 3. Resultado de las dieciséis corridas

Ejecutadas y verificadas. Las dieciséis pasaron el guardarraíl —los tres hashes coincidieron en
cada una— antes de escribir cualquier archivo.

| Sitio | 12 (static) | 18 (ordered) | 39 (ordered) | 116 (ordered) |
|---|---:|---:|---:|---:|
| NYU | 0.575 | 0.561 | 0.530 | 0.563 |
| Peking | 0.592 | 0.504 | 0.624 | 0.592 |
| NeuroIMAGE | 0.632 | 0.565 | 0.511 | 0.550 |
| OHSU | 0.496 | 0.629 | 0.363 | 0.546 |

AUC out-of-fold media por repetición, agregada igual que en §2.7 del manuscrito. Se muestran aquí
como verificación de que el mecanismo produce números con sentido (todas entre 0 y 1, ninguna en
un extremo degenerado); **no** son una lectura del resultado del baseline frente a BrainNetCNN —
eso requiere el contraste pareado del paso siguiente, que no se ha calculado.

Los dieciséis directorios están bajo `results/runs/{roi_set}/{site}_rois{roi_set}_static_logreg_baseline_<hash8>/`,
sin manifiesto propio todavía: no han pasado por una revisión del equipo ni se han incluido en
ningún `run_manifest.csv`.

---

## 4. Validación ligera, no `build_analysis_dataset.py`

Ese validador exige `history.csv` con curva de épocas completa por pliegue y campos de
`EarlyStopping` —conceptos de Keras que una regresión logística no produce—, y está construido
específicamente para las 16 corridas de la comparación primaria de tamaño de ROI (con
`roi_indices_hash` y conteos de sujetos hardcodeados para ese propósito). Reutilizarlo tal cual
fallaría, no por un defecto del baseline sino porque comprueba algo que este modelo no tiene. Para
esto sirve la celda de validación ligera del notebook (§6 más abajo): cobertura de sujetos,
ausencia de solapamiento entre particiones, rango válido de las probabilidades.

---

## 5. Cómo ejecutarlo — notebook

`tdha_baseline_ml.ipynb`, en la raíz del repositorio, junto a `tdha_experimentos.ipynb` pero
deliberadamente más simple: sin celda de GPU, sin bloques de arquitectura o enventanado, una sola
celda de configuración con las combinaciones a correr. Trae, en orden: preparación del entorno,
configuración, comprobación previa de qué comparador usará cada combinación (distingue `static` de
`ordered` antes de correr nada), la corrida, resultados, validación ligera, y el mismo flujo de
subida a GitHub que el notebook original.

---

## 6. Lo que falta

**El contraste estadístico — hecho el 3 de agosto de 2026.** `analysis/roi_comparison/scripts/
build_baseline_contrast.py` implementa exactamente el patrón descrito abajo (que quedó como
especificación antes de ejecutar nada, por eso se conserva sin editar): por sitio y grupo de ROIs,
`ref_reps` (las 5 AUC por repetición de la corrida BrainNetCNN pareada) frente a `new_reps` (las 5
del baseline), bootstrap pareado estratificado por clase, PCG64, seed=42, reset por sitio, 2.000
remuestreos. Salida real: `analysis/roi_comparison/outputs/baseline/data/baseline_contrast_results.json`
(mismo esquema que `new_contrasts_results.json`) y la tabla
`outputs/baseline/tables/baseline_contrast.csv`. Las 16 filas pasaron el guardarraíl (mismo
`split_fingerprint` y `roi_indices_hash` entre baseline y comparador) y verificación cruzada: los
`new_auc_point` reproducen a tres decimales los valores de la tabla de §3, calculados por una vía
distinta. Para los doce casos con comparador `ordered`, el contraste hereda la confusión declarada
(`representation_confound: true` en el JSON) y debe reportarse con esa salvedad explícita, no como
una comparación limpia de arquitectura. Documentación completa en
`analysis/roi_comparison/README.md`, sección "Contraste del baseline de regresión logística".
Especificación original, conservada como referencia:

**La integración en el manuscrito — pendiente.** En `build_ms_en2.js`, una fila más en la lista de `T7` de §3.3,
con `kind: "new"`. En §2.6, la quinta dimensión de sensibilidad, distinguiendo las cuatro corridas
sin confusión (12 ROIs) de las doce con confusión declarada (18, 39, 116) — la misma distinción de
la tabla de §0 de esta guía. En la carta de respuesta, la cronología completa: que se decidió
después de conocer el resto de la campaña, que el alcance creció de ocho a dieciséis corridas
porque la primera versión dejaba sin comparador la sensibilidad de dimensionalidad, y que el
resultado se reporta en cualquier dirección.

**Revisión del equipo antes de tratar estos dieciséis resultados como definitivos.** Son corridas
reales, con el guardarraíl verificado, pero no han pasado por la misma revisión que las diez
corridas de §8: nadie más que yo ha mirado los números todavía.
