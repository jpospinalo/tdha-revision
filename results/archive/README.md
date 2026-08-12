# Archivo de corridas históricas

## Propósito

Esta carpeta guarda corridas que ya cumplieron su propósito experimental y no son
referencias activas del compendio, pero que siguen siendo **científicamente válidas**: no se
archivan por estar rotas o ser inválidas, sino porque ya se compararon contra el baseline
correspondiente y la decisión metodológica ya quedó registrada en `results/README.md`.

`compile_results.py` usa por defecto únicamente `results/runs`; las corridas archivadas aquí
no entran en `collect()`, `--strict` ni en ninguna tabla comparativa a menos que se apunte
explícitamente `--root results/archive/<roi_set>`.

**No borrar ni editar los artefactos de estas corridas.** Se movieron con `git mv` para
conservar el historial de versiones; sus siete artefactos (`config.json`, `folds.csv`,
`history.csv`, `metrics_train.csv`, `metrics_val.csv`, `predictions_val.csv`, `resumen.md`)
y sus hashes son idénticos a como estaban en `results/runs/39`.

## Corridas archivadas

| run_id | Diferencia frente al baseline vigente | Accuracy | AUC | F1-macro | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| `NYU_rois39_w60s6_brainnetcnn_3925cca4` | `dropout=0.75`, `l2_reg=0.08` (vs. baseline `1521c348`) | 52.88% | 53.08% | 52.72% | 52.92% |
| `Peking_rois39_w60s6_brainnetcnn_0885abc4` | `dropout=0.8`, `l2_reg=0.15` (vs. baseline `396e34d2`) | 57.70% | 60.06% | 56.92% | 57.34% |

Ambas son variantes metodológicas válidas —no duplicados— de la exploración de capacidad de
BrainNetCNN en el conjunto de 39 ROIs: se ejecutaron con un `dropout`/`l2_reg` distinto al
baseline, no mejoraron frente a él, y además se ejecutaron con árbol de trabajo sucio
(`git.clean=false`), por lo que no son la referencia activa. Se conservan porque documentan
un punto ya explorado del espacio de hiperparámetros.

### Nota sobre `Peking_rois39_..._5506e815`

La especificación que motivó esta reorganización esperaba encontrar una tercera corrida,
`Peking_rois39_w60s6_brainnetcnn_control_base_line_5506e815` (réplica metodológica del
baseline anterior a `396e34d2`), y archivarla también. Al revisar el repositorio, esa carpeta
ya no existe: fue eliminada en un commit anterior a esta implementación (`delete: exp peking
5506e815`), fuera de este trabajo. No se ha recreado ni restaurado a partir del historial de
git — hacerlo habría revertido una decisión que el propio equipo ya tomó y confirmó con un
commit explícito. Esta carpeta de archivo contiene únicamente las dos corridas que sí
existían físicamente en `results/runs/39` al momento de archivar.

## Corridas archivadas — sensibilidad de arquitectura, NYU / 12 ROIs (depuración de repositorio, 2026-08-11)

Cinco corridas del lote de sensibilidad de arquitectura (GRU/LSTM/DeepSets/BrainNetCNN, semilla 42, NYU, 12 ROIs). Se movieron aquí tras una auditoría cruzada exhaustiva contra todos los manifiestos de `analysis/roi_comparison/`, `analysis/loso/` y toda la documentación de `docs/`: son las únicas 5 de un total de 111 directorios de corrida en todo el repositorio (62 en `results/runs` + 49 en `results/loso`) sin ninguna cita externa — no aparecen en `run_manifest.csv`, `baseline_manifest.csv`, `algorithm_comparison_deepsets_audit.csv`, `figure4_v6_audit.csv` ni en `docs/paper_reference_configuration.md`.

Su equivalente en Peking (mismo lote de sensibilidad de arquitectura) sí quedó documentado en `algorithm_comparison_deepsets_audit.csv` y en `docs/paper_reference_configuration.md` como parte de la cronología del bug de `class_weight`; el de NYU nunca se incorporó a ninguna tabla del manuscrito ni a ningún manifiesto formal. No están rotas ni son inválidas — simplemente no llegaron a citarse en ningún resultado publicado.

| run_id | Modelo | Ventana | Semilla |
|---|---|---|---|
| `NYU_rois12_static_lstm_reviewer_sensitivity_6686b406` | LSTM | estática | 42 |
| `NYU_rois12_w30s6_brainnetcnn_reviewer_sensitivity_642e1ea6` | BrainNetCNN | 30 TR / 60 s, paso 6 TR | 42 |
| `NYU_rois12_w30s6_gru_reviewer_sensitivity_945b2b57` | GRU | 30 TR / 60 s, paso 6 TR | 42 |
| `NYU_rois12_w60s6_deepsets_reviewer_sensitivity_c1063217` | DeepSets | 60 TR / 120 s, paso 6 TR | 42 |
| `NYU_rois12_w60s6_gru_reviewer_sensitivity_0fa455ae` | GRU | 60 TR / 120 s, paso 6 TR | 42 |

Movidas con `git mv` desde `results/runs/12/`; sus siete artefactos y hashes son idénticos a como estaban antes de moverse (verificado SHA-256 antes/después).

## Corrida histórica Peking–18 fuera de esta versión

`Peking_rois18_w60s6_brainnetcnn_control_baseline_v13_b8e8a44d` (ablación con
`class_weight=False`) tampoco está presente en el repositorio: no forma parte de la versión
14 y no se restauró aquí. Si el equipo decide conservarla, debe guardarse en un archivo
histórico externo o crearse explícitamente bajo `results/archive/18/`, marcada con claridad
como ablación `class_weight=False` — no se creó esa carpeta en esta implementación porque la
corrida no está disponible en el checkout actual.
