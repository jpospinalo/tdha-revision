# Entornos de software usados para el paper

**Tres** entornos distintos produjeron resultados usados en el manuscrito, no
dos. Ninguno regenera a otro: los modelos originalmente entrenados bajo A y C
no se reentrenan con B. Pero B no es "solo análisis": también entrenó las
corridas de sensibilidad neuronal posteriores (`*reviewer_sensitivity*`,
detalle en Environment B más abajo), además de ejecutar los scripts de
análisis/estadística/bootstrap sobre las predicciones ya almacenadas de A, B
y C.

Verificado recorriendo los `config.json` reales bajo `results/runs/**`
(no es una simplificación de memoria): 26 corridas en Environment A, 20 en
Environment B (actualizado 2026-08-07 tras la corrección de class_weight de
Peking — ver más abajo; eran 14 antes de esa corrección), 16 en
Environment C. 62 corridas en total. `results/runs/**` **no** pertenece
íntegramente a un único entorno.

## Environment A — entrenamiento neural original (BrainNetCNN, LSTM)

26 corridas: las 16 combinaciones sitio × panel de referencia (Table 4) más
las corridas `rev32` de comparación estática/LSTM. Hardware mixto: **17
sobre Tesla T4, 9 sobre CPU** (por `config["env"]["gpu"]`; el hardware real
está registrado corrida por corrida, no es uniforme dentro del grupo).

| Paquete | Versión |
|---|---|
| Python | 3.12.13 |
| NumPy | 2.0.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.6.1 |
| TensorFlow | 2.20.0 |
| Keras | 3.13.2 |
| Plataforma | Linux-6.6.122+-x86_64-with-glibc2.35 |
| GPU | mixto — ver arriba; el `config.json` de cada corrida registra el dispositivo real |

## Environment B — sensibilidades neuronales posteriores y análisis derivado

**20** corridas neuronales `*reviewer_sensitivity*` bajo `results/runs/12/`,
todas en CPU — no 14. Desglose (actualizado 2026-08-07, corrección
class_weight de Peking, ver `docs/paper_reference_configuration.md` §5):

- **8** condiciones no afectadas por la corrección (NYU ×6, NeuroIMAGE ×1,
  OHSU ×1): DeepSets, LSTM-static, GRU, ventanas 60s/12 alternativas.
- **6** corridas históricas de Peking (`class_weight=False`, ejecutadas
  antes de la corrección) — **presentes** en `results/runs/` como
  provenance, pero **no usadas** por ningún artefacto canónico del
  manuscrito desde la corrección.
- **6** corridas corregidas de Peking (`*_reviewer_sensitivity_weighted_fix_*`,
  `class_weight=True`, mismas particiones que las históricas) — las
  **canónicas** para Figure 3/Table 5 desde la corrección.

De las 20, **14 son canónicas** (8 no afectadas + 6 corregidas); no confundir
"presente en `results/runs/`" con "usado por el manuscrito".

**Este entorno no es exclusivamente de análisis**: además de los scripts de
`analysis/roi_comparison/scripts/*` y `analysis/finalization/*.py`
(cálculo de AUC agregada, bootstrap, figuras, CSV canónicos derivados),
también entrenó las 20 corridas de sensibilidad que aparecen en
`results/runs/**` (las 14 originales más las 6 corregidas el 2026-08-07).

| Paquete | Versión |
|---|---|
| Python | 3.13.2 |
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| scikit-learn | 1.8.0 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| SciPy | no registrado en la metadata de ninguna corrida (`config.json` no captura paquetes de análisis, solo los de entrenamiento); usado por los scripts derivados (`scipy.stats.rankdata`) pero su versión exacta no se reconstruye por inferencia |
| GPU | sin GPU (CPU), en las 20 corridas verificadas |
| Platform | `macOS-26.5.2-arm64-arm-64bit-Mach-O` — verificado en `config["env"]["platform"]` de las 20 corridas `*reviewer_sensitivity*`, sin excepción (no se documenta por inferencia: es el valor real de las 20) |

## Environment C — baseline de regresión logística

16 corridas (`*_static_logreg_baseline_*`, las 4 combinaciones de panel ROI
× 4 sitios). Metadata **parcialmente registrada**: cada `config.json`
guarda `python` y `sklearn` en la raíz. NumPy/pandas y TensorFlow/Keras se
tratan por separado, porque no están en la misma situación:

- **NumPy/pandas: se usan** (la regresión logística y el manejo de datos de
  este pipeline los requieren), pero su versión exacta no quedó registrada
  en la metadata de estas 16 corridas. No se completa por inferencia.
- **TensorFlow/Keras: no aplica.** Esta familia es regresión logística
  (scikit-learn), no usa red neuronal.

| Paquete | Versión |
|---|---|
| Python | 3.10.12 |
| scikit-learn | 1.7.2 |
| NumPy/pandas | usados, pero versión exacta no registrada en la metadata de estas corridas — no inferida |
| TensorFlow/Keras | no aplica (esta familia no usa red neuronal) |

## Qué resultado proviene de cuál

| Resultado | Entorno |
|---|---|
| Predicciones OOF, `metrics_train.csv`, `metrics_val.csv`, `history.csv`, pesos entrenados — 16 combos de referencia (Table 4) y corridas `rev32` | **A** |
| Predicciones OOF de las 20 corridas `*reviewer_sensitivity*` (14 canónicas + 6 históricas de Peking, superseded) | **B** (entrenamiento) |
| Predicciones OOF de las 16 corridas de regresión logística estática | **C** |
| `descriptive_performance.csv`, `primary_12_vs_116.csv` (Table 4, contraste primario) | **B** (análisis), sobre predicciones de A |
| `figure4_v6_audit.csv`, `algorithm_comparison_deepsets_audit.csv`, `manuscript_bootstrap_10k.csv` (Table 5, Figure 3) | **B** (análisis), sobre predicciones de A, B y C combinadas según el contraste. Para Peking, las 5+1 celdas afectadas por la corrección de class_weight usan las 6 corridas `*_weighted_fix_*` de B, no las 6 históricas |
| `demographics_by_site_dx.csv`, `cohort_audit.csv` | **B** (análisis), sobre predicciones de C + fenotípico externo (no de A: estas dos auditorías usan la cohorte del baseline logístico, no las corridas neuronales) |

No todas las predicciones OOF del manuscrito proceden del mismo entorno de
entrenamiento: Table 4 y las filas `rev32` vienen de A; las filas de
sensibilidad de Figure 3/Table 5 vienen de B; la fila de regresión logística
de Table 5 viene de C.

## Advertencia de reproducibilidad

El hardware (CPU vs. GPU/T4) y las versiones menores de NumPy/TensorFlow/Keras
entre A y B pueden producir diferencias numéricas pequeñas (redondeo de punto
flotante, no diferencias de metodología). El chequeo interno de
`generate_manuscript_figures_v6.py` que compararía filas contra
`figure4_v5_audit.csv` no puede ejecutarse actualmente: ese archivo nunca se
comprometió a git y se perdió en la limpieza del repositorio del 2026-08-06
(`c2c78f0`). El script sigue escribiendo su CSV de salida correctamente antes
de llegar a esa verificación (confirmado: el CSV es byte-idéntico al
canónico cuando no depende de una corrida corregida); la comparación
fila-por-fila contra el estado previo se hizo manualmente vía `git diff`
durante la corrección de class_weight de Peking (2026-08-07) en su lugar.
Pendiente para el equipo, fuera de esta corrección: reconstruir o retirar
esa verificación interna.

## Nota sobre `requirements.txt`

No se sustituyen los mínimos de `requirements.txt` por versiones exactas de A o
B: el archivo sigue documentando límites mínimos compatibles, no una
fotografía exacta de ningún entorno. Este documento es la fuente de verdad
para "qué versión exacta produjo qué resultado del paper"; `requirements.txt`
sigue siendo la fuente de verdad para "qué versión mínima hace falta para que
el repositorio funcione".
