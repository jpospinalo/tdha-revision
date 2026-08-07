# Entornos de software usados para el paper

**Tres** entornos distintos produjeron resultados usados en el manuscrito, no
dos. Ninguno regenera a otro: los modelos de A y C no se reentrenan con B; B
solo lee predicciones ya almacenadas y calcula estadística
descriptiva/bootstrap sobre ellas.

Verificado recorriendo los 56 `config.json` reales bajo `results/runs/**`
(no es una simplificación de memoria): 26 corridas en Environment A, 14 en
Environment B, 16 en Environment C. `results/runs/**` **no** pertenece
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

14 corridas neuronales (`*_reviewer_sensitivity_*`: DeepSets, LSTM-static,
GRU, ventanas 60s/12 alternativas), todas en CPU. **Este entorno no es
exclusivamente de análisis**: además de los scripts de
`analysis/roi_comparison/scripts/*` y `analysis/finalization/*.py`
(cálculo de AUC agregada, bootstrap, figuras, CSV canónicos derivados),
también entrenó las 14 corridas de sensibilidad que sí aparecen en
`results/runs/**`.

| Paquete | Versión |
|---|---|
| Python | 3.13.2 |
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| scikit-learn | 1.8.0 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| GPU | sin GPU (CPU), en las 14 corridas verificadas |

## Environment C — baseline de regresión logística

16 corridas (`*_static_logreg_baseline_*`, las 4 combinaciones de panel ROI
× 4 sitios). Metadata **parcialmente registrada**: cada `config.json`
guarda `python` y `sklearn` en la raíz, pero no NumPy/pandas/TensorFlow/Keras
(no aplica: estas corridas no usan red neuronal). No se completa lo ausente
por inferencia.

| Paquete | Versión |
|---|---|
| Python | 3.10.12 |
| scikit-learn | 1.7.2 |
| NumPy/pandas/TensorFlow/Keras | *environment metadata partially recorded in baseline_ml_v1 configs* — no registrado, no inferido |

## Qué resultado proviene de cuál

| Resultado | Entorno |
|---|---|
| Predicciones OOF, `metrics_train.csv`, `metrics_val.csv`, `history.csv`, pesos entrenados — 16 combos de referencia (Table 4) y corridas `rev32` | **A** |
| Predicciones OOF de las 14 corridas de sensibilidad (`*_reviewer_sensitivity_*`) | **B** (entrenamiento) |
| Predicciones OOF de las 16 corridas de regresión logística estática | **C** |
| `descriptive_performance.csv`, `primary_12_vs_116.csv` (Table 4, contraste primario) | **B** (análisis), sobre predicciones de A |
| `figure4_v6_audit.csv`, `algorithm_comparison_deepsets_audit.csv`, `manuscript_bootstrap_10k.csv` (Table 5, Figure 3) | **B** (análisis), sobre predicciones de A, B y C combinadas según el contraste |
| `demographics_by_site_dx.csv`, `cohort_audit.csv` | **B** (análisis), sobre predicciones de A/C + fenotípico externo |

No todas las predicciones OOF del manuscrito proceden del mismo entorno de
entrenamiento: Table 4 y las filas `rev32` vienen de A; las filas de
sensibilidad de Figure 3/Table 5 vienen de B; la fila de regresión logística
de Table 5 viene de C.

## Advertencia de reproducibilidad

El hardware (CPU vs. GPU/T4) y las versiones menores de NumPy/TensorFlow/Keras
entre A y B pueden producir diferencias numéricas pequeñas (redondeo de punto
flotante, no diferencias de metodología). Esto ya se verificó explícitamente
al menos una vez: `generate_manuscript_figures_v6.py` reproduce con diferencia
0 los valores de `figure4_v5_audit.csv` que dependían de la misma cadena de
cálculo, a pesar de estar generado en Environment B y no en el entorno
original de esas filas.

## Nota sobre `requirements.txt`

No se sustituyen los mínimos de `requirements.txt` por versiones exactas de A o
B: el archivo sigue documentando límites mínimos compatibles, no una
fotografía exacta de ningún entorno. Este documento es la fuente de verdad
para "qué versión exacta produjo qué resultado del paper"; `requirements.txt`
sigue siendo la fuente de verdad para "qué versión mínima hace falta para que
el repositorio funcione".
