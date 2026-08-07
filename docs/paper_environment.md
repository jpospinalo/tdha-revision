# Entornos de software usados para el paper

Dos entornos distintos produjeron resultados usados en el manuscrito. Ninguno
de los dos regenera al otro: los modelos de la campaña experimental (Environment A)
no se reentrenan con Environment B; Environment B solo lee predicciones ya
almacenadas y calcula estadística descriptiva/bootstrap sobre ellas.

## Environment A — campaña experimental oficial

Todas las corridas en `results/runs/**` (entrenamiento de modelos, predicciones OOF).

| Paquete | Versión |
|---|---|
| Python | 3.12.13 |
| NumPy | 2.0.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.6.1 |
| TensorFlow | 2.20.0 |
| Keras | 3.13.2 |
| Plataforma | Linux-6.6.122+-x86_64-with-glibc2.35 |
| GPU | sin GPU (CPU) para las corridas verificadas en `docs/paper_reference_configuration.md` §9 |

## Environment B — análisis derivado y auditoría

Scripts en `analysis/roi_comparison/scripts/*` y `analysis/finalization/*.py`:
cálculo de AUC agregada por repetición, bootstrap de participantes,
generación de figuras y de los CSV canónicos derivados.

| Paquete | Versión |
|---|---|
| Python | 3.13.2 |
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| scikit-learn | 1.8.0 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |

## Qué resultado proviene de cuál

| Resultado | Entorno |
|---|---|
| Predicciones OOF, `metrics_train.csv`, `metrics_val.csv`, `history.csv`, pesos entrenados | **A** |
| `descriptive_performance.csv`, `primary_12_vs_116.csv` (Table 4, contraste primario) | **B**, sobre predicciones de A |
| `figure4_v6_audit.csv`, `algorithm_comparison_deepsets_audit.csv`, `manuscript_bootstrap_10k.csv` (Table 5, Figure 3) | **B**, sobre predicciones de A |
| `demographics_by_site_dx.csv`, `cohort_audit.csv` | **B**, sobre predicciones de A + fenotípico externo |

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
