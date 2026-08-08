# IMPLEMENTATION_SPEC — `loso_static_v1`

Especificación autocontenida del diseño que **realmente produjo** las 48
corridas formales de `loso_static_v1`. No es un diseño nuevo ni aspiracional:
cada valor listado abajo está respaldado por `results/loso/_design/loso_static_v1_design.json`,
por los 48 `config.json` de las corridas, o por `analysis/loso/config/loso_analysis_config.json`.
Este documento no se modifica para reflejar preferencias posteriores; si el
diseño cambiara, sería una campaña nueva con otro `campaign_id`.

## Identidad de campaña

- `campaign_id`: `loso_static_v1`
- Estimando: transporte cross-site "crudo" (raw cross-site transport) — un
  modelo entrenado con tres sitios ADHD-200 evaluado en un cuarto sitio
  completamente excluido de ajuste e inner validation, sin ninguna técnica de
  adaptación de dominio.
- Sitios held-out (una rotación por sitio): NYU, Peking, NeuroIMAGE, OHSU.
- Cada rotación entrena con los otros tres sitios (`training_sites`) y evalúa
  únicamente en el sitio held-out.

## Representación de datos

- `representation`: `static` (una sola ventana; `n_windows = 1`).
- `fisher_z`: `False` — sin transformación Fisher-z de la matriz de
  conectividad.
- `constant_policy`: `zero` — política de reemplazo para columnas
  constantes/degeneradas en la FC.
- ROI sets: `12` (66 features triangulares) y `116` (6670 features
  triangulares), vía `build_flat_static_connectivity()`.
- Sin harmonización (`harmonization = "none"`), sin ponderación de clase
  (`class_weight = False`), sin ponderación de sitio (`site_weighting =
  False`), sin ponderación de muestra (`sample_weight = False`).

## Modelos

### BrainNetCNN

Arquitectura congelada (`BNN_ARCH_KWARGS`):

```text
e2e = 4
e2n = 8
dense = 8
dropout = 0.7
leaky = 0.33
l2_reg = 0.05
inter_dropout = 0.6
```

Seeds: 42, 43, 44, 45, 46 (cinco corridas independientes por
sitio-held-out x ROI, agregadas metric-then-mean, nunca prob-then-metric).

Entrenamiento: Adam, lr=1e-4, batch_size=32, epochs=300, patience=25,
`early_stopping_monitor="val_loss"`, `early_stopping_min_delta=1e-5`,
`restore_best_weights=True` verificado mediante un gate de restauración no
circular (re-evaluar el modelo restaurado sobre inner_val y comparar contra
`early_stopping.best`).

### Regresión logística

Configuración congelada (`LOGREG_CONFIG`), una sola corrida determinista por
sitio-held-out x ROI (sin seed):

```text
penalty = l2
C = 1.0
class_weight = null
solver = lbfgs
max_iter = 2000
```

## Split (outer LOSO + inner site×diagnóstico)

- Outer: el sitio held-out va 100% a test; los otros tres sitios forman
  `train_pool`.
- Inner: `StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)`
  sobre `train_pool`, estratificado por sitio×diagnóstico (6 estratos:
  3 sitios x 2 diagnósticos), produciendo `fit`/`inner_val`.
- `split_seed = 42`, `inner_val_frac = 0.15`.
- El split es **idéntico** entre los dos ROI sets, los 5 seeds de BrainNetCNN
  y la regresión logística dentro de cada rotación (`build_rotation_split()`
  no recibe `roi_set`/`model`/`seed` como argumento — verificado por
  `rotation_split_fingerprint()` y por T10SplitInvariance/T21).
- Tamaños por rotación (`fit` / `inner_val` / `test`):

| Held-out   | Fit | Inner val | Test |
|---|---:|---:|---:|
| NYU        | 244 | 44 | 177 |
| Peking     | 239 | 43 | 183 |
| NeuroIMAGE | 362 | 64 | 39  |
| OHSU       | 339 | 60 | 66  |

## Métricas y análisis

- Métrica primaria: AUC, umbral no aplicable (AUC no depende de threshold);
  métricas secundarias (`balanced_accuracy`, `f1_macro`, `sensitivity`,
  `specificity`) calculadas a `threshold = 0.5`.
- `sensitivity` es sinónimo de `recall`; no existe una columna `sensitivity`
  cruda en `metrics_test.csv` — se deriva de `recall`.
- Bootstrap: 10,000 iteraciones, `numpy.random.Generator(PCG64(42))`,
  reiniciado por sitio (`reset_per_site`), draws pareados por clase
  (resample con reemplazo dentro de control y dentro de ADHD por separado),
  el mismo conjunto de draws reutilizado por las 8 condiciones de un sitio
  (2 ROI x (5 seeds BNN + 1 logistic)) y por los contrastes de ese sitio —
  esto es lo que hace pareados los 3 contrastes.
- Agregación BrainNetCNN: metric-then-mean (AUC por seed, luego promedio de
  las 5 AUC), nunca prob-then-metric (promediar probabilidades y luego
  calcular AUC) — verificado por T25MetricThenMean.
- 12 contrastes preespecificados (3 tipos x 4 sitios):
  - `dimensionality`: `brainnetcnn_116 − brainnetcnn_12`
  - `model_family_at_12`: `logreg_12 − brainnetcnn_12`
  - `model_family_at_116`: `logreg_116 − brainnetcnn_116`
- CI: percentil, 95%, sin ajuste por comparaciones múltiples
  (`ci_adjustment = "none_pointwise"`).

## Caveat de desarrollo (NYU)

La arquitectura de BrainNetCNN (hiperparámetros de `BNN_ARCH_KWARGS`) se fijó
históricamente usando datos de NYU antes de esta campaña LOSO. Por lo tanto,
la rotación con NYU held-out es una re-evaluación dentro de la campaña sobre
el sitio de desarrollo histórico del modelo, no una validación en un sitio
totalmente ajeno al desarrollo. Las otras tres rotaciones (Peking,
NeuroIMAGE, OHSU) no tienen esta salvedad.

## Fuera de alcance (requieren un `campaign_id` y plan nuevos)

BrainNetCNN con ventanas (no estático), DeepSets/LSTM/GRU en modo LOSO,
ROI 18/39, harmonización (p. ej. ComBat), adaptación de dominio, balanceo de
sitios, ponderación de clase, búsqueda de hiperparámetros, umbral de
clasificación distinto de 0.5, p-values o permutation tests. Ninguno de estos
se implementó, se probó parcialmente, ni se mezcló con `loso_static_v1`.
