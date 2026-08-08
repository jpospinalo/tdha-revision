# LOSO_STATIC_V1_REPORT

Campaña `loso_static_v1`: 4 sitios held-out x 2 ROI sets x 2 familias de modelo, conectividad estática, sin harmonización ni ponderación de clase/sitio. Diseño completo autocontenido en `analysis/loso/IMPLEMENTATION_SPEC.md` (dentro de este repositorio).

## Diseño

Una rotación por sitio held-out (LOSO exhaustivo, 4 rotaciones); entrenamiento con los otros tres sitios; ROI 12 y 116; BrainNetCNN (5 seeds) + regresión logística L2 (determinista); FC estática, sin Fisher-z; sin harmonización; sin ponderación de clase/sitio/muestra.

## Cohorte

| Site | n | Control | ADHD |
|---|---:|---:|---:|
| NYU | 177 | 87 | 90 |
| Peking | 183 | 109 | 74 |
| NeuroIMAGE | 39 | 22 | 17 |
| OHSU | 66 | 38 | 28 |
| **Total** | **465** | **256** | **209** |

## Splits

| Held-out | Fit | Inner val | Test |
|---|---:|---:|---:|
| NYU | 244 | 44 | 177 |
| Peking | 239 | 43 | 183 |
| NeuroIMAGE | 362 | 64 | 39 |
| OHSU | 339 | 60 | 66 |

Estratificación inner: sitio x diagnóstico. `split_seed = 42`. Mismo split entre los dos ROI sets, los 5 seeds de BrainNetCNN y la regresión logística dentro de cada rotación.

## Environments y source SHAs

### Training

- `training_source_git_sha`: `428cbc18f9b7e099d56bed91acd2fbc4f18ee6e8`
- `training_environment_signature`: `0951e380a4901bc2`
- Training environment:
  - python: 3.13.2
  - platform: macOS-26.5.2-arm64-arm-64bit-Mach-O
  - numpy: 2.5.1
  - pandas: 3.0.5
  - scikit_learn: 1.8.0
  - tensorflow: 2.21.0
  - keras: 3.15.1
  - gpu: sin GPU

### Original analysis

- `original_analysis_source_git_sha`: `428cbc18f9b7e099d56bed91acd2fbc4f18ee6e8`
- Solo se registraron `numpy`/`scikit-learn` de este environment original (ver `loso_bootstrap_manifest.json` -> `original_analysis_environment_partial` en `loso_provenance_manifest.json`); no se inventan las demás versiones.

### Closeout analysis

- `closeout_analysis_source_git_sha`: `be674c5373f4cf5b00105076c21ecaa4aa6e5998`
- Closeout analysis environment:
  - python: 3.10.12
  - platform: Linux-6.8.0-124-generic-aarch64-with-glibc2.35
  - numpy: 2.2.6
  - pandas: 2.3.3
  - sklearn: 1.7.2
  - scipy: 1.15.3

Training y analysis environments se registran por separado; no se exige que sean idénticos (Sección 33.3) — la validez del rerun del analyzer se decide por el primary-result regression gate (Sección 45), no por identidad de environments.

## Completitud: 48/48 corridas formales

40 BrainNetCNN + 8 logistic = 48 total, 5580 predicciones, 0 faltantes, 0 duplicadas, 0 parciales (verificado en Gates A-Q antes de calcular cualquier output; ver `LOSO_STATIC_V1_QA.md`).

## AUC por condición (95% CI, percentil, sin ajustar) — primario, sin cambios

| held_out_site   |   roi_set | model       |   auc_point |   auc_ci_low |   auc_ci_high |     seed_sd |   seed_min |   seed_max |
|:----------------|----------:|:------------|------------:|-------------:|--------------:|------------:|-----------:|-----------:|
| NYU             |        12 | brainnetcnn |    0.575683 |     0.508556 |      0.641891 |   0.0446013 |   0.535504 |   0.649042 |
| NYU             |        12 | logreg      |    0.560026 |     0.471644 |      0.64636  | nan         | nan        | nan        |
| NYU             |       116 | brainnetcnn |    0.545415 |     0.481277 |      0.607536 |   0.0305273 |   0.503704 |   0.570115 |
| NYU             |       116 | logreg      |    0.556577 |     0.470115 |      0.639084 | nan         | nan        | nan        |
| Peking          |        12 | brainnetcnn |    0.562237 |     0.486163 |      0.635234 |   0.0235023 |   0.542896 |   0.592859 |
| Peking          |        12 | logreg      |    0.528391 |     0.440612 |      0.615175 | nan         | nan        | nan        |
| Peking          |       116 | brainnetcnn |    0.518249 |     0.458239 |      0.578701 |   0.0277932 |   0.47694  |   0.550459 |
| Peking          |       116 | logreg      |    0.542648 |     0.452886 |      0.628691 | nan         | nan        | nan        |
| NeuroIMAGE      |        12 | brainnetcnn |    0.633155 |     0.478075 |      0.773262 |   0.0179762 |   0.612299 |   0.657754 |
| NeuroIMAGE      |        12 | logreg      |    0.639037 |     0.457219 |      0.804813 | nan         | nan        | nan        |
| NeuroIMAGE      |       116 | brainnetcnn |    0.500535 |     0.388222 |      0.614439 |   0.130008  |   0.291444 |   0.644385 |
| NeuroIMAGE      |       116 | logreg      |    0.572193 |     0.387701 |      0.746056 | nan         | nan        | nan        |
| OHSU            |        12 | brainnetcnn |    0.631015 |     0.503943 |      0.752632 |   0.0372375 |   0.571429 |   0.662594 |
| OHSU            |        12 | logreg      |    0.532895 |     0.391917 |      0.671053 | nan         | nan        | nan        |
| OHSU            |       116 | brainnetcnn |    0.496053 |     0.392857 |      0.600752 |   0.0755992 |   0.428571 |   0.612782 |
| OHSU            |       116 | logreg      |    0.443609 |     0.301692 |      0.589286 | nan         | nan        | nan        |

## Métricas secundarias (threshold=0.5; BNN: media de 5 seeds + SD; logistic: valor único determinista; sin CI adicionales)

| held_out_site   |   roi_set | model       |   balanced_accuracy_point |   balanced_accuracy_seed_sd |   f1_macro_point |   f1_macro_seed_sd |   sensitivity_point |   sensitivity_seed_sd |   specificity_point |   specificity_seed_sd |
|:----------------|----------:|:------------|--------------------------:|----------------------------:|-----------------:|-------------------:|--------------------:|----------------------:|--------------------:|----------------------:|
| NYU             |        12 | brainnetcnn |                  0.515594 |                   0.0226644 |         0.393073 |          0.085442  |            0.162222 |             0.296648  |            0.868966 |             0.2673    |
| NYU             |        12 | logreg      |                  0.547318 |                 nan         |         0.546851 |        nan         |            0.588889 |           nan         |            0.505747 |           nan         |
| NYU             |       116 | brainnetcnn |                  0.541609 |                   0.0257641 |         0.519764 |          0.0470774 |            0.386667 |             0.156426  |            0.696552 |             0.125282  |
| NYU             |       116 | logreg      |                  0.550383 |                 nan         |         0.54026  |        nan         |            0.411111 |           nan         |            0.689655 |           nan         |
| Peking          |        12 | brainnetcnn |                  0.514009 |                   0.0129546 |         0.482022 |          0.0443707 |            0.372973 |             0.246524  |            0.655046 |             0.243387  |
| Peking          |        12 | logreg      |                  0.526965 |                 nan         |         0.5201   |        nan         |            0.310811 |           nan         |            0.743119 |           nan         |
| Peking          |       116 | brainnetcnn |                  0.526209 |                   0.0365626 |         0.518695 |          0.032802  |            0.443243 |             0.12619   |            0.609174 |             0.164959  |
| Peking          |       116 | logreg      |                  0.555418 |                 nan         |         0.555604 |        nan         |            0.459459 |           nan         |            0.651376 |           nan         |
| NeuroIMAGE      |        12 | brainnetcnn |                  0.525668 |                   0.0220365 |         0.446225 |          0.0569325 |            0.305882 |             0.370635  |            0.745455 |             0.347066  |
| NeuroIMAGE      |        12 | logreg      |                  0.592246 |                 nan         |         0.588318 |        nan         |            0.411765 |           nan         |            0.772727 |           nan         |
| NeuroIMAGE      |       116 | brainnetcnn |                  0.508021 |                   0.0795595 |         0.489553 |          0.095292  |            0.270588 |             0.121979  |            0.745455 |             0.0518262 |
| NeuroIMAGE      |       116 | logreg      |                  0.47861  |                 nan         |         0.47861  |        nan         |            0.411765 |           nan         |            0.545455 |           nan         |
| OHSU            |        12 | brainnetcnn |                  0.587594 |                   0.0209357 |         0.556794 |          0.0394376 |            0.685714 |             0.21488   |            0.489474 |             0.206876  |
| OHSU            |        12 | logreg      |                  0.510338 |                 nan         |         0.480556 |        nan         |            0.678571 |           nan         |            0.342105 |           nan         |
| OHSU            |       116 | brainnetcnn |                  0.505075 |                   0.0784834 |         0.493095 |          0.080361  |            0.578571 |             0.0774267 |            0.431579 |             0.0995141 |
| OHSU            |       116 | logreg      |                  0.49718  |                 nan         |         0.463664 |        nan         |            0.678571 |           nan         |            0.315789 |           nan         |

`sensitivity` es sinónimo de `recall` (no existe columna `sensitivity` cruda en `metrics_test.csv`).

## Variabilidad entre seeds (BrainNetCNN, AUC)

| held_out_site   |   roi_set |   auc_point |   seed_sd |   seed_min |   seed_max |
|:----------------|----------:|------------:|----------:|-----------:|-----------:|
| NYU             |        12 |    0.575683 | 0.0446013 |   0.535504 |   0.649042 |
| NYU             |       116 |    0.545415 | 0.0305273 |   0.503704 |   0.570115 |
| Peking          |        12 |    0.562237 | 0.0235023 |   0.542896 |   0.592859 |
| Peking          |       116 |    0.518249 | 0.0277932 |   0.47694  |   0.550459 |
| NeuroIMAGE      |        12 |    0.633155 | 0.0179762 |   0.612299 |   0.657754 |
| NeuroIMAGE      |       116 |    0.500535 | 0.130008  |   0.291444 |   0.644385 |
| OHSU            |        12 |    0.631015 | 0.0372375 |   0.571429 |   0.662594 |
| OHSU            |       116 |    0.496053 | 0.0755992 |   0.428571 |   0.612782 |

## Contrastes preespecificados (95% CI) — primario, sin cambios, sin p-values

| contrast            | held_out_site   | condition_a     | condition_b     |   delta_point |   delta_ci_low |   delta_ci_high |
|:--------------------|:----------------|:----------------|:----------------|--------------:|---------------:|----------------:|
| dimensionality      | NYU             | brainnetcnn_116 | brainnetcnn_12  |   -0.0302682  |     -0.110425  |       0.0486609 |
| dimensionality      | Peking          | brainnetcnn_116 | brainnetcnn_12  |   -0.0439871  |     -0.111555  |       0.0276996 |
| dimensionality      | NeuroIMAGE      | brainnetcnn_116 | brainnetcnn_12  |   -0.13262    |     -0.303222  |       0.0465241 |
| dimensionality      | OHSU            | brainnetcnn_116 | brainnetcnn_12  |   -0.134962   |     -0.285526  |       0.018797  |
| model_family_at_12  | NYU             | logreg_12       | brainnetcnn_12  |   -0.0156577  |     -0.109733  |       0.0751986 |
| model_family_at_12  | Peking          | logreg_12       | brainnetcnn_12  |   -0.0338458  |     -0.100153  |       0.0329308 |
| model_family_at_12  | NeuroIMAGE      | logreg_12       | brainnetcnn_12  |    0.00588235 |     -0.170602  |       0.179679  |
| model_family_at_12  | OHSU            | logreg_12       | brainnetcnn_12  |   -0.0981203  |     -0.217862  |       0.0169173 |
| model_family_at_116 | NYU             | logreg_116      | brainnetcnn_116 |    0.0111622  |     -0.0413295 |       0.0642158 |
| model_family_at_116 | Peking          | logreg_116      | brainnetcnn_116 |    0.0243987  |     -0.0377641 |       0.0865361 |
| model_family_at_116 | NeuroIMAGE      | logreg_116      | brainnetcnn_116 |    0.0716578  |     -0.0679278 |       0.211243  |
| model_family_at_116 | OHSU            | logreg_116      | brainnetcnn_116 |   -0.0524436  |     -0.138727  |       0.0308318 |

## Convergencia (BrainNetCNN)

| held_out_site   |   roi_set |   n_runs |   n_seeds |   epochs_ran_mean |   epochs_ran_sd |   epochs_ran_min |   epochs_ran_max |   best_epoch_mean |   best_epoch_sd |   best_epoch_min |   best_epoch_max |   n_hit_epoch_300 |   n_stopped_before_300 |
|:----------------|----------:|---------:|----------:|------------------:|----------------:|-----------------:|-----------------:|------------------:|----------------:|-----------------:|-----------------:|------------------:|-----------------------:|
| NYU             |        12 |        5 |         5 |             300   |          0      |              300 |              300 |             300   |          0      |              300 |              300 |                 5 |                      0 |
| NYU             |       116 |        5 |         5 |             234.2 |         75.9026 |              149 |              300 |             218.2 |         86.0041 |              124 |              300 |                 2 |                      3 |
| Peking          |        12 |        5 |         5 |             300   |          0      |              300 |              300 |             300   |          0      |              300 |              300 |                 5 |                      0 |
| Peking          |       116 |        5 |         5 |             237.6 |         94.9358 |               80 |              300 |             219.2 |        100.758  |               55 |              299 |                 2 |                      3 |
| NeuroIMAGE      |        12 |        5 |         5 |             300   |          0      |              300 |              300 |             300   |          0      |              300 |              300 |                 5 |                      0 |
| NeuroIMAGE      |       116 |        5 |         5 |             235.8 |         36.5199 |              190 |              279 |             210.8 |         36.5199 |              165 |              254 |                 0 |                      5 |
| OHSU            |        12 |        5 |         5 |             300   |          0      |              300 |              300 |             300   |          0      |              300 |              300 |                 5 |                      0 |
| OHSU            |       116 |        5 |         5 |             295.6 |          9.8387 |              278 |              300 |             281.4 |         17.7989 |              253 |              300 |                 4 |                      1 |

No se infiere overfitting automáticamente a partir de estos valores (Sección 29/40.10).

## QA

Ver `LOSO_STATIC_V1_QA.md` para la tabla completa de gates (A-X) tras auditoría PASS. Resumen: 48/48 config PASS, 48/48 split PASS, 48/48 prediction schema PASS, 48/48 métricas reproducidas de forma independiente, 40/40 convergencia BNN PASS, 8/8 configuración logistic PASS, 5580/5580 predicciones completas, hashes de campaña cruda y del repositorio histórico intactos.

## Terminología

cross-site transport / LOSO cross-site evaluation / internal-external validation-style analysis / held-out site / transportability. Evitar sin matiz: external validation, independent validation, generalizes across sites, clinically validated, robust biomarker.

## Caveat NYU

La configuración de BrainNetCNN se desarrolló/fijó históricamente usando NYU antes de la evaluación multisitio. La rotación con NYU held-out es 'development-site held-out re-evaluation within the LOSO campaign', no una evaluación en un sitio totalmente ajeno al desarrollo del modelo.

## No harmonización / no ponderación

class_weight=false, site_weighting=false, sample_weight=false, harmonization=none en las 48 corridas formales.
