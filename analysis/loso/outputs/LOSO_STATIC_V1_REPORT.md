# LOSO_STATIC_V1_REPORT

Campaña `loso_static_v1`: 4 sitios held-out x 2 ROI sets x 2 familias de modelo, conectividad estática, sin harmonización ni ponderación de clase/sitio. Ver `PLAN_FINAL_LOSO_STATIC_V1_IA_REVISADO.md` para el diseño completo.

## Completitud: 48/48 corridas formales

## AUC por condición (95% CI, percentil, sin ajustar)

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

## Contrastes preespecificados (95% CI)

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

## Terminología

cross-site transport / LOSO cross-site evaluation / internal-external validation-style analysis / held-out site / transportability. Evitar sin matiz: external validation, independent validation, generalizes across sites, clinically validated, robust biomarker.

## Caveat NYU

La configuración de BrainNetCNN se desarrolló/fijó históricamente usando NYU antes de la evaluación multisitio. La rotación con NYU held-out es 'development-site held-out re-evaluation within the LOSO campaign', no una evaluación en un sitio totalmente ajeno al desarrollo del modelo.

## No harmonización / no ponderación

class_weight=false, site_weighting=false, sample_weight=false, harmonization=none en las 48 corridas formales.
