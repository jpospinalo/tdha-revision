# Configuración de referencia del manuscrito

**Fuente canónica única** para la configuración experimental reportada en el
manuscrito vigente, `docs/manuscrito_revisado/Manuscript_Methods_Results_English_Working_v9_10_LOSO_V3_2_1_R8_clean.docx`
(la campaña base descrita aquí no cambió al pasar de la generación `v9_9` a
la integración LOSO `v9_10`; solo se añadieron las secciones LOSO — ver
`analysis/loso/README.md`). Si algo en el README, en las guías de
experimentación, o en la memoria de alguien del equipo contradice este
documento, **este documento tiene prioridad** para lo que respecta al
paper: el README describe el comportamiento general del runner, que
incluye defaults legado no usados en la campaña oficial.

---

## 1. Cohorte

| Sitio | n | Control | TDAH | Fuente OOF (cohorte) |
|---|---|---|---|---|
| NYU | 177 | 87 | 90 | `results/runs/12/NYU_rois12_static_logreg_baseline_aafd45ca` |
| Peking | 183 | 109 | 74 | `results/runs/12/Peking_rois12_static_logreg_baseline_1e9626ad` |
| NeuroIMAGE | 39 | 22 | 17 | `results/runs/12/NeuroIMAGE_rois12_static_logreg_baseline_bc5c92b2` |
| OHSU | 66 | 38 | 28 | `results/runs/12/OHSU_rois12_static_logreg_baseline_a59dca47` |
| **Total** | **465** | **256** | **209** | verificado en `analysis/finalization/cohort_audit.csv` |

Diagnóstico binarizado: `DX==0` → control; `DX∈{1,2,3}` (ADHD-Combined,
Hyperactive/Impulsive, Inattentive) → TDAH. Ver `docs/data_provenance/adhd200_phenotypics.md`.

## 2. Paneles de ROI

Cuatro paneles fijos sobre el atlas AAL116: 12, 18, 39, 116 regiones.
Definidos en `data/atlas/roi_sets.json`. **Procedencia: Gate G1 = G1-A,
cerrado el 2026-08-07** — paneles prespecificados, informados por literatura
previa y juicio neuroanatómico experto (confirmación directa del equipo).
Ver `docs/finalization/f1_gates.md` §1.1.

## 3. Ventana y número de ventanas por sitio — configuración de la campaña oficial

**Todos los sitios usan el mismo parámetro *solicitado* de ventana: 120 s,
paso 12 s** (`requested_window_seconds`/`requested_step_seconds` en
`config.json`). El valor *efectivo* difiere ligeramente entre sitios porque
la ventana y el paso se discretizan en unidades de TR
(`window_tr`/`step_tr`), y el TR no es igual en los cuatro sitios. No decir
que los cuatro sitios tuvieron exactamente el mismo paso físico efectivo:

| Sitio | TR (s) | Puntos temporales | Duración escaneo (s) | Ventana solicitada/efectiva | Paso solicitado/efectivo | Ventanas resultantes (`n_windows`) |
|---|---|---|---|---|---|---|
| NYU | 2.00 | 172 | 344 | 120 s / 120.00 s | 12 s / 12.00 s | **19** |
| Peking | 2.00 | 232 | 464 | 120 s / 120.00 s | 12 s / 12.00 s | **29** |
| NeuroIMAGE | 1.96 | 257 | 504 | 120 s / 119.56 s | 12 s / 11.76 s | **33** |
| OHSU | 2.50 | 74 | 185 | 120 s / 120.00 s | 12 s / 12.50 s | **6** |

(`window_tr`/`step_tr` × `tr_seconds` de cada sitio: NYU 60×2.00/6×2.00;
Peking 60×2.00/6×2.00; NeuroIMAGE 61×1.96/6×1.96; OHSU 48×2.50/5×2.50.)

**Esto corrige la tabla de "ventana por defecto" del README** (§7 más abajo),
que describe el comportamiento del runner sin argumentos explícitos, no la
configuración realmente usada en la campaña. En particular, OHSU **no** se
evaluó como estática por defecto en la campaña oficial: se evaluó con
ventaneado (6 ventanas), igual que los otros tres sitios.

## 4. Validación

10 pliegues externos × 5 repeticiones (`n_splits=10`, `n_repeats=5`) en los
**cuatro** sitios, incluidos NeuroIMAGE (n=39) y OHSU (n=66) — verificado en
los `config.json` de las corridas de referencia. Validación interna al
15% del particionamiento externo (`inner_val_frac=0.15`). Toda la
validación es interna al sitio; ningún sitio funciona como validación externa
de otro.

## 5. Class weighting

| Sitio | `class_weight` | Base |
|---|---|---|
| NYU | `False` | — |
| Peking | **`True`** | desbalance de clases (Gate G2 = PASS, prespecificado; ver `docs/finalization/f1_gates.md` §1.3) |
| NeuroIMAGE | `False` | — |
| OHSU | `False` | — |

### Corrección class_weight Peking (2026-08-07)

Las seis corridas `reviewer_sensitivity` de Peking usadas en Figure 3 / Table 5
fueron ejecutadas originalmente sin `--class-weight` en
`run_reviewer_sensitivity.sh`, violando la política de esta tabla. Se
detectó, verificó contra los seis `config.json` (`class_weight=False`,
`split_fingerprint=1e9626ad3839ff46` en las seis) y se corrigió en la rama
`fix/peking-class-weight-consistency`. Las corridas históricas se conservan
como provenance pero **no se usan** en ningún artefacto canónico del
manuscrito desde esta corrección:

| Condición | Corrida histórica (superseded, `class_weight=False`) | Corrida corregida (canónica, `class_weight=True`) |
|---|---|---|
| LSTM estático | `Peking_rois12_static_lstm_reviewer_sensitivity_da7862e4` | `Peking_rois12_static_lstm_reviewer_sensitivity_weighted_fix_f0640423` |
| GRU 120 s / 12 s | `Peking_rois12_w60s6_gru_reviewer_sensitivity_60c51708` | `Peking_rois12_w60s6_gru_reviewer_sensitivity_weighted_fix_8a4926b2` |
| DeepSets estático | `Peking_rois12_static_deepsets_reviewer_sensitivity_1b2c1963` | `Peking_rois12_static_deepsets_reviewer_sensitivity_weighted_fix_e7e3d566` |
| DeepSets 120 s / 12 s | `Peking_rois12_w60s6_deepsets_reviewer_sensitivity_b954a3cf` | `Peking_rois12_w60s6_deepsets_reviewer_sensitivity_weighted_fix_fbe99635` |
| BrainNetCNN 60 s / 12 s | `Peking_rois12_w30s6_brainnetcnn_reviewer_sensitivity_9070ebdd` | `Peking_rois12_w30s6_brainnetcnn_reviewer_sensitivity_weighted_fix_2bfac330` |
| GRU 60 s / 12 s | `Peking_rois12_w30s6_gru_reviewer_sensitivity_a2065789` | `Peking_rois12_w30s6_gru_reviewer_sensitivity_weighted_fix_2e23b0b9` |

Las seis corridas corregidas comparten `split_fingerprint=1e9626ad3839ff46`
con sus contrapartes históricas (mismas particiones; solo cambia
`class_weight`), verificado fold-a-fold (folds.csv, `predictions_val.csv`,
`class_weight_0`/`class_weight_1`) antes de sustituir los selectores en
`generate_manuscript_figures_v6.py` y `generate_algorithm_comparison_audit.py`.
Efecto neto en las seis celdas de Table 5/Figure 3 afectadas: ningún cambio
de signo ni de inclusión de cero; solo magnitud e IC.

## 6. Configuración de BrainNetCNN

Congelada antes del lote multisitio (ver `docs/finalization/f1_gates.md` §1.2)
y aplicada sin cambios en los cuatro sitios:

```
e2e=4  e2n=8  dense=8  dropout=0.7  leaky=0.33  l2_reg=0.05  inter_dropout=0.6
lr=0.0001  batch_size=32  epochs=300  patience=25  inner_val_frac=0.15
early_stopping_monitor=val_loss  early_stopping_min_delta=1e-05  seed=42
```

`runner_code_hash` y `data_code_hash` idénticos en los cuatro sitios para
esta familia de corridas.

## 7. Entornos

Ver `docs/paper_environment.md` para la tabla completa. **Tres** entornos,
no dos — `results/runs/**` no pertenece íntegramente a uno solo:

- **Environment A** (26 corridas: las 16 combinaciones de referencia de
  Table 4 más las `rev32`): Python 3.12.13, TensorFlow 2.20.0, Keras 3.13.2,
  hardware mixto (17 sobre Tesla T4, 9 sobre CPU).
- **Environment B**: Python 3.13.2, NumPy 2.5.1, pandas 3.0.5,
  scikit-learn 1.8.0, TensorFlow 2.21.0, Keras 3.15.1, todas en CPU. No es
  exclusivamente un entorno de análisis: entrenó redes neuronales y corre
  los scripts de análisis. Desde la corrección de class_weight de Peking
  (2026-08-07, ver §5) hay **20** corridas `*reviewer_sensitivity*` bajo
  `results/runs/12/` en este entorno, no 14: las 8 condiciones no
  afectadas (NYU ×6, NeuroIMAGE ×1, OHSU ×1) más las 6 corridas
  históricas de Peking (`class_weight=False`, superseded, solo
  provenance) más las 6 corridas corregidas de Peking
  (`*_weighted_fix_*`, `class_weight=True`, canónicas). De esas 20,
  **14** son las canónicas usadas por el manuscrito (8 no afectadas + 6
  corregidas); las 6 históricas de Peking están *presentes* en
  `results/runs/` pero no son *usadas* por ningún artefacto canónico. No
  confundir "presente en results/runs" con "usado por el manuscrito".
  Los scripts derivados (`generate_manuscript_figures_v6.py`,
  `generate_algorithm_comparison_audit.py`, etc.) también corren en este
  entorno y usan, entre otros paquetes, `scipy.stats.rankdata`; la versión
  exacta de SciPy de Environment B no quedó registrada en la metadata de
  ninguna corrida (SciPy no es un paquete de entrenamiento, así que
  `config.json` no la captura) y no se reconstruye por inferencia.
- **Environment C** (16 corridas de regresión logística estática,
  `*_static_logreg_baseline_*`): Python 3.10.12, scikit-learn 1.7.2.
  NumPy/pandas: usados (la regresión logística y el manejo de datos los
  requieren), pero sus versiones exactas no quedaron registradas en la
  metadata de `config.json` de estas corridas; no se completan por
  inferencia. TensorFlow/Keras: no aplica — esta familia no usa red
  neuronal.

## 8. Comando reconstruible (plantilla)

```bash
python run_experiment.py --site {SITIO} --roi-set {12|18|39|116} \
    --model brainnetcnn --representation ordered \
    --window-seconds 120 --step-seconds 12 \
    --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 \
    --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 \
    --inner-val-frac 0.15 --start-from-epoch 0 \
    --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 \
    --seed 42 --n-splits 10 --n-repeats 5 --tag control_baseline_v13 --verbose
```

Reconstruido literalmente desde `config.json["command"]` de
`NYU_rois12_w60s6_brainnetcnn_control_baseline_v13_3e220e5c`; los demás sitios
usan la misma plantilla con `--site` distinto (mismos hiperparámetros).

## 9. Run IDs oficiales — las 16 combinaciones sitio × panel (BrainNetCNN, Table 4)

| Panel | NYU | Peking | NeuroIMAGE | OHSU |
|---|---|---|---|---|
| 12 | `..._control_baseline_v13_3e220e5c` | `..._control_baseline_v13_bc841110` | `..._control_baseline_v13_2b729a8c` | `..._control_baseline_v13_1a7c37ce` |
| 18 | `..._control_baseline_v13_662d71a9` | `..._control_baseline_v13_0bf7fa0e` | `..._control_baseline_v13_93342cf0` | `OHSU_rois18_w48s5_brainnetcnn_2ce6c48e` ⚠ |
| 39 | `NYU_rois39_w60s6_brainnetcnn_control_base_line_1521c348` ⚠ | `..._control_baseline_v13_396e34d2` | `..._control_baseline_v13_dc028168` | `..._control_baseline_v13_299719fe` |
| 116 | `..._control_baseline_v13_160b89cd` | `Peking_rois116_w60s6_brainnetcnn_240732d1` ⚠ | `..._control_baseline_v13_669d72bd` | `..._control_baseline_v13_f82f17b4` |

Prefijo común `{SITIO}_rois{PANEL}_w{ventana_tr}s{paso_tr}_brainnetcnn`, omitido
por espacio.

**⚠ Nota de trazabilidad (no una corrección, una advertencia):** 3 de las 16
corridas no llevan el tag `control_baseline_v13` (una tiene el tag mal escrito
`control_base_line`, dos no llevan tag de campaña en absoluto). Se verificó
que sus hiperparámetros, `n_windows`, `class_weight`, `n_splits`/`n_repeats` y
el AUC resultante en `descriptive_performance.csv` son consistentes con el
resto de la familia — son las corridas oficiales, solo con un nombre
inconsistente. **No se renombran** (el plan lo prohíbe explícitamente: no
alterar `run_id` de corridas históricas). Si el equipo repite esta campaña en
el futuro, usar el tag `control_baseline_v13` de forma consistente en las 16
combinaciones.

Baseline logístico estático (12/18/39/116 ROI, por sitio):
`{SITIO}_rois{PANEL}_static_logreg_baseline_{hash}` — hash de
`split_fingerprint` compartido dentro de cada sitio (`aafd45ca` NYU,
`1e9626ad` Peking, `bc5c92b2` NeuroIMAGE, `a59dca47` OHSU).

## 10. Salidas usadas por el manuscrito y su procedencia

| Archivo | Contenido | Procedencia |
|---|---|---|
| `analysis/roi_comparison/outputs/tables/descriptive_performance.csv` | Table 4 (AUC + IC + métricas secundarias, 16 combos) | *canonical derived analysis artifact*: predicciones OOF → script de análisis → bootstrap 10k → CSV |
| `analysis/roi_comparison/outputs/tables/primary_12_vs_116.csv` | Contraste primario 116 vs 12 ROI | idem |
| `analysis/roi_comparison/outputs/tables/figure4_v6_audit.csv` | Filas de Figure 3 / Table 5 no algorítmicas (paneles, representación, arquitectura, ventaneado) | `analysis/roi_comparison/scripts/generate_manuscript_figures_v6.py`, Environment B. Peking usa las 6 corridas `reviewer_sensitivity_weighted_fix` corregidas (§5), no las históricas |
| `analysis/roi_comparison/outputs/tables/algorithm_comparison_deepsets_audit.csv` | Fila "DeepSets, 12 ROIs (static comparator)" de Table 5 | `analysis/roi_comparison/scripts/generate_algorithm_comparison_audit.py`, Environment B. Peking usa la corrida `reviewer_sensitivity_weighted_fix` corregida (§5) |
| `analysis/roi_comparison/outputs/tables/manuscript_bootstrap_10k.csv` | Contrastes de regresión logística (`baseline__*`) de Table 5 | script de bootstrap 10k, Environment B; predicciones de entrada de las corridas `*_static_logreg_baseline_*`, Environment C |
| `analysis/roi_comparison/outputs/analysis_manifest.json` | Reconciliación global | `reconciliation_status: PASS (16/16)` |
| `analysis/finalization/cohort_audit.csv` | Auditoría de cohorte (n/control/TDAH por sitio) | script de análisis en Environment B (`analysis/finalization/build_cohort_audit.py`); lee `predictions_val.csv` ya almacenados de las corridas `*_static_logreg_baseline_*`, generadas en Environment C |
| `analysis/finalization/demographics_by_site_dx.csv` | Demografía por sitio × diagnóstico | script de análisis en Environment B (`analysis/finalization/build_demographics.py`); fuente de datos: fenotípico externo con hash verificado (ver `docs/data_provenance/adhd200_phenotypics.md`), no una corrida de `results/runs/` |

Ningún archivo de esta lista se modifica sin regenerar por script y sin
volver a certificar F3.
