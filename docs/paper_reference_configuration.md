# Configuración de referencia del manuscrito

**Fuente canónica única** para la configuración experimental reportada en
`Manuscript_Methods_Results_English_Working_v9_9.docx`. Si algo en el README,
en las guías de experimentación, o en la memoria de alguien del equipo
contradice este documento, **este documento tiene prioridad** para lo que
respecta al paper: el README describe el comportamiento general del runner,
que incluye defaults legado no usados en la campaña oficial.

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
Definidos en `data/atlas/roi_sets.json`. **Procedencia: ver Gate G1 en
`docs/finalization/f1_gates.md` — no resuelta a la fecha de este documento.**

## 3. Ventana y número de ventanas por sitio — configuración de la campaña oficial

**Todos los sitios usan el mismo parámetro físico de ventana: 120 s, paso 12 s.**
El número resultante de ventanas difiere solo porque el TR y la duración del
escaneo difieren entre sitios — no porque se haya usado una ventana distinta
por sitio.

| Sitio | TR (s) | Puntos temporales | Duración escaneo (s) | Ventana física | Ventanas resultantes (`n_windows`) |
|---|---|---|---|---|---|
| NYU | 2.00 | 172 | 344 | 120 s / 12 s | **19** |
| Peking | 2.00 | 232 | 464 | 120 s / 12 s | **29** |
| NeuroIMAGE | 1.96 | 257 | 504 | 120 s / 12 s | **33** |
| OHSU | 2.50 | 74 | 185 | 120 s / 12 s | **6** |

**Esto corrige la tabla de "ventana por defecto" del README** (§7 más abajo),
que describe el comportamiento del runner sin argumentos explícitos, no la
configuración realmente usada en la campaña. En particular, OHSU **no** se
evaluó como estática por defecto en la campaña oficial: se evaluó con
ventaneado (6 ventanas), igual que los otros tres sitios.

## 4. Validación

10 pliegues externos × 5 repeticiones (`n_splits=10`, `n_repeats=5`) en los
**cuatro** sitios, incluidos NeuroIMAGE (n=39) y OHSU (n=66) — verificado en
los `config.json` de las corridas de referencia. Validación interna al
10.15% del particionamiento externo (`inner_val_frac=0.15`). Toda la
validación es interna al sitio; ningún sitio funciona como validación externa
de otro.

## 5. Class weighting

| Sitio | `class_weight` | Base |
|---|---|---|
| NYU | `False` | — |
| Peking | **`True`** | desbalance de clases (Gate G2 = PASS, prespecificado; ver `docs/finalization/f1_gates.md` §1.3) |
| NeuroIMAGE | `False` | — |
| OHSU | `False` | — |

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

Ver `docs/paper_environment.md` para la tabla completa. Resumen:

- **Environment A** (campaña experimental oficial: todas las corridas de
  `results/runs/`): Python 3.12.13, TensorFlow 2.20.0, Keras 3.13.2.
- **Environment B** (scripts de análisis derivado y auditoría:
  `analysis/roi_comparison/scripts/*`): Python 3.13.2, TensorFlow 2.21.0,
  Keras 3.15.1.

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
| `analysis/roi_comparison/outputs/tables/figure4_v6_audit.csv` | Filas de Figure 3 / Table 5 no algorítmicas (paneles, representación, arquitectura, ventaneado) | `analysis/roi_comparison/scripts/generate_manuscript_figures_v6.py`, Environment B |
| `analysis/roi_comparison/outputs/tables/algorithm_comparison_deepsets_audit.csv` | Fila "DeepSets, 12 ROIs (static comparator)" de Table 5 | `analysis/roi_comparison/scripts/generate_algorithm_comparison_audit.py`, Environment B |
| `analysis/roi_comparison/outputs/tables/manuscript_bootstrap_10k.csv` | Contrastes de regresión logística (`baseline__*`) de Table 5 | script de bootstrap 10k, Environment B |
| `analysis/roi_comparison/outputs/analysis_manifest.json` | Reconciliación global | `reconciliation_status: PASS (16/16)` |
| `analysis/finalization/cohort_audit.csv` | Auditoría de cohorte (n/control/TDAH por sitio) | `analysis/finalization/build_cohort_audit.py`, solo lee `predictions_val.csv` ya almacenados |
| `analysis/finalization/demographics_by_site_dx.csv` | Demografía por sitio × diagnóstico | `analysis/finalization/build_demographics.py`, fuente externa con hash verificado (ver `docs/data_provenance/adhd200_phenotypics.md`) |

Ningún archivo de esta lista se modifica sin regenerar por script y sin
volver a certificar F3.
