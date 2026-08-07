# Fase 1 — Hechos bloqueantes

**Baseline:** HEAD `4c15d39`, rama `paper/finalization-2026-08`.

---

## 1.1 Procedencia de los paneles ROI — Gate G1

**Estado: G1-A. Cerrado.**

La búsqueda dentro del repositorio estaba agotada (ver Plan v1.1 §1.1): el repositorio arranca el 2026-07-20 con `roi_sets.json` ya completo en el commit inicial del pipeline (`3a9424a`), una sola modificación posterior sin registro de derivación (`543fe01`, 2026-07-29), y ningún mensaje de commit que mencione selección o ablación de paneles. El historial de git no podía resolver G1 por sí solo.

**Confirmación directa del equipo (2026-08-07):** los paneles de ROI fueron seleccionados antes de los experimentos, con base en las redes funcionales que el propio manuscrito ya menciona en §2.2 (DMN, ECN, SN, DAN, FST; Figure 1). Esto resuelve G1 como **G1-A** — evidencia suficiente de definición a priori, informada por literatura previa y juicio neuroanatómico experto — y no como G1-C.

**Consecuencia práctica:** la Fase 3 ya se ejecutó con la narrativa G1-A (ver `docs/finalization/f3_terminologia/`). Se retiró `derived by progressive ablation` de Methods §2.1 y `ablation-derived` de Results §3.1, y se separó explícitamente la procedencia de los paneles (a priori, por literatura) de la del entrenamiento del modelo (desarrollado en NYU) en Methods §2.6 (párr. 26) y Results §3.2 (párr. 55), conforme a §1.2 más abajo.

---

## 1.2 Separar procedencia de paneles y de hiperparámetros

**Verificado: los hiperparámetros de BrainNetCNN se congelaron antes del lote multisitio y se aplicaron sin cambios en los cuatro sitios.**

Evidencia:

| Hecho | Fuente |
|---|---|
| Último cambio al código del modelo (`src/kerasmodels/brainnetcnn.py`) | commit `39e5218`, 2026-07-23 11:33:06 -0500 |
| Corridas estáticas de 12 ROI (`rev32`) de los cuatro sitios, ejecutadas en una ventana continua de 68 minutos, 8 días después del último cambio de código | NYU 00:04:56 · Peking 00:27:55 · NeuroIMAGE 01:03:24 · OHSU 01:12:06 (2026-07-31) |
| `lr`, `dropout`, `arch_json`, `patience`, `clipnorm`, `min_delta`, `epochs`, `seed` | **idénticos** en los cuatro `config.json` |
| `runner_code_hash`, `data_code_hash` | **idénticos** en los cuatro sitios |

No hay tuning por sitio: el mismo código, con los mismos hiperparámetros, se ejecutó secuencialmente sobre los cuatro sitios sin modificación entre corridas.

**Frase factual reutilizable en Methods (§2.6):**

> *The BrainNetCNN architecture and its hyperparameters (learning rate, patience, regularization) were fixed prior to the multi-site evaluation and were not modified between sites; the same configuration and code version were applied to NYU, Peking, NeuroIMAGE, and OHSU.*

**Distinción explícita que no debe perderse:** esto resuelve la procedencia de los *hiperparámetros del modelo*, que es una pregunta independiente de la procedencia de los *paneles de ROI* (G1). Ambas quedan cerradas hoy — hiperparámetros por evidencia directa de código/config (arriba), paneles como G1-A por confirmación del equipo (§1.1) — pero se cerraron por vías distintas y no deben fusionarse en el texto: "el modelo no se ajustó por sitio" no implica, por sí solo, "los paneles se definieron a priori".

---

## 1.3 Cronología de `class_weight` de Peking — Gate G2

**Estado: G2 = PASS.** La política es prespecificada y documentada antes de la corrida oficial. **No se activa la contingencia de sensibilidad (§2.4).**

Cadena temporal verificada:

| Fecha | Evento | Evidencia |
|---|---|---|
| 2026-07-24 15:35:53 | Se añade el fenotípico crudo de Peking al repositorio; el desbalance de clases (DX=0: 61, DX=3: 17, DX=1: 7) es visible directamente en los datos, sin requerir ningún resultado de modelo | commit `aa3f885` |
| 2026-07-26 11:39:47 | Primera versión de `docs/guia-experimentacion-colaborativa.md` — **ya contiene** la tabla de política por sitio: `Peking → CLASS_WEIGHT = True (desbalanceado)`; los otros tres sitios en `False` | commit `991abf5` |
| 2026-07-28 17:17:39 | Corrida oficial de Peking BrainNetCNN (12 ROI, `control_baseline_v13`) con `class_weight: True` en su `config.json` | `results/runs/12/Peking_rois12_w60s6_brainnetcnn_control_baseline_v13_bc841110/config.json` |

La política se documentó en un archivo versionado y con fecha **dos días antes** de que se ejecutara la corrida que la usa. La regla no se infirió del desempeño del modelo: se derivó de una característica fija y observable de los datos (la proporción de clases), fijada por escrito antes de entrenar. Esto satisface el criterio de prespecificación que exige G2.

**Conclusión:** la corrida original de Peking es la *reference analysis from the completed experimental campaign* sobre bases fácticas verificadas, no por invocación circular. No se ejecuta ninguna corrida adicional de sensibilidad de `class_weight`.

Texto para Methods:

> *For Peking, class weighting was applied during training to address the class imbalance present in that site's cohort; this policy was documented prior to the corresponding experimental run and applied uniformly across the site's evaluations. No other site required this adjustment.*

**Nota append-only (2026-08-07) — implementación de la política, no el gate:**

La política Peking `class_weight=True` permaneció prespecificada. Durante la auditoría posterior se detectó que seis corridas `reviewer_sensitivity` omitieron accidentalmente el flag en el script. Esas corridas se conservaron como provenance y fueron sustituidas en los análisis canónicos por seis corridas corregidas con weighting, sin cambiar folds, seeds ni las demás especificaciones. Detalle completo: rama `fix/peking-class-weight-consistency`, mapeo run-a-run en `docs/paper_reference_configuration.md` §"Corrección class_weight Peking (2026-08-07)". Esto no reabre ni reescribe la cronología original de G2 (arriba): la política siempre fue `True` para Peking; lo que falló fue la ejecución del script de sensibilidad para seis condiciones, no la especificación de la política.

---

## Checkpoint F1 (parcial)

| Punto | Estado |
|---|---|
| G1 (procedencia ROI) | **G1-A — cerrado**, confirmación directa del equipo (2026-08-07) |
| Procedencia de hiperparámetros | **Documentada** — congelados 8 días antes del lote, idénticos entre sitios |
| G2 (`class_weight` Peking) | **PASS** — prespecificado, documentado 2 días antes de la corrida |
| Referencias pendientes | Ver Fase 6 |
| Inventario de covariables (G3/G4) | Ver §F2 |

**Fase 1 cerrada por completo.** La Fase 3 (edición de Methods/Results sobre terminología ROI) ya se ejecutó — ver `docs/finalization/f3_terminologia/informe_f3.md`.
