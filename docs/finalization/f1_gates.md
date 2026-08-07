# Fase 1 — Hechos bloqueantes

**Baseline:** HEAD `4c15d39`, rama `paper/finalization-2026-08`.

---

## 1.1 Procedencia de los paneles ROI — Gate G1

**Estado: sin resolver. No se activa G1-C en este documento.**

La búsqueda dentro del repositorio ya estaba agotada antes de esta fase (ver Plan v1.1 §1.1): el repositorio arranca el 2026-07-20 con `roi_sets.json` ya completo en el commit inicial del pipeline (`3a9424a`), una sola modificación posterior sin registro de derivación (`543fe01`, 2026-07-29), y ningún mensaje de commit que mencione selección o ablación de paneles.

Esta fase no añade una nueva búsqueda dentro del repositorio porque no hay nada nuevo que revisar: el historial ya se agotó. La búsqueda **fuera** del repositorio (notebooks históricos, correos, notas de reunión) sigue pendiente del equipo — ítem 1 de la sección "Pendiente del equipo" del plan. G1-C no se activa unilateralmente aquí porque el plan lo condiciona a que se agote esa ventana externa o a una autorización explícita del equipo; ninguna de las dos ha ocurrido todavía.

**Consecuencia práctica:** la Fase 3 (edición de terminología ROI en Methods/Results) permanece bloqueada hasta que se resuelva G1.

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

**Distinción explícita que no debe perderse:** esto resuelve la procedencia de los *hiperparámetros del modelo*. No dice nada sobre la procedencia de los *paneles de ROI* (G1, que sigue abierto). Son dos preguntas independientes y no deben fusionarse en el texto: "el modelo no se ajustó por sitio" no implica "los paneles se definieron a priori".

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

---

## Checkpoint F1 (parcial)

| Punto | Estado |
|---|---|
| G1 (procedencia ROI) | **Abierto** — pendiente de evidencia externa o autorización de G1-C |
| Procedencia de hiperparámetros | **Documentada** — congelados 8 días antes del lote, idénticos entre sitios |
| G2 (`class_weight` Peking) | **PASS** — prespecificado, documentado 2 días antes de la corrida |
| Referencias pendientes | Ver Fase 6 |
| Inventario de covariables (G3/G4) | Ver §F2 |

**Por lo anterior, la Fase 3 (edición de Methods/Results sobre terminología ROI) sigue bloqueada por G1.** El resto de la Fase 1 queda cerrado.
