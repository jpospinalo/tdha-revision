# Fase 3 — segunda ronda (comentarios 2–5 del equipo)

Todos los cambios son de texto (Methods/Results) o de un nuevo CSV de respaldo no destructivo. Ningún valor de AUC, intervalo de confianza, métrica secundaria, tabla o figura fue alterado: 108 números decimales en el cuerpo del texto, idéntico antes y después. Copia previa: `docs/finalization/f3_terminologia/PREVIO_antes_de_ronda2.docx`.

## Comentario 2 — Construcción de la conectividad estática, no descrita

**Aceptado sin reservas.** Verificado contra el código (`src/data.py::build_static_connectivity`): usa `window=step=duración total de la serie` → una sola ventana por participante, y el cálculo dentro de esa ventana (`_correlation_windows`, rama `rectangular`) es Pearson estándar (centrado + normalización por desviación estándar). Se añadió al final de Methods §2.3 (párr. 10):

> *"For the static-connectivity conditions, Pearson correlations were instead computed over the complete postprocessed regional time series, yielding one connectivity matrix per participant."*

## Comentario 3 — `class_weight` documentado pero no justificado

**Aceptado.** El plan ya había resuelto este punto como Gate G2 = PASS (`docs/finalization/f1_gates.md` §1.3: política documentada el 2026-07-26, dos días antes de la corrida oficial de Peking del 2026-07-28, derivada del desbalance de clases visible en los datos desde el 2026-07-24). Lo que faltaba era trasladar esa resolución al propio manuscrito — no se había hecho. Se añadió a Methods §2.4 (párr. 35):

> *"This policy addressed a class imbalance present in Peking's cohort and was documented prior to the corresponding experimental runs; no other site required this adjustment."*

## Comentario 4 — Restricción a NYU/Peking, posible razonamiento post hoc

**Aceptado, con verificación cronológica propia.** Se reconstruyó la cronología real:

| Fecha | Evento |
|---|---|
| 2026-07-30 11:33 | `primary_12_vs_116.csv` generado — el contraste primario (y sus anchos de IC por sitio) ya estaba calculado |
| 2026-07-31 00:09–02:58 | Corridas de arquitectura/ventaneado restringidas a NYU/Peking (LSTM, ventanas 140s/120s-24) |
| 2026-08-04 | Primer documento del repositorio que describe la restricción como "decisión preespecificada de precisión" |

La "decisión preespecificada" se documentó **después** de ejecutar las corridas que supuestamente restringía, y el contraste primario ya era conocido un día antes de esas corridas. No hay evidencia contemporánea de que el criterio de "intervalos más estrechos" se fijara antes de verlos — es indistinguible de una racionalización posterior. El tamaño de sitio (NYU+Peking = 360 sujetos vs NeuroIMAGE+OHSU = 105) sí es un hecho fijo desde el inicio, no depende de ningún resultado.

Se reformuló Methods §2.6 (párr. 31), retirando el criterio de "intervalos más estrechos" como justificación de diseño y marcando la restricción como exploratoria:

> *"Architecture and windowing analyses were exploratory and were subsequently restricted to NYU and Peking, the two largest sites, which also provided the most precise primary-comparison estimates."*

## Comentario 5 — Lenguaje causal de "memorization" y falta de evidencia de convergencia

**Aceptado, y verificado con datos reales antes de aplicar el cambio.** Se comprobó el patrón descrito con `history.csv` de NYU (12 vs 116 ROI): a la época 200, la brecha train−val accuracy es −0.021 en 12 ROI pero +0.397 en 116 ROI (train 0.946 vs val 0.549) — el patrón es real, solo la especificidad causal del lenguaje era excesiva. Se corrigió Results §3.4 (párr. 67), separando también la descripción de `loss` de la de `accuracy`:

> *"For the 39- and 116-ROI panels, training accuracy continued to increase after roughly the first third of training while inner-validation accuracy remained comparatively flat, a pattern consistent with increasing overfitting rather than improved generalization."*

### Hallazgo adicional (no señalado por el equipo, encontrado al verificar la evidencia de respaldo)

Para responder a la segunda parte del comentario 5 —falta de evidencia visible para las cifras de convergencia— se construyó `analysis/finalization/convergence_summary.csv` (script `build_convergence_summary.py`, no reentrena nada, lee solo `metrics_val.csv`/`history.csv` ya almacenados, con verificación cruzada interna: `n_epochs` debe coincidir con el máximo `epoch` de `history.csv`, 0 discrepancias en las 16 combinaciones).

Al construirlo se encontró que **el rango publicado para el panel de 116 ROI era incorrecto**:

| Panel | Rango publicado (antes) | Rango real verificado |
|---|---|---|
| 12 ROI (referencia) | "44 and 50" | **44–50** ✓ coincide exactamente |
| 116 ROI | "19 to 28" | **19–43** ✗ no coincide — Peking real es 43/50, fuera del rango declarado |

Verificado por dos vías independientes (columna `n_epochs` de `metrics_val.csv`, y máximo de `epoch` en `history.csv` — coinciden exactamente en los 4 sitios) y confirmado que la corrida de Peking/116 usada es la oficial (AUC = 0.60329779320605, idéntico a 12 decimales con `descriptive_performance.csv`). Se corrigió Results §3.4 (párr. 68):

> *"...at the 116-ROI panel, 19 to 43 of 50 folds reached the ceiling depending on the site, and for the LSTM architecture early stopping triggered in every fold."*

No se insertó una tabla nueva en el cuerpo del manuscrito: la Fase 5 (reparto main/suplemento) sigue bloqueada por falta de revista/límite de páginas, y crear una tabla ahora implicaría renumerarla dos veces. `convergence_summary.csv` queda como artefacto derivado canónico, listo para convertirse en tabla suplementaria cuando F5 se desbloquee.

## Verificación estructural final

- 85 párrafos, 5 tablas, 4 imágenes — sin cambio.
- 108 números decimales en el cuerpo — sin cambio (las cifras corregidas, 19/28/43, son conteos enteros de pliegues, no valores de resultado).
- `ablation`, `memorization`, `discovered`, `discriminative weight` — ninguno presente.
