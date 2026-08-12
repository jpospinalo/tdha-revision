# HANDOFF_FOR_NEXT_PHASE.md — LOSO_METHODS_RESULTS_INTEGRATION_V3_2_1

Este documento traspasa al equipo, y a cualquier fase futura, el contexto interpretativo que **no** debe insertarse en Methods/Results del manuscrito en esta fase, junto con las tareas explícitamente diferidas. No duplica el contenido narrativo de `IMPLEMENTATION_REPORT.md`; lo complementa.

## H5 — Within-site vs. LOSO: por qué no son comparables numéricamente

- **Estimandos distintos:** within-site es una validación cruzada repetida y estratificada *dentro* de cada sitio; LOSO es una evaluación *site-held-out*, con el sitio evaluado completamente fuera del entrenamiento.
- **Representación de conectividad distinta:** within-site (Tables 4–5) evalúa dynamic/windowed y static connectivity a través de varias condiciones de sensibilidad; LOSO usa exclusivamente static Pearson connectivity.
- **La política de class-weighting en Peking difiere entre análisis:** within-site aplica weighting inverso a la frecuencia de clase en Peking (y no en los otros tres sitios); el procedimiento LOSO congelado no aplica ningún class/site/sample weighting en ninguna rotación. Cualquier comparación numérica directa entre un AUC within-site de Peking y su AUC LOSO confundiría el efecto de esa política con el efecto del cambio de estimando.
- **No son directamente comparables:** ninguna cifra de este informe, del manuscrito o del Supplement resta, promedia o combina un AUC within-site con un AUC LOSO.
- **No existe un "generalization gap" numérico:** no se calculó, no se reportó, y no debe calcularse en una fase futura sin una justificación metodológica separada que resuelva las dos diferencias anteriores (representación y weighting), no solo el cambio de estimando.

## H6 — Limitación de desarrollo/configuración (NYU / 12-ROI / windowed connectivity)

- **Desarrollo histórico verificado:** la configuración de entrenamiento y regularización de BrainNetCNN fue desarrollada y fijada usando NYU, en la condición 12-ROI, con conectividad windowed — antes de la evaluación multi-sitio.
- **Decisiones retenidas por out-of-fold performance, verificadas históricamente:** L2 = 0.05 retenido sobre una alternativa probada de 0.01; intermediate dropout = 0.6 retenido sobre una alternativa probada de 0.3. Solo estas dos decisiones tienen evidencia histórica verificable; el informe no extrapola a otras posibles decisiones no documentadas.
- **La misma especificación preespecificada se aplicó sin retuning por dimensionalidad ni por etapa LOSO:** ni la comparación 12-vs-116 ROI (within-site o LOSO) ni la propia campaña LOSO reoptimizaron ningún hiperparámetro.
- **No afirmar arquitectura completa idéntica entre 12 y 116 ROI:** aunque la especificación de entrenamiento/regularización (Table 3) es la misma, las dimensiones de la red difieren materialmente:
  - Parámetros de BrainNetCNN: 1,361 (12 ROI) vs. 12,177 (116 ROI).
  - Edge features / edges triangulares superiores: 66 (12 ROI) vs. 6,670 (116 ROI).
- **Tamaños FIT de LOSO por rotación:** 239–362 participantes (rango completo de las cuatro rotaciones), consistente con `results/loso/_design/loso_static_v1_design.json`.
- **Consecuencia para NYU específicamente:** el resultado LOSO de NYU (held-out) es una re-evaluación del sitio de desarrollo, no una validación independiente — declarado en §2.6 del manuscrito y en el caption de Table 6, y reiterado aquí para que ninguna fase futura lo pierda de vista al reutilizar estos resultados.

## H7 — Patrones direccionales (observación cautelar, sin interpretación causal)

Registrado únicamente como observación descriptiva, **sin** implicar causalidad ni un mecanismo subyacente:

- Los contrastes logistic-vs-BrainNetCNN LOSO (`S_LOSO_Contrasts`) cambian de signo entre sitios held-out, tanto a 12 como a 116 ROI (por ejemplo, Logistic−BrainNetCNN a 12 ROI es negativo en NYU/Peking/OHSU y positivo en NeuroIMAGE; a 116 ROI el patrón se invierte parcialmente). Los 8 intervalos de confianza correspondientes incluyen cero.
- Peking es, en el análisis within-site, el único sitio con class-weighting aplicado y también el sitio cuyo patrón de intervalos (dentro de esa misma familia within-site) difiere visiblemente de los otros tres. Esto se señala como una co-ocurrencia a vigilar, no como una relación causal establecida; separar el efecto de weighting del efecto de composición de sitio requeriría un diseño experimental adicional fuera del alcance de esta fase.
- Ninguna de estas observaciones debe citarse en Methods/Results como una conclusión sobre el comportamiento del modelo; su único uso legítimo es informar el diseño de una fase de análisis futura, si el equipo decide abrirla.

## H8 — Límites futuros (fuera del alcance de esta fase; NO insertar en Methods/Results ahora)

El siguiente párrafo se traspasa como límite interpretativo pendiente. **No fue insertado** en Methods, Results, ni en ninguna otra sección del manuscrito durante esta fase; queda documentado aquí para que una fase futura decida, con su propio análisis y revisión, si y cómo incorporarlo:

> The frozen LOSO campaign evaluates discrimination and prespecified threshold-based classification metrics across the four observed sites. It does not establish calibration, clinical net benefit, prospective utility, fairness across demographic subgroups, or performance in an independent future cohort; no such claims should be introduced without additional analyses.

## H9 — Tareas diferidas

| Tarea | Estado |
|---|---|
| PROBAST+AI | Diferido a una futura auditoría interna de riesgo de sesgo; explícitamente **sin score** en esta fase. |
| Interpretación de dispersión entre seeds / convergencia LOSO | `S_LOSO_Seeds` y `S_LOSO_Convergence` están reportadas en el Supplement, pero su interpretación (p. ej. si la mayor dispersión de seeds en NeuroIMAGE-116 refleja el tamaño muestral held-out más pequeño, n=39) queda diferida. |
| Target journal | Pendiente de decisión del equipo; no bloqueó esta fase de integración, pero sí bloqueará el submission. |
| Response letter a revisores | Diferido; fuera del alcance Methods/Results/Table 6/Supplement de esta fase. |
| Higiene de campos `INCLUDEPICTURE` | Tarea separada, explícitamente no ejecutada en esta fase. Procedimiento para una fase futura: (1) registrar el hash de cada imagen actualmente embebida vía el campo; (2) desvincular el campo externo `INCLUDEPICTURE` sin eliminar la imagen ya embebida; (3) verificar que las relaciones (`_rels`) del documento queden consistentes; (4) reabrir el documento resultante y renderizarlo para confirmar que la imagen sigue apareciendo correctamente. No ejecutar como parte de un cambio LOSO. |

## Referencias mínimas (bibliografía de contexto para el handoff; no todas están citadas en el manuscrito)

1. TRIPOD+AI 2024 — *BMJ* 385:e078378.
2. TRIPOD-Cluster 2023 — *BMJ* 380:e071058.
3. CLAIM 2024 — https://doi.org/10.1148/ryai.240300
4. PROBAST+AI 2025 — *BMJ* 388:e082505.
5. REFORMS 2024 — https://doi.org/10.1126/sciadv.adk3452
6. Zhang-James et al. 2023 — https://doi.org/10.1177/10870547221146256
7. Wang et al. 2023 — https://doi.org/10.1016/j.psychres.2023.115453
8. Marek et al. 2022 — https://doi.org/10.1038/s41586-022-04492-9
9. Spisak et al. 2023 — https://doi.org/10.1038/s41586-023-05745-x
10. Kang et al. 2024 — https://doi.org/10.1038/s41586-024-08260-9
11. Bayer et al. 2022 — https://doi.org/10.3389/fneur.2022.923988
12. Marek & Laumann 2025 — https://doi.org/10.1038/s41386-024-01960-w
13. Richter et al. 2025 — https://doi.org/10.1038/s41380-025-02950-0

Ninguna de estas referencias se añadió a la lista de References del manuscrito en esta fase; se traspasan como contexto para decisiones futuras de citación, no como una instrucción de edición inmediata.
