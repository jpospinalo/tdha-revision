# Revisión del plan de análisis estadístico (v5.6) — registro de discusión y confirmación de decisiones

**Alcance de este documento:** registro de una revisión crítica del plan y de la discusión posterior con el equipo. No modifica `analysis_plan.md` (el plan canónico sigue vigente con SHA-256 `199857a46006a082d97f6a055ffdaaa075fd25be87bbb4147e806aae28367163`) ni ningún otro archivo del repositorio. **Conclusión: el plan y la implementación actuales no requieren cambios.**

**Versión 3** — incorpora las correcciones de dos rondas de revisión del equipo. El historial de cambios se detalla al final.

## Veredicto general

El plan v5.6 es sólido en su núcleo y no se recomienda rediseñarlo. El estimando (media de los cinco AUC OOF por repetición), el bootstrap pareado y estratificado por sujeto, la decisión de no agrupar los cuatro sitios (D3) y la disciplina de comunicación de la sección 15 son respuestas correctas a las restricciones reales del proyecto (corridas ya existentes, sitios no intercambiables, exposición previa a los resultados).

La revisión evaluó cuatro posibles aperturas. **Ninguna se recomienda implementar.** Este documento las deja registradas junto con la razón técnica de por qué se descartaron, de modo que no vuelvan a plantearse sin argumentos nuevos.

## Lo que se preserva sin cambios

- Media de los cinco AUC OOF por repetición como estimando primario (D1).
- Bootstrap pareado, estratificado por clase, condicionado a las particiones existentes (D4).
- Ningún efecto combinado ni prueba de heterogeneidad entre sitios (D3): con `class_weight=true` únicamente en Peking (verificado en las 16 corridas reales) y tamaños muestrales entre 39 y 183 sujetos, agrupar escondería diferencias de protocolo detrás de un número falsamente preciso.
- La distinción entre "compatible con no inferioridad" y "superioridad", y la prohibición de "óptimo"/"ganador" sin criterio preespecificado (sección 15). Estas reglas se conservan como criterios conceptuales **condicionales**: dado que no existe un margen aprobado, ninguna afirmación de no inferioridad se aplica a los resultados actuales.
- El registro explícito de que el plan se cerró después de una revisión de factibilidad con exposición a resultados (sección 2) — poco común y valioso para la credibilidad del análisis.

## D2 (margen de no inferioridad): decisión ya cerrada, se confirma

**D2 no es una decisión pendiente.** `noninferiority_margin = null` no significa "sin resolver": significa, de forma deliberada y registrada, análisis de estimación pura, con diferencias puntuales e intervalos bilaterales, sin dictamen binario de no inferioridad y sin margen elegido retrospectivamente.

La decisión está registrada en tres registros complementarios y mutuamente consistentes:

1. `analysis/roi_comparison/README.md`, sección «Resolución D1–D5»: "La ausencia de un margen no es una omisión: es la decisión del equipo", con sus tres justificaciones.
2. `analysis_manifest.json`: `"D2_scope": "estimacion_pura_sin_dictamen_binario"`.
3. La implementación misma, que no genera límites unilaterales, márgenes ni etiquetas de no inferioridad. Una prueba de regresión (`test_no_one_sided_columns_in_precision_diagnostics`) impide que esas columnas reaparezcan.

El plan canónico conserva la discusión condicional de D2 (sección 4) porque fue congelado **antes** de que se registraran las decisiones finales; el README y el manifiesto lo complementan deliberadamente, sin modificarlo ni reinterpretarlo. Por eso **no conviene** modificar `analysis_plan.md`, retirar secciones históricas ni alterar su hash.

Tampoco conviene buscar ahora una justificación clínica para δ: el equipo ya conoce las diferencias observadas y no existe un uso clínico o de cribado predefinido que permita justificar un margen con independencia de estos resultados. Cualquier margen fijado en este punto sería retrospectivo.

### Diagnóstico de precisión

Con la corrida productiva real más reciente, los intervalos bilaterales del contraste primario tienen estas amplitudes:

| Sitio | N | Ancho completo del IC | Mitad del ancho total |
|---|---:|---:|---:|
| NYU | 177 | 0.118 | 0.059 |
| Peking | 183 | 0.129 | 0.065 |
| OHSU | 66 | 0.182 | 0.091 |
| NeuroIMAGE | 39 | 0.303 | 0.151 |

Se presenta **únicamente como diagnóstico de precisión**: describe cuánta resolución ofrecen estos datos, no un criterio de decisión. El orden coincide con el que ya anticipaba la sección 7.3 del plan. Nótese que los intervalos bootstrap percentiles no son necesariamente simétricos respecto de la estimación puntual (se verificó que ninguno de los cuatro lo es), por lo que la última columna es la mitad del ancho total, no una semiamplitud ±.

Deliberadamente **no se calcula qué margen permitiría alcanzar una conclusión de no inferioridad en cada sitio.** Aunque pueda presentarse como descriptivo, ese cálculo entrega al lector el valor exacto de δ que produciría el dictamen deseado, que es precisamente la selección retrospectiva de margen que D2 prohíbe. Adicionalmente, el procedimiento del plan (sección 7.1) requiere un límite inferior unilateral del 95 %, que la implementación no produce por diseño cuando el margen es `null`; derivar el margen a partir del límite inferior del intervalo bilateral equivaldría a un criterio unilateral distinto (más conservador, cercano al 97,5 %) y no al especificado.

## Aperturas evaluadas y descartadas

### Síntesis descriptiva entre sitios — ya cubierta

Podría pensarse que falta una forma de resumir el patrón entre los cuatro sitios sin obligar a comparar cuatro paneles a ojo. Ya existe: `generate_d3_narrative()` en `run_statistical_analysis.py` deriva de `primary_12_vs_116.csv` una narrativa que reporta signos por sitio y si los cuatro intervalos comparten una región común, dejando explícito en ambas ramas que "el análisis no estima ni contrasta heterogeneidad ni un efecto común". **Se conserva únicamente la narrativa D3; no se añade nada.**

### Metaanálisis formal de efectos aleatorios entre sitios — no se incorpora

El metaanálisis no es "imposible", pero **no produciría un estimando claramente interpretable para la pregunta actual**. Las razones, en orden de peso:

- No se ha definido una población de sitios sobre la cual interpretar un efecto promedio. Sin ese estimando, el número resultante no responde a ninguna pregunta bien planteada. Esta es la razón principal, y es anterior a cualquier consideración de precisión.
- Los tamaños muestrales son muy diferentes (39 a 183 sujetos).
- Peking usa `class_weight=true` y los otros tres sitios no.
- Existen diferencias de adquisición y protocolo entre sitios.
- Con cuatro sitios no es posible separar esas diferencias metodológicas de heterogeneidad real.

Como contexto adicional, las estimaciones de heterogeneidad son imprecisas y sesgadas con pocos estudios: von Hippel (2015) muestra que con 7 estudios —ya más que los 4 sitios de este proyecto— I² puede sobreestimar la heterogeneidad en 12 puntos porcentuales cuando la real es baja, y subestimarla en 28 puntos cuando es alta. Conviene precisar que ese artículo **recomienda cautela e intervalos de incertidumbre, no establece una prohibición de metaanalizar**; el Cochrane Handbook advierte en la misma línea que con pocos estudios se estima mal la varianza entre estudios y que la diversidad metodológica puede confundirse con heterogeneidad real. La razón para no combinar aquí es la ausencia de un estimando interpretable, no un veto estadístico.

### Tabla de diagnóstico por pliegue — no se añade en esta fase

El plan prohíbe construir `metrics_by_fold.csv` **para inferencia**, correctamente, porque reintroduciría el problema de AUC diminutas y discretas que motivó la v5.6 (sección 1). Se evaluó si una tabla de control de calidad —nunca usada para ningún Δ, IC o conclusión— aportaría algo, y se concluye que no en este momento:

- La auditoría actual ya controla 50 pares distintos de repetición–fold, diez folds por repetición, correspondencia exacta entre predicciones y sujetos de validación, duplicados, faltantes, clases y probabilidades inválidas, e identidad de particiones entre tamaños de ROI.
- Un AUC por pliegue mostraría valores extremadamente discretos, especialmente con 3–4 sujetos por pliegue en NeuroIMAGE, y podría interpretarse accidentalmente como evidencia estadística. El precedente pertinente es el diseño anterior basado en AUC por pliegue, descartado por su alta variabilidad y discretización extrema en los sitios pequeños. Friedman/Wilcoxon constituye una advertencia distinta: las repeticiones de validación cruzada tampoco pueden tratarse como observaciones independientes.

**Condición de disparo para reconsiderarlo:** solo se añadiría una tabla de conteos y composición por fold si aparece una sospecha concreta de integridad no cubierta por la auditoría actual.

## Punto de acuerdo: trabajo futuro

Tanto la revisión como la respuesta del equipo convergen en el mismo punto, que conviene destacar como conclusión operativa:

> Una evaluación confirmatoria y cualquier margen de no inferioridad deben reservarse para una cohorte externa, con protocolo y criterio práctico definidos prospectivamente, no usada en ninguna de estas decisiones.

Esto también explica por qué las solicitudes recurrentes de "agregar alguna prueba estadística formal" no se resuelven bien agregando pruebas internas sobre las mismas particiones de validación cruzada — ya se intentó con Friedman/Wilcoxon y hubo que retirarlo por dependencia entre repeticiones (además de un error en la fórmula del estadístico). El plan ya identifica la salida correcta en sus secciones 2, 7.2 y 15.

## Resolución

1. **D2:** confirmado como cerrado — estimación sin margen. No se reabre.
2. **Síntesis entre sitios:** se conserva únicamente la narrativa D3.
3. **Metaanálisis:** no se incorpora.
4. **QC por pliegue:** no se añade en esta fase, salvo que surja la condición de disparo indicada.
5. **`analysis_plan.md`:** no se modifica. Su hash canónico se mantiene.
6. **Implementación:** sin cambios requeridos.

## Historial de cambios de este documento

**De la versión 2 a la versión 3** (segunda ronda de revisión del equipo):

- **Se eliminó la tabla de "δ mínimo requerido" y su párrafo interpretativo.** Contenía tres defectos: (a) derivaba el margen del límite inferior del IC bilateral, equivalente a un criterio unilateral cercano al 97,5 %, no al límite unilateral del 95 % que especifica la sección 7.1 del plan; (b) los valores redondeados no satisfacían estrictamente la desigualdad (NYU requería δ > 0.00723, no 0.007; NeuroIMAGE δ > 0.200, no 0.200); y (c) —la razón determinante, que ningún ajuste numérico habría resuelto— calcular el margen que superaría el intervalo observado invita a escogerlo retrospectivamente, justo lo que D2 prohíbe. Se sustituyó por una nota explícita de por qué ese cálculo no se hace.
- Se retiró la afirmación de que δ = 0.20 "dejaría el desempeño en torno a 0.32": esa cifra era una tolerancia hipotética derivada del margen frente a 116, no el desempeño estimado de 12 ROIs en NeuroIMAGE, que es aproximadamente 0.474.
- La tabla de amplitudes se conserva, ahora etiquetada explícitamente como diagnóstico de precisión y no como criterio de decisión.
- "Semiamplitud bilateral" pasó a "mitad del ancho total", más exacto porque los intervalos bootstrap percentiles pueden ser asimétricos (se verificó que los cuatro lo son).
- "Tres lugares independientes" pasó a "tres registros complementarios": README, manifiesto e implementación son consistentes entre sí, pero no constituyen verificaciones estadísticamente independientes.
- Se corrigió el precedente citado para el riesgo de la tabla por pliegue: el precedente pertinente es el diseño anterior basado en AUC por pliegue, no Friedman/Wilcoxon, que fue retirado por una razón distinta (dependencia entre repeticiones).
- Se aclaró que la distinción no inferioridad/superioridad se conserva como regla conceptual condicional y no se aplica a los resultados actuales.

**De la versión 1 a la versión 2** (primera ronda de revisión del equipo):

- La sección sobre D2 pasó de proponer "resolver una decisión pendiente" a **confirmar una decisión ya cerrada**. La versión 1 partía de una lectura incorrecta: interpretaba la discusión condicional del plan congelado (sección 4) como estado vigente, sin cruzarla contra el README y el manifiesto, que ya registran la resolución.
- Se corrigió la tabla de precisión: los valores presentados como "semi-amplitudes" eran anchos completos de intervalo.
- Se retiró la afirmación "en los otros tres sitios la conclusión sería inconclusa casi con cualquier valor razonable de δ": sin definir científicamente qué es razonable, mezclaba una afirmación matemática con un juicio de aceptabilidad no fundamentado.
- Se reformuló la justificación para no metaanalizar: la razón principal es la ausencia de una población de sitios definida (y por tanto de un estimando interpretable), no la imprecisión de I²/τ² con k=4.

## Referencias

1. Von Hippel PT. The heterogeneity statistic I² can be biased in small meta-analyses. *BMC Medical Research Methodology*. 2015;15:35. [PubMed](https://pubmed.ncbi.nlm.nih.gov/25880989/)
2. Higgins JPT, Thomas J, et al. (eds). Chapter 10: Analysing data and undertaking meta-analyses. *Cochrane Handbook for Systematic Reviews of Interventions*. [Cochrane Handbook](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10)
