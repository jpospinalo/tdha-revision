# IMPLEMENTATION_REPORT.md — LOSO_METHODS_RESULTS_INTEGRATION_V3_2_1

**Fase:** Integración de la campaña congelada `loso_static_v1` en Methods, Statistical Analysis y Results del manuscrito ADHD, más nueva Table 6 y Supplement.
**Plan ejecutado:** PLAN EJECUTABLE V3.2.1 PARA IA — Integración LOSO en Methods y Results.
**Autorización de ejecución:** mensaje del usuario "empieza con la ejecución del plan." (9 de agosto de 2026), tras verificación independiente de V3.1, V3.2 y V3.2.1.
**Estado final:** **PASS FINAL** (ver §10 y criterio §18 del plan).

---

## 1. Preflight y fuentes

| Campo | Valor |
|---|---|
| Tag canónico de la campaña LOSO | `loso-static-v1-complete-v3` |
| Commit apuntado por el tag / `PRE_INTEGRATION_HEAD` | `491d33bdc93480fc3a780bdc7c9d8b4beae5f9fe` |
| Ruta del plan ejecutado | `PLAN_EJECUTABLE_FINAL_IA_LOSO_METHODS_RESULTS_DISTRIBUCION_V3_2_1.md` (guardado como insumo externo para hashing de CP1) |
| SHA-256 del plan | `6c7cbfd3821a1a2eb093175027cc983bd5a34cd20d1b3e3c0a09b021e939d71f` |
| Ruta de `v9_9` | `docs/manuscrito_revisado/Manuscript_Methods_Results_English_Working_v9_9.docx` |
| SHA-256 de `v9_9` (antes y después — sin cambios) | `5bd4a6bff80c380cbabc6a2579c39225b6516f759f911b4fdca20a605224bbe3` |
| Directorio de fase | `docs/manuscrito_revisado/loso_integration_v3_2_1/` (no existía antes de CP1; verificado) |

Inputs canónicos leídos (nunca modificados):

- `analysis/loso/outputs/loso_metrics_summary.csv` (16 filas), `loso_contrasts.csv` (12 filas), `loso_metrics_by_run.csv` (48 filas), `loso_predictions_long.csv` (5580 filas), `loso_convergence_summary.csv` (8 filas).
- `analysis/loso/config/loso_analysis_config.json` (`bootstrap_seed_scope = "reset_per_site"`).
- `results/loso/_design/loso_static_v1_design.json` (tamaños de rotación, conteos de clase por sitio, `representation="static"`, `fisher_z=false`, `harmonization="none"`, sin weighting).
- `src/run_loso.py` (comparador logístico determinista L2/C=1.0, `StandardScaler` fit-only, split único fit/inner/test por sitio, docstring que confirma reutilización del mismo split para las dos representaciones ROI y los cinco seeds de BrainNetCNN).

Jerarquía de verdad respetada: en todo desacuerdo aparente entre el plan y estas fuentes canónicas, las fuentes canónicas prevalecieron (no se detectó ningún desacuerdo real; solo imprecisiones editoriales menores, documentadas en §9).

## 2. Decision Registry D01–D43

Registro completo con `decision_id`, `status`, `evidence`, `implementation_location`, `QA_result`, `superseded_by`:

| ID | Estado | Evidencia | Ubicación de implementación | QA |
|---|---|---|---|---|
| D01 | ACTIVE_LOCKED | `FROZEN_STATE_BEFORE.sha256` = `FROZEN_STATE_AFTER.sha256` (409/409 líneas) | N/A (no-op por diseño) | PASS |
| D02 | ACTIVE_LOCKED | Cambios limitados a §2.6, §2.7, §3.5, Table 6, Supplement, y ajuste de alcance de Table 3 | `work/unpacked/word/document.xml`; scope diff §8 | PASS |
| D03 | ACTIVE_LOCKED | Abstract/Introduction/Discussion/Limitations/Conclusion/Highlights/Title no aparecen en el diff de 7 bloques | `MANUSCRIPT_DIFF.md` §1–2 | PASS |
| D04 | SUPERSEDED_BY_D43 | Registro histórico de una v3.5 planeada que nunca existió | `docs/PLAN_RESPUESTA_REVISORES.md`, enmienda v3.6 | PASS |
| D05 | SUPERSEDED_BY_D33 | Registro histórico de una formulación amplia de transportability | Enmienda v3.6, D33 | PASS |
| D06 | ACTIVE_LOCKED | §2.6/§2.7 declaran explícitamente within-site y LOSO como estimandos distintos | Párrafos LOSO Methods/Statistical | PASS |
| D07 | ACTIVE_LOCKED | Ningún cálculo de pooled/mean LOSO AUC en ningún script | `build_loso_reporting.py`, `qa_loso_reporting.py` (scan) | PASS |
| D08 | ACTIVE_LOCKED | Ningún cálculo within-site−LOSO en ningún script ni en prosa | Idem; verificado por scan de términos prohibidos | PASS |
| D09 | ACTIVE_LOCKED | Todos los scripts solo leen artefactos congelados; ningún nuevo entrenamiento/tuning/bootstrap | `build_loso_reporting.py`, `insert_results_section.py` | PASS |
| D10 | ACTIVE_LOCKED | "44.4% to 63.9%" en §3.5 y Table 6, verificado contra `auc_point.min()/max()` | §3.5 párrafo 1; `qa_loso_reporting.py` check | PASS |
| D11 | ACTIVE_LOCKED | "Fourteen of the sixteen ... included AUC = 0.50" verificado contra `loso_metrics_summary.csv` | §3.5 párrafo 1 | PASS |
| D12 | ACTIVE_LOCKED | "treated descriptively and were not interpreted as a family of 16 hypothesis tests" | §3.5 párrafo 1, última frase | PASS |
| D13 | ACTIVE_LOCKED | Los cuatro deltas BNN116−12 por sitio (NYU −3.0, Peking −4.4, NeuroIMAGE −13.3, OHSU −13.5 pp) en §3.5 párrafo 2 | §3.5 párrafo 2 | PASS |
| D14 | ACTIVE_LOCKED | 12 contrastes completos en `S_LOSO_Contrasts` | `Supplementary_LOSO_Tables.docx`, tabla 3 de 5 | PASS |
| D15 | ACTIVE_LOCKED | Interpretación mediante IC únicamente; ninguna declaración de equivalence/non-inferiority en el texto LOSO | Scan de prohibited claims | PASS |
| D16 | ACTIVE_LOCKED | Caveat NYU en §2.6 párrafo Methods y en el caption de Table 6 | §2.6; caption Table 6 | PASS |
| D17 | ACTIVE_LOCKED | Solo se declara la historia verificable (L2 y dropout intermedio retenidos por out-of-fold performance) | §2.6, frase "During NYU 12-ROI windowed-connectivity development..." | PASS |
| D18 | ACTIVE_LOCKED | "historical development = NYU / 12 ROI / windowed connectivity" declarado explícitamente | §2.6 | PASS |
| D19 | ACTIVE_LOCKED | El comparador logístico LOSO (determinista, FIT-only, sin tuning) se distingue del comparador logístico within-site (§3.3) | §2.6 vs §2.7/§3.3 (secciones separadas, terminología distinta) | PASS |
| D20 | ACTIVE_LOCKED | Table 6 es la única tabla LOSO nueva del cuerpo principal; ninguna figura LOSO en main text | `insert_results_section.py`; D21 | PASS |
| D21 | ACTIVE_LOCKED | Ningún forest plot LOSO insertado; Figuras 1–4 sin cambios (verificado por hash MD5 de las 4 imágenes embebidas) | `MANUSCRIPT_DIFF.md` §1 | PASS |
| D22 | ACTIVE_LOCKED | Figuras 1–4 y Tablas 1, 2, 4, 5 idénticas byte-a-byte en texto y celdas (verificado) | `MANUSCRIPT_DIFF.md` §1 | PASS |
| D23 | ACTIVE_LOCKED | REFORMS no aparece como criterio de aceptación de ningún resultado, solo como referencia interna de buenas prácticas de reporting (heredado del plan, no de esta fase) | N/A — no introducido en esta fase | PASS |
| D24 | ACTIVE_LOCKED | TRIPOD+AI/CLAIM 2024 no requirieron mapping exhaustivo en esta fase; solo TRIPOD-Cluster se usó como crosswalk (§5 de este informe) | §5 de este informe | PASS |
| D25 | SUPERSEDED_BY_D38 | Registro histórico de TRIPOD-Cluster como opcional | D38 | PASS |
| D26 | ACTIVE_LOCKED | PROBAST+AI no se usó; diferido explícitamente en HANDOFF §H9 | `HANDOFF_FOR_NEXT_PHASE.md` | PASS |
| D27 | ACTIVE_LOCKED | Target journal pendiente; no bloqueó esta fase | `HANDOFF_FOR_NEXT_PHASE.md` | PASS |
| D28 | ACTIVE_LOCKED | INCLUDEPICTURE no se tocó; tarea separada diferida | `HANDOFF_FOR_NEXT_PHASE.md` §H9 | PASS |
| D29 | ACTIVE_LOCKED | El heading de §3.5 usa `<w:numPr><w:ilvl w:val="1"/><w:numId w:val="7"/></w:numPr>`, sin texto literal "3.5"; render confirma numeración automática correcta tras §3.4 | `insert_results_section.py::heading_paragraph`; render p.17 | PASS |
| D30 | ACTIVE_LOCKED | Caption de Table 6 es texto literal "Table 6. ..." coherente con Tables 1–5, sin campo SEQ | `insert_results_section.py::caption_paragraph` | PASS |
| D31 | ACTIVE_LOCKED | §2.6 declara expresamente: "this does not remove the historical exposure of NYU described above" | §2.6 | PASS |
| D32 | ACTIVE_LOCKED | §2.7 declara la frase obligatoria de emparejamiento: PCG64 reset per held-out site, mismos remuestreos reutilizados entre condiciones/contrastes | §2.7 párrafo LOSO, penúltima frase | PASS |
| D33 | ACTIVE_LOCKED | "does not equate to testing in a completely independent future cohort" (§2.6, última frase) | §2.6 | PASS |
| D34 | ACTIVE_LOCKED | Título de §3.5 = "Site-Held-Out Performance under Leave-One-Site-Out Evaluation" (verificado carácter por carácter) | `insert_results_section.py` | PASS |
| D35 | ACTIVE_LOCKED | Table 6 muestra `(held-out n=...)` en cada fila y el caption remite a `Supplementary Table S_LOSO_Design`; no se creó tabla de características nueva (`s_loso_design.csv` marca "NOT AVAILABLE IN FROZEN SCOPE") | Table 6; `S_LOSO_Design` | PASS |
| D36 | ACTIVE_LOCKED | Intro y caption de Table 3 reformuladas a "prespecified ... without dimensionality-specific retuning ... applied without LOSO-stage retuning"; ningún texto afirma arquitectura completa idéntica | Table 3 intro + caption (2 reemplazos verificados byte-exactos antes de aplicar) | PASS |
| D37 | ACTIVE_LOCKED | "These interval crossings were treated descriptively and were not interpreted as a family of 16 hypothesis tests" | §3.5 párrafo 1 | PASS |
| D38 | ACTIVE_LOCKED | Ver §5 de este informe (crosswalk dirigido); sin pooling, heterogeneity tests, ni análisis nuevo | §5 | PASS |
| D39 | ACTIVE_LOCKED | Caption de Table 6: "pointwise, unadjusted, and conditional on the fixed source-site composition and frozen LOSO predictions for that rotation" | Caption Table 6 | PASS |
| D40 | ACTIVE_LOCKED | §2.6: "this describes a property of the evaluated procedure and does not by itself demonstrate the absence of site effects" | §2.6 | PASS |
| D41 | ACTIVE_LOCKED | `HANDOFF_FOR_NEXT_PHASE.md` contiene §H8 (límites futuros) y §H9 (tareas diferidas); ese párrafo NO se insertó en Methods/Results | `HANDOFF_FOR_NEXT_PHASE.md`; confirmado ausente en `clean.docx` | PASS |
| D42 | ACTIVE_LOCKED | `loso_integration_v3_2_1/` y `v9_10_LOSO_V3_2_1_*` son artefactos nuevos; `v9_9` intacto (hash idéntico) | §1 de este informe | PASS |
| D43 | ACTIVE_LOCKED | Enmienda v3.6 consolidada añadida tras detectar ausencia de v3.5 | `docs/PLAN_RESPUESTA_REVISORES.md` | PASS |

**Conteo verificado:** 43 decisiones registradas; 40 `ACTIVE_LOCKED`; 3 `SUPERSEDED` (D04→D43, D05→D33, D25→D38). Coincide exactamente con el conteo exigido por el plan (§4). Ninguna contradicción no resuelta, ninguna decisión activa sin evidencia, ninguna supersesión rota.

## 3. Enmienda de gobernanza

- **Documento:** `docs/PLAN_RESPUESTA_REVISORES.md` (el único archivo de gobernanza cuya edición directa permite el plan).
- **Detección previa:** v3.5 confirmada ausente (no hay ninguna referencia a "v3.5" ni "3.5" como versión en el documento antes de la edición) → se aplicó una única enmienda v3.6 consolidada, documentando explícitamente el salto v3.4→v3.6 para que no se asuma una versión intermedia perdida.
- **Cambios aplicados (verificados con `git diff`, 18 inserciones / 6 eliminaciones, un solo archivo):**
  1. Encabezado de versión 3.4 → 3.6 y fecha de enmienda añadida.
  2. Nuevo párrafo de enmienda v3.6 completo (distingue estimandos, prohíbe pooling/within-site−LOSO, limita transportability a los cuatro sitios, preserva caveat NYU, fija TRIPOD-Cluster como crosswalk no analítico, limita alcance a Methods/Results/Table 6/Supplement).
  3. Dos correcciones de colisión de nombre: las referencias previas a "la Tabla 6" (que designaban una fila de `run_manifest.csv` de la campaña de diez corridas, no la nueva Table 6 del manuscrito) se reformularon como "la referencia primaria del comparador de paneles", con nota explícita a la enmienda v3.6.
  4. Nota aclaratoria en §9.2 (la lista de exclusiones "LOSO o transporte entre sitios" y "static con 116 ROIs" describía correctamente el alcance de la campaña de diez corridas cerrada en v3.4, no un alcance global; no se reescribe, se anota).
  5. Nota parentética en D10 (§12), documentando sin corregir una discrepancia editorial preexistente y no relacionada con LOSO (activación de la contingencia de baseline lineal), y aclarando que el comparador logístico LOSO es una configuración distinta ya implementada.
  6. Nueva fila D14 en la tabla de decisiones registrando la reconciliación de gobernanza.
- **Barrido obligatorio de términos** (LOSO, transport/cross-site, Table 6/Tabla 6, static+116, external/independent/generaliz*): ejecutado; las únicas coincidencias nuevas son las de la propia enmienda v3.6 (uso permitido y con el alcance correcto); no se detectaron usos residuales problemáticos en el resto del documento.
- **Historia científica previa:** preservada sin reescritura; los cambios se documentan como anotaciones y adiciones, nunca como sustitución silenciosa de una decisión histórica.

## 4. Cambios del manuscrito

Manuscrito autoritativo nuevo: `docs/manuscrito_revisado/Manuscript_Methods_Results_English_Working_v9_10_LOSO_V3_2_1_clean.docx` (SHA-256 `eafa9e3fd49a42f70f10483fc382c58a2c01f34d98ff8adfaf9e9e9acd6727e5`).

Los 7 bloques de cambio (detalle completo, con texto antes/después, en `MANUSCRIPT_DIFF.md`):

1. Table 3 — frase introductoria y caption ampliadas para cubrir el uso LOSO de la especificación congelada, sin tocar valores.
2. "The analysis comprised three components" → "The within-site analysis comprised three components".
3. "Evaluation Setup" → "Within-Site Evaluation" (subtítulo no numerado).
4. Nuevo subtítulo "Cross-Site Transportability Evaluation across the Observed Sites (Leave-One-Site-Out)" + párrafo Methods LOSO (308 palabras).
5. "The primary metric is the AUC..." → "For the within-site analyses, the primary metric is the AUC...".
6. Reformulación del estimando within-site + nuevo párrafo Statistical Analysis LOSO (158 palabras), con la frase obligatoria de emparejamiento D32.
7. Nueva subsección §3.5 completa (heading numerado automáticamente, 4 párrafos de prosa — 247 palabras —, Table 6 de 4×5, y su caption), insertada antes de References.

Resultado estructural: 85 → 94 párrafos, 5 → 6 tablas, 4/4 imágenes embebidas MD5-idénticas. 78/85 párrafos de v9_9 se preservan sin cambios y en el mismo orden.

## 5. TRIPOD-Cluster targeted crosswalk

Ejecutado como control interno dirigido, sin autorizar pooling, heterogeneity tests, calibration, harmonization, subgroup analysis ni retraining (D38):

| Dominio | Evidencia | Resultado |
|---|---|---|
| Definición de site | §2.6 declara los cuatro sitios y la lógica source/held-out por rotación | PASS |
| Características | Table 6 muestra held-out N por sitio; `S_LOSO_Design.participant_characteristics_cross_reference = "NOT AVAILABLE IN FROZEN SCOPE"` (ninguna tabla de características verificada existía para referenciar; no se creó una nueva) | PASS |
| Diseño | §2.6 y `S_LOSO_Design` documentan FIT/inner-validation/test y las cuatro rotaciones | PASS |
| Performance | Table 6 y `S_LOSO_FullMetrics` reportan AUC e IC por sitio/ROI/modelo | PASS |
| Incertidumbre | Caption de Table 6 y §2.7 declaran explícitamente pointwise, unadjusted, y condicional a la composición fija de sitios fuente | PASS |
| Heterogeneidad | Tratada solo descriptivamente (§3.5 párrafo 1, "treated descriptively"); ningún test de heterogeneidad nuevo | PASS |

## 6. QA científico y numérico (CP7)

- **Regresiones numéricas:** `scripts/qa_loso_reporting.py` re-deriva cada cifra citada directamente desde `analysis/loso/outputs/*.csv` y `results/loso/_design/loso_static_v1_design.json` (no desde memoria ni desde el plan) y las contrasta contra `loso_table6_source.csv`, la Table 6 embebida en el `clean.docx`, y la prosa de §3.5. Resultado: **ALL CHECKS PASSED** (ver salida completa de la ejecución en el historial de la fase).
  - Conteos de filas: 16/12/48/5580/8 — todos verificados.
  - AUC min/max: 44.4% / 63.9% — verificado con tolerancia `1e-9` contra el valor crudo `0.443609022556391` / `0.6390374331550802`.
  - 14/16 IC incluyen 0.50; excepciones exactamente {NYU/12/brainnetcnn, OHSU/12/brainnetcnn} — verificado.
  - 4 contrastes de dimensionalidad, los 4 con IC que incluyen cero, deltas redondeados a −3.0/−4.4/−13.3/−13.5 pp — verificado.
  - 8 contrastes de model family, los 8 con IC que incluyen cero — verificado.
  - Held-out N por sitio (177/183/39/66) verificado contra `design.json` y contra la etiqueta de fila embebida en Table 6.
- **Claim map:** `loso_methods_results_claim_map.csv`, 34 filas (21 Methods M01–M21 + 13 Results R01–R13), `QA_status=PASS` en el 100% de las filas.
- **Scan de overclaims prohibidos:** términos escaneados (`external validation`, `independent validation`, `generalizes`, `future cohort`, `pooled`, `non-inferior`, `identical architecture`, etc.) sobre la prosa nueva/editada de §3.5. Se hallaron 3 coincidencias totales; las 3 se clasificaron manualmente:
  - "pooled" → coincide con "No pooled estimate across sites was calculated" (negación requerida por D7/D11, no un overclaim).
  - "future cohort" → coincide con "does not equate to testing in a completely independent future cohort" (negación requerida por D33).
  - "non-inferior" → texto within-site preexistente y no relacionado con LOSO ("no non-inferiority margin was specified").
  - **0 hits sin clasificar.**
- **Table 3 scope wording:** verificado byte-exacto antes de cada uno de los 2 reemplazos (intro sentence y caption); confirmado tras el reemplazo que ningún valor de celda de Table 3 cambió.

## 7. QA documental y visual (CP8)

- **QA estructural del `clean.docx`:** `word/_rels/document.xml.rels`, `styles.xml`, `numbering.xml`, `settings.xml`, `webSettings.xml`, `fontTable.xml`, `theme/theme1.xml` — byte-identical a `v9_9`. Solo `word/document.xml` cambió.
- **Render:** `Manuscript_v9_10_LOSO_V3_2_1_clean_QA.pdf`, 20 páginas (vs. 17 en `v9_9`), generado con `soffice --headless --convert-to pdf`.
- **Inspección página por página:** las 20 páginas fueron inspeccionadas visualmente (no solo por texto extraído). Resultado: **0 defectos visuales.**
  - Sin números de sección duplicados o ausentes: "2.6"→"2.8", "3.1"→"3.5" corren correctamente; "3.5 Site-Held-Out Performance under Leave-One-Site-Out Evaluation" se numera automáticamente justo después de "3.4 Convergence and Model Behavior", sin texto "3.5" hard-codeado (confirmado en el XML fuente y en el render).
  - Sin table clipping ni overflow: Table 3 (p.4), Table 4 (p.10), Table 5 (p.15–16), Table 6 (p.18) renderizan completas, con columnas y celdas legibles.
  - Sin headings huérfanos introducidos por esta fase: se verificó que "2.5 BrainNetCNN" sin cuerpo de texto antes de "2.6" (p.4) es una condición **preexistente e idéntica en `v9_9`** (confirmado por `pdftotext` sobre el PDF baseline de `v9_9`), no una regresión de esta fase.
  - Sin captions separadas de su tabla/figura: cada tabla y figura (1–4, y la nueva Table 6) mantiene su caption inmediatamente debajo, en la misma página o en continuación natural.
  - Sin figuras desplazadas: Figuras 1–4 en las mismas posiciones relativas que en `v9_9`.
  - Sin páginas en blanco inesperadas.
  - Sin font/style drift: "Within-Site Evaluation" y "Cross-Site Transportability Evaluation across the Observed Sites (Leave-One-Site-Out)" renderizan como subtítulos en negrita no numerados, con el mismo estilo que "Experimental Design" preexistente.
  - Sin cross-references rotas: References comienza inmediatamente después del último párrafo de §3.5, con la bibliografía completa e intacta.
- **Supplement:** `Supplementary_LOSO_Tables.docx` renderizado a PDF (3 páginas) e inspeccionado visualmente. Las 5 tablas (`S_LOSO_Design` 4 filas, `S_LOSO_FullMetrics` 16 filas, `S_LOSO_Contrasts` 12 filas, `S_LOSO_Seeds` 8 filas, `S_LOSO_Convergence` 8 filas) están completas y legibles, sin clipping.
- **Corrección aplicada durante esta inspección:** ver §9 (Checkpoints y correcciones).
- **Mecanismo de revisión:** ver §8 y `MANUSCRIPT_DIFF.md`.

## 8. Integridad y scope diff (CP9)

- **`FROZEN_STATE_BEFORE.sha256` vs `FROZEN_STATE_AFTER.sha256`:** 409/409 líneas idénticas, mismo conjunto de rutas y mismos hashes. Cobertura: `analysis/loso/**` (incluyendo outputs, config, scripts, tests y `__pycache__`), `results/loso/**`, `src/run_loso.py`, `results/README.md`, `v9_9.docx`, y el propio archivo del plan. **Resultado: IDÉNTICO.**
- **Scope diff (`git status`/`git diff` sobre el repositorio completo):**
  - Único archivo modificado (`M`) en todo el repositorio: `docs/PLAN_RESPUESTA_REVISORES.md` — exactamente el permitido, y el diff completo (18 inserciones/6 eliminaciones) coincide exactamente con la enmienda v3.6 descrita en §3.
  - Archivos nuevos (`??`): `docs/manuscrito_revisado/Manuscript_Methods_Results_English_Working_v9_10_LOSO_V3_2_1_clean.docx`, todo `docs/manuscrito_revisado/loso_integration_v3_2_1/**`, y una serie de documentos `docs/REVISION_*`/`docs/REVISION_CRITICA_*` que son los artefactos de las rondas de revisión previas a la autorización de ejecución (no forman parte de esta fase de implementación, no tocan ningún archivo protegido, y no requieren acción).
  - **Ningún archivo bajo `analysis/loso/**`, `results/loso/**`, `src/run_loso.py` ni `results/README.md` aparece como modificado o nuevo.**
  - `v9_9.docx` no aparece en absoluto en el diff (ni modificado ni eliminado) — confirmado también por hash idéntico en §1.
- **Conclusión CP9:** protected BEFORE = AFTER; `v9_9` intacto; plan V3.2.1 intacto; scope diff sin cambios no autorizados; todos los cambios intencionales están trazados en este informe y en `MANUSCRIPT_DIFF.md`. **Gate CP9: PASS.**

## 9. Checkpoints y correcciones

| Checkpoint | Resultado |
|---|---|
| CP1 — Preflight | PASS |
| CP2 — Gobernanza, copia de trabajo, estructura | PASS |
| CP3 — Methods §2.6/§2.7 | PASS |
| CP4 — Table 6 source + Supplement | PASS (1 CORRECT, ver abajo) |
| CP5 — Results §3.5 + Table 6 | PASS |
| CP6 — Clean + mecanismo de revisión | PASS (ver decisión de gobernanza abajo) |
| CP7 — QA numérico, claim map, overclaim scan | PASS (1 CORRECT, ver abajo) |
| CP8 — QA OOXML/render/visual | PASS (1 CORRECT, ver abajo) |
| CP9 — Integridad final y scope diff | PASS |
| CP10 — Informe final y entrega | PASS (este documento) |

**Correcciones (CORRECT, no STOP) realizadas:**

1. **CP7 — falso positivo en el scan de overclaims.** La primera versión de `qa_loso_reporting.py` marcaba "pooled" como overclaim prohibido sin distinguir que la ocurrencia real era "No pooled estimate across sites was calculated" — una negación exigida por D7/D11, no una afirmación. Se corrigió el script para clasificar cada coincidencia (`allowed_and_scoped` / overclaim sin clasificar) en vez de un booleano ciego, siguiendo la misma metodología que exige el plan en §12.3. Re-ejecutado: `ALL CHECKS PASSED`, 0 overclaims sin clasificar.
2. **CP8 — typo bilingüe en el Supplement.** La inspección visual página por página del `Supplementary_LOSO_Tables.docx` encontró que el caption de `Table S_LOSO_Convergence` decía "already-congelado convergence records" (palabra en español dentro de una leyenda en inglés). Corregido a "already-frozen" en `scripts/build_supplement_docx.js`; el documento se regeneró y se re-renderizó para confirmar la corrección (hash del `.docx` recalculado: `ac1484891963c37b1e5e42e99a77a67c`). No afecta ninguna cifra científica ni ningún dato de la tabla, solo texto de leyenda.
3. **Autocorrección de integridad de artefactos, previa a CP1.** Al guardar el texto del plan V3.2.1 como insumo externo para hashing, la primera escritura omitió por error las secciones 4–19. Se detectó antes de proceder y se reescribió el archivo completo y verbatim antes de calcular ningún hash sobre él.

**Ningún STOP fue necesario durante la implementación.** Un punto de gobernanza (no un STOP científico) se elevó al usuario durante CP6: la ausencia en este entorno de una herramienta que genere y verifique de forma independiente revisiones OOXML reales (`w:ins`/`w:del`) activó el fallback previsto por el plan (`STRUCTURED_DIFF_FALLBACK` → `MANUSCRIPT_DIFF.md`), que requiere aprobación de un PLAN_OWNER identificable antes de cerrar CP6. Se presentó la pregunta al usuario mediante `AskUserQuestion`; el usuario seleccionó la opción recomendada, "Apruebo yo el fallback". Registro formal:

> **Estado del riesgo residual:** `ACCEPTED_BY_PLAN_OWNER`.
> **Aprobado por:** Juan Pablo Ospina (jpospinalo@gmail.com).
> **Rol:** plan owner / usuario que autorizó la ejecución de esta fase ("empieza con la ejecución del plan.").
> **Fecha:** 9 de agosto de 2026.
> **Evidencia:** selección explícita de la opción "Apruebo yo el fallback (Recomendado)" en respuesta a la pregunta "El fallback STRUCTURED_DIFF_FALLBACK ... ¿Cómo quieres resolverlo?", presentada durante la ejecución de CP6.
> **Justificación del fallback (no simulación):** este entorno cuenta con LibreOffice headless y utilidades OOXML (`python-docx`, `lxml`, los scripts de la skill `docx`), pero ninguna herramienta produce y verifica de forma independiente revisiones `w:ins`/`w:del` reales entre dos `.docx` arbitrarios. Construir ese redline manualmente habría sido posible en principio, pero lo habría hecho la misma IA que introdujo los cambios, sin la garantía independiente que un `review.docx` real está pensado para dar — por eso se usó el fallback estructurado en vez de simular tracked changes con colores o resaltado (expresamente prohibido por el plan).
> **Respaldo real del mecanismo:** el scope diff independiente de CP9 (§8 de este informe), que compara `v9_9` contra `clean` con el mismo método reproducible (`python-docx` + `difflib.SequenceMatcher` + comparación celda-por-celda + hashes de medios) documentado en `MANUSCRIPT_DIFF.md` §4, y que se ejecutó después de que todos los entregables científicos estuvieran finales, sin encontrar ningún cambio fuera del alcance permitido.

## 10. Entregables y hashes

Ver `DELIVERABLES.sha256` (calculado como último paso de CP10, excluyéndose a sí mismo) para la lista completa con ruta, tamaño y SHA-256 de cada entregable final. Lista de los 10 grupos del conjunto mínimo de entrega (§5.3 del plan), todos presentes:

1. `docs/manuscrito_revisado/Manuscript_Methods_Results_English_Working_v9_10_LOSO_V3_2_1_clean.docx`
2. `MANUSCRIPT_DIFF.md` (mecanismo de revisión, `STRUCTURED_DIFF_FALLBACK`, aprobado por PLAN_OWNER — ver §9)
3. `Manuscript_v9_10_LOSO_V3_2_1_clean_QA.pdf`
4. `Supplementary_LOSO_Tables.docx`
5. `loso_table6_source.csv`
6. `loso_methods_results_claim_map.csv`
7. `IMPLEMENTATION_REPORT.md` (este documento)
8. `HANDOFF_FOR_NEXT_PHASE.md`
9. `scripts/build_loso_reporting.py`, `scripts/build_supplement_docx.js`, `scripts/insert_results_section.py`, `scripts/qa_loso_reporting.py`
10. `FROZEN_STATE_BEFORE.sha256`, `FROZEN_STATE_AFTER.sha256`, `DELIVERABLES.sha256`

No review.docx real fue producido (limitación de entorno documentada en §9; no bloqueante por diseño del plan). No placeholders pendientes. Ningún STOP abierto. Los 3 CORRECT están cerrados y re-verificados.
