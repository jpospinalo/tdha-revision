# Informe de entrega — Manuscript_Methods_Results_English_Working_v9_9.docx
## Revisión editorial de Methods y Results (eliminación de sección duplicada, renumeración, referencias)

## 1. Contexto y verificación de entrada

Ronda de edición editorial controlada sobre el mismo archivo oficial, sin crear versión numerada nueva, conforme a la instrucción explícita: *"editar ese mismo archivo; no crear v9_10 ni otra versión numerada por esta intervención."*

- Archivo base: `Manuscript_Methods_Results_English_Working_v9_9.docx` (candidata previa, ronda puramente estética de Figuras 3/4).
- SHA-256 verificado al iniciar esta ronda: `e7543e9195f7edb35628842dc96b484ad95ca5d3834116ea1f6e9c29a2003919` — coincide con el valor registrado al cierre de la ronda anterior.
- Copia de respaldo sin numerar creada fuera de los entregables oficiales: `recovery_copy_UNNUMBERED.docx` (idéntica, mismo SHA-256). Se mantiene fuera de `docs/manuscrito_revisado/` y solo debe eliminarse tras confirmación del equipo.
- Copia de trabajo sin numerar (también fuera de entregables oficiales): `working_copy_UNNUMBERED.docx` → editada iterativamente hasta `working_copy_EDITED.docx`.
- Todas las ediciones se aplicaron y validaron primero en la copia de trabajo. El archivo oficial no se tocó hasta que **todos** los controles obligatorios (redacción, científicos, visuales, bibliográficos) pasaron.

## 2. Cambios realizados

### 2.1 Eliminación de la sección duplicada

Se eliminó por completo la sección "3.1 Functional-Network Composition of the Evaluated ROI Panels": encabezado, párrafo, la Figura 2 duplicada (que usaba la misma imagen anatómica que la Figura 1) y su leyenda. El contenido de este párrafo repetía, sin aportar información metodológica adicional, lo ya expuesto en Métodos 2.4 ("Functional Networks Involved in ADHD") y en la Figura 1; no se identificó ninguna información metodológica única en el texto eliminado, por lo que su eliminación no requirió preservar ni trasladar contenido a otra sección.

No se usó contenido de versiones anteriores del documento para reconstruir párrafos ni resultados; no se recalculó ninguna métrica, intervalo o prueba estadística; no se modificó ninguna cifra de las Tablas 1–5, ningún texto de cuerpo distinto de los cambios explícitamente instruidos, ni ninguna imagen de figura.

### 2.2 Renumeración (tabla de correspondencia exacta)

| Elemento | Numeración anterior | Numeración nueva |
|---|---|---|
| Figura 1 (redes funcionales, panel anatómico) | Figura 1 | Figura 1 (sin cambio) |
| Figura 2 duplicada (mismo contenido anatómico que Fig. 1) | Figura 2 | **Eliminada** |
| Figura de AUC por panel de ROI | Figura 3 | Figura 2 |
| Figura de análisis de sensibilidad | Figura 4 | Figura 3 |
| "Functional-Network Composition of the Evaluated ROI Panels" | 3.1 | **Eliminada** |
| "Performance of the Ablation-Derived ROI Panels" | 3.2 | 3.1 (mismo título) |
| "Primary 12-versus-116 ROI Comparison" | 3.3 | 3.2 — **renombrada** a "Paired ROI-Panel Comparisons" |
| "Sensitivity Analyses" | 3.4 | 3.3 (mismo título) |
| "Convergence and Model Behavior" | 3.5 | 3.4 (mismo título) |

La numeración de los encabezados de sección (Methods 2.x, Results 3.x) usa numeración automática multinivel real de Word (`numId=7`), no números tecleados: al eliminar los 5 elementos de la sección duplicada, Word renumeró automáticamente el resto de subsecciones de Resultados sin intervención manual, confirmado en el render (sin saltos, sin numeración duplicada). Las leyendas de Figuras/Tablas son texto literal y se actualizaron mediante barrido de búsqueda-y-reemplazo global, verificado para evitar colisiones encadenadas (uso de token temporal en el barrido "Figura 4 → 3" y "Figura 3 → 2" simultáneo) y excluyendo explícitamente el único párrafo que ya contenía intencionalmente el número final correcto.

Todas las menciones en cuerpo de texto, leyendas y referencias cruzadas fueron actualizadas; no quedan huecos de numeración.

### 2.3 Sección 2.3 (BOLD Signal Extraction and Functional Connectivity Construction)

Se reemplazó el párrafo de justificación de la ventana temporal por el texto exacto indicado, incorporando la cita "(Leonardi and Van De Ville, 2015)" dentro del alcance estrictamente delimitado por el plan. Se agregó esta obra como 14.ª entrada bibliográfica.

### 2.4 Sección "Functional Networks Involved in ADHD" (antes 2.4, sin cambio de posición numérica)

- Título renombrado exactamente a "Functional Networks Involved in ADHD".
- Se preservó el orden de contenido existente: cinco redes → relación con TDAH → Figura 1 → procedencia AAL116 → relaciones de inclusión entre paneles → Tabla 2.
- Se agregó una única oración de salvaguarda: las asignaciones de red no se interpretan como biomarcadores diagnósticos a nivel de región.
- No se repitió el listado de redes después de la Figura 1.
- La Figura 1 (imagen, número, leyenda, contenido anatómico) no se tocó en esta ronda.

### 2.5 Sección 2.6 (Experimental and Evaluation Setup)

Cinco sustituciones aplicadas con el texto exacto proporcionado:
1. Simplificación de la oración sobre "complete pipelines".
2. Reemplazo de la descripción de la condición LSTM.
3. Reemplazo de la descripción de la línea base de regresión logística.
4. Reemplazo del párrafo "Five dimensions…" por un resumen de tres componentes (diseño experimental / articulación de resultados).
5. Inserción del párrafo nuevo inmediatamente posterior, listando los niveles de sensibilidad evaluados.

Se conservó el contenido de "Evaluation Setup" (dividido en un párrafo adicional, sin subtítulos ni listas nuevas) y todo el contenido correspondiente a las antiguas §6.1/§6.6, condensado pero no eliminado.

### 2.6 Sección 2.7 (Statistical Analysis)

Tres sustituciones aplicadas con el texto exacto proporcionado:
1. Simplificación de la oración sobre métricas dependientes de umbral.
2. Eliminación de la oración sobre proporción de clases dentro de la justificación de AUC.
3. Reemplazo del contenido final del párrafo de bootstrap y reemplazo completo del párrafo siguiente por un nuevo párrafo compacto de estimando/alcance.

Se preservó explícitamente: ausencia de reentrenamiento o nuevas particiones; el estimando como el procedimiento de validación cruzada repetida; intervalos de sensibilidad puntuales y sin ajuste por multiplicidad; ausencia de margen de no-inferioridad; folds/repeticiones no tratados como réplicas independientes. **No se restauró** la enumeración "superiority, equivalence, or non-inferiority" (verificado en la auditoría de escritura, §4 de este informe: 0 coincidencias).

### 2.7 Resultados (secciones renumeradas 3.1–3.4)

Ediciones por párrafo, todas con el texto exacto proporcionado: eliminación de oraciones anticipatorias o de referencia hacia adelante (incluida "ROI-panel comparisons are reported in Section 3.3" en la sección de Sensitivity Analyses), condensación de cierres interpretativos, reemplazo de una oración vaga por una oración cuantitativa sobre los contrastes de regresión logística, actualización de números de figura, y renombrado de un encabezado ("Primary 12-versus-116 ROI Comparison" → "Paired ROI-Panel Comparisons").

### 2.8 Tabla 5

- Título y nota separados en dos elementos claramente diferenciados (título en texto normal, nota en cursiva).
- Reetiquetado compacto de filas (p. ej. "ROI panel size" → "ROI count"; "18-ROI panel" → "18 ROIs"; "Model architecture" → "Architecture"; "LSTM (128 hidden units)" → "LSTM (128 units)"; "Window configuration" → "Window / step (s)"; "140-s window, 12-s step" → "140 / 12"; "Linear baseline (logistic regression)" → "Logistic regression"; "12-ROI panel (vs. static)" → "12 ROIs (static comparator)", etc.).
- Eliminados todos los símbolos "†" (0 ocurrencias verificadas).
- Formato ya conforme preservado: celdas combinadas "estimate [CI]", Times New Roman ≥9 pt, encabezados en negrita, columnas numéricas centradas, ancho de página completo.
- Se agregaron únicamente separadores horizontales discretos; no se introdujeron rellenos, colores, bordes pesados, negrita en celdas de datos, símbolos de significancia ni fuentes menores a 9 pt.
- Tabla + nota permanecen juntas en una sola página (página 11 del render final).
- **Verificación numérica:** las 20 celdas de datos (estimate [CI]) de la Tabla 5 son idénticas byte a byte antes/después del reetiquetado; el único cambio detectado por comparación automática de la grilla completa fue en las columnas de etiqueta ("Dimension"/"Condition"), exactamente como se esperaba.

### 2.9 Figura 3 (antes Figura 4, análisis de sensibilidad)

No se regeneró ni alteró: datos, diseño, dimensiones, resolución ni paleta. Únicamente cambiaron el número de la leyenda, las referencias textuales y el texto alternativo (donde contenía "Figure 4"). Figura y leyenda completa permanecen en la misma página (página 12). **Verificación:** SHA-256 del binario de imagen idéntico antes/después (`d5415de46d6c9f1727d73d493ee003bffe48925e99bb02e527feaeb9cda9b046`).

### 2.10 Sección de referencias

Se agregó el encabezado "References" (sin numeración automática, clonado del estilo de encabezado de "Results" y despojado de su `numPr`) inmediatamente después de "3.4 Convergence and Model Behavior", con una lista de 14 entradas — únicamente obras citadas en Methods/Results — en orden alfabético por apellido del primer autor, con sangría francesa y DOI en formato `https://doi.org/...`.

## 3. Inventario y verificación bibliográfica (§11)

| Referencia | Estado | Fuente de verificación |
|---|---|---|
| ADHD-200 Consortium (2011) | Verificada (con nota) | Frontiers/PMC — DOI 10.3389/fnsys.2012.00062. **Nota:** el artículo aparece indexado con fecha 2012; el manuscrito lo cita como 2011 (posible fecha de aceptación/online-first en Frontiers). No se alteró el año de la cita ya establecida en rondas previas sin evidencia concluyente de error. |
| Bellec et al. (2017) | Verificada | NeuroImage 144(B), 275–286. DOI 10.1016/j.neuroimage.2016.06.034 |
| Blomberg et al. (2022) | Verificada | Frontiers in Neuroscience 16:972730. DOI 10.3389/fnins.2022.972730 (DOI verificado según advertencia explícita del plan) |
| Damiani et al. (2021) | Verificada | Eur Child Adolesc Psychiatry 30, 619–631. DOI 10.1007/s00787-020-01545-0 |
| Francx et al. (2015) | Verificada | Cortex 73, 62–72. DOI 10.1016/j.cortex.2015.08.012 |
| **Hale et al. (2014)** | **No resuelta** | Sin metadatos completos verificables contra fuente autoritativa; no se completó por inferencia. Placeholder entregado con nota explícita, por decisión del equipo (ver §5). |
| Koirala et al. (2024) | Confianza moderada-alta | Nat Rev Neurosci 25, 759–775. DOI 10.1038/s41583-024-00869-z — temáticamente compatible; no verificado contra el manuscrito original completo. |
| Leonardi & Van De Ville (2015) | Verificada | NeuroImage 104, 430–436. DOI 10.1016/j.neuroimage.2014.09.007 — nueva entrada, exactamente como especifica el plan. |
| Parlatini et al. (2023) | Verificada | Mol Psychiatry 28, 4098–4123. DOI 10.1038/s41380-023-02173-1 |
| **Reimann et al. (2024)** | **No resuelta** | No identificada en ninguna búsqueda. Placeholder entregado con nota explícita, por decisión del equipo (ver §5). |
| **Singh et al. (2024)** | **No resuelta** | Ortografía de autor(es) no confirmable; nota interna del proyecto advierte posibles errores de metadatos históricos. Placeholder entregado con nota explícita, por decisión del equipo (ver §5). |
| Sutcubasi et al. (2020) | Verificada | World J Biol Psychiatry 21(9), 662–672. DOI 10.1080/15622975.2020.1775889 |
| Tzourio-Mazoyer et al. (2002) | Verificada | NeuroImage 15(1), 273–289. DOI 10.1006/nimg.2001.0978 |
| **Yu et al. (2018)** — control bloqueante §11.4 | **Resuelto** | Human Brain Mapping 39(11), 4213–4227. DOI 10.1002/hbm.24241. Confirmado contra nota histórica del propio proyecto (`Comentarios_secciones_2.1_a_2.4_y_recorte.md`, líneas 82-84); verificado como obra real y distinta de cualquier "Yu et al., 2023". No se sustituyó el año ni se inventó el dato. |

**Reconciliación cita↔referencia:** 14 grupos de cita distintos detectados en el texto final (búsqueda automatizada) = 14 entradas en la lista de referencias. Cero citas huérfanas, cero referencias no citadas, cero duplicados.

**No se inventaron datos ni se completaron metadatos por inferencia** para Hale (2014), Reimann (2024) ni Singh (2024).

## 4. Auditoría de escritura (§12)

Búsqueda automatizada de las 13 frases/símbolos prohibidos o sospechosos sobre el texto completo del documento final (párrafos + celdas de tabla):

| Frase | Ocurrencias |
|---|---|
| "reported in Section" | 0 |
| "described in Section" | 0 |
| "as shown later" | 0 |
| "will be presented" | 0 |
| "does not define" | 0 |
| "should not be interpreted" | 0 |
| "without threshold optimization" | 0 |
| "with no hyperparameter search" | 0 |
| "No clinically justified" | 0 |
| "superiority, equivalence, or non-inferiority" | 0 |
| "Figure 4" | 0 |
| "Section 3.5" | 0 |
| "†" | 0 |

**Resultado: PASSED — 0 coincidencias en todos los ítems.**

## 5. Resolución de las 3 referencias no verificables

Se consultó al usuario sobre cómo proceder con Hale (2014), Reimann (2024) y Singh (2024), dado que el plan solo exige bloqueo explícito de entrega para Yu et al. (2018) (ya resuelto) y únicamente prohíbe inventar datos para el resto. **Decisión del usuario: entregar con placeholders**, dejando el texto entre corchetes que indica verificación pendiente, documentado en este informe. El equipo deberá completar los metadatos de estas 3 entradas en una ronda posterior.

## 6. Controles científicos obligatorios (§13) — comparación automatizada antes/después

| Control | Resultado |
|---|---|
| Tablas 1–3 | Idénticas byte a byte |
| Tabla 4 (todas las celdas) | Idéntica byte a byte |
| Tabla 5 (columnas numéricas 2–5) | 0 discrepancias; solo cambiaron etiquetas de columnas 1 (Dimension/Condition) y se eliminaron los "†" |
| Imágenes (`image1.png`, `image2.png`, `image3.png`) | SHA-256 idéntico antes/después en las 3 imágenes — 0 cambio a nivel de píxel en ninguna figura |
| Nuevos valores p | 0 detectados |
| Tamaños de muestra, direcciones de contraste, parámetros de ventana/bootstrap | Sin cambios (verificado por igualdad de Tablas 1–4 y ausencia de ediciones en esos párrafos) |
| Cambios esperados vs. observados | Coinciden exactamente: renumeración (Figuras/Secciones/Tabla), eliminación de sección duplicada, cita de Leonardi & Van De Ville, y las sustituciones de texto explícitamente instruidas. Ningún otro cambio detectado. |

## 7. Revisión visual (§14) — 100% de páginas

Documento renderizado a PDF: **14 páginas** (igual que la versión de entrada). Revisión página por página:

- Numeración continua de secciones y figuras confirmada de principio a fin, sin huecos.
- Ausencia confirmada de la Figura 2 duplicada (contenido anatómico repetido de la Figura 1).
- Figura 1 preservada sin cambios (imagen, número, leyenda, contenido anatómico) — página 3.
- Nueva Figura 2 (antes Figura 3, AUC por panel) — página 9, datos idénticos, leyenda renumerada.
- Nueva Figura 3 (antes Figura 4, sensibilidad) y su leyenda completa en la misma página — página 12.
- "3.1 Performance of the Ablation-Derived ROI Panels", "3.2 Paired ROI-Panel Comparisons", "3.3 Sensitivity Analyses", "3.4 Convergence and Model Behavior": numeración automática correcta, sin huecos.
- Tabla 5 reformada — página 11, título+nota+tabla juntos en una sola página.
- Encabezado "References" sin numerar, inmediatamente después de "3.4 Convergence and Model Behavior".
- 14 entradas de referencia en orden alfabético, sangría francesa correcta, incluidas las 3 entradas con placeholder visibles con su texto de advertencia — páginas 13–14.
- Sin páginas en blanco, saltos de página accidentales ni etiquetas superpuestas en ninguna página.

**Resultado: PASSED.**

## 8. Reconstrucción y reemplazo del archivo oficial

- Validación estructural XML (`validate.py --original recovery_copy_UNNUMBERED.docx`): **"All validations PASSED!"**
- Párrafos: 418 → 429 (+11), coincide exactamente con la expectativa algebraica: −5 (sección duplicada eliminada) +1 (párrafo nuevo de niveles de sensibilidad, §2.6) +1 (encabezado "References") +14 (entradas de referencia) = **+11**.
- SHA-256 antes del reemplazo: `e7543e9195f7edb35628842dc96b484ad95ca5d3834116ea1f6e9c29a2003919`
- SHA-256 después del reemplazo (archivo oficial actual): `35834b5cf02bf3fa6f9c62853a5f6f5064ff043f1ee21c8ba92dd0879267ad55`
- El archivo oficial `Manuscript_Methods_Results_English_Working_v9_9.docx` fue reemplazado **en el mismo nombre de archivo**, sin crear `v9_10` ni ninguna copia con sufijo "final"/"corrected"/"new" ni ningún otro DOCX numerado.
- Verificado que el archivo reemplazado abre correctamente (python-docx: 84 párrafos de nivel superior detectados vía la API estándar, 5 tablas) y pasa la validación estructural completa.

## 9. Lista de referencias cruzadas actualizadas

- Todas las menciones de "Figure 3" (antes "Figure 4") y "Figure 2" (antes "Figure 3") en cuerpo, leyendas y texto alternativo.
- Referencia cruzada en la sección "Sensitivity Analyses" (antes: "ROI-panel comparisons are reported in Section 3.3. Figure 4 summarizes…"; ahora: mención directa a "Figure 3" sin anticipación hacia otra sección).
- Nombres internos de imagen no requirieron cambio (no contienen el número de figura en el nombre de archivo `image1.png`/`image2.png`/`image3.png`).

## 10. Entregables

Actualizados en `docs/manuscrito_revisado/`:

- `Manuscript_Methods_Results_English_Working_v9_9.docx` (reemplazado en el mismo archivo, mismo nombre).
- Este informe: `Informe_de_entrega_English_Working_v9_9.md` (sobrescrito).

Fuera de entregables oficiales (carpeta de trabajo temporal, no se copian a `docs/manuscrito_revisado/`):

- `recovery_copy_UNNUMBERED.docx` — copia de respaldo pre-edición; **solo debe eliminarse tras confirmación del equipo**.
- `working_copy_UNNUMBERED.docx`, `working_copy_EDITED.docx`, checkpoints intermedios (`stage_methods.docx`, `stage_results.docx`), scripts de edición y de verificación, notas de verificación bibliográfica.

No se entregó `v9_10`, ninguna copia con sufijo "final"/"corrected"/"new", ni ningún otro DOCX numerado.

## 11. Desviaciones respecto al plan

- Las 3 referencias (Hale 2014, Reimann 2024, Singh 2024) no pudieron verificarse contra una fuente autoritativa dentro del alcance de esta ronda; se entregan con placeholder explícito por decisión del usuario, en lugar de bloquear la entrega completa (el plan solo exige bloqueo formal para Yu et al. 2018).
- Nota de fuente ya documentada en rondas anteriores del proyecto y aún vigente: se usan Liberation Sans/Times New Roman según disponibilidad en este entorno de render; no afecta el documento entregado en sí, solo la vista previa de verificación.
- Ninguna otra desviación identificada.

## 12. Declaración final

`Manuscript_Methods_Results_English_Working_v9_9.docx` (archivo oficial, reemplazado en el mismo nombre) incorpora exactamente los cambios instruidos: eliminación de la sección/figura duplicada, renumeración completa y sin huecos, cinco sustituciones de texto en Métodos 2.6, tres en Métodos 2.7, un párrafo nuevo, el reemplazo de la justificación de ventana en 2.3 con la nueva cita, el renombrado y salvaguarda en 2.4, las ediciones de Resultados, la reforma de la Tabla 5, y la nueva sección de referencias con 14 entradas. Todos los controles obligatorios (auditoría de escritura, controles científicos automatizados, revisión visual al 100%, reconciliación bibliográfica) pasaron. Las tres referencias no verificables se entregan transparentemente marcadas, por decisión explícita del usuario. No se creó ninguna versión numerada nueva ni copia con sufijo alternativo.

## 13. Addendum — Reconciliación Tabla 4/Tabla 5 con las Figuras 2 y 3 (ronda posterior)

Tras una revisión editorial del equipo, se reportaron discrepancias entre las Figuras 2 y 3 (generadas por el equipo) y los valores entonces vigentes en las Tablas 4 y 5 (procedentes de una corrección solicitada en una ronda intermedia, con n = 166/165/35/66). El equipo confirmó que el tamaño de muestra oficial es **n = 177 (NYU), 183 (Peking), 39 (NeuroIMAGE), 66 (OHSU)**.

**Verificación (análisis de píxeles de las imágenes vigentes en ese momento, sin modificar datos):**
- Figura 2: las filas de 18/39/116 ROIs, en los cuatro sitios, y la fila de 12 ROIs en OHSU coincidían con los valores de la Tabla 4 **anteriores** a la corrección intermedia (n=177/183/39/66). Solo la fila de 12 ROIs en NYU/Peking/NeuroIMAGE coincidía con la corrección intermedia.
- Figura 3: los dos ejemplos señalados por el equipo (NYU 18 ROIs; Peking 116 ROIs) coincidían con los valores de Tabla 5 **anteriores** a la corrección intermedia, no con los valores corregidos.
- Esta medición por píxeles tiene un margen de error de ±0.5–1 punto porcentual (confirmado con las filas de OHSU, sin cambios entre versiones) y no sustituye a una fuente de datos verificada; no se usó para inventar ni ajustar cifras de tabla.

**Decisión aplicada** (siguiendo la recomendación del equipo — tablas como fuente, figuras regeneradas a partir de ellas — y el tamaño de muestra confirmado):
- **Tabla 4**: revertida a los valores originales (n=177/183/39/66; AUC/métricas asociadas). Párrafo idx48 (3.1) revertido en consonancia.
- **Tabla 5**: se conservaron los valores de la corrección intermedia (el equipo confirmó que la narrativa de 3.3 ya coincidía con ellos).
- **Sección 2.7 (Statistical Analysis)**: corregido "three and eighteen participants" → "**three and nineteen participants**" (consistente con n=183 confirmado, 10 folds).
- **3.3 Sensitivity Analyses**: aplicadas las dos mejoras de redacción solicitadas — dirección explícita en la oración de Peking/windowing ("...yielded AUC values 1.3 and 2.9 percentage points **lower than the reference**...") y adición del intervalo de confianza `[−23.3, −4.7]` al final de la oración de regresión logística (OHSU, 39 ROIs), por simetría con el resto de la oración.
- **Figuras 2 y 3**: regeneradas con Matplotlib replicando el estilo del equipo (tipografía, color de marcador/línea `#195D96`, línea de referencia punteada gris, disposición en 4 paneles, encabezados de grupo en cursiva/negrita en la Figura 3), usando exactamente los valores finales de Tabla 4 (Figura 2) y Tabla 5 (Figura 3). Dimensiones en píxeles idénticas a las imágenes reemplazadas (1444×481 y 1672×941) para no requerir cambios en `wp:extent`. Se detectó y corrigió un recorte (`a:srcRect`) heredado en la relación de la Figura 3 que cortaba ~36% de la parte inferior de la imagen; se restableció sin recorte.
- Se repitieron todos los controles: auditoría de escritura (0/13 coincidencias), validación estructural (`All validations PASSED!`, 429 párrafos sin cambio), comparación automatizada de tablas/párrafos contra el estado pre-ronda, y revisión visual de las páginas afectadas (7–12).

**Limitación declarada:** las Figuras 2 y 3 fueron reconstruidas a partir de los valores de tabla, no recuperadas del archivo de figura original del equipo; el estilo se imitó visualmente (fuente Liberation Sans como sustituto, igual que en rondas previas) pero no es un archivo fuente idéntico al del equipo.

SHA-256 antes de esta ronda: `760fd6296e46707d629e23432ec29d363df6941f6842ef92d712597ef0526cf2`
SHA-256 después (archivo oficial actual): `094f7537ef1f267462619a0ced4af6d2d031ed069353b0d55601729475d8dfaf`
