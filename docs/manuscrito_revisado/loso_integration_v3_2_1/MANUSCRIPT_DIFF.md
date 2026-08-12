# MANUSCRIPT_DIFF.md — v9_9 → v9_10_LOSO_V3_2_1_clean

**Fase:** LOSO_METHODS_RESULTS_INTEGRATION_V3_2_1, CP6.
**Mecanismo de revisión:** `STRUCTURED_DIFF_FALLBACK` (no hay en este entorno una herramienta de comparación de Word que produzca y verifique revisiones OOXML reales `w:ins`/`w:del`; ver §3 de este documento).
**Método de generación:** comparación estructurada, párrafo por párrafo y tabla por tabla, mediante `python-docx` + `difflib.SequenceMatcher` sobre el texto extraído de ambos documentos — no de memoria. Script fuente: ver bloque reproducible en §4.

---

## 1. Resumen cuantitativo

| | v9_9 | clean (v9_10_LOSO_V3_2_1) |
|---|---:|---:|
| Párrafos | 85 | 94 |
| Tablas | 5 | 6 |
| Imágenes embebidas | 4 | 4 (MD5-idénticas a v9_9) |

`SequenceMatcher` sobre las listas de texto de párrafo encontró **7 bloques de diferencia**; los **78 párrafos restantes de v9_9 son idénticos, en el mismo orden, en `clean`**. Las Tablas 1–5 son textualmente idénticas entre ambos documentos (verificado celda por celda). La Tabla 6 es nueva y no existe en v9_9.

## 2. Los 7 cambios, uno por uno

### 2.1 — Table 3, frase introductoria y caption (párrafos 27–28)

**Antes:**
> Table 3 reports the reference BrainNetCNN configuration used across ROI panels and across the representation and windowing sensitivity conditions.
>
> Table 3. Reference BrainNetCNN training configuration used unchanged across sites, ROI panels, and representation and windowing sensitivity conditions.

**Después:**
> Table 3 reports the prespecified BrainNetCNN training and regularization specification used without dimensionality-specific retuning across ROI panels and the representation and windowing sensitivity conditions, and applied without LOSO-stage retuning to the static-connectivity LOSO conditions.
>
> Table 3. Prespecified BrainNetCNN training and regularization specification used without dimensionality-specific retuning across sites, ROI panels, and representation and windowing sensitivity conditions, and applied without LOSO-stage retuning to the static-connectivity LOSO conditions.

**Por qué:** extiende la enumeración de alcance de Table 3 para incluir las condiciones LOSO que ahora también usan esa especificación congelada, sin tocar ningún valor de la tabla y sin implicar que la arquitectura completa sea idéntica entre 12 y 116 ROI (D36).

### 2.2 — Alcance de "The analysis comprised three components" (párrafo 31)

**Antes:** "The analysis comprised three components."
**Después:** "The within-site analysis comprised three components."

**Por qué:** acota la frase al análisis within-site, ahora que existe un segundo análisis (LOSO) en la misma sección.

### 2.3 — Renombrado de "Evaluation Setup" (párrafo 34)

**Antes:** "Evaluation Setup"
**Después:** "Within-Site Evaluation"

**Por qué:** diferencia el subtítulo del dedicado a LOSO que se añade a continuación.

### 2.4 — Nuevo subtítulo y bloque LOSO en Methods (inserción tras el párrafo 35 de v9_9)

**Nuevo (2 párrafos, 308 palabras el segundo):**
> Cross-Site Transportability Evaluation across the Observed Sites (Leave-One-Site-Out)
>
> To complement the within-site evaluation, we performed a separate leave-one-site-out analysis as a site-held-out assessment of cross-site transportability across the four observed acquisition sites. [...]

**Por qué:** cubre los 26 elementos obligatorios del §8.1 del plan (rotaciones, sitios fuente, partición, estratificación, semilla, representación estática, ROI, edges, Fisher z, harmonización, weighting, especificación congelada de BrainNetCNN, seeds, comparador logístico determinista, StandardScaler, límites LOSO-stage, caveat de NYU, ausencia de retuning, límite a los cuatro sitios observados, y la declaración de no equivalencia con una cohorte futura independiente).

### 2.5 — Alcance de la apertura de Statistical Analysis (párrafo 37 de v9_9)

**Antes:** "The primary metric is the AUC of the out-of-fold predictions, ..."
**Después:** "For the within-site analyses, the primary metric is the AUC of the out-of-fold predictions, ..."

### 2.6 — Reformulación del estimando within-site + nuevo bloque estadístico LOSO (tras el párrafo 40 de v9_9)

**Antes (frase reemplazada dentro del párrafo 40):** "The estimand is therefore the repeated-cross-validation procedure rather than a single fitted model."
**Después:** "For the within-site analyses, the reported AUC summarizes performance of the repeated-cross-validation procedure based on the stored out-of-fold predictions, rather than performance of a single fitted model."

**Nuevo párrafo insertado a continuación (158 palabras):**
> For each LOSO rotation, the site-specific estimand was AUC in the held-out site for the frozen procedure trained on the other three sites. [...] The same class-stratified participant resamples were reused across all conditions and contrasts within each held-out site; the PCG64 generator was reset per held-out site rather than per comparison, preserving paired contrasts. [...]

**Por qué:** declara el estimando LOSO, las métricas secundarias, el bootstrap (10.000 remuestreos, PCG64, semilla 42, reset per held-out site) y la frase obligatoria de emparejamiento exigida por D32/M18.

### 2.7 — Nueva subsección 3.5 completa, con Table 6 (inserción tras el párrafo 72 de v9_9, antes de References)

**Nuevo (heading numerado automáticamente como 3.5, 4 párrafos de prosa — 247 palabras en total —, 1 tabla nueva de 4 filas × 5 columnas, y su caption):**
> Site-Held-Out Performance under Leave-One-Site-Out Evaluation
>
> Table 6 summarizes held-out-site AUC for the 12- and 116-ROI BrainNetCNN and LOSO logistic-regression conditions across the four observed sites. [...]
>
> [Table 6 — ver `loso_table6_source.csv` para los valores exactos sin redondear]
>
> Table 6. Leave-one-site-out AUC by held-out site, ROI dimensionality, and model across the four observed acquisition sites. [...]
>
> For BrainNetCNN, the 116-minus-12 ROI AUC point estimate was negative in all four held-out-site rotations [...]
>
> The LOSO logistic-regression-versus-BrainNetCNN contrasts varied in direction across held-out sites [...]
>
> Because the within-site and LOSO analyses differ in target estimand and connectivity representation [...]

## 3. Por qué se usa STRUCTURED_DIFF_FALLBACK y no un review.docx real

Este entorno dispone de LibreOffice (`soffice`, modo headless) y de utilidades de manipulación OOXML (`python-docx`, `lxml`, los scripts de la skill `docx`: `merge_runs.py`, `comment.py`, `accept_changes.py`), pero **no de una herramienta de comparación de documentos que genere automáticamente revisiones `w:ins`/`w:del` verificables** entre dos archivos `.docx` arbitrarios (el equivalente a "Word Compare"). Construir manualmente ese redline envolviendo cada cambio en `<w:ins>`/`<w:del>` sería posible en principio, pero el propio texto de la skill señala que es "easy to do by accident and invisible in the accepted view" — es decir, un redline hecho a mano por la misma IA que hizo el cambio no aporta la garantía independiente que un review.docx real está pensado para dar. Por eso se activa el fallback previsto en el plan (§11.2–§11.3), y no se simula con colores o resaltado.

**Estado del riesgo residual:** `ACCEPTED_BY_PLAN_OWNER`. Aprobado por Juan Pablo Ospina (jpospinalo@gmail.com), plan owner, el 9 de agosto de 2026, en respuesta directa a esta pregunta durante la ejecución de la fase. Evidencia y registro completos en `IMPLEMENTATION_REPORT.md`, §6.

El respaldo real de este mecanismo no es este documento por sí solo, sino el `scope diff` independiente del CP9 (§14.2 del plan), que compara `v9_9` contra `clean` de forma estructurada — exactamente el mismo método que generó este archivo — y que se ejecutará después de que el resto de los entregables científicos estén finales.

## 4. Reproducibilidad

Este diff se generó con el siguiente procedimiento (no de memoria):

```python
import docx, difflib
d9 = docx.Document('Manuscript_Methods_Results_English_Working_v9_9.docx')
dc = docx.Document('Manuscript_Methods_Results_English_Working_v9_10_LOSO_V3_2_1_clean.docx')
t9 = [p.text for p in d9.paragraphs]
tc = [p.text for p in dc.paragraphs]
sm = difflib.SequenceMatcher(a=t9, b=tc, autojunk=False)
# + comparación celda-por-celda de d9.tables vs dc.tables[:5]
# + comparación MD5 de word/media/image{1..4}.png
```

Resultado verificado: 7 bloques de diferencia, 78/85 párrafos de v9_9 preservados sin cambios y en el mismo orden, Tablas 1–5 textualmente idénticas, las 4 imágenes embebidas MD5-idénticas.
