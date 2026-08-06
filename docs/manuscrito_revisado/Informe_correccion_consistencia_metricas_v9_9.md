# Informe de corrección de consistencia de métricas — Manuscript_Methods_Results_English_Working_v9_9.docx

Fecha: 2026-08-05

## 1. Identificación del archivo y control de integridad

- Ruta oficial (única, sin duplicados ni versiones numeradas nuevas):
  `docs/manuscrito_revisado/Manuscript_Methods_Results_English_Working_v9_9.docx`
- SHA-256 **antes** de la intervención: `91e699f7f2d14c347a92d6d36f4e33cb8884b51cbc3e8b64598f36643c83e131`
  (verificado idéntico al valor informado en las instrucciones antes de tocar el archivo).
- SHA-256 **después** de la intervención: `83003683b0b4d01dec95c16c8d1e17a4d523d472880cd0ff0f077966807715b0`
- Todo el trabajo se realizó sobre una copia temporal fuera de la carpeta oficial
  (`working_copy_UNNUMBERED.docx` → `working_copy_FIXED.docx`); el archivo oficial no se modificó
  hasta que todos los controles (estructurales, numéricos y visuales) pasaron. No queda ninguna
  copia temporal dentro de la carpeta de entrega oficial.

## 2. Fuentes numéricas oficiales utilizadas (con SHA-256)

| Fuente | Ruta | SHA-256 | Uso |
|---|---|---|---|
| Bootstrap maestro (10,000 iteraciones) | `analysis/roi_comparison/outputs/tables/manuscript_bootstrap_10k.csv` | `dc436761e70caca91af60c80b7bf15829143892a5b81154694e7119e74fd59f4` | Fuente única para Tabla 5 (todas las filas, incluida regresión logística) y para regenerar las afirmaciones cuantitativas de 3.2/3.3 |
| Manifiesto del bootstrap | `analysis/roi_comparison/outputs/logs/manuscript_bootstrap_10k_manifest.json` | `4aaf8a081ded61fadb1d9798977d8a60747fd5aa6af20d1f79858918bd853304` | Confirma `retraining: false`, `resampling: paired_by_subject_stratified_by_class`, `ci_method: percentile_2.5_97.5`, seed=42, generator=PCG64, n_iter=10000 — coherente con el método descrito en 2.7 |
| Auditoría de Figura 2 | `analysis/roi_comparison/outputs/tables/figure3_v5_audit.csv` | `601f72750ba995523b57d9a937a4b6bebbe1ba2c5d2b32bc21351bea3b331709` | Verificación de correspondencia Tabla 4 ↔ Figura 2 (16 pares sitio–panel) |
| Auditoría de Figura 3 | `analysis/roi_comparison/outputs/tables/figure4_v5_audit.csv` | `b67eae688c742cbda768a1595ddf55194b433251dc4150892c35e56dcb7446cc` | Verificación de correspondencia Tabla 5 ↔ Figura 3, incluida columna `status: evaluated/not_evaluated` para condiciones no ejecutadas (LSTM y ventanas en NeuroIMAGE/OHSU) |
| Fuente alternativa revisada y **no usada** | `analysis/roi_comparison/outputs/baseline/tables/baseline_contrast.csv` | `12110622c600b3d855d7fd782ff2fb9716e2b713d20fd44087f4e7d0abf6c52d` | Bootstrap de 2,000 iteraciones marcado `team_review_status: pendiente`; descartado por menor número de iteraciones y por no estar validado por el equipo. Las filas de regresión logística equivalentes en `manuscript_bootstrap_10k.csv` (mismo seed/generador que el resto del documento) se usaron en su lugar, por consistencia metodológica plena. |

No se generó, recalculó ni reentrenó ningún modelo. Todos los valores usados ya existían en los
archivos de resultados almacenados; el único procesamiento fue lectura, conversión a puntos
porcentuales (×100) y redondeo a la presentación (1 decimal), aplicado únicamente al final.

## 3. Cambios realizados

### 3.1 Tabla 1 (obligatorio, §5)
Corregidos Participants/Control/ADHD en las filas NYU, Peking y NeuroIMAGE:

| Sitio | Antes | Después |
|---|---|---|
| NYU | 166 / 82 / 78 (inconsistente: 82+78=160≠166) | 177 / 87 / 90 |
| Peking | 165 / 100 / 65 | 183 / 109 / 74 |
| NeuroIMAGE | 35 / 19 / 16 | 39 / 22 / 17 |
| OHSU | 66 / 38 / 28 (ya correcto) | sin cambio |

Verificado: Control + ADHD = Participants en las cuatro filas; los tamaños 177/183/39/66 ahora
coinciden con la Tabla 4 y la Figura 2. Búsqueda completa del documento confirmó que no quedan
otras apariciones de 166/165/35 usadas como tamaño de muestra.

### 3.2 Tabla 5 (corrección bloqueante, §7)
Las 11 filas de datos se reconstruyeron a partir de `manuscript_bootstrap_10k.csv` (no por
aproximación ni lectura de la Figura 3). Se confirmó que estos valores reproducen exactamente los
puntos e intervalos usados para generar la Figura 3 actual (`figure4_v5_audit.csv`), incluidas las
celdas no evaluadas (Architecture y Window/step en NeuroIMAGE y OHSU, mantenidas como guiones "—",
sin imputación).

Las cuatro filas de "Logistic regression" se verificaron contra los contrastes
`baseline__SITE__roiN` del mismo archivo maestro de 10,000 iteraciones (misma semilla/generador que
el resto del documento), no contra la fuente de 2,000 iteraciones pendiente de revisión.

### 3.3 Sección 3.2 (párrafos 54 y 56)
Reescritos para reportar exactamente los valores reconciliados de la fila "116 ROIs (primary)" y
"18/39 ROIs" de la Tabla 5 corregida (−5.1/+4.0/+4.5/+0.9 pp; −6.2 en NYU 39-ROI; +14.6 en
NeuroIMAGE 18-ROI), con lectura de inclusión/exclusión de cero verificada algorítmicamente contra
los intervalos, no copiada de memoria.

### 3.4 Sección 3.3 (párrafos 59, 60, 61, 62)
Reescritos para reportar los valores reconciliados de Static FC, LSTM (128 units), 140/12, 120/24 y
regresión logística vs. BrainNetCNN emparejado. El conteo "quince de dieciséis intervalos de
regresión logística incluyeron cero, con una excepción (39 ROIs en OHSU)" se derivó contando
directamente sobre los 16 intervalos de la Tabla 5 corregida, no se fijó como texto arbitrario.

### 3.5 Elementos explícitamente NO modificados (protegidos, §4/§6)
Tabla 2, Tabla 3, Figura 1: sin cambios. Tabla 4, Figura 2 y la narrativa de la Sección 3.1: sin
cambios (ya coherentes con n=177/183/39/66 y con `figure3_v5_audit.csv`). El texto de la Sección
2.7 sobre el tamaño de los folds externos ("between three and nineteen participants") ya estaba
correcto en el archivo de entrada verificado; no requirió cambio en esta ronda. Las imágenes
insertadas de la Figura 2 y la Figura 3 (estilo, paleta, tipografía del equipo) no se tocaron: al
restaurarse manualmente por el equipo antes de esta ronda, ya son consistentes con las tablas
corregidas sin necesidad de regenerarlas.

## 4. Control de reconciliación automatizado (§9)

Script de verificación (comparación numérica, tolerancia 0.05 puntos porcentuales):

- Tabla 1 vs. tamaños confirmados vs. Tabla 4: **12 comparaciones, 0 diferencias**.
- Tabla 5 vs. `manuscript_bootstrap_10k.csv` (44 celdas: 11 filas × 4 sitios): **44 comparaciones, 0 diferencias**.
- Total: **56 comparaciones, 0 fallos** dentro de la tolerancia de redondeo.

No se detectaron: sitio/condición incorrecta, dirección de contraste invertida, valor de punto
distinto, límites de intervalo distintos, redondeo inconsistente, o uso de un valor de un
experimento no ejecutado.

Auditoría adicional de texto: búsqueda de las cifras heredadas incorrectas mencionadas en las
instrucciones (9.0/1.2/0.0 de la comparación 116−12; 9.5/12.5 de los paneles intermedios;
5.5/7.5/0.6 de Static FC; 7.7/1.9 de LSTM; 4.4/6.2/1.3/2.9 de windowing; 8.4 de la excepción Peking
18-ROI) confirmó **cero apariciones residuales** en el documento final.

## 5. Inspección visual (§10)

Documento renderizado íntegramente a PDF/JPEG (14 páginas) e inspeccionadas el 100% de las páginas:

- Página 1: Tabla 1 con las cuatro filas corregidas, sin cortes ni desbordes.
- Páginas 2–6: Métodos (2.2–2.7), Tabla 2, Tabla 3, Figura 1 — sin cambios, renderizado correcto.
- Páginas 7–9: Tabla 4, Figura 2, Sección 3.1 — sin cambios, coherentes.
- Páginas 9–11: Sección 3.2, Tabla 5 (11 filas, cabecera y nota completas, sin dividirse entre
  páginas), Sección 3.3.
- Página 12: Figura 3 con su leyenda completa en la misma página.
- Página 13–14: Sección 3.4 y Referencias, sin cambios.

No se observaron textos cortados, superposiciones, tablas fuera de márgenes, figuras borrosas,
saltos anómalos ni cambios de fuente. La paginación se mantiene en **14 páginas**, igual que la
versión de entrada — no hubo cambio de paginación que requiera justificación.

## 6. Confirmaciones exigidas por las instrucciones

- No se reentrenó ningún modelo ni se repitió el análisis estadístico.
- No se recalcularon intervalos con un método distinto al bootstrap de 10,000 iteraciones ya
  documentado.
- No se introdujeron comparaciones, pruebas ni conclusiones nuevas.
- Ningún valor de la Tabla 5 se calculó a partir de cifras redondeadas de la Tabla 4 ni se leyó de
  los píxeles de la Figura 3; todos provienen de `manuscript_bootstrap_10k.csv`.
- El archivo oficial sigue siendo la única versión de trabajo: no se creó `v9_10`, "final",
  "corrected" ni ninguna copia numerada adicional.

## 7. Pendientes fuera de alcance (sin resolver en esta intervención)

- Sección 3.4 (Convergence and Model Behavior): sus afirmaciones sobre convergencia quedan
  explícitamente fuera de alcance de esta corrección, a auditar en una ronda posterior.
- Tres referencias bibliográficas siguen marcadas como pendientes de verificación contra fuente
  autoritativa: Hale et al. (2014), Reimann et al. (2024), Singh et al. (2024). No se inventaron ni
  completaron citas.

## 8. Declaración final

`Manuscript_Methods_Results_English_Working_v9_9.docx` permanece como la única versión oficial de
trabajo, en la misma ruta. Todas las afirmaciones cuantitativas modificadas son trazables al
archivo `manuscript_bootstrap_10k.csv` (SHA-256 arriba) y a su manifiesto de metodología, lo que
permite una auditoría futura completa.
