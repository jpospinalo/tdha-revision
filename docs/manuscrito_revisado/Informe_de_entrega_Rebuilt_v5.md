# Informe de entrega v5 — Manuscript_Methods_Results_Rebuilt_v5.docx

Ronda menor: 2 correcciones de precisión textual sobre v4, sin cambios numéricos, estadísticos ni de datos.

## 1. Hashes

**Entradas (verificadas sin modificar):**
- `Manuscript_Methods_Results_Rebuilt_v4.docx`: `a62c3d6de1ea1b663985f81b92cb3ee10489fd6e9c29f93ca77dc4d725f9af8a`
- `Informe_de_entrega_Rebuilt_v4.md`: `4cd1b5293ee5f3676af21a0e3bbe187c4fd14b46dbb30048c43b7de39c08b2ac`

**Salida:**
- `Manuscript_Methods_Results_Rebuilt_v5.docx`: `1c5491de27fb49b9042e6a2483ae6ee8a9714d87c3ba98ba710ace7719425af9`

## 2. Correcciones aplicadas

| # | Hallazgo | Verificación previa | Corrección |
|---|---|---|---|
| 1 | 2.3 afirmaba que "Table 1 gives their translation into volumes", pero la Tabla 1 no traduce ventana/paso a volúmenes: sus columnas son Site, Participants, Control, ADHD, TR, Volumes, Window, Step, Windows — `Volumes` es la longitud total de la serie tras el recorte de las primeras 4, no una conversión de 120 s/12 s a número de volúmenes; `Window`/`Step` quedan expresados en segundos (con la variación específica por sitio ya visible: NYU/Peking exactos, NeuroIMAGE ≈11.8 s, OHSU 12.5 s) | Confirmado leyendo `t1_header` del script y el texto renderizado de la Tabla 1 en v4 | Reemplazada la oración por: *"The reference window was rectangular and nominally 120 s long, with a nominal 12-s step. Table 1 reports the site-specific effective values and the resulting number of windows."* |
| 2 | 3.3 atribuía el cambio de solapamiento únicamente a la condición de paso ("the step condition additionally changes overlap"), pero alargar la ventana de 120 a 140 s manteniendo el paso en 12 s también cambia el solapamiento | Calculado desde los mismos `config.json` ya citados en v3/v4: overlap de referencia (120 s/12 s) = 1 − 12/120 = 0.900; ventana 140 s/paso 12 s → 1 − 12/140 = 0.9143 (cambia); ventana 120 s/paso 24 s → 1 − 24/120 = 0.800 (cambia). Las dos condiciones cambian el solapamiento, no solo la de paso | Reemplazadas las tres oraciones finales del párrafo por: *"Each contrast changes one configured windowing parameter while holding the other fixed. However, changing either window length or step also changes the overlap fraction and the number of windows, and may therefore change effective model capacity. These results describe complete windowing conditions and are not attributed to temporal scale or overlap alone."* Los valores del contraste (−3.3 [−6.5, +0.1] y −4.7 [−8.6, −0.8]) no se tocaron. |

No se modificó ninguna cifra, tabla, figura, ni ningún otro texto.

## 3. Comparación dirigida v4 vs. v5

Diff de párrafo (`document.xml`, 382 líneas en ambas versiones): **exactamente 2 bloques de diferencia**, uno por cada corrección de la tabla anterior. Sin ninguna otra diferencia de texto (aparte de `docProps/core.xml`: `cp:revision` 4→5 y `dcterms:modified` actualizado).

## 4. Verificación estructural y visual

- `validate.py --original original_manuscript.docx`: 0 errores XSD; 412→401 párrafos, idéntico a v4.
- 13 páginas (idéntico a v4), renderizadas a PDF/JPG.
- Página 2 (§2.3) y página 10 (§3.3) inspeccionadas visualmente: ambas oraciones aparecen exactamente como se especificó, sin cortes ni defectos de maquetación; Tabla 1 y los valores del contraste de enventanado permanecen sin cambios.

## 5. Estado editorial

Sin cambios respecto a v4: sección 2.5 vacía, Figura 1 como marcador, integración bibliográfica de la Tabla 2 pendiente. **Este documento no debe considerarse listo para envío** por esas mismas razones ya señaladas en informes anteriores.
