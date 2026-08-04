# Evaluación de la segunda respuesta del equipo

**Documento evaluado:** «Respuesta al equipo sobre la revisión del manuscrito y la carta», 2 de agosto de 2026
**Estado:** se aceptan todos los puntos salvo uno, que debe corregirse antes de aplicarse.

> **Aviso sobre el estado de aplicación.** Este documento registra **aceptaciones, no
> aplicaciones**. Ninguna de las correcciones aquí aceptadas se ha incorporado todavía a
> `Manuscrito_Metodos_Resultados.docx` ni a `Respuesta_Revisores.docx`: las cinco
> inconsistencias del §4 siguen presentes en los archivos vigentes, tal como el equipo las
> encontró. Se hace explícito precisamente porque el defecto señalado en la ronda anterior fue
> declarar como terminadas correcciones aplicadas solo en parte. Las correcciones se aplicarán
> en la Etapa 2 del plan de trabajo, después de la auditoría, y su cumplimiento se verificará
> contra el texto extraído de los DOCX, no contra la intención de haberlas hecho.

---

## 1. Concepto general

La revisión es correcta y detectó un problema real en la ronda anterior: **cinco correcciones se
declararon completas cuando solo se habían aplicado parcialmente**. Se comprobaron una por una
sobre el texto extraído de los DOCX y las cinco existen. La causa fue un patrón de trabajo
defectuoso —corregir la tabla sin corregir la prosa que la describe, o corregir la carta sin
corregir el manuscrito— y no una discrepancia de criterio.

De los tres desacuerdos sustantivos que plantea el equipo, **se aceptan los tres**. Dos de ellos
se resolvieron con verificación empírica, no con argumento. Se detectó además **un error en la
corrección §4.4 propuesta**, que de aplicarse introduciría una afirmación falsa en la carta.

---

## 2. Las cinco inconsistencias del §4: confirmadas

Verificadas sobre el texto extraído de los DOCX vigentes.

| Punto | Texto que sobrevive | Confirmado |
|---|---|---|
| §4.1 | §2.7: «La exactitud, la precisión, la sensibilidad y la especificidad se reportan como métricas secundarias» | **Sí** |
| §4.2 | §3.3.1: «La sensibilidad a la dimensionalidad del panel es, por tanto, dependiente del sitio» | **Sí** |
| §4.3 | R2.5: «Para los demás sitios y paneles, la configuración se aplicó ya congelada» | **Sí** |
| §4.4 | R2.13: «Se retiran el ANOVA y la prueba de Tukey de la versión anterior» | **Sí, pero ver §5** |
| §4.5 | §2.8: «ventanas por debajo del límite del filtro pasa-banda» | **Sí** |

Se aceptan las redacciones propuestas para §4.1, §4.2, §4.3 y §4.5 sin modificación.

---

## 3. Los tres desacuerdos: se aceptan los tres

### 3.1 Tabla 6 (§5.1) — aceptado

Se verificó el render: la tabla ocupa las páginas 6, 7 y 8, y un bloque lógico de tres filas
—entrenamiento, validación y brecha de un mismo sitio y panel— queda partido en el salto de
página. La recomendación de compactar se acepta y el punto queda cerrado; ya no requiere
decisión adicional del equipo.

*Precisión menor:* el encabezado sí se repite en las páginas 7 y 8, porque la tabla está marcada
como encabezado repetible. El defecto real no es la ausencia de encabezado sino la
fragmentación del bloque de tres filas, que la compactación resuelve igualmente.

*Corrección de cronología:* el §10 del plan no se aprobó después de la instrucción de imitar la
tabla histórica, sino antes —el plan es del 30 de julio y la enmienda del 1 de agosto no tocó
§10—. La conclusión no cambia: el plan congelado gobierna, y la instrucción de formato no fue
una decisión metodológica registrada.

### 3.2 Bootstrap de 10 000 remuestreos (§5.2) — aceptado, con la prueba empírica hecha

El equipo ofrecía dos vías: recalcular a 10 000, o demostrar empíricamente estabilidad. **Se
ejecutó la segunda** sobre el contraste crítico, `static` en NYU, repitiendo el procedimiento de
2000 remuestreos con cuatro semillas independientes:

| Semilla | IC del contraste `static` en NYU | ¿Incluye cero? |
|---|---|---|
| 42 | [−0.0816, **+0.0014**] | sí |
| 1 | [−0.0824, **+0.0013**] | sí |
| 7 | [−0.0825, **+0.0002**] | sí |
| 2026 | [−0.0849, **+0.0019**] | sí |

La lectura cualitativa es estable en las cuatro repeticiones: el intervalo incluye el cero
siempre. Pero el límite superior oscila entre +0.0002 y +0.0019, y esa dispersión **es del mismo
orden que la distancia al cero**. Con la semilla 7 el margen es de dos diezmilésimas.

**El argumento de la ronda anterior era incorrecto.** Se sostuvo que el error de Monte Carlo solo
afecta la tercera cifra decimal y que ningún contraste está cerca de un umbral. Lo primero es
cierto; lo segundo no: la distancia al cero de este contraste está precisamente en la tercera
cifra. El equipo tenía razón y el recálculo a 10 000 se incorpora como paso previo a congelar
cifras.

### 3.3 Separación de manifiestos (§5.3) — aceptado

Se revisó `validate_manifest_structure()` en `build_analysis_dataset.py`. El pipeline contiene
dos comprobaciones que hacen inviable la propuesta anterior:

```
if len(included) != 16:
    raise ValidationError("se esperaban 16 filas con include=true, hay {n}")
...
dup = included.duplicated(subset=["site", "roi_set"])
```

Las diez corridas nuevas tienen todas `roi_set = 12`, de modo que añadirlas al manifiesto
primario dispararía **ambas** validaciones: el conteo dejaría de ser 16 y aparecerían
combinaciones sitio–panel duplicadas en los cuatro sitios. El pipeline no se contaminaría
silenciosamente: fallaría al arrancar.

La recomendación anterior de «extender `run_manifest.csv` a las 26 corridas» era, por tanto,
inviable además de indeseable. Se adopta el diseño de dos manifiestos separados con salidas
independientes, unidas solo al construir la Tabla 7 y la Figura 2.

---

## 4. Los hallazgos documentales del §6: aceptados

- **§6.1 Condiciones de acceso.** Correcto. La formulación «sin restricción para uso
  investigativo» debe sustituirse por la propuesta, que refleja el uso no comercial y el
  registro en NITRC.
- **§6.2 `QC_Rest_1`.** Correcto. Atribuir al indicador la detección de información clínica
  incompleta no está documentado. Se separarán los tres criterios.
- **§6.3 ATHENA.** Verificado contra la descripción del repositorio preprocesado: los datos
  funcionales se escriben en MNI152 a 4×4×4 mm, mientras que el manuscrito solo declara la
  resolución estructural de 1×1×1 mm. También es correcto que los mapas estructurales suavizados
  son mapas de densidad de materia gris, y que atribuirles el propósito de «enmascarar las
  regiones de interés» no está respaldado. Se corregirá y se identificará el derivado exacto del
  que se extrajeron las series AAL116.
- **§6.4 Ética.** Correcto: la afirmación absoluta se sustituye por la formulación propuesta.
- **§6.5 Disponibilidad.** Correcto: la carta debe citar un commit o DOI y no rutas locales.

---

## 5. El único punto que debe corregirse antes de aplicarse

### §4.4 — La corrección propuesta para R2.13 introduciría un error

El equipo sostiene que «la historia documentada indica que el manuscrito anterior tenía ANOVA y
el revisor solicitó añadir Tukey», y propone escribir: *«Se retiró el ANOVA y no se añadió la
prueba de Tukey solicitada»*.

**Esa formulación es falsa.** El manuscrito anterior sí reportaba Tukey. Párrafo 68 de
`Manuscript.docx`:

> «Tukey's post-hoc analysis showed significant differences between the model with 116 ROIs
> (p=0.0139) and the models with 12 and 18 ROIs (p=0.010), with an advantage for the 12 ROI
> configuration. No significant differences were found between the configurations with 12, 18
> and 39 ROIs.»

De modo que la versión anterior contenía **ambos** análisis, ANOVA y Tukey, y ambos se retiran.
Escribir que Tukey «no se añadió» sugeriría al revisor que nunca estuvo, lo que es incorrecto y
además desdibuja el alcance real de la corrección: no se está declinando una adición, se está
**retirando un análisis publicado**.

Conviene notar que el comentario R2.13 pide *extender* el análisis estadístico con una prueba
post-hoc, pese a que el manuscrito ya la reportaba. Esa discrepancia merece señalarse en la
respuesta, porque explica por qué la contestación no puede limitarse a declinar una solicitud.

**Redacción propuesta:**

> La versión anterior del manuscrito reportaba tanto el ANOVA como el análisis post-hoc de Tukey
> sobre los valores de exactitud en validación. Ambos se retiran en esta versión. Los pliegues y
> las repeticiones de la validación cruzada comparten sujetos y conjuntos de entrenamiento, de
> modo que no constituyen observaciones independientes; el ANOVA no cumple ese supuesto y una
> prueba post-hoc aplicada sobre él heredaría el mismo problema en lugar de resolverlo. En su
> lugar se adoptan diferencias pareadas a nivel de sujeto sobre las predicciones fuera de
> pliegue, con remuestreo bootstrap estratificado dentro de cada sitio, procedimiento coherente
> con la dependencia presente en los datos.

Esta versión mantiene la justificación técnica que el propio equipo pide conservar en su §3.4,
sin el tono comparativo y sin afirmar algo que el manuscrito anterior contradice.

---

## 6. Estado de la ronda

Se aceptan todos los puntos del documento del equipo, con la única salvedad de §4.4, que debe
aplicarse con la redacción corregida de §5 de este informe.

Ningún punto exige nuevos entrenamientos. El trabajo restante es el que el propio equipo
enumera en su §8, con una corrección de secuencia: el recálculo a 10 000 remuestreos y la
separación de manifiestos son ahora parte de la Etapa 1 y condicionan las Etapas 2 y 3, porque
las tablas y figuras deben regenerarse desde los resultados auditados.

**Criterios de cierre añadidos a los del equipo:**

- [ ] R2.13 refleja que la versión anterior reportaba ANOVA **y** Tukey, y que ambos se retiran.
- [ ] Se corrigió el *docstring* de `src/kerasmodels/brainnetcnn.py`, origen documental del error
      sobre el tamaño muestral de la arquitectura original.
