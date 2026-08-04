# Revisión de inglés académico tras la reestructuración de secciones

**Documento:** `Manuscript_Methods_Results_EN.docx` (versión reestructurada, 3 de agosto de 2026)
**Guía aplicada:** prompt maestro de revisión de inglés académico v2.0
**Alcance:** forma. Esta revisión no reabre decisiones metodológicas.
**Nota:** el documento ya había pasado la guía v2.0 antes de reestructurarse. Esta pasada se
centra en lo que la reestructuración pudo romper —referencias cruzadas, aperturas de sección,
antecedentes de pronombres y colisiones léxicas— y reverifica el resto.

---

## A. Diagnóstico general

**Publicable con correcciones menores.** Se detectaron cuatro problemas, todos aplicados. Uno
era obligatorio.

El patrón que los une no es gramatical: los cuatro son efectos secundarios de mover texto. Al
cambiar de sitio, un párrafo arrastra referencias que ya no apuntan donde deben, queda encabezando
una sección cuyo título no anticipa, o pone en contacto dos palabras que antes estaban lejos y
ahora compiten por el mismo sentido.

**Riesgo principal para publicación:** ninguno de idioma.
**Grado de intervención: bajo.** Una referencia corregida, tres oraciones ajustadas.
**Tipo de corrección:** lingüística y de función retórica, en proporciones similares.

---

## B. Cambios aplicados

| Ubicación | Texto original | Problema observable | Tipo | Severidad | Corrección |
|---|---|---|---|---|---|
| §2.1, 2.º párrafo | «…were not pooled into a common effect **(Section 2.7)**» | La referencia apunta a «Statistical Analysis», que no justifica el no agrupar. La justificación —«prevents treating the four procedures as interchangeable and supports the decision not to combine them into a common effect»— está en §2.6. La reestructuración movió el contenido y dejó la llamada atrás | Technical precision | **Obligatorio** | «…**(Section 2.6)**» |
| §2.1, 2.º párrafo | «were not **pooled** into a common effect» | `pooled` ya tiene sentido técnico fijo en el documento: agrupar las predicciones out-of-fold de los diez pliegues (§2.7 y pie de la Tabla 6). Usarlo también en sentido meta-analítico crea dos lecturas | Terminology consistency | Recomendable | «were not **combined** into a common effect», que además es la palabra que usa §2.6 |
| §2.1, 1.ª oración | «An empirical-analytical design of comparative scope and cross-sectional nature was used, based on retrospective, publicly available rs-fMRI data. Images from four sites…» | La sección se titula «Participants» y abre con el diseño del estudio; la primera oración no orienta (§17.7, §18.D). Es consecuencia de reubicar aquí la frase de diseño que quedó suelta al retirar los tres párrafos | Rhetorical function | Recomendable | Se invierte el orden: la sección abre con la muestra y el diseño la sigue. «Resting-state functional magnetic resonance imaging (rs-fMRI) images from four sites of the public ADHD-200 repository were used: NYU, Peking, NeuroIMAGE, and OHSU. The study followed an empirical-analytical design of comparative scope and cross-sectional nature, based on retrospective data.» |
| §3.3.1, 2.º párrafo | «39 ROIs in NYU, against the **wider** group, and 18 ROIs in NeuroIMAGE, **in its favor**» | Dos problemas en la misma oración. `wider` designa la amplitud de los intervalos en otras cinco apariciones, una de ellas en la oración siguiente («their intervals wide»); y el antecedente de `its` queda a doce palabras, tras una coordinación | Word choice / Grammar | Recomendable | «39 ROIs in NYU, against the **larger** group, and 18 ROIs in NeuroIMAGE, **in favor of the larger group**» |

Al reordenar la primera oración de §2.1 se eliminó «publicly available», que quedaba redundante
con «the public ADHD-200 repository» de la misma oración (§10.14). En la versión anterior los dos
sintagmas estaban en párrafos distintos y la redundancia no se percibía.

Ninguna cifra, cita, etiqueta ni valor estadístico cambió.

---

## C. Comprobaciones que pasaron sin hallazgos

Las de la pasada anterior se repitieron sobre el texto reestructurado; se añaden las específicas
de la reorganización.

| Criterio | Resultado |
|---|---|
| **Referencias cruzadas internas** | 9 llamadas a sección; las 9 apuntan a secciones existentes y, tras la corrección, a la que contiene el material |
| **Tablas y figuras** | 7 tablas y 3 figuras presentes; las 10 citadas en el texto. `Table S1` citada 4 veces |
| **Numeración** | §2.1–2.8, §3.1–3.4, §3.3.1–3.3.4. Sin saltos ni duplicados |
| §5 Terminología «ROI group» | 38 apariciones; `ROI panel` 0. Las 19 apariciones sueltas de `group` son anafóricas y remiten a grupos de ROI. Sin colisión con grupos de participantes: el texto los llama `classes`, `cases and controls` |
| §11 `panel` | 3 apariciones, todas de paneles de figura —«panel (a)», «panel (b)»—, que no debían tocarse |
| §4 Denominación, `contrast` | 5 apariciones, todas verbales o adjetivales: «contrasts complete pipelines», «as a contrast condition». Ninguna en sentido nominal, que un lector de fMRI leería como contraste del GLM |
| §15.11 Variante de inglés | Estadounidense sin mezcla: `behavior` 5, `behaviour` 0, `analyzed`, `normalization`. Los aparentes `analyse` son `analysis`/`analyses` |
| §9 Conectores | `however` 1 en todo el documento; `therefore` 6, nunca dos veces en el mismo párrafo; `moreover`, `furthermore`, `in addition`, `notably`, `importantly`, `thus`: 0 |
| §9.5 `since` / `while` | `since` causal 0. `while` 2, ambas de simultaneidad: «lengthens the window to 140 s while keeping the step». Los contrastes usan `whereas` (4) y `although` (4) |
| §11.5 Verbos de sobreafirmación | `demonstrate`, `prove`, `confirm`, `underscore`, `showcase`, `leverage`, `delve`: 0. `establish` 2, ambas negadas: «does not establish the existence», «does not establish general superiority» |
| §14.2 Léxico inflado | `novel`, `state-of-the-art`, `comprehensive`, `crucial`, `pivotal`, `insights`, `realm`, `landscape`, `meticulous`, `intricate`: 0. `robust` 1, en «the robustness of its estimates», uso técnico |
| §19.8 `significant` | 0 apariciones. Coherente con un estudio sin pruebas de hipótesis |
| §12 Word usage | `compared with` 4, `compared to` 0. `due to`, `the former`, `the latter`, `a number of`, `in the case of`, `etc.`, `and so on`, `in order to`, `prior to`: 0 |
| §12.21 `respectively` | 1 aparición, al final y precedida de coma |
| §12.11 `percent` | Sin usos desnudos: las 4 apariciones de la raíz son `percentile` y `percentage` |
| §6.26 Siglas | `rs-fMRI`, `ROI`, `AUC`, `TR`, `FWHM`, `LSTM` y `BOLD` definidas en su primera aparición en el cuerpo. `BOLD` encabeza §2.4 sin definir, pero se desarrolla en la primera oración de la sección |
| §12.8 `this` aislado | 3 casos, todos con antecedente inmediato e inequívoco |
| §18.E Resultados | 0 citas bibliográficas. Verbos de evidencia: `show` 6, `indicate` 5, `support` 1 |
| §18.H Pies | «Figure 2 shows», «Table 6 shows». Sin «demonstrates» ni «As can be seen from…» |

**Verificación de contenido asociada.** La reestructuración introdujo en §3.3.1 la frase «Only two
of the twelve dimensionality comparisons have intervals that exclude zero». Se comprobó contra
`dim_contrasts.json`: de las doce comparaciones, solo NYU-39 ([−12.3, −0.3]) y NeuroIMAGE-18
([+3.8, +25.8]) excluyen el cero, y ninguna de las cuatro comparaciones primarias 116-frente-a-12
lo hace. La cifra es correcta y sigue siéndolo aunque el párrafo remita el contraste principal a
§3.2, porque ninguno de los cuatro entra en la cuenta de los que excluyen cero.

---

## D. Lo que no se cambió, y por qué

- **Los dos párrafos de más de 160 palabras.** Son la declaración de exposición al ajuste y el
  apartado de dimensiones no evaluadas: unidades argumentativas con entradilla propia (§17.10).
- **Las seis oraciones de más de 40 palabras.** Ninguna supera las 50. Son enumeraciones de
  hiperparámetros y de criterios de restricción; la extensión viene del contenido, no de la
  sintaxis.
- **La voz pasiva de Métodos.** Convención para describir procedimientos, con agente irrelevante
  (§8.1).
- **`Deep Learning Architectures` en plural.** El original decía «LSTM Architecture» en singular,
  pero la versión nueva compara dos arquitecturas y el singular ya no describe el contenido.
- **El corchete de §2.5.** «[ Section reserved… ]» es andamiaje editorial, no texto del
  manuscrito, y desaparece cuando el equipo redacte la sección. Repite la fórmula «will be written
  at a later stage» de la nota de cabecera, cosa que en texto corrido marcaría §14.1.3, pero en
  una acotación entre corchetes la repetición es funcional.
- **`Model behavior` en el título de §3.4 y en el pie de la Figura 3.** Es vago, y así lo señalé
  en el documento de denominación, pero el equipo decidió conservar la nomenclatura acordada.
- **Las cifras de tabla, figura y sección en dígitos.** Son etiquetas, no cantidades: §6.25 no
  aplica.

---

## E. Evaluación por función retórica

| Sección | Función esperada | Estado | Comentario |
|---|---|---|---|
| Título | Breve, específico, sin adjetivos promocionales | Cumple | Delimita objeto, método y alcance multisitio |
| §2.1 Participants | Presentar la muestra y su procedencia | Cumple | Tras la corrección abre con la muestra; el diseño la sigue como calificación |
| §2.2–2.4 Preprocesamiento y conectividad | Procedimental y reproducible, en pasado | Cumple | Pasado simple; nombres de software, atlas y parámetros conservados |
| §2.5 Architectures | — | No aplica | Reservada por decisión del equipo |
| §2.6 Experimental and Evaluation Setup | Declarar protocolo y diseño de la sensibilidad | Cumple | Contiene la declaración de exposición y el apartado de dimensiones no evaluadas |
| §2.7 Statistical Analysis | Métodos analíticos, al final de Métodos | Cumple | Métrica primaria, bootstrap y reglas de inferencia, sobre datos ya definidos |
| §3.1–3.4 Results | Hallazgos sin discusión prematura ni literatura | Cumple | Cero citas; verbos de evidencia precisos; §3.3.1 remite a §3.2 en vez de repetir |
| Pies de tabla y figura | Autosuficientes | Cumple | La Tabla 7 declara la convención de signo y marca la fila primaria |
| Abstract, Introduction, Discussion, Conclusions | — | No aplica | No redactadas |

---

## F. Revisión anti-escritura genérica

| Criterio | Estado |
|---|---|
| No usa conectores en exceso | Cumple — `however` 1 en 38 párrafos |
| No usa transiciones artificiales | Cumple |
| No contiene relleno académico | Cumple |
| No usa lenguaje promocional | Cumple |
| No usa fórmulas metatextuales | Cumple — sin `it should be noted`, `plays a crucial role`, `sheds light on` |
| No abusa de adjetivos evaluativos vagos | Cumple |
| No usa verbos de estilo inflado | Cumple |
| No usa estructuras decorativas | Cumple — sin `not only… but also` |
| No sobreexplica ideas evidentes | Cumple — §3.3.1 remite al contraste primario en vez de reexponerlo |
| No cambia el grado de certeza | Cumple — ningún cambio toca modales ni verbos de evidencia |
| Mantiene terminología técnica consistente | Cumple — tras unificar `larger` frente a `wide` |
| Conserva claridad y concisión | Cumple |
| No realiza cambios innecesarios | Cumple — cuatro cambios, ninguna oración reescrita desde cero |
| Mantiene inglés académico internacional | Cumple |

---

## G. Indicadores y prioridades

| Indicador | Español | Inglés previo | Inglés reestructurado |
|---|---:|---:|---:|
| Longitud media de oración | 24.0 | 22.2 | **22.3** |
| Oraciones de más de 40 palabras | 15 | 10 | **6** |
| Oraciones de más de 50 palabras | — | — | **0** |
| Párrafos de más de 160 palabras | 3 | 2 | **2** |
| Longitud media de párrafo | 101 | 96 | **95** |

La reestructuración mejoró la métrica que más pesa en legibilidad: las oraciones largas bajaron de
diez a seis, y ninguna llega a cincuenta palabras. No es efecto de reescribir, sino de que al
repartir el contenido entre §2.6 y §2.7 se deshicieron dos enumeraciones que antes convivían en la
misma oración.

**Obligatorios por resolver:** ninguno. El único detectado está aplicado.

**Recomendables:** ninguno pendiente. Los tres están aplicados.

**Opcionales, no necesarios:** partir los dos párrafos de más de 160 palabras.

**Patrón que conviene vigilar en las secciones que faltan.** Los cuatro hallazgos de esta pasada
son del mismo tipo y no habrían aparecido sin mover texto: llamadas a sección que apuntan al lugar
donde el contenido estaba, no donde quedó; primeras oraciones que ya no anticipan el título de su
sección; y pares de palabras que solo compiten cuando quedan cerca. Cuando se integren
Introducción, Discusión y la §2.5, conviene reverificar estas tres cosas —y no solo la gramática—
en todo párrafo que cambie de lugar.
