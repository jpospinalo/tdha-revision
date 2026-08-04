# Revisión del inglés de la versión traducida

**Documento:** `Manuscript_Methods_Results_EN.docx`
**Guía aplicada:** prompt maestro de revisión de inglés académico v2.0
**Alcance:** forma. La revisión no reabre decisiones metodológicas.

---

## A. Diagnóstico general

**Publicable con correcciones menores.** Se detectaron tres errores aislados, todos corregidos, y
ninguno afectaba la comprensión.

El problema más frecuente no fue gramatical sino de precisión en el uso de conectores y
convenciones: un `since` causal donde la guía prescribe `because`, un `respectively` sin coma
previa y un `in percent` sin cifra acompañante.

**Riesgo principal para publicación:** ninguno de idioma.
**Grado de intervención: bajo.** Tres cambios obligatorios, ninguna oración reescrita.
**Tipo de corrección:** lingüística. No requirió ajuste retórico.

---

## B. Cambios aplicados

| Ubicación | Texto original | Problema | Tipo | Severidad | Corrección |
|---|---|---|---|---|---|
| §2.8, 2.º párrafo | «…not artificially equalized, **since** equalizing it would introduce…» | `since` tiene connotación temporal; la relación es causal (§9.5) | Cohesion | Obligatorio | «…equalized, **because** equalizing it would introduce…» |
| §3.2, 1.ª oración | «…in Peking, OHSU, and NeuroIMAGE **respectively**.» | `respectively` va al final precedido de coma (§12.21) | Punctuation | Obligatorio | «…and NeuroIMAGE**, respectively**.» |
| Pie de la Tabla 6 | «…by site and ROI panel, **in percent**.» | `percent` solo acompañado de cifra; sin número corresponde `percentage` (§12.11) | Word choice | Obligatorio | «…by site and ROI panel, **expressed as percentages**.» |

Se verificó que ninguna cifra cambió entre la versión anterior y la corregida.

---

## C. Comprobaciones que pasaron sin hallazgos

| Criterio | Resultado |
|---|---|
| §5.19 / §15.11 Variante de inglés | Estadounidense, sin mezcla: `behavior` 5, `behaviour` 0, `modeling` 1, `-ise` 0 |
| §11.5 Verbos de sobreafirmación | `demonstrate` 0, `prove` 0, `underscore` 0, `showcase` 0, `leverage` 0, `delve` 0 |
| §11.5 `establish` | 3 apariciones, **todas en construcción negada**: «does not establish clinical utility», «does not establish the existence», «does not establish general superiority» |
| §11.7 / §19.8 `significant` | 0 apariciones. Coherente con un estudio que no realiza pruebas de hipótesis |
| §14.2 Adjetivos evaluativos vagos | `novel` 0, `state-of-the-art` 0, `robust framework` 0, `comprehensive` 0 |
| §12.2 `compare with` / `compare to` | `compared with` 4, `compared to` 0 |
| §12.4 `different from` | Sin `different than` |
| §12.5 `due to` | 0 apariciones; se usa `because of` donde corresponde |
| §12.20 `the former` / `the latter` | 0 apariciones |
| §12.24 / §12.25 `a number of`, `in the case of` | 0 apariciones |
| §12.18 `etc.`, `and so on` | 0 apariciones |
| §6.13 Colocación de `only` | 8 apariciones, todas adyacentes a lo que modifican: «ran on CPU only», «computed only on the fitting data», «retained only as an audit metric», «the only case» |
| §6.14 Comparaciones completas | Sin comparaciones truncadas; se usa «exceeds that of 18 ROIs», no «exceeds 18 ROIs» |
| §6.15 `that` / `which` | Los 4 casos de `which` sin coma previa son construcciones con preposición: «the extent to which», «access to which», «the only case in which» |
| §6.16 `fewer` / `less` | 0 apariciones; no aplica |
| §6.4 Plurales técnicos | `data were` en plural, `the use of these data conforms` en singular por el sujeto |
| §16.2 Guiones compuestos | Consistentes: `out-of-fold` 10, `cross-validation` 6, `class-stratified` 2, `two-sided` 3, `resting-state`, `attention-deficit` |
| §6.11 Punto y coma | 22 apariciones, todas uniendo cláusulas estrechamente ligadas o separando ítems complejos de una lista |
| §8.6 / §8.10 Primera persona y titularidad | `we` 0, `our` 0. `the authors` aparece una vez, en la declaración de conflicto de interés, sin ambigüedad de referente |
| §9.4 Acumulación de conectores | `therefore` aparece 7 veces pero **nunca dos veces en el mismo párrafo**; `however` 1 vez en todo el documento |
| §9.5 `while` | 2 apariciones, ambas de simultaneidad y no de contraste: «lengthens the window to 140 s while keeping the step» |
| §6.22 Arranque con gerundio | «Holding the hyperparameters fixed…» y «Aggregating by repetition…» funcionan como sujeto de la oración; no son modificadores colgantes |
| §14.1 Fórmulas metatextuales | Sin `it should be noted`, `it is worth noting`, `plays a crucial role`, `sheds light on` |

---

## D. Lo que no se cambió, y por qué

- **`confirms` en §3.3** («which confirms the redundancy expected with 90% overlap»). La guía
  desaconseja `confirm` cuando el diseño solo permite describir, pero aquí la redundancia se mide
  directamente: la similitud entre ventanas adyacentes es el dato. No es una inferencia causal.
- **La voz pasiva de Métodos.** Es la convención para describir procedimientos y el agente no
  aporta información (§8.1).
- **Los 54 paréntesis.** Contienen siglas, citas, unidades o valores estadísticos, que §16.4
  protege expresamente.
- **Los seis pares de raya.** Delimitan aclaraciones técnicas breves y no constituyen un tic:
  seis pares en cuarenta y cuatro párrafos.
- **Dos párrafos de más de 160 palabras.** Son la declaración de exposición al ajuste y el
  apartado de dimensiones no evaluadas: unidades argumentativas con entradilla propia (§17.10).
- **Las diez oraciones de más de 40 palabras.** Son enumeraciones técnicas de hiperparámetros,
  regiones y pasos de preprocesamiento; la extensión proviene del contenido.
- **Los números de tabla, figura y sección en dígitos.** Son etiquetas, no cantidades, de modo
  que §6.25 no aplica.

---

## E. Evaluación por función retórica

| Sección | Función esperada | Estado | Comentario |
|---|---|---|---|
| Título | Breve, específico, sin adjetivos promocionales | Cumple | Delimita objeto, método y alcance multisitio; sin «A study of» ni equivalentes |
| §2 Methods | Procedimental, reproducible, en pasado | Cumple | Pasado simple para procedimientos; pasiva adecuada; nombres de software y versiones conservados |
| §3 Results | Hallazgos sin discusión prematura ni literatura | Cumple | Cero citas en Resultados; verbos de evidencia precisos (`show`, `indicate`, `range`, `exceed`) |
| Pies de tabla y figura | Autosuficientes, sin `Figure X demonstrates` | Cumple | Se usa `Figure 2 shows`, `Table 6 shows`; sin «As can be seen from…» |
| Abstract, Introduction, Discussion, Conclusions | — | No aplica | No redactadas |

---

## F. Revisión anti-escritura genérica

| Criterio | Estado |
|---|---|
| No usa conectores en exceso | Cumple — `however` 1 en todo el documento; `therefore` nunca repetido en un párrafo |
| No usa transiciones artificiales | Cumple |
| No contiene relleno académico | Cumple |
| No usa lenguaje promocional | Cumple |
| No usa fórmulas metatextuales | Cumple |
| No abusa de adjetivos evaluativos vagos | Cumple |
| No usa verbos de estilo inflado | Cumple |
| No usa estructuras decorativas | Cumple — sin `not only… but also` |
| No sobreexplica ideas evidentes | Cumple |
| No cambia el grado de certeza | Cumple — los tres cambios son de conector, coma y una palabra de unidad |
| Mantiene terminología consistente | Cumple |
| Conserva claridad y concisión | Cumple |
| No realiza cambios innecesarios | Cumple — ninguna oración reescrita |
| Mantiene inglés académico internacional | Cumple |

---

## G. Indicadores y prioridades

| Indicador | Español | Inglés |
|---|---:|---:|
| Longitud media de oración | 24.0 palabras | **22.2** |
| Oraciones de más de 40 palabras | 15 | **10** |
| Párrafos de más de 160 palabras | 3 | **2** |
| Longitud media de párrafo | 101 palabras | **96** |

**Obligatorios por resolver:** ninguno. Los tres detectados están aplicados.

**Recomendables:** ninguno pendiente.

**Opcionales, no necesarios:** partir los dos párrafos restantes de más de 160 palabras; los
conservo porque su unidad argumentativa es clara.

**Patrones a vigilar al traducir lo que falta.** Cuando se traduzcan Introducción y Discusión, el
riesgo se desplaza a tres puntos que en Métodos y Resultados no llegan a manifestarse: el uso de
`since` y `while` como conectores causales o adversativos, donde el inglés académico prefiere
`because` y `although`; el modal de la Discusión, donde la presión por concluir empuja de
`is compatible with` hacia `indicates`; y la distinción entre `future work should` —recomendación
a la comunidad— y `future work will` —compromiso de los autores—, que en español no se marca.
