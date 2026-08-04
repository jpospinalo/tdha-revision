# Revisión de escritura académica — versión final de Métodos y Resultados

**Documento:** `Manuscrito_Metodos_Resultados.docx`, 2 de agosto de 2026
**Guía aplicada:** prompt maestro de revisión de inglés académico v2.0, adaptado al español
**Alcance:** forma, no fondo. Los criterios de idioma inglés (variante US/UK, artículos,
preposiciones, falsos amigos, plurales latinos) no aplican; sí aplican los de estructura
oracional, cohesión, precisión técnica, tono, patrones genéricos y función retórica.

---

## A. Diagnóstico general

**Publicable con correcciones menores.** El texto es gramaticalmente correcto, técnicamente
preciso y libre de lenguaje promocional: no aparece «significativo» en sentido no estadístico,
«demuestra» solo se usa en construcciones negadas, y no se detectaron intensificadores vacíos ni
fórmulas metatextuales repetidas.

El problema más frecuente era de **párrafo, no de oración**. La longitud media de oración ya
estaba en 24 palabras, dentro del rango habitual, pero seis párrafos superaban las 160 palabras
y uno alcanzaba 292, acumulando hasta seis ideas distintas. Era una regresión introducida al
incorporar las correcciones de las dos rondas anteriores dentro de párrafos existentes.

**Riesgo principal para publicación:** ninguno de idioma. El texto no requería reescritura.

**Grado de intervención: bajo.** Nueve cambios, todos de división de párrafo o expansión de
siglas. Ninguna oración se reescribió.

---

## B. Cambios aplicados

| Ubicación | Problema observable | Tipo | Severidad | Corrección |
|---|---|---|---|---|
| §2.7, párrafo del bootstrap | 292 palabras con seis ideas: mecánica del remuestreo, magnitud de la incertidumbre, pruebas no aplicadas y alcance de los intervalos | Paragraphing | Obligatorio | Dividido en cuatro párrafos: especificación, magnitud, lo que no se aplicó, alcance |
| §2.7, párrafo de la métrica | 195 palabras: definición de la métrica primaria y justificación del esquema de agregación | Paragraphing | Recomendable | Dividido en dos |
| §2.6, arquitectura | 195 palabras: regularización de la implementación y conteo de parámetros entre paneles | Paragraphing | Recomendable | Dividido en dos |
| §3.4, convergencia | 215 palabras: descripción de las curvas y comparabilidad con la Tabla 6 | Paragraphing | Recomendable | Dividido en dos |
| §2.3, ATHENA | 198 palabras: procesamiento anatómico y funcional en un solo bloque | Paragraphing | Recomendable | Dividido en dos |
| §2.4, paneles | 176 palabras: definición de los paneles y su consecuencia metodológica | Paragraphing | Recomendable | Dividido en dos |
| §3.3.1, dimensionalidad | 190 palabras: lecturas por sitio y recuento global | Paragraphing | Recomendable | Dividido en dos |
| §2.3 | «Las imágenes anatómicas (T1) se reorientaron al sistema RPI» — dos siglas sin expandir | Word choice | Obligatorio | «imágenes anatómicas ponderadas en T1… sistema de orientación derecha-posterior-inferior (RPI)» |
| §2.4 | «La señal BOLD» sin expandir en primera aparición | Word choice | Obligatorio | «La señal dependiente del nivel de oxígeno en sangre (BOLD)» |
| §2.8 | «arquitecturas de tipo GRU o Transformer» sin expandir | Word choice | Recomendable | «GRU —unidad recurrente con compuertas— o Transformer» |
| §3.4 | «Conviene precisar que…» repetido en dos secciones | Artificial/generic writing pattern | Recomendable | Reformulado como «no son, sin embargo, directamente comparables» |

---

## C. Efecto medido

| Indicador | Antes | Después |
|---|---:|---:|
| Párrafos de prosa | 40 | 49 |
| Longitud media de párrafo | 123 palabras | **101** |
| Párrafo más largo | 292 palabras | **182** |
| Párrafos de más de 160 palabras | 10 | **3** |
| Oraciones | 204 | 205 |
| Longitud media de oración | 24.1 palabras | **24.0** |
| Oraciones de más de 40 palabras | 16 (8 %) | **15 (7 %)** |

Se verificó que el conjunto de cifras del texto es idéntico antes y después: ninguna perdida,
ninguna añadida. No se modificaron tiempos verbales ni modales, de modo que el alcance y el
grado de certeza de las afirmaciones se conservan sin cambio.

---

## D. Lo que no se cambió, y por qué

- **Tres párrafos siguen por encima de 160 palabras** y se dejan así deliberadamente. «Exposición
  al ajuste de hiperparámetros» (182) y «Dimensiones no evaluadas» (174) son declaraciones
  unitarias con entradilla propia: partirlas rompería la unidad argumentativa que la guía pide
  preservar. El párrafo del protocolo de validación (163) describe un procedimiento continuo.
- **Las quince oraciones de más de 40 palabras** son enumeraciones técnicas —listas de
  hiperparámetros, de regiones, de pasos de preprocesamiento— donde la extensión proviene del
  contenido y no de la sintaxis.
- **La voz pasiva de Métodos** se conserva: es la convención para describir procedimientos y el
  agente no aporta información.
- **Los términos técnicos centrales** —panel, representación multicanal, fuera de pliegue,
  partición exterior, remuestreo pareado— se mantienen estables y sin sinónimos.
- **«Statistically significant» y equivalentes** no aparecen; no había nada que preservar ni que
  eliminar en ese frente.
- **Los guiones de modificador compuesto** («fuera de pliegue», «arista-a-arista») se conservan
  porque cambian el significado si se retiran.
- **Las rayas de inciso** se mantienen donde delimitan una aclaración técnica breve; solo hay
  tres pares en todo el documento, lejos de constituir un tic.

---

## E. Evaluación por función retórica

| Sección | Función esperada | Estado | Comentario |
|---|---|---|---|
| §2 Método | Lenguaje procedimental, reproducible, en pasado | Cumple | Pasiva adecuada; secuencia clara; nombres de software y versiones conservados |
| §3 Resultados | Hallazgos sin discusión prematura ni literatura | Cumple | Cero citas en Resultados; verbos de evidencia precisos; sin «demuestra» ni «confirma» |
| Pies de tabla y figura | Autosuficientes, sin repetir el texto | Cumple | Los tres pies permiten leer la figura sin recurrir al cuerpo |
| Título | Breve, específico, sin adjetivos promocionales | Cumple | Delimita objeto, método y alcance multisitio |
| Resumen, Introducción, Discusión, Conclusiones | — | No aplica | No redactadas |

---

## F. Revisión anti-escritura genérica

| Criterio | Estado |
|---|---|
| No usa conectores en exceso | Cumple — máximo cinco apariciones de un mismo conector en todo el documento |
| No usa transiciones artificiales | Cumple |
| No contiene relleno académico | Cumple — sin «cabe señalar», «es importante mencionar», «con el fin de» |
| No usa lenguaje promocional | Cumple |
| No usa fórmulas metatextuales | Cumple — se retiró la repetición de «conviene precisar» |
| No abusa de adjetivos evaluativos vagos | Cumple — «relevante» y «sustancial» aparecen una vez cada uno, con referente |
| No usa verbos de estilo inflado | Cumple |
| No usa estructuras decorativas | Cumple — sin «no solo… sino también» ni tríadas ornamentales |
| No sobreexplica ideas evidentes | Cumple |
| No cambia el grado de certeza | Cumple — verificado: ningún tiempo verbal ni modal fue alterado |
| Mantiene terminología consistente | Cumple |
| Conserva claridad y concisión | Cumple |
| No realiza cambios innecesarios | Cumple — ninguna oración reescrita por preferencia estilística |

---

## G. Patrones que conviene vigilar al redactar lo que falta

1. **El párrafo acumulativo.** Las tres rondas de revisión produjeron el mismo efecto: las
   correcciones se añadieron dentro del párrafo existente en lugar de abrir uno nuevo. Al
   escribir Introducción y Discusión, y al aplicar futuras correcciones, conviene comprobar la
   longitud del párrafo después de cada añadido.
2. **Siglas en primera aparición.** Aparecieron cinco sin expandir pese a dos revisiones previas.
   Al incorporar secciones nuevas, revisar la primera aparición de cada sigla en el documento
   completo, no dentro de la sección que se está escribiendo.
3. **Repetición de valores ya tabulados.** §3.3.1 recita los seis valores de la Tabla 7. Se dejó
   como está porque sostiene la lectura analítica, pero es un hábito a vigilar en Discusión,
   donde repetir cifras en lugar de interpretarlas es el error más frecuente.
4. **Cautela proporcional.** El texto actual acierta en no exagerar. El riesgo se desplaza a la
   Discusión y las Conclusiones, donde la presión por concluir empuja a sustituir «es compatible
   con» por «indica» y «no permite resolver» por «sugiere».
