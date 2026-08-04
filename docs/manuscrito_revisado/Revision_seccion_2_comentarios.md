# Revisión de la sección 2 según los comentarios del equipo

**Fecha:** 3 de agosto de 2026
**Documento:** `Manuscript_Methods_Results_EN.docx`
**Estado:** los siete comentarios están aplicados. Hay una consecuencia que conviene que vean
antes de escribir §2.5.

---

## 1. §2.1 Referencia complementaria y más reciente

Añadida **Hu et al. (2023)**, *NeuroImage* 274:120125, «Image harmonization: A review of
statistical and deep learning methods for removing batch effects and evaluation metrics for
effective harmonization». La frase queda así:

> …removing it requires harmonization procedures that carry assumptions of their own and that
> remain an active methodological problem (Yu et al., 2018; Hu et al., 2023).

El par funciona bien porque cubre dos cosas distintas: Yu et al. documentan el fenómeno en medidas
de conectividad funcional; Hu et al. revisan el estado del arte de los métodos de corrección y de
sus métricas de evaluación, que es lo que sostiene «sigue siendo un problema metodológico abierto».

**Una alternativa que descarté y por qué.** Consideré **Yamashita et al. (2019)**, *PLoS Biology*,
que aporta el dato más contundente del área: los tamaños de efecto del sesgo de medición sobre la
conectividad funcional son iguales o mayores que los de los trastornos psiquiátricos. Es un
argumento más fuerte para no agrupar, pero es de 2019 y ustedes pidieron algo más reciente. Si
prefieren el argumento sobre la actualidad, la cambio.

---

## 2. §2.2 La última línea no pertenecía a esta sección

Tenían razón. «The AAL116 time series were extracted from that filtered and smoothed functional
derivative» describe extracción de señal, no preprocesamiento. Se movió a §2.3 y quedó fusionada
con la primera oración, que ahora dice de dónde sale la señal sin repetir nada:

> The blood-oxygen-level-dependent (BOLD) signal was extracted from the filtered and smoothed
> functional derivative with the AAL116 atlas…

---

## 3. §2.3 Reescrita

### Referencia a los grupos de ROI, sin paréntesis

> The same extraction was applied to each of the ROI groups described in Section 2.4.

### Lo que dependía de la arquitectura, retirado

Este es el comentario más importante de la ronda y era correcto. La sección afirmaba cosas que no
son propiedades de la señal sino del modelo que la consume:

| Retirado de §2.3 | Por qué no iba ahí | Dónde quedó |
|---|---|---|
| «successive matrices were **stacked as channels** in fixed order» | «Canal» es un formato de entrada, no una propiedad de la conectividad | §2.5, cuando se describan las arquitecturas |
| «This is a **multichannel** representation… the architecture applies no recurrence, attention, or memory across them, so it should not be described as a sequence processed over time» | Exactamente su observación: con una LSTM o un Transformer la misma serie sí se trata como secuencia | §2.5 |
| «As a **contrast condition**, a single Pearson correlation matrix per subject was also computed… referred to as static connectivity» | Es una condición experimental, no un paso de extracción | §2.6, donde se define la segunda dimensión de sensibilidad |

§2.3 termina ahora en una descripción neutral respecto del modelo:

> …within each window the Pearson correlation matrix between all regions of the group was computed,
> so that every subject is represented by an **ordered series of connectivity matrices** covering
> the scan.

«Serie ordenada de matrices de conectividad» es un hecho del dato. Que se consuma como secuencia,
como canales o como grafo es la decisión que describirá §2.5.

**Consecuencia que deben tener presente:** el término «multichannel» desapareció del documento, así
que §3.3.2 y el pie de la Figura 2 pasaron a decir «windowed representation». **Cuando escriban
§2.5 hay que reintroducir el término allí**, o esas dos menciones quedarán sin antecedente. Dejé la
nota necesaria dentro del marcador de §2.5 para que no se pierda.

También quité «channels» del encabezado de la Tabla 3, que ahora dice «Windows per subject», y de
la Tabla 5 y §3.3.4, por la misma razón.

### Paréntesis para tablas y figuras

Eliminados en toda la sección de Método. No queda ninguno:

| Antes | Ahora |
|---|---|
| «depends on the TR of each site (Table 3)» | «**Table 3 gives** their translation into volumes, which depends on the TR of each site» |
| «relative to the reference (Table 5)» | «relative to the reference, **as Table 5 sets out**» |
| «identical for all conditions (Table 4)» | «**Table 4 lists** the training configuration, held identical for every condition» |

Las citas bibliográficas entre paréntesis se conservan: son convención y no es lo que ustedes
señalaban.

---

## 4. §2.4 Acortada de 390 a 304 palabras

Retiré dos bloques defensivos y uno que repetía información de otra sección:

| Retirado | Motivo |
|---|---|
| «They were not derived from a **performance-guided selection procedure**» | Escritura defensiva. El hecho positivo —«definidos por expertos a partir de la literatura, antes de entrenar ningún modelo»— dice lo mismo sin ponerse a la defensiva |
| «Because the groups were defined a priori, **no supervised region-selection step is repeated within each cross-validation fold**» | Es información del protocolo experimental. Quedó en §2.6, en una sola frase |
| «This avoids the bias of choosing regions by their performance… **but not the optimism** associated with the modeling decisions developed on NYU» | Doblemente defensiva, y anticipa la declaración de exposición que ya está en §2.6 |

Lo que se conserva: las cinco redes con sus citas, la función de cada sistema, la heterogeneidad
del trastorno (Schleim, 2022) y toda la estructura de anidamiento de los grupos. Las llamadas a la
Figura 1 y a la Tabla 2 quedaron integradas en el texto: «Figure 1 shows the five networks…»,
«Table 2 lists the composition of the 12-ROI group».

---

## 5. §2.6 Setup general, y separación entre lector y revisor

### El sesgo hacia una configuración

Existía, y no solo en el tono. La Tabla 4 se titulaba «**Reference** training configuration» y el
texto hablaba de «canales de entrada», que es una propiedad de la arquitectura de referencia.
Ahora la sección abre declarando su alcance:

> The protocol in this section applies to **every run of the study**: the same partitioning scheme,
> training configuration, and seed were used for each site, ROI group, and sensitivity condition.

El título de la Tabla 4 pasó a «Training configuration, held identical across the four sites, the
four ROI groups, and all sensitivity conditions», y el párrafo que la acompaña explica ahora qué es
lo único que varía entre corridas y por qué no es una elección libre:

> Two quantities still vary between runs, and neither is a free choice: the number of windows per
> subject, set by the TR and the series length of each site, and the number of model parameters,
> set by the ROI group and that number of windows.

La configuración de referencia sigue existiendo, porque un análisis de sensibilidad necesita un
punto de comparación, pero ahora se presenta como tal y no como el sujeto de la sección.

### Lector frente a revisor

Quedaron separados por marca tipográfica, no por tono:

- **Para el lector, en la ilación lógica:** protocolo de validación cruzada, ponderación de clase,
  configuración de entrenamiento, las cuatro dimensiones de sensibilidad y la restricción a NYU y
  Peking. Todo en texto corrido, sin justificarse ante nadie.
- **Para el revisor, con entradilla en negrita:** un solo bloque, «*Exposure to hyperparameter
  tuning*», que declara que la configuración se eligió viendo resultados en NYU con 12 ROIs. Es una
  declaración de transparencia exigida por TRIPOD+AI y por eso se queda en el artículo, pero está
  visualmente marcada como lo que es. El detalle de las diez configuraciones comparadas se fue a la
  Tabla S2.

---

## 6. §2.7 Justificación de la elección del AUC

Reescrita y dividida en dos párrafos, con referencia verificada:

> AUC was chosen as the primary metric for two reasons. It summarizes the receiver operating
> characteristic curve over all decision thresholds, so it does not require committing to an
> operating point that this study has no clinical basis for choosing. And it is not distorted by
> the class balance of each site, which ranges from 87:90 in NYU to 109:74 in Peking. Reviews of
> single-subject prediction of brain disorders in neuroimaging identify overall accuracy reported
> without regard to unequal class sizes as a recurrent source of uninformative results, because
> assigning every subject to the majority class can already produce a high value
> (Arbabshirani et al., 2017).

**La referencia es Arbabshirani, Plis, Sui & Calhoun (2017)**, *NeuroImage* 145:137–165, «Single
subject prediction of brain disorders in neuroimaging: Promises and pitfalls». Es la mejor opción
disponible para este proyecto por tres motivos: revisa la predicción a nivel de sujeto en los
trastornos que nos ocupan, TDAH incluido; es la referencia que un revisor de este campo reconoce; y
sostiene textualmente lo que le atribuimos. Verifiqué el pasaje en el propio artículo antes de
citarlo:

> «Some of the studies in this review just reported the overall accuracy, which can be very
> uninformative especially when classes have unequal sample sizes… Reporting 80% accuracy is
> completely uninformative since the classification of all subjects as healthy could also result in
> 80%… The ROC curve is the plot of sensitivity against 1-specificity by changing the discrimination
> threshold and therefore provides a complete picture of classifier's performance. The ROC curve is
> usually summarized by the area under the curve (AUC).»

La misma fuente respalda otras dos decisiones que ya estaban tomadas y que ahora quedan citadas de
paso: reportar intervalos de confianza en lugar de comparar el punto contra el azar, y conservar la
exactitud simple solo como métrica de auditoría.

---

## 7. Efecto sobre la extensión

| | Antes de esta ronda | Ahora |
|---|---:|---:|
| Texto corrido | 3.540 | **3.601** |
| Contra la línea base de 4.046 | −12,5% | **−11,0%** |

Subió 61 palabras, y la razón es que esta ronda pedía dos cosas que suman texto: la justificación
del AUC con su referencia, unas 90 palabras, y la declaración de alcance de §2.6, unas 30. §2.4
devolvió 86 y §2.3 otras 30, pero no alcanzan a compensarlas.

No forcé el porcentaje recortando en otro sitio, porque lo que se añadió responde a un comentario
suyo y lo que quedaría por cortar ya no es grasa. Las tres opciones para cerrar la diferencia
siguen siendo las del documento anterior, y la primera —llevar §2.8 Ética a las declaraciones
finales— sigue siendo gratis y da 82 palabras.

**La escritura mejoró en el camino:**

| Indicador | Antes | Ahora |
|---|---:|---:|
| Párrafos de más de 160 palabras | 1 | **0** |
| Párrafo más largo | 182 | **160** |
| Oraciones de más de 40 palabras | 7 | **5** |
| Oraciones de más de 50 palabras | 0 | 0 |
| Longitud media de oración | 23,7 | **23,4** |

Verifiqué además que las dieciséis cifras clave siguen presentes, que ninguna referencia cruzada
quedó rota y que no hay pies de tabla ni de figura duplicados.
