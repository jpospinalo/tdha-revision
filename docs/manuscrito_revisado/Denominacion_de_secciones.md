# Denominación de secciones y subsecciones: revisión contra la convención del campo

**Fecha:** 3 de agosto de 2026
**Alcance:** solo los títulos. El contenido de cada sección queda como está, según acordó el equipo.
**Referencias consultadas:** TRIPOD+AI, la norma de reporte vigente para modelos predictivos, y
la convención de secciones en artículos de clasificación con rs-fMRI y aprendizaje profundo.

---

## 1. Marco de referencia

Dos convenciones concurren sobre este manuscrito y conviene distinguirlas, porque no coinciden
en todo.

**TRIPOD+AI** organiza los Métodos por función de reporte: fuente de datos, participantes,
resultado, predictores, tamaño muestral, datos faltantes, métodos analíticos, desbalance de
clases, y evaluación del modelo. Es la norma aplicable porque el estudio desarrolla y evalúa un
modelo predictivo.

**La convención de neuroimagen computacional** organiza los Métodos por etapa del pipeline:
participantes, adquisición, preprocesamiento de imagen, extracción de características o
conectividad, arquitectura, y entrenamiento y evaluación.

El manuscrito actual sigue mayoritariamente la segunda, que es la adecuada para el público
lector. Las observaciones siguientes buscan alinear los títulos con esa convención sin
reorganizar el contenido.

---

## 2. Métodos

| Actual | Observación | Sugerencia |
|---|---|---|
| 2.1 Design, objective, and tools | **El título más débil del conjunto.** Agrupa tres cosas sin relación: el diseño del estudio, el objetivo y el software. El objetivo pertenece a la Introducción, no a Métodos, y las versiones de software no suelen titular una subsección: van al final de Métodos o a la declaración de disponibilidad. | **Study design and data source**, moviendo el software a un párrafo de implementación dentro de la misma subsección o al apartado de disponibilidad |
| 2.2 Participants | Convencional en ambas normas. Sin observaciones | *Sin cambio* |
| 2.3 Preprocessing | Correcto pero escueto. La convención de neuroimagen distingue el preprocesamiento de imagen del de datos | **Image preprocessing** |
| 2.4 Anatomical panels of regions of interest | Describe lo que TRIPOD+AI llama predictores y la neuroimagen llama ROIs. Largo y con «of» duplicado | **Anatomical ROI panels**, que además coincide con el título del artículo |
| 2.5 Functional connectivity construction | «Construction» es poco habitual para conectividad; el verbo convencional es *estimate* | **Functional connectivity estimation** |
| 2.6 BrainNetCNN architecture | Convencional. Nombrar el modelo en el título es frecuente e informativo | *Sin cambio* |
| 2.7 Validation protocol and statistical analysis | Reúne dos bloques que TRIPOD+AI separa —métodos analíticos y evaluación del modelo— y es una de las subsecciones más largas | Dividir en **Cross-validation protocol** y **Statistical analysis**, o renombrar a **Model training and evaluation** |
| 2.8 Sensitivity analyses | Convencional. Que se repita como título en Resultados es correcto: Métodos describe el diseño, Resultados reporta el desenlace | *Sin cambio* |
| 2.9 Ethical considerations | En revistas de neuroimagen y aprendizaje automático esto no suele ser una subsección numerada de Métodos. Va en las declaraciones finales del artículo, junto a conflicto de interés y financiación | Mover a **Ethics statement**, fuera de la numeración de Métodos |

---

## 3. Resultados

| Actual | Observación | Sugerencia |
|---|---|---|
| 3.1 Classification performance by panel and site | Convencional y preciso | *Sin cambio* |
| 3.2 Primary contrast between ROI panels | **Ver §4. Riesgo terminológico específico del campo** | **Primary comparison between ROI panels** |
| 3.3 Sensitivity analyses | Convencional | *Sin cambio* |
| 3.3.1 Panel dimensionality | Claro | *Sin cambio* |
| 3.3.2 Connectivity representation | Claro | *Sin cambio* |
| 3.3.3 Architecture | Demasiado escueto como título de subsección; no dice de qué | **Model architecture** |
| 3.3.4 Window length and step | Claro | *Sin cambio* |
| 3.4 Convergence and model behavior | «Model behavior» es vago y no anticipa qué se reporta: convergencia, exactitud y capacidad discriminativa | **Training convergence and discrimination** |

---

## 4. El hallazgo más importante: «contrast»

El manuscrito usa **contrast** treinta y una veces para designar la comparación entre dos
configuraciones. Es el uso estándar en estadística, y es el que traía la versión en español.

En fMRI, sin embargo, **contrast** tiene un significado técnico distinto y muy establecido: una
combinación lineal de columnas de la matriz de diseño del modelo lineal general, evaluada
mediante un estadístico t. Un lector de neuroimagen encuentra «primary contrast» y su primera
lectura será la del GLM, no la de una comparación entre condiciones experimentales.

El riesgo es mayor precisamente porque este manuscrito **retiró el ANOVA** de la versión
anterior: un revisor podría leer «contrast» como el vestigio de un análisis inferencial que
declaramos haber eliminado.

**Recomendación:** sustituir *contrast* por *comparison* en los títulos y en el texto corrido,
y conservarlo únicamente donde la frase ya especifica que se trata de una diferencia de AUC
—por ejemplo en los pies de tabla, donde aparece junto a «Δ = condition − reference»—.

Es la única sugerencia de esta lista que afecta al texto y no solo a los títulos, y por eso
conviene decidirla antes de aplicar las demás.

---

## 5. Estructura resultante propuesta

```
2. Methods
   2.1  Study design and data source
   2.2  Participants
   2.3  Image preprocessing
   2.4  Anatomical ROI panels
   2.5  Functional connectivity estimation
   2.6  BrainNetCNN architecture
   2.7  Cross-validation protocol
   2.8  Statistical analysis
   2.9  Sensitivity analyses

3. Results
   3.1  Classification performance by panel and site
   3.2  Primary comparison between ROI panels
   3.3  Sensitivity analyses
        3.3.1  Panel dimensionality
        3.3.2  Connectivity representation
        3.3.3  Model architecture
        3.3.4  Window length and step
   3.4  Training convergence and discrimination

Declaraciones finales
   Ethics statement
   Data and code availability
   Conflict of interest
   Funding
```

La numeración de Métodos se mantiene en nueve subsecciones: entra una nueva por la división de
la antigua 2.7 y sale una por el traslado de Ética a las declaraciones.

---

## 6. Lo que no recomiendo cambiar

- **«Methods» en plural.** Es la forma dominante en revistas de neuroimagen, aunque el
  manuscrito original usaba «Method» en singular. Si la revista objetivo exige «Materials and
  Methods», el cambio es trivial y conviene consultarlo en las normas de la revista.
- **Repetir «Sensitivity analyses» en Métodos y en Resultados.** Es la práctica correcta y no
  una redundancia: una sección declara el diseño y la otra reporta el desenlace.
- **Nombrar BrainNetCNN en el título de la subsección.** Alternativamente podría titularse
  «Model architecture», pero nombrar el modelo es frecuente y ayuda a la recuperación en bases
  de datos.
- **Los títulos de 3.3.1, 3.3.2 y 3.3.4.** Son precisos y no requieren ajuste.

---

## 7. Nota sobre los comentarios pendientes

Los comentarios generales sobre el estado actual **no llegaron con el mensaje**; solo se recibió
la instrucción. Este documento cubre únicamente la consulta sobre denominación de secciones, que
era autocontenida. La evaluación del resto queda pendiente de recibir el texto.
