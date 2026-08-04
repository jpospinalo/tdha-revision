# Evaluación de la reestructuración de secciones propuesta

**Fecha:** 3 de agosto de 2026
**Estado:** evaluación. No se aplicó ningún cambio.

---

## 1. Resumen

De las nueve indicaciones, **seis se aceptan sin reservas**, **una requiere una decisión previa
del equipo** y **cuatro observaciones deben resolverse antes de aplicar**, porque la estructura
propuesta pierde elementos o los ubica en un lugar que genera un problema de coherencia. El
propio encargo autoriza a apartarse de la estructura original cuando eso ocurra, siempre que se
justifique; los cuatro casos siguientes son de ese tipo.

---

## 2. Aceptado sin reservas

**Retirar los objetivos de Métodos.** Correcto y convencional: los objetivos pertenecen a la
Introducción. El párrafo se retira de §2.1 y no se reubica dentro de Métodos.

**Conservar los nombres del original.** Es la decisión adecuada. Los revisores ya leyeron esa
estructura, y cambiarla sin necesidad les obliga a reorientarse. Las excepciones se argumentan en
§4 de este documento.

**Crear §2.7 «Experimental and evaluation setup».** Es la mejor idea del conjunto. El original no
tenía dónde alojar los parámetros comunes a todas las arquitecturas, y esa ausencia obligaba a
repartir la validación cruzada entre secciones. Con esta subsección, el protocolo de validación,
las particiones, la semilla y el balanceo de clases quedan en un solo lugar.

**Renombrar §2.6 y dejarla en blanco por ahora.** Coherente con que la versión nueva compara dos
arquitecturas. El nombre debería ser **plural**: «Deep learning architectures» o «Model
architectures». El singular del original —«LSTM Architecture»— ya no describe el contenido.

**Llevar «Functional Networks Involved in ADHD» a Métodos como §2.4.** Correcto, y coincide con
lo que ya se había aplicado. En el original estaba en Resultados, donde no reportaba ningún
hallazgo. Al ser los paneles definidos a priori, la justificación anatómica es descripción del
instrumento y pertenece a Métodos.

**Mantener «Ethical considerations» como subsección numerada.** El original la tenía así. Se
retira la sugerencia previa de moverla a las declaraciones finales: si el equipo prefiere
conservar la estructura del original, esta es una diferencia de convención editorial y no un
defecto.

---

## 3. Requiere decisión previa del equipo

### 3.1 Los tres párrafos de §2.1: distinguir «retirar» de «reubicar»

La indicación dice que los tres párrafos son innecesarios «para esta parte del trabajo». La
interpretación cambia el resultado, y dos de los tres contienen material que **el propio equipo
pidió añadir en rondas anteriores**:

| Párrafo | Contenido | Estado |
|---|---|---|
| 1.º | Objetivos y diseño | **Retirar.** Los objetivos van a la Introducción; el diseño cabe en una frase de §2.2 |
| 2.º | Versiones de software, entorno y determinismo | **Reubicar, no retirar.** Es la corrección de la 2.ª ronda sobre CPU/GPU y `deterministic=False`, y el ítem de configuración experimental de TRIPOD+AI. Su lugar natural es §2.7 |
| 3.º | Disponibilidad de datos y código | **Reubicar, no retirar.** Contiene las condiciones de acceso corregidas en la 4.ª ronda —uso no comercial y registro NITRC— y la cita del commit. Su lugar es una declaración final o §2.7 |

Retirar los párrafos segundo y tercero deshace correcciones aceptadas y verificadas. Reubicarlos
cumple la indicación sin perderlas.

---

## 4. Observaciones que deben resolverse antes de aplicar

### 4.1 «Statistical Analysis» como §2.1 es anómalo

En el manuscrito original, esa subsección **no contenía análisis estadístico**. Su primer
párrafo dice: «An empirical analytical approach with explanatory scope was employed… Development
was carried out in the Python 3.13 programming language…». Es decir, el título era un nombre
equivocado para «diseño y herramientas».

Si ahora se llena con el análisis estadístico real —métricas, bootstrap, reglas de inferencia—,
queda situado **antes** de que se hayan presentado los participantes, el preprocesamiento y el
modelo. La convención, tanto en TRIPOD+AI como en neuroimagen computacional, sitúa los métodos
analíticos al final de Métodos, porque describen qué se hace con los datos ya definidos.

**Tres salidas posibles**, en orden de preferencia:

1. Renombrar §2.1 a **«Study design»**, con un párrafo breve, y llevar el análisis estadístico a
   §2.7 junto con el protocolo experimental. Conserva el orden del original y resuelve el
   contenido.
2. Conservar «Statistical Analysis» como título pero moverla al final, como §2.8. Cambia el
   orden del original.
3. Conservar título y posición, y aceptar que la sección anticipa un análisis sobre datos que el
   lector aún no conoce.

### 4.2 El contraste principal 12 frente a 116 pierde su sección

La estructura propuesta de Resultados no incluye el contraste principal. Quedaría absorbido en
§3.2.1 «Grupo de ROIs», como una más de las cinco comparaciones de sensibilidad.

Es un problema de fondo, no de forma. Ese contraste es el **análisis primario preespecificado**
del plan congelado, y el resto son análisis de sensibilidad declarados como tales. Presentarlos
al mismo nivel invierte la jerarquía del plan y debilita la respuesta a los revisores, que
distingue explícitamente lo primario de lo exploratorio.

**Sugerencia:** conservar una subsección propia —§3.2 «Primary comparison between ROI panels»—
y renumerar la sensibilidad como §3.3.

### 4.3 «Convergence and model behavior» no es un análisis de sensibilidad

En la estructura propuesta aparece como §3.2.5, es decir, como la quinta dimensión de
sensibilidad. No lo es: las otras cuatro varían una decisión de diseño y miden el efecto sobre el
AUC; esta reporta curvas de entrenamiento, exactitud por época y capacidad discriminativa de la
configuración de referencia. Es un diagnóstico del modelo.

Además es la sección que responde a R1.4 y a buena parte de R2.11. Si un revisor la busca bajo
«sensitivity analyses», no la encuentra donde espera.

**Sugerencia:** dejarla como subsección de primer nivel, §3.4.

### 4.4 «Sensitivity analyses» desaparece de Métodos

La lista propuesta de Métodos no tiene dónde describir el **diseño** de los análisis de
sensibilidad: las cuatro dimensiones, la restricción a NYU y Peking, las condiciones de ventana y
el apartado «Dimensiones no evaluadas». Resultados sí los reporta, en §3.2.

**Sugerencia:** alojarlos en §2.7 «Experimental and evaluation setup», que por su definición
—parámetros y métodos comunes a todas las arquitecturas— es el lugar natural.

---

## 5. Dos detalles menores

**Numeración.** La propuesta pasa de §3.2.2 a §3.3.3, §3.3.4 y §3.3.5. Presumo que se quiso
escribir §3.2.3, §3.2.4 y §3.2.5.

**Terminología: «grupo de ROI» frente a «panel».** El original usaba «ROI group» cinco veces y
nunca «panel». La versión actual usa «ROI panel» veinticuatro veces y nunca «group». La
estructura propuesta vuelve a «grupo de ROI».

Hay que elegir uno y aplicarlo en todo el documento, incluidos títulos de tabla, pies de figura y
la carta de respuesta. Mi recomendación es **panel**, por tres razones: designa un conjunto
curado y no una partición arbitraria, ya está en el título del artículo, y evita la colisión con
«group» en su sentido de grupo de participantes, que aparece en el mismo texto al hablar de
control y TDAH. Si el equipo prefiere «group» por continuidad con lo que leyeron los revisores,
la sustitución es mecánica pero debe hacerse completa.

---

## 6. Estructura resultante si se aceptan las sugerencias

```
2. Method
   2.1  Study design                          ← renombrada; sin objetivos
   2.2  Participants
   2.3  Preprocessing
   2.4  Functional Networks Involved in ADHD  ← incluye la definición de los paneles
   2.5  BOLD Signal Extraction and Functional Connectivity Construction
   2.6  Deep learning architectures           ← en blanco por ahora
   2.7  Experimental and evaluation setup     ← validación cruzada, análisis estadístico,
                                                 software y entorno, diseño de sensibilidad
   2.8  Ethical considerations

3. Results
   3.1  Performance by ROI panel and site
   3.2  Primary comparison between ROI panels
   3.3  Sensitivity analyses
        3.3.1  ROI panel
        3.3.2  Signal representation
        3.3.3  Model architecture
        3.3.4  Window length and step
   3.4  Convergence and model behavior
```

Ocho subsecciones en Métodos, igual que la propuesta. Los cambios respecto de ella son: §2.1
renombrada, el contraste principal recupera sección propia, la convergencia sale de sensibilidad,
y §2.7 absorbe el software, el análisis estadístico y el diseño de la sensibilidad.

---

## 7. Qué falta decidir antes de que yo aplique

1. Si los párrafos segundo y tercero de §2.1 se **reubican** o se **retiran** (§3.1).
2. Cuál de las tres salidas para «Statistical Analysis» se prefiere (§4.1).
3. Si el contraste principal conserva sección propia (§4.2).
4. «Panel» o «grupo de ROI» (§5).

Las demás sugerencias no requieren decisión y pueden aplicarse en cuanto se resuelvan estas.
