# Respuesta a la revisión del equipo sobre el manuscrito y la carta

**Documento revisado:** «Revisión del manuscrito y de la respuesta a los revisores»
**Fecha:** 2 de agosto de 2026
**Alcance:** qué se acepta, qué se acepta con modificación, qué se rechaza y con qué argumento.

---

## Resumen

De los comentarios recibidos, **se aceptan 24 sin reservas**, **4 con modificación** y **2 se
rechazan**. Cinco afirmaciones factuales del equipo se verificaron contra el repositorio o la
fuente primaria antes de aceptarlas: **las cinco resultaron correctas**, y tres de ellas
señalaban errores reales del manuscrito que ya están corregidos.

La revisión es de alta calidad. El hallazgo más valioso no es ninguna de las correcciones de
redacción, sino la tabla de conteos de parámetros de §3.6: identifica un confundido que
afecta a la pregunta principal del estudio y que el manuscrito no declaraba.

---

## 0. Lo que necesitamos del equipo en esta ronda

### 0.1 Una decisión

**Estructura de la Tabla 6.** Es el único punto que no se resolvió y que no puede resolverse
sin el equipo. La propuesta de compactarla coincide con el plan congelado, pero contradice la
instrucción previa de replicar el formato de tres filas del manuscrito original. Se aplicó el
cambio de *qué métricas* y se dejó intacta la estructura de *qué filas*. El detalle y la
recomendación están en §3.1 de este documento.

### 0.2 Cuatro bloques pendientes, que no son omisiones

Se enumeran para que no vuelvan a reportarse como hallazgos:

1. **Auditoría de las 26 corridas** (§3.9 de la revisión). Bloquea la redacción de R2.6 en
   pasado y es el único bloque que condiciona a otros.
2. **Recálculo del bootstrap a 10 000 remuestreos.** Se resuelve dentro de la auditoría
   anterior, sin trabajo adicional ni reentrenamiento.
3. **Descripción del diagnóstico en Participantes.** Requiere acceso a la documentación
   clínica de ADHD-200; es la única omisión de fondo que queda.
4. **Portada, lista de referencias, Resumen, Introducción, Discusión y Conclusiones.** Nunca
   se escribieron: el documento entregado cubre solo Métodos y Resultados.

### 0.3 Dos comentarios que no se aceptaron

Para que se conozcan de antemano y no se descubran leyendo:

- el encuadre del veredicto «no enviar todavía» (§4.1 de este documento);
- la condición de recalcular con 10 000 remuestreos antes del envío (§4.2). La acción se
  acepta; lo que se rechaza es que sea criterio de aprobación.

---

## 1. Verificación de las afirmaciones factuales

Antes de aceptar nada se comprobó lo comprobable.

| Afirmación del equipo | Verificación | Resultado |
|---|---|---|
| El plan define como secundarias exactitud balanceada, F1-macro, sensibilidad y especificidad; la exactitud es métrica de auditoría | `analysis_config.json`: `secondary_metrics: [balanced_accuracy, f1_macro, sensitivity, specificity]`, `audit_metrics: [accuracy]` | **Correcta.** El manuscrito usaba exactitud y precisión; la precisión no figura en el plan |
| FP32 y semilla no garantizan reproducibilidad con `deterministic=False` | `deterministic: False` en las 26 corridas | **Correcta.** El manuscrito afirmaba invariancia |
| BrainNetCNN no fue diseñada para muestras de miles de sujetos | Kawahara et al. (2017): 168 escaneos de ~115 participantes prematuros | **Correcta.** El error venía del *docstring* de `src/kerasmodels/brainnetcnn.py` |
| Los parámetros cambian entre paneles según la tabla aportada | Modelos instanciados con el constructor real | **Correcta en los 16 valores** |
| Las cifras 54.0 % y 62.4 % no corresponden al mismo conjunto | `run_experiment.py:1001`: `train_metrics = evaluate(model, Xf_fold[outer_train], ...)`; `history.csv` registra el subconjunto de ajuste en modo entrenamiento | **Correcta.** Difieren en conjunto y en modo |

---

## 2. Aceptado sin reservas y ya aplicado

Estas correcciones ya están incorporadas al manuscrito y a la figura.

**§3.1 — Métricas conformes al plan congelado.** La Tabla 6 sustituye exactitud y precisión
por exactitud balanceada y F1-macro, conserva sensibilidad y especificidad, y mantiene el AUC.
La exactitud pasa a métrica de auditoría en el suplemento. Este es el comentario más importante
del bloque: el manuscrito reportaba una métrica que el plan no contempla (precisión) y omitía
dos que sí (exactitud balanceada y F1-macro).

**§3.3 — Terminología de conjuntos.** Las filas ahora se llaman «Ext. de entrenamiento» y
«Ext. de validación», y el pie define la primera como ajuste más validación interna evaluada
en modo inferencia. Se retiró la afirmación del 54.0 % → 62.4 % «sobre el mismo conjunto sin
dropout» y se sustituyó por una advertencia de que las curvas de la Figura 3 y las filas de la
Tabla 6 no son directamente comparables, mencionando además la normalización por lotes.

**§3.4 — Determinismo.** Se adoptó la redacción propuesta, prácticamente literal.

**§3.5 — BrainNetCNN.** Se retiró la afirmación sobre miles de sujetos y la de sobreajuste
necesario. *Acción adicional recomendada:* corregir también el *docstring* de
`src/kerasmodels/brainnetcnn.py`, que es el origen del error y sigue afirmándolo.

**§3.6 — Capacidad entre paneles.** Aceptado íntegramente. El manuscrito ahora declara que
con 116 ROIs el modelo alcanza entre 16 817 y 41 873 parámetros, unas nueve veces más que con
12, y que mantener fijos los hiperparámetros no mantiene fijo el número de parámetros. La
comparación entre paneles se presenta explícitamente como comparación entre pipelines
completos. Los 16 conteos van al suplemento.

**§3.8 — Exposición de NYU.** Se declara ahora que las estimaciones de NYU están expuestas en
distinto grado según el panel, y que los demás paneles de NYU tampoco son completamente ajenos
al desarrollo.

**§4.3 — Ventanas.** Se adoptó la formulación conservadora: «pueden producir estimaciones más
variables, aunque no son formalmente inválidas por sí mismas».

**§4.4 — Ponderación de clases.** Añadida a §2.7, con la precisión de que los pesos se calculan
solo sobre los datos de ajuste de cada pliegue y de que la diferencia impide tratar los cuatro
procedimientos como intercambiables.

**§5.1, §5.3, §5.5, §5.6, §5.7 — Correcciones de interpretación.** Aplicadas todas: «separada»
en lugar de «independiente»; «magnitud muy distinta» en lugar de «resultados opuestos»; OHSU
descrito como impreciso y no como nulo; «cuatro sitios de adquisición»; «alcance comparativo».

**§6.3 — Figura de convergencia.** La línea de azar dejó de ser discontinua, que era el estilo
de la curva de validación. Se añadió el conteo de pliegues activos por época sobre el eje
derecho, y la leyenda distingue «Ajuste (modo entrenamiento)» de «Validación interna
(inferencia)». El resultado es informativo: en NeuroIMAGE los pliegues activos caen de 50 a 44
a lo largo del entrenamiento, mientras que en NYU se mantienen en 50.

**§3.7 — Heterogeneidad.** Se retira la expresión de la respuesta R1.2. El manuscrito no la
usaba; la carta sí.

---

## 3. Aceptado con modificación

### 3.1 Simplificar la Tabla 6 (§3.2) — **decisión pendiente del equipo**

El equipo propone dejar en el texto principal una tabla compacta solo de desempeño fuera de
pliegue y trasladar entrenamiento y brechas al suplemento. La propuesta **coincide con el plan
congelado** (§10.1 y §10.2), que sitúa la tabla de las 16 corridas en el texto principal y las
brechas completas en el suplemento.

Sin embargo, la estructura actual de tres filas responde a una instrucción explícita previa:
replicar el formato de la Tabla 2 del manuscrito original, que reportaba entrenamiento,
validación y sobreajuste. Esa instrucción se dio antes de contrastar el formato contra el plan.

**No se aplica de forma unilateral porque son dos decisiones distintas y solo una está en
conflicto.** Se aplicó la corrección de *qué métricas* (§3.1, indiscutible por conformidad con
el plan) y se dejó intacta la estructura de *qué filas*, que es la que el equipo quiere cambiar.
La tabla actual tiene 48 filas y ocupa dos páginas; la versión compacta tendría 16.

Recomendación: seguir el plan y compactar. Requiere confirmación de quien fijó el formato
original.

### 3.2 Tono de la carta (§7)

Se aceptan las correcciones de R1.3 («atendimos la alternativa propuesta») y de R2.2 (reconocer
que el contraste no reproduce el factorial SFC × LSTM solicitado).

Sobre «moderar expresiones adversariales como *más exigente, no menos*»: se acepta suavizar la
formulación, **pero no eliminar la afirmación**. R2.13 es el único comentario que se declina, y
declinar sin justificar por qué el sustituto es metodológicamente superior debilita la respuesta
justo donde más apoyo necesita. El bootstrap pareado a nivel de sujeto sí impone condiciones que
el ANOVA sobre pliegues no cumple. Se reformulará como afirmación técnica y no comparativa.

### 3.3 Alcance de los datos fenotípicos (§4.1)

Se aceptan sin reserva: flujo de inclusión y exclusión por sitio, criterio exacto de
`QC_Rest_1`, construcción de la etiqueta binaria, tratamiento de subtipos, edad y sexo por
clase, y referencias de adquisición y ética. También se acepta no atribuir a `QC_Rest_1` la
detección de información clínica incompleta sin evidencia documental: esa afirmación se
heredó del manuscrito anterior y no se verificó.

Sobre IQ, medicación y comorbilidades: se aceptan **al suplemento y como caracterización de la
muestra**, no al texto principal. Ninguna de esas variables entra en el análisis, y presentarlas
en el cuerpo del artículo junto a las métricas sugeriría un control que no se ejerció. Como
limitación en la Discusión sí son pertinentes.

### 3.4 Figuras del texto principal (§6.1)

El equipo enumera cuatro elementos: perfiles de AUC, forest plot del contraste 12−116, tabla
compacta de contrastes nuevos y párrafo de brechas. Se aceptan los cuatro contenidos, pero
**no como cuatro objetos separados**: la Figura 2 actual ya los cubre en dos paneles —perfiles
en 2a, contrastes con su intervalo en 2b, incluido el 12−116—. Separarlos consumiría dos de las
tres figuras disponibles, y la restricción de espacio fue el motivo declarado para fusionarlas.

---

## 4. Rechazado

### 4.1 «No enviar todavía» como veredicto sobre este documento

**Se rechaza el encuadre, no la conclusión.** El documento entregado contiene únicamente
Métodos y Resultados, y lo declara en su primera línea. No existían Resumen, Introducción,
Discusión, Conclusiones, lista de referencias ni portada, de modo que la posibilidad de
enviarlo nunca estuvo sobre la mesa.

Esto importa porque el veredicto puede leerse como si el trabajo experimental estuviera en
duda, y no lo está: el propio documento del equipo concluye en §11 que la evidencia es
suficiente y que no hay razón para reabrir el entrenamiento. La formulación precisa es que
Métodos y Resultados están listos para validación de fondo, y que faltan las secciones que
nunca se escribieron.

### 4.2 Recalcular los contrastes con 10 000 remuestreos como condición de envío (§4.5)

**Se acepta la acción; se rechaza que sea condición de aprobación.**

El equipo ofrece la alternativa correcta —«o demostrar que los resultados con 2000 son
estables»—, y esa es la vía razonable. El bootstrap percentil con 2000 remuestreos tiene un
error de Monte Carlo en los límites del intervalo del orden de la tercera cifra decimal. Las
conclusiones del manuscrito no dependen de esa cifra: ningún contraste queda cerca de un umbral
de decisión, entre otras cosas porque el estudio no define ninguno. El caso más ajustado es
`static` en NYU, con intervalo [−0.082, +0.001], y ahí la conclusión que se reporta es
precisamente que el intervalo incluye el cero por muy poco margen.

Se recalculará a 10 000 por homogeneidad con los contrastes de dimensionalidad, que ya lo
usan. Pero condicionar el envío a ese recálculo confunde precisión numérica con validez, y
ninguna cifra publicada cambiará más allá de la tercera decimal.

---

## 5. Comentarios que ya estaban resueltos

Se señalan para evitar trabajo redundante.

| Comentario | Estado previo |
|---|---|
| §5.8 no usar equivalencia, no inferioridad, superioridad, biomarcador ni validación externa | Ya cumplido; verificado por búsqueda de términos |
| §5.4 dos variaciones de ventana no demuestran robustez general | Ya redactado así en §3.3.4 |
| §5.2 los paneles a priori no eliminan todo sesgo de selección | Ya declarado en §2.6 |
| §2.3 no añadir campaña de capacidad igualada | Coincide con la campaña cerrada del plan |
| Cierre de la campaña experimental | Decisión D13 del plan, vigente |
| Ausencia de efecto combinado entre sitios | Ya implementado |

---

## 6. Trabajo pendiente, en orden

**Bloque 1 — auditoría (§3.9).** Es el único bloque que bloquea la redacción de R2.6 en pasado.
Requiere extender `run_manifest.csv` a las 26 corridas y ejecutar el pipeline oficial, que ya
verifica artefactos, conteos, particiones y reproducción de métricas. Es la vía que además
resuelve el recálculo a 10 000 remuestreos sin trabajo adicional.

**Bloque 2 — participantes (§4.1).** Necesita a alguien con acceso a la documentación clínica
de ADHD-200. Es la única omisión de fondo que queda.

**Bloque 3 — preprocesamiento (§4.2).** Verificar la descripción de ATHENA contra la
documentación de los derivados. Nota: el manuscrito describe dos suavizados distintos —uno
sobre los mapas anatómicos para enmascarar ROIs y otro sobre las imágenes funcionales—, que no
son una duplicación sino dos pasos del pipeline. Lo que sí falta es confirmar que ambos
corresponden a lo que ATHENA documenta.

**Bloque 4 — suplemento (§6.2).** Diez productos, ninguno bloqueante para la validación de
fondo.

**Bloque 5 — decisión de la Tabla 6.** Ver §3.1 de este documento.
