# Evaluación del manuscrito según la rúbrica v3.5 — versión posterior a las dos rondas de revisión

**Documento evaluado:** `Manuscrito_Metodos_Resultados.docx` (secciones 2 y 3), 2 de agosto de 2026
**Modo:** diagnóstico estándar
**Evaluación anterior:** `Evaluacion_rubrica_v3.5.md`, 1 de agosto de 2026

---

## 1. Diagnóstico ejecutivo

El manuscrito mejoró en trazabilidad y precisión pero **el puntaje apenas se movió: de 3.60 a
3.64 sobre 4**. El motivo es informativo: las dos rondas de revisión corrigieron criterios que
ya puntuaban alto o que la rúbrica no discrimina, mientras que lo que deprime el puntaje
—ausencia de ground truth, de portada y de referencias— sigue igual.

La fortaleza sigue siendo la trazabilidad metodológica: el dominio D mantiene 98 % de cobertura
y ahora incluye la especificación completa del bootstrap, las versiones exactas del entorno, la
salvedad de determinismo y el cambio de capacidad entre paneles.

La debilidad principal no cambió: **el diagnóstico de TDAH, que es la variable de resultado,
sigue sin describirse**. Es el único cero de fondo del informe.

Se detectó además **una regresión introducida en esta ronda**: los párrafos de más de 160
palabras pasaron de 6 a 10, y el que especifica el bootstrap alcanza 292 palabras.

**Confianza: media-baja (cobertura global 63 %)**, por la ausencia deliberada de Resumen,
Introducción, Discusión y Conclusiones.

---

## 2. Clasificación inicial

Sin cambios respecto de la evaluación anterior, salvo:

| Elemento | Estado |
|---|---|
| Tablas | 7 (la Tabla 6 pasó de 48 a 16 filas) |
| Figuras | 3 (la Figura 1 sigue siendo un marcador de posición) |
| Referencias | 14 citas en texto; **sigue sin lista de referencias** |
| Declaraciones editoriales | Ética ✓, conflicto de interés ✓, disponibilidad ✓ con mención de commit; financiación ✗, contribuciones ✗, autoría ✗ |

---

## 3. Fallas críticas

| Grupo | Falla | Estado | Tipo | Evidencia | Acción |
|---|---|---|---|---|---|
| 8.3 | **22.** Referencias no verificables | Sí | Corregible | 14 citas en texto, sin lista de referencias | Incorporar la lista |
| 8.3 | **23.** Marcas internas de borrador | Sí | Corregible | Nota «Versión candidata…» y bloque «[ Figura 1 — reservada… ]» | Retirar e insertar la figura |
| 8.3 | **24.** Autoría y financiación omitidas | Sí (parcial) | Corregible | Conflicto de interés declarado; sin autores, afiliación ni financiación | Añadir portada |
| 8.1 y 8.2 | 1–18 | No | — | Cadena objetivo–método–resultado completa; fuga tratada; comparaciones con referencia explícita | — |

Las tres son las mismas de la evaluación anterior y siguen derivando del alcance parcial del
documento. Ninguna falla científica ni metodológica crítica está presente.

---

## 4. Resumen cuantitativo

| Dominio | Peso | Aplicables | Evaluados | No verif. | Puntaje | Anterior | Confianza |
|---|---:|---:|---:|---:|---:|---:|---|
| A. Problema, brecha, contribución | 15 % | 20 | 7 | 13 | 3.57 | 3.43 | Baja |
| B. Estructura científica | 15 % | 35 | 10 | 25 | **4.00** | 3.80 | Baja |
| C. Argumentación científica | 20 % | 20 | 13 | 7 | 3.46 | 3.46 | Media-baja |
| D. Metodología y trazabilidad | 25 % | 44 | 43 | 1 | 3.53 | 3.60 | **Alta** |
| E. Resultados y conclusiones | 15 % | 45 | 16 | 29 | 3.88 | 3.88 | Baja |
| F. Comunicación y edición | 10 % | 56 | 50 | 6 | 3.44 | 3.40 | Media-alta |

- **Puntaje ponderado total: 3.64 / 4 (91 %)** — anterior: 3.60 (90 %)
- **Cobertura global: 139 / 220 = 63 %** → confianza media-baja
- **Advertencia:** cobertura inferior al 80 %. El puntaje describe el documento entregado, no el
  artículo completo.

El dominio D baja siete centésimas pese a las mejoras. La razón es que esta evaluación puntúa
con más rigor la definición de métricas (D16 y D35): la exactitud balanceada, el F1-macro, la
sensibilidad y la especificidad se nombran y se tabulan, pero no se definen. La evaluación
anterior no lo había separado del criterio general de variables.

---

## 5. Criterios que no alcanzan «Cumple»

| Criterio | Estado | Pts | Evidencia | Acción |
|---|---|---:|---|---|
| **D18** Ground truth descrito o validado | No cumple | 0 | Ninguna sección indica cómo se estableció el diagnóstico | Añadir instrumento y criterios por sitio |
| **D19** Quién etiquetó o verificó | No cumple | 0 | No se menciona | Ídem |
| **F49** Referencias con datos mínimos | No cumple | 0 | Sin lista | Incorporar |
| **F52** Autoría, afiliación, correspondencia | No cumple | 0 | Ausentes | Portada |
| **F55** Sin marcas de borrador | No cumple | 0 | Nota de portada y marcador de Figura 1 | Retirar |
| **D10 / D11** Inclusión y exclusión | Parcial | 2 | §2.2 nombra `QC_Rest_1` y separa los tres criterios, pero no da umbral ni conteo de descartes | Flujo de participantes por sitio |
| **D15** Datos faltantes o atípicos | Parcial | 2 | Solo el control de calidad general | Declarar tratamiento |
| **D16 / D35** Definición de métricas | Parcial | 2–3 | AUC definido en §2.7; las cuatro secundarias solo se nombran | Definirlas en §2.7 |
| **D33 / D34** Baselines | Parcial | 3 | Referencia interna y LSTM histórica; sin método no profundo, declarado en §2.8 | Suficiente si se mantiene la declaración |
| **F21** Figuras citadas presentes | Parcial | 2 | La Figura 1 es un marcador | Insertar el archivo |
| **F3 / F4** Siglas y términos | Parcial | 3 | Sin expandir: BOLD, GRU, RPI, T1, `QC_Rest_1` | Expandir en primera aparición |
| **F12 / F13** Longitud de oración y párrafo | Parcial | 3 | Media 24.1 palabras (dentro del rango), pero **10 párrafos superan 160 palabras** y el de §2.7 alcanza 292 | Partir los cuatro párrafos más largos |
| **F19** Distinción entre métricas afines | Parcial | 3 | Se usan correctamente, pero sin definirse | Ligado a D35 |
| **F37** Crédito de material reusado | Parcial | 2 | La Figura 1 procede de la versión anterior sin declararlo | Nota de procedencia |
| **E18** No repetir valores tabulados | Parcial | 2 | §3.3.1 recita los seis valores de la Tabla 7 | Conservar solo los que sostienen la lectura |
| **A1** Problema enunciado | Parcial | 2 | §2.1 declara el objetivo; el porqué importa está ausente | Introducción |
| **C1 / C20** Apertura y aporte | Parcial | 2 | Dependen de Introducción y Discusión | — |

---

## 6. Matriz de coherencia interna

| Objetivo (§2.1) | Método | Datos | Resultado | Interpretación | Conclusión | Estado |
|---|---|---|---|---|---|---|
| Estimar cuánto cambia la discriminación al reducir a paneles de menor dimensionalidad, por sitio | §2.4 paneles; §2.7 validación 10×5 y bootstrap especificado | 4 sitios, 465 sujetos (Tabla 1) | Tabla 6; §3.2 contraste 12 vs 116 con IC | §3.2: los 4 IC incluyen cero; sin declarar equivalencia ni superioridad | Ausente | **Parcial** |
| Evaluar si esa estimación depende de representación, arquitectura y enventanado | §2.8, cuatro dimensiones | Mismas particiones y sujetos | §3.3.1–3.3.4; Tabla 7; Figura 2b | §3.3: variación entre sitios sin afirmar heterogeneidad; confundidos declarados | Ausente | **Parcial** |

Ambas filas son parciales solo por la ausencia de Conclusiones. La cadena objetivo → método →
datos → resultado → interpretación está completa y sin inconsistencias.

---

## 7. Evaluación metodológica y de reproducibilidad

**Lo que mejoró en esta ronda.** La especificación del bootstrap pasó de cuatro a los ocho
elementos exigibles: remuestreo de sujetos con reemplazo estratificado por clase, pareo entre
condiciones y repeticiones, percentiles 2.5 y 97.5, generador PCG64 con semilla 42 reiniciada
por sitio, número de remuestreos, condicionamiento a particiones existentes, exclusión explícita
de la variabilidad de reentrenamiento y definición del estimando como pipeline de validación
cruzada. Se corrigió la afirmación de invariancia entre CPU y GPU, incompatible con
`deterministic=False`. Se declaró el cambio de capacidad entre paneles, que es un confundido de
la pregunta principal. Se corrigieron las resoluciones de ATHENA y se identificó el derivado del
que se extrajeron las series.

**Lo que sigue igual.** El ground truth no está descrito: el diagnóstico que define la variable
de resultado no tiene instrumento, criterio ni responsable declarados. El flujo de participantes
carece de conteos de exclusión. No hay evaluación de calibración, solo de discriminación, lo que
se aparta de lo que TRIPOD+AI espera de un modelo predictivo.

---

## 8. Evaluación de escritura científica

La media de 24.1 palabras por oración está dentro del rango habitual en revistas científicas, y
las oraciones de más de 40 palabras bajaron al 8 %. No se detectan afirmaciones absolutas:
«significativo» no aparece y «demuestra» solo se usa en construcciones negadas.

**Regresión de esta ronda.** Los párrafos de más de 160 palabras pasaron de 6 a 10 porque las
correcciones se añadieron a párrafos existentes en lugar de abrir párrafos nuevos. Los cuatro
más largos:

| Sección | Palabras | Contenido |
|---|---:|---|
| §2.7 | 292 | Especificación del bootstrap |
| §3.4 | 215 | Convergencia y comportamiento del modelo |
| §2.6 | 195 | Regularización y capacidad |
| §2.7 | 195 | Métrica primaria y agregación |

El de 292 palabras concentra ocho elementos distintos de la especificación estadística y debería
partirse en dos o tres.

---

## 9. Tablas, figuras y visualizaciones

Los valores del texto coinciden con las tablas; la numeración es secuencial; las tres figuras
reportan incertidumbre e indican qué representa; los pies son autosuficientes. La Tabla 6
compactada resolvió la fragmentación entre páginas y ahora identifica el resultado principal sin
mezclarlo con el diagnóstico de entrenamiento. La Figura 3 distingue el modo de evaluación de
cada curva y muestra los pliegues activos por época.

Persisten dos puntos: la Figura 1 no existe como archivo, y la Tabla 7 duplica los datos del
panel b de la Figura 2 —duplicación defendible, porque la tabla da los valores exactos y la
figura el patrón, pero candidata al suplemento si el espacio aprieta.

---

## 10. Evaluación funcional de citas

Las 14 citas son funcionales y ninguna decorativa: sustentan el pipeline, el atlas, la
arquitectura, el límite de ventana, la interpretación de la incertidumbre y las herramientas con
su versión. No hay autocitas ni acumulaciones sin función. Las citas de antecedentes clínicos
están en §2.4, dentro de Métodos, y Resultados no contiene ninguna.

El problema sigue siendo de verificabilidad: sin lista de referencias, ninguna puede resolverse.

---

## 11. Problemas prioritarios

**Críticos** — impiden someter

1. Sin lista de referencias.
2. Marcas de borrador visibles y Figura 1 ausente.
3. Sin autoría, afiliación ni financiación.

**Altos** — afectan validez o interpretación

4. Ground truth no descrito (D18 / D19).
5. Flujo de participantes sin conteos de exclusión.
6. Sin evaluación de calibración.

**Medios**

7. Métricas secundarias no definidas.
8. Cuatro párrafos de entre 195 y 292 palabras.
9. Siglas sin expandir: BOLD, GRU, RPI, T1, `QC_Rest_1`.
10. §3.3.1 repite los valores de la Tabla 7.

**Bajos**

11. Procedencia de la Figura 1 sin declarar.

---

## 12. Recomendaciones

| Prioridad | Sección | Problema | Acción | Resultado esperado |
|---|---|---|---|---|
| Crítica | Front matter | Sin referencias, autoría ni financiación | Portada y lista de referencias | Documento sometible |
| Crítica | §2.4 | Figura 1 es un marcador | Insertar el archivo y retirar la nota | Figura citada presente |
| Alta | §2.2 | Ground truth no descrito | Instrumento y criterios diagnósticos por sitio | Variable de resultado trazable |
| Alta | §2.2 | Exclusiones sin conteo | Flujo evaluados → excluidos → analizados | Selección auditable |
| Alta | Suplemento | Sin calibración | Curva de calibración con las probabilidades OOF | Alineación con TRIPOD+AI |
| Media | §2.7 | Métricas secundarias sin definir | Definir las cuatro | Métricas interpretables |
| Media | §2.7, §2.6, §3.4 | Párrafos de 195–292 palabras | Partir los cuatro más largos | Legibilidad |
| Baja | §3.3.1 | Repite la Tabla 7 | Conservar solo los valores que sostienen el argumento | Menos redundancia |

---

## 13. Veredicto editorial simulado

### Requiere revisión mayor

El puntaje de 3.64 correspondería a «listo para revisión menor», pero la regla de la rúbrica
asigna el veredicto más severo ante fallas críticas, y hay tres del grupo 8.3.

Las tres son editoriales y derivan de que el documento cubre solo Métodos y Resultados. La
lectura útil es que **el núcleo metodológico está por encima del umbral de revisión menor**, y
que lo que impide someter es el material que nunca se escribió, más una omisión de fondo —el
ground truth— que sí exige trabajo sustantivo.

---

## 14. Lista de chequeo antes del envío

Pendientes de la evaluación anterior:

- [ ] Insertar la lista de referencias
- [ ] Insertar la Figura 1 y retirar el marcador
- [ ] Retirar la nota «Versión candidata…»
- [ ] Añadir autores, afiliaciones, correspondencia y financiación
- [ ] Describir el ground truth: instrumento y criterios diagnósticos por sitio
- [ ] Añadir el flujo de participantes con conteos de exclusión
- [ ] Definir exactitud balanceada, F1-macro, sensibilidad y especificidad
- [ ] Expandir BOLD, GRU, RPI, T1 y `QC_Rest_1`
- [ ] Declarar la procedencia de la Figura 1
- [ ] Añadir la curva de calibración al suplemento
- [ ] Escribir Resumen, Introducción, Discusión y Conclusiones
- [ ] Traducir al inglés una vez cerrado el contenido

Nuevos en esta evaluación:

- [ ] Partir los cuatro párrafos de 195–292 palabras
- [ ] Reducir la repetición de valores tabulados en §3.3.1

Resueltos desde la evaluación anterior:

- [x] ~~Mover a Resultados las cifras que estaban en Métodos~~
- [x] ~~Dividir las oraciones de más de 40 palabras~~
- [x] ~~Alinear las métricas con el plan congelado~~
- [x] ~~Corregir la afirmación de invariancia CPU/GPU~~
- [x] ~~Corregir la descripción histórica de BrainNetCNN~~
- [x] ~~Declarar el cambio de parámetros entre paneles~~
- [x] ~~Especificar el bootstrap en sus ocho elementos~~
- [x] ~~Compactar la Tabla 6~~
