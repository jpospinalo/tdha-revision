# Evaluación del manuscrito según la rúbrica v3.5

**Documento evaluado:** `Manuscrito_Metodos_Resultados.docx` (secciones 2 y 3)
**Fecha:** 1 de agosto de 2026
**Modo:** diagnóstico estándar

> **Nota de versión.** Tras esta evaluación se aplicó una revisión de estilo que resolvió
> tres de los criterios listados más abajo. Quedan marcados como **[resuelto]** en la
> tabla de §5 y descontados de la lista de chequeo final. El puntaje de los dominios no se
> recalculó: la revisión de estilo afecta a F12, F13 y B31, cuyo efecto conjunto sobre el
> total ponderado es inferior a 0.05 puntos.

---

## 1. Diagnóstico ejecutivo

El manuscrito presenta un núcleo metodológico sólido y una sección de resultados
disciplinada. La fortaleza principal es la trazabilidad: el dominio D obtiene 3.60/4 con
98 % de cobertura, con versiones de librerías, semillas, huellas criptográficas de datos y
código, tratamiento explícito de fuga de información y declaración de exposición al ajuste
de hiperparámetros. Los resultados reportan intervalos de confianza en todos los
contrastes, no usan el término «significativo» en ningún punto y reportan resultados
nulos y adversos con el mismo detalle que los favorables.

La debilidad principal es que **el ground truth no está descrito**: las etiquetas
TDAH/control son la variable de resultado del estudio y el manuscrito nunca indica cómo se
estableció el diagnóstico. Es una omisión de fondo, no de forma, y precede a cualquier
discusión sobre el desempeño del clasificador.

El riesgo editorial más importante es que el documento contiene tres fallas críticas de
transparencia —sin lista de referencias, sin autoría ni financiación, con marcas de
borrador visibles—, todas atribuibles a que las secciones frontales aún no se escriben.

**Confianza: media-baja (cobertura global 63 %)**, porque Resumen, Introducción, Discusión
y Conclusiones están ausentes por diseño. Los dominios D y F sí alcanzan confianza alta.

---

## 2. Clasificación inicial

| Elemento | Estado |
|---|---|
| Tipo declarado | No indicado en el manuscrito |
| Tipo inferido | Investigación empírica computacional; ML aplicado a neuroimagen |
| Área | Neuroimagen computacional / aprendizaje automático en salud |
| Idioma | Español (la versión previa del artículo está en inglés) |
| Estructura | IMRaD parcial: solo Método y Resultados |
| Secciones presentes | §2.1–2.9 (Método), §3.1–3.4 (Resultados) |
| Secciones ausentes | Título con metadatos, Resumen, Palabras clave, Introducción, Discusión, Conclusiones, Referencias, Highlights |
| Palabras clave | Ausentes |
| Tablas | 7 |
| Figuras | 3 (la Figura 1 es un marcador de posición, no el archivo) |
| Referencias | 12 citas en texto; **sin lista de referencias** |
| Declaraciones editoriales | Ética ✓, conflicto de interés ✓, disponibilidad de datos y código ✓; financiación ✗, contribuciones de autor ✗, autoría/afiliación ✗ |
| Limitaciones | Integradas en el texto; sin sección consolidada (corresponde a Discusión) |
| Venue objetivo | No proporcionado |
| Tipo de datos | Humanos, públicos, anonimizados (ADHD-200) |
| Contribución declarada | Objetivo declarado en §2.1; la contribución como tal corresponde a Introducción/Discusión |

**Ruta activada:** investigación empírica + machine learning/IA.

---

## 3. Fallas críticas

| Grupo | Falla | Estado | Tipo | Evidencia | Impacto | Acción |
|---|---|---|---|---|---|---|
| 8.3 | **22.** Referencias imposibles de verificar | Sí | Corregible | 12 citas en texto (Bellec 2017, Kawahara 2017, Varoquaux 2018…) sin lista de referencias | El lector no puede resolver ninguna fuente | Incorporar la lista; la de la versión anterior cubre 10 de las 12 |
| 8.3 | **23.** Marcas internas de borrador | Sí | Corregible | Nota de portada «Versión candidata…» y bloque «[ Figura 1 — reservada… ]» | Inaceptable en versión de envío | Retirar ambas e insertar el archivo de la Figura 1 |
| 8.3 | **24.** Omisión de autoría y financiación | Sí (parcial) | Corregible | §2.9 declara conflicto de interés; no hay autores, afiliación, correspondencia ni financiación | Impide someter | Añadir portada completa |
| 8.1 | 1–10 (científicas) | No | — | Objetivo en §2.1 con método, resultado e interpretación asociados | — | — |
| 8.2 | 11–18 (metodológicas) | No | — | Muestra (Tabla 1), procedimiento (§2.7), fuga tratada explícitamente | — | — |

Las tres fallas son editoriales y derivan del alcance parcial del documento. Ninguna falla
científica ni metodológica crítica está presente.

---

## 4. Resumen cuantitativo

| Dominio | Peso | Aplicables | Evaluados | No verificables | Puntaje | Confianza | Justificación |
|---|---:|---:|---:|---:|---:|---|---|
| A. Problema, brecha, contribución | 15 % | 20 | 7 | 13 | 3.43 | Baja | Brecha y contribución dependen de la Introducción |
| B. Estructura científica | 15 % | 35 | 10 | 25 | 3.80 | Baja | Solo evaluables título, Métodos y Resultados |
| C. Argumentación científica | 20 % | 20 | 13 | 7 | 3.46 | Media-baja | Falta el cierre del arco argumentativo |
| D. Metodología y trazabilidad | 25 % | 44 | 43 | 1 | **3.60** | **Alta** | Cobertura 98 % |
| E. Resultados y conclusiones | 15 % | 45 | 16 | 29 | 3.88 | Baja | Resultados completos; discusión ausente |
| F. Comunicación y edición | 10 % | 56 | 50 | 6 | 3.40 | Media-alta | Cobertura 89 % |

- **Puntaje ponderado total: 3.60 / 4 (90 %)**
- **Cobertura global: 139 / 220 = 63 % → confianza media-baja**
- **Advertencia:** cobertura inferior al 80 %. El puntaje describe el documento entregado,
  no el artículo completo. Los dominios A, B y E están evaluados sobre menos de la mitad
  de sus criterios.

---

## 5. Criterios que no alcanzan «Cumple»

| Criterio | Estado | Pts | Evidencia | Acción |
|---|---|---:|---|---|
| **D18** Ground truth descrito o validado | No cumple | 0 | Ninguna sección indica cómo se estableció el diagnóstico | Añadir a §2.2 el instrumento y los criterios diagnósticos por sitio |
| **D19** Quién etiquetó o verificó los datos | No cumple | 0 | No se menciona | Ídem |
| **F52** Autoría, afiliación, correspondencia | No cumple | 0 | Ausentes | Portada |
| **F55** Sin marcas de borrador | No cumple | 0 | Nota de portada y marcador de Figura 1 | Retirar |
| **F49** Referencias con información mínima | No cumple | 0 | No hay lista | Incorporar |
| **D10/D11** Criterios de inclusión y exclusión | Parcial | 2 | «control de calidad basado en el indicador QC_Rest_1» (§2.2), sin umbral ni conteo de descartes | Añadir flujo de participantes: evaluados → excluidos → analizados, por sitio |
| **D15** Manejo de datos faltantes o atípicos | Parcial | 2 | Solo el QC general | Declarar si hubo imputación o descarte adicional |
| **B31** Métodos no incluye resultados | **[resuelto]** | 4 | La cifra de similitud entre ventanas se trasladó a §3.3; las amplitudes de §2.8 se reformularon como precisión ya conocida de las corridas base | — |
| **F12/F13** Longitud de oración | **[resuelto]** | 3 | Media reducida de 30.3 a 23.4 palabras; oraciones de más de 40 palabras, de 34 a 9 | Quedan seis párrafos de más de 160 palabras, todos enumeraciones con unidad argumentativa clara |
| **F21** Figuras citadas presentes | Parcial | 2 | Figura 1 es un marcador | Insertar el archivo |
| **E18** No repetir valores ya tabulados | Parcial | 2 | §3.3.1 recita +1.0, −6.2, −5.1, +5.8, +4.0, +14.6, todos en la Tabla 7 | Conservar solo los valores que sostienen la lectura analítica |
| **F37** Crédito de material reusado | Parcial | 2 | La Figura 1 procede de la versión anterior sin declararlo | Añadir nota de procedencia |
| **D16/D35/F19** Definición de métricas secundarias | Parcial | 3 | AUC definido en §2.6; exactitud, precisión, sensibilidad y especificidad se usan sin definir | Definirlas; «precisión» es ambigua en español |
| **F3** Siglas en primera aparición | Parcial | 3 | Sin definir: BOLD, GRU, RPI, T1, QC_Rest_1 | Expandir |
| **A1** Problema enunciado | Parcial | 2 | §2.1 declara el objetivo; el porqué importa está ausente | Introducción |
| **D33/D34** Baselines | Parcial | 3 | Referencia interna y LSTM histórica; sin método no profundo | Declarado en «Dimensiones no evaluadas»; suficiente si se mantiene |

---

## 6. Matriz de coherencia interna

| Objetivo (§2.1) | Método | Datos | Resultado | Interpretación | Conclusión | Estado |
|---|---|---|---|---|---|---|
| Estimar cuánto cambia la discriminación al reducir a paneles de menor dimensionalidad, por sitio | §2.4 paneles; §2.7 validación 10×5 y bootstrap pareado | 4 sitios, 465 sujetos (Tabla 1) | Tabla 6; §3.2 contraste 12 vs 116 con IC | §3.2: los 4 IC incluyen cero; no se declara equivalencia ni superioridad | Ausente (sección no escrita) | **Parcial** |
| Evaluar en qué medida esa estimación depende de representación, arquitectura y enventanado | §2.8 cuatro dimensiones | Mismas particiones y sujetos | §3.3.1–3.3.4; Tabla 7; Figura 2b | §3.3: patrón dependiente del sitio; confundidos declarados | Ausente | **Parcial** |

Ambas filas son *parciales* únicamente por la ausencia de Conclusiones. La cadena
objetivo → método → datos → resultado → interpretación está completa y sin
inconsistencias.

---

## 7. Evaluación metodológica y de reproducibilidad

**Fortalezas verificables.** El pipeline es trazable de extremo a extremo: cada corrida
almacena huellas criptográficas de señal, código de datos, código de ejecución y
particiones (§2.1). La fuga de información se aborda explícitamente por construcción
—«todas las ventanas de un mismo sujeto permanecen siempre en la misma partición»
(§2.7)—. La configuración experimental está completa: versiones exactas de seis librerías,
semilla, precisión numérica y entorno de ejecución. La elección de la métrica primaria
está justificada por el desbalance entre sitios. Los tres niveles de reproducibilidad
—conceptual, metodológica y computacional— se cumplen.

**Debilidades.** La construcción del ground truth es el hueco más serio: el diagnóstico
que define la variable de resultado no se describe en ningún punto. El flujo de
participantes carece de conteos de exclusión, de modo que la Tabla 1 muestra la muestra
final sin permitir reconstruir cuántos registros se descartaron ni por qué. No hay
evaluación de calibración, solo de discriminación, lo que se aparta de lo que espera
TRIPOD+AI para modelos predictivos.

---

## 8. Evaluación de escritura científica

La terminología es consistente y no se detectaron afirmaciones absolutas: «significativo»
no aparece, y «demuestra» solo se usa en construcciones negadas.

El problema medible es la longitud de oración: **media de 36.1 palabras y 41 de 152
oraciones por encima de 40**. La oración que enuncia el objetivo tiene 66 palabras y
concentra dos objetivos distintos; conviene partirla. Seis párrafos de prosa superan las
160 palabras (§2.3, §2.4, §2.7, §2.8, §3.3.1, §3.4).

No hay ecuaciones, por lo que los criterios de notación no aplican.

---

## 9. Evaluación de tablas, figuras y visualizaciones

Los valores del texto coinciden con las tablas y figuras; la numeración es secuencial;
todas las figuras reportan incertidumbre e indican qué representa; los pies son
autosuficientes. La Figura 3c ahora anota el AUC medio, que coincide con la fila de
validación de la Tabla 6.

Dos observaciones. La Figura 1 no existe como archivo. Y la Tabla 7 y la Figura 2b
presentan exactamente los mismos datos: es duplicación defendible —la tabla da los valores
exactos y la figura el patrón— pero si el espacio aprieta, la tabla puede ir al
suplemento.

---

## 10. Evaluación funcional de citas

Las 12 citas son funcionales y ninguna es decorativa: sustentan el pipeline (ATHENA), el
atlas (AAL116), la arquitectura (Kawahara), el límite de ventana (Leonardi & Van De Ville),
la interpretación de la incertidumbre (Varoquaux) y las herramientas con su versión. No se
detectan autocitas ni acumulaciones sin función.

El problema no es funcional sino de verificabilidad: sin lista de referencias, ninguna
puede resolverse. Las citas de antecedentes clínicos (Singh, Blomberg, Damiani, Francx,
Hale) están correctamente ubicadas en §2.4 y no en Resultados.

---

## 11. Problemas prioritarios

**Críticos** — impiden someter

1. Sin lista de referencias.
2. Marcas de borrador visibles y Figura 1 ausente.
3. Sin autoría, afiliación ni financiación.

**Altos** — afectan validez o interpretación

4. Ground truth no descrito (D18/D19).
5. Flujo de participantes sin conteos de exclusión (D10/D11).
6. Sin evaluación de calibración.

**Medios**

7. Resultados reportados dentro de Métodos (§2.5, §2.8).
8. Métricas secundarias sin definir; «precisión» ambigua.
9. Oraciones y párrafos largos.
10. Duplicación entre §3.3.1 y la Tabla 7.

**Bajos**

11. Siglas sin expandir: BOLD, GRU, RPI, T1, QC_Rest_1.
12. Procedencia de la Figura 1 sin declarar.

---

## 12. Recomendaciones

| Prioridad | Sección | Problema | Acción | Resultado esperado |
|---|---|---|---|---|
| Crítica | Front matter | Sin referencias, autoría ni financiación | Portada y lista de referencias | Documento sometible |
| Crítica | §3.1 (Fig. 1) | Marcador en vez de figura | Insertar el archivo y retirar la nota | Figura citada presente |
| Alta | §2.2 | Ground truth no descrito | Añadir instrumento y criterios diagnósticos por sitio, desde la documentación de ADHD-200 | Variable de resultado trazable |
| Alta | §2.2 | Exclusiones sin conteo | Flujo evaluados → excluidos → analizados por sitio | Selección auditable |
| Alta | Suplemento | Sin calibración | Curva de calibración con las probabilidades OOF | Alineación con TRIPOD+AI |
| Media | §2.5, §2.8 | Resultados en Métodos | Mover a §3.1 o reformular | Separación de funciones |
| Media | §2.7 / Tabla 6 | Métricas secundarias sin definir | Definirlas en §2.7 | Métricas interpretables |
| Media | Todo | Oraciones de 36 palabras de media | Dividir las 41 oraciones largas | Legibilidad |
| Baja | §3.3.1 | Repite la Tabla 7 | Conservar solo los valores que sostienen el argumento | Menos redundancia |

---

## 13. Veredicto editorial simulado

### Requiere revisión mayor

El puntaje ponderado de 3.60/4 correspondería a «listo para revisión menor», pero la regla
de la rúbrica es explícita: ante fallas críticas se asigna el veredicto más severo, y hay
tres del grupo 8.3.

La lectura correcta es que **ninguna de las tres es científica**. Las tres derivan de que
el documento cubre deliberadamente solo Métodos y Resultados, y se resuelven escribiendo
las secciones pendientes. El único hallazgo de fondo que exige acción sustantiva es la
descripción del ground truth.

Sobre el documento como candidato a artículo completo, la evaluación aplicable es: núcleo
metodológico listo, front matter y back matter pendientes, una omisión de fondo por
cerrar.

---

## 14. Lista de chequeo antes del envío

- [ ] Insertar la lista de referencias completa
- [ ] Insertar el archivo de la Figura 1 y retirar el marcador
- [ ] Retirar la nota de portada «Versión candidata…»
- [ ] Añadir autores, afiliaciones, correspondencia y financiación
- [ ] Describir el ground truth: instrumento y criterios diagnósticos por sitio
- [ ] Añadir el flujo de participantes con conteos de exclusión
- [ ] Definir exactitud, precisión, sensibilidad y especificidad
- [x] ~~Mover a Resultados las cifras de §2.5 y §2.8~~
- [ ] Expandir BOLD, GRU, RPI, T1 y QC_Rest_1
- [ ] Declarar la procedencia de la Figura 1
- [ ] Añadir la curva de calibración al suplemento
- [x] ~~Dividir las oraciones de más de 40 palabras~~
- [ ] Escribir Resumen, Introducción, Discusión y Conclusiones
- [ ] Traducir al inglés una vez cerrado el contenido
