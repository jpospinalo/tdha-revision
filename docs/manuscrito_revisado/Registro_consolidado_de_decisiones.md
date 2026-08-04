# Registro consolidado de decisiones sobre el manuscrito

**Fecha:** 2 de agosto de 2026
**Cubre:** las cuatro rondas de revisión del equipo y las evaluaciones internas por rúbrica y por
guía de escritura.
**Propósito:** dejar en un solo lugar qué se implementó, qué no y por qué, para que los puntos ya
resueltos o descartados no vuelvan a plantearse.

> **Por qué existe este documento.** Hasta ahora las decisiones se comunicaron ronda por ronda,
> pero **los puntos rechazados nunca se consolidaron**. Sin ese registro, un comentario descartado
> con argumento reaparece en la ronda siguiente como hallazgo nuevo.

---

## 1. Resumen

| Categoría | Nº |
|---|---:|
| Implementados y verificados | 43 |
| Rechazados con argumento | 5 |
| Concesiones: donde nuestro argumento era incorrecto | 2 |
| Aceptados y aplicados con modificación | 4 |
| Aceptados pero **todavía no aplicados** | 11 |

---

## 2. Rechazados — no volver a plantear sin evidencia nueva

Cinco puntos se examinaron, se verificaron contra la fuente y se descartaron. Cada uno incluye la
evidencia que sostiene el rechazo.

### 2.1 «Se retiró el ANOVA y no se añadió la prueba de Tukey solicitada»

**Propuesto en:** segunda revisión, §4.4. **Rechazado.**
El manuscrito anterior **sí reportaba Tukey**. Párrafo 68 de `Manuscript.docx`: «Tukey's post-hoc
analysis showed significant differences between the model with 116 ROIs (p=0.0139) and the models
with 12 and 18 ROIs (p=0.010)…». La formulación propuesta afirmaría que Tukey nunca estuvo, lo que
es falso, y desdibuja el alcance real: no se declina una adición, se retira un análisis publicado.
**Resolución:** el equipo aceptó el rechazo en la tercera ronda. Redacción alternativa aplicada.

### 2.2 «El encabezado de la Tabla 6 no se repite en la página 8»

**Propuesto en:** tercera revisión, §11.1, como rectificación a un punto nuestro. **Rechazado.**
Se amplió el render de la página 8 al 150 % y el encabezado aparece completo, con su fila
sombreada, antes del primer dato.
**Estado de comunicación:** este es el punto que el equipo nunca recibió y que motivó el registro.
**Consecuencia práctica:** ninguna. La Tabla 6 se compactó y ya no alcanza la página 8.

### 2.3 «No enviar todavía» como veredicto sobre el documento

**Propuesto en:** primera revisión, encabezado. **Rechazado el encuadre, no la conclusión.**
El documento contiene solo Métodos y Resultados y lo declara en su primera línea; enviarlo nunca
estuvo sobre la mesa. El veredicto podía leerse como duda sobre el trabajo experimental, cuando el
propio documento del equipo concluía lo contrario.
**Resolución:** el equipo aceptó la observación y propuso una formulación de consenso, adoptada.

### 2.4 Cuatro figuras separadas en el texto principal

**Propuesto en:** primera revisión, §6.1. **Rechazado parcialmente.**
Los cuatro contenidos se aceptan; su separación en cuatro objetos no. La Figura 2 ya los cubre en
dos paneles —perfiles en 2a, contrastes con intervalo en 2b, incluido el 12−116—. Separarlos
consumiría dos de las tres figuras disponibles, y la restricción de espacio fue el motivo
declarado para fusionarlas.
**Resolución:** el equipo aceptó conservar la figura compuesta con cinco condiciones de
legibilidad, que se cumplen.

### 2.5 IQ, medicación y comorbilidades en la caracterización principal

**Propuesto en:** primera revisión, §4.1. **Rechazado para el texto principal, aceptado para el
suplemento.** Ninguna de esas variables entra en el análisis; presentarlas en el cuerpo junto a
las métricas sugeriría un control estadístico que no se ejerció. Como limitación en la Discusión
sí son pertinentes.
**Resolución:** el equipo aceptó la distinción y la reiteró en su cuarta revisión, §3.5.7, con la
salvedad correcta de no imputar información ausente.

---

## 3. Concesiones: puntos donde nuestro argumento era incorrecto

### 3.1 Extender `run_manifest.csv` a las 26 corridas

Recomendación nuestra en la primera ronda. **Era inviable.** `validate_manifest_structure()`
contiene `if len(included) != 16: raise ValidationError` y un chequeo de duplicados sobre
`(site, roi_set)`. Las diez corridas nuevas tienen todas `roi_set = 12`, de modo que dispararían
ambas validaciones y el pipeline fallaría al arrancar.
**Adoptado:** el diseño de dos manifiestos separados propuesto por el equipo.

### 3.2 «El error de Monte Carlo solo afecta la tercera cifra decimal»

Argumento nuestro para no condicionar el envío al recálculo. **Incorrecto en su conclusión.**
Se ejecutó la prueba empírica que el equipo ofrecía como alternativa: cuatro semillas
independientes sobre el contraste `static` de NYU con 2000 remuestreos.

| Semilla | IC | ¿Incluye cero? |
|---|---|---|
| 42 | [−0.0816, **+0.0014**] | sí |
| 1 | [−0.0824, **+0.0013**] | sí |
| 7 | [−0.0825, **+0.0002**] | sí |
| 2026 | [−0.0849, **+0.0019**] | sí |

La lectura cualitativa es estable, pero el límite superior oscila entre +0.0002 y +0.0019: la
dispersión es del mismo orden que la distancia al cero. El error afecta la tercera cifra, y la
distancia al cero **también está** en la tercera cifra.
**Adoptado:** recálculo a 10 000 remuestreos antes de congelar cifras.

---

## 4. Aceptados con modificación

| Punto | Propuesta original | Cómo se aplicó |
|---|---|---|
| Tono de R2.13 | Suprimir «más exigente, no menos» por adversarial | Se retiró la comparación pero **se conservó la justificación técnica**: eliminarla dejaría sin fundamento el único comentario que se declina |
| Recálculo a 10 000 | Condición de aprobación | Se aceptó la acción; se rechazó que fuera criterio de aprobación y después se concedió tras la prueba empírica (§3.2) |
| Estructura de la Tabla 6 | Compactar | Se separó en dos decisiones: *qué métricas* se aplicó de inmediato por conformidad con el plan; *qué filas* esperó confirmación del equipo |
| Reproducción íntegra de los comentarios | Reproducir completos | Se conservan como extractos claramente marcados; la reproducción íntegra corresponde al formato final de la revista |

---

## 5. Implementados y verificados

Verificados por extracción de texto de los DOCX, comprobando que la formulación antigua
desapareció y no solo que la nueva apareció.

**Correcciones de fondo (17)**

1. Métricas conformes al plan congelado; exactitud simple como métrica de auditoría.
2. Retirada la afirmación de invariancia entre CPU y GPU, incompatible con `deterministic=False`.
3. Retirada la atribución de BrainNetCNN a «muestras de miles de sujetos»: el original usó 168
   escaneos de unos 115 participantes.
4. Declarado el cambio de capacidad entre paneles: de 1841–4433 parámetros con 12 ROIs a
   16 817–41 873 con 116.
5. Corregida la terminología de particiones: `fit`, `inner_val`, partición exterior de
   entrenamiento y predicciones fuera de pliegue.
6. Retirada la comparación 54.0 % → 62.4 % «sobre el mismo conjunto sin dropout».
7. Retirada toda afirmación de heterogeneidad entre sitios.
8. Delimitada la exposición de NYU por grado, no solo para el panel de 12.
9. Documentada la ponderación de clases por sitio.
10. Especificado el bootstrap en sus ocho elementos.
11. Terminología conservadora de ventanas, consistente en todo el documento.
12. Condiciones de acceso a ADHD-200: investigación no comercial y registro en NITRC.
13. `QC_Rest_1` separado en tres criterios independientes.
14. ATHENA: estructural a 1×1×1 mm, funcional a 4×4×4 mm, derivado identificado.
15. Ética reformulada sin afirmación absoluta de no reidentificación.
16. Añadido el apartado «Dimensiones no evaluadas».
17. Añadida la referencia a Varoquaux (2018) sobre la amplitud de los intervalos en neuroimagen.

**Cuarta ronda (7)**

18. §3.4 ya no atribuye a la Tabla 6 las brechas de entrenamiento: remite a la Tabla S1.
19. §3.1 remite también a la Tabla S1 en las dos menciones de brechas.
20. Moderada la afirmación sobre sesgo de selección en §2.4, distinguiendo el sesgo evitado del
    optimismo que persiste por el desarrollo con NYU y 12 ROIs.
21. §2.9 ya no describe el repositorio como «de acceso abierto»: coherente con §2.1.
22. La carta usa «evaluación multisitio» en lugar de «multi-cohorte».
23. La carta remite a la Tabla S1 en R1.4, en lugar de «el material suplementario» genérico.
24. Retirada la fecha fija de la carta; ahora remite al identificador del Apéndice B.

**Tablas y figuras (8)**

25. Tabla 6 compactada de 48 a 16 filas.
26. Tabla de entrenamiento y brechas trasladada al suplemento como Tabla S1.
27. Figura 2 con filas alineadas entre sitios y encabezados de grupo legibles.
28. Figura 3 con la línea de azar diferenciada de la curva de validación.
29. Figura 3 con el conteo de pliegues activos por época.
30. Figura 3 con el AUC medio anotado en cada panel ROC.
31. Número de remuestreos declarado en los pies de la Tabla 7 y la Figura 2.
32. **Filas de tabla protegidas contra división entre páginas**: la fila `Insula_R` de la Tabla 2
    ya no se parte y la palabra «atención» no queda huérfana.

**Estructura y escritura (11)**

33. Contenido anatómico trasladado de Resultados a Métodos: Resultados no contiene ninguna cita.
34. Cifras que estaban en Métodos trasladadas a Resultados.
35. Longitud media de oración reducida de 30.3 a 24.0 palabras.
36. Oraciones de más de 40 palabras reducidas de 34 a 15.
37. Párrafo más largo reducido de 292 a 182 palabras; los de más de 160, de 10 a 3.
38. Siglas expandidas en primera aparición: AUC, TR, FWHM, MNI, BOLD, RPI, T1, GRU.
39. Corregida una referencia cruzada rota introducida al renumerar las secciones.
40. Sincronizadas todas las referencias cruzadas entre manuscrito y carta.
41. Añadido el objetivo explícito al inicio de Métodos.
42. Añadida la declaración de disponibilidad con mención del commit.
43. Retirada la repetición de la fórmula metatextual «conviene precisar».

---

## 6. Aceptados pero todavía no aplicados

Verificado por extracción de texto sobre los DOCX vigentes.

| # | Elemento | Origen | Bloqueado por | Responsable |
|---|---|---|---|---|
| 1 | Recálculo de los diez contrastes con 10 000 remuestreos | 4.ª ronda §3.1 | Cómputo | Máquina del equipo |
| 2 | Regeneración de Tabla 7, Figura 2b y las frases de §3.3 sobre cruce del cero | 4.ª ronda §3.1 | Recálculo | Ídem |
| 3 | Retirada de toda mención a 2000 remuestreos en §2.7, Figura 2 y Tabla 7 | 4.ª ronda §3.1 | Recálculo | Ídem |
| 4 | Auditoría de comparabilidad de las 26 corridas y su artefacto | 4.ª ronda §3.3 | — | Ídem |
| 5 | Creación de los dos manifiestos separados | 2.ª ronda §5.3 | — | Ídem |
| 6 | R2.6 en pasado, citando la auditoría ejecutada | 4.ª ronda §3.3 | Auditoría | Redacción |
| 7 | Descripción del ground truth: instrumento y criterios diagnósticos por sitio | 1.ª ronda §4.1 | Documentación clínica de ADHD-200 | Equipo |
| 8 | Flujo de participantes: conteo inicial, exclusiones por causa, muestra final | 4.ª ronda §3.5 | Ídem | Equipo |
| 9 | Edad y sexo por sitio y clase; subtipos; IQ, medicación y comorbilidades al suplemento | 4.ª ronda §3.5 | Ídem | Equipo |
| 10 | Figura 1: insertar el archivo, verificar contra `roi_sets.json`, declarar procedencia | 4.ª ronda §3.4 | Archivo anatómico | Equipo |
| 11 | Definición de exactitud balanceada, F1-macro, sensibilidad y especificidad | Rúbrica D35 | — | Redacción |

**Además, pendientes de las secciones no escritas:** portada con autoría y financiación, lista de
referencias, retirada de las marcas de borrador, curva de calibración en el suplemento, tabla
comparativa de literatura con las seis columnas de la 4.ª ronda §4.4, conversión de la carta a
formato editorial, y corrección del `docstring` de `src/kerasmodels/brainnetcnn.py` en un commit
documental posterior al registro de procedencia.

---

## 7. Observación sobre la cuarta ronda

Los cinco puntos verificables de esta ronda eran correctos, y el más importante —§3.2— vuelve a
señalar el mismo fallo: **una corrección aplicada a un documento y no al otro**. R1.4 se corrigió
en la carta para remitir al suplemento, pero §3.4 del manuscrito siguió atribuyendo las brechas a
la Tabla 6. Es la tercera vez que aparece este patrón.

La medida adoptada es verificar cada corrección por extracción de texto **sobre los dos documentos
a la vez**, comprobando que la formulación antigua desapareció, y no solo que la nueva apareció.

---

## 8. Cómo usar este registro

Al recibir una nueva ronda, comprobar primero si el punto figura en §2 o §3. Si está en §2, se
descartó con evidencia y solo debe reabrirse aportando evidencia nueva. Si está en §3, ya se
concedió y la acción correspondiente está en §6.
