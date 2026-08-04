# Comentarios sobre §2.1–§2.4 y reducción de extensión

**Fecha:** 3 de agosto de 2026
**Documento:** `Manuscript_Methods_Results_EN.docx`
**Estado:** aplicado. Incluye una decisión que necesita confirmación del equipo (§2.1, punto 3) y un
menú de recortes adicionales que no ejecuté por cuenta propia (§4).

---

## 1. Comentarios aplicados

### §2.1 Frase de control de calidad, restaurada con una salvedad

Se restauró la formulación del artículo original. Retiré una de sus tres cláusulas:

> Each image underwent an initial quality control check based on the QC_Rest_1 indicator, used to
> assess the validity of resting-state acquisitions; this procedure identified records with motion
> artifacts or capture failures.

La cláusula retirada es «**or incomplete clinical information**». El motivo es que el mismo mensaje
del equipo pide eliminar la frase sobre disponibilidad de diagnóstico y variables fenotípicas
porque no se usó en los experimentos. Si se restaura íntegra la frase original, esa misma
información vuelve a entrar por la puerta de atrás y, además, mal atribuida: QC_Rest_1 es una
calificación de la serie funcional obtenida por inspección visual y no evalúa la completitud de la
información clínica. Dejarla produciría una descripción inexacta del procedimiento.

**Confirmen si aceptan esta salvedad** o si prefieren la frase íntegra. Si la quieren íntegra, lo
coherente sería reponer también la frase sobre disponibilidad, porque es lo que la cláusula
describe.

### §2.1 Argumento sobre la combinación de sitios, ampliado y con referencia

Antes remitía a otra sección. Ahora el argumento está donde se necesita y se apoya en tres piezas:

1. **Variabilidad no biológica.** Las diferencias de escáner y protocolo introducen variabilidad no
   biológica en las estimaciones de conectividad, que limita la potencia estadística y puede
   producir hallazgos espurios; corregirla exige procedimientos de armonización con supuestos
   propios. Referencia añadida: **Yu et al. (2018)**, *Human Brain Mapping* 39(11):4213–4227, que
   documenta este efecto específicamente en medidas de conectividad funcional y evalúa ComBat como
   corrección.
2. **Confusión sitio–composición.** En ADHD-200 el sitio está confundido con el balance de clases y
   con el protocolo de adquisición, de modo que una estimación agrupada mezclaría el efecto del
   grupo de ROI con el del sitio. Esto no necesita cita: se lee en la Tabla 1.
3. **Ponderación de clase.** Peking aplica pesos inversos a la frecuencia de clase y los otros tres
   sitios no. Ese hecho seguía en §2.6 y ahora remite a §2.1, que carga el argumento.

Redacté la afirmación pegada a lo que la fuente sostiene. Yu et al. documentan pérdida de potencia
y riesgo de hallazgos espurios; **no** afirman que el efecto de sitio iguale en magnitud al efecto
biológico, así que el texto no lo dice.

### §2.1 Frase sobre disponibilidad de diagnóstico y fenotipo, eliminada

Hecho.

### §2.2 Preprocesamiento

La versión nueva ya conservaba el contenido del original salvo tres puntos. Repuse dos y mantengo
el tercero corregido:

| Del original | Estado |
|---|---|
| «to remove low-frequency components and physiological noise» (justificación del filtro) | **Repuesto** |
| «to improve the signal-to-noise ratio» (justificación del suavizado) | **Repuesto** |
| «the resulting maps were smoothed … **and used to mask ROIs**» | **No repuesto.** Los mapas de densidad de sustancia gris que produce ATHENA no intervienen en la extracción de series de este trabajo: las series AAL116 salen del derivado funcional filtrado y suavizado. La frase original describía un paso que el pipeline no ejecuta aquí |

Lo que la versión nueva **aporta** sobre el original: que el pipeline se aplicó de forma idéntica en
los cuatro sitios; la resolución funcional de 4 × 4 × 4 mm, que el original omitía; y de qué
derivado exacto se extrajeron las series AAL116.

### Reordenamiento de §2.3 y §2.4

Aplicado. El orden queda: BOLD y conectividad en §2.3, redes funcionales en §2.4. Es coherente,
porque la extracción BOLD usa el atlas AAL116 completo y los grupos reducidos son una decisión
posterior. Una sola referencia hacia adelante quedó necesaria, en §2.3: «between all regions of the
ROI group under evaluation (Section 2.4)».

### §2.4 Contenido recuperado del original

Del original se recuperaron dos bloques que estaban en la sección de ablación y que la versión
anterior había perdido:

- **La función de las cinco redes:** autorreferencia, control inhibitorio, priorización de
  estímulos, regulación atencional y modulación motora, con las cuatro citas del original
  (Koirala et al., 2024; Parlatini et al., 2023; Reimann et al., 2024; Sutcubasi et al., 2020).
- **La heterogeneidad del trastorno:** no todas las estructuras de una red muestran el mismo grado
  de disfunción (Schleim, 2022), lo que explica por qué los grupos reducidos conservan un
  subconjunto de cada red y no la red completa.

**Un bloque del original que no recuperé, y es importante que lo sepan.** El original justificaba
la reducción a 18 y 12 ROIs diciendo que se retuvieron «the nodes with the greatest **discriminatory
weight**, functional stability and pathophysiological coherence». Esa formulación describe una
selección guiada por desempeño, que es exactamente lo que toda la revisión niega y lo que sostiene
la respuesta a los revisores sobre el sesgo de selección. Restaurarla contradiría §2.4, §3.2 y la
carta de respuesta. En su lugar, el texto dice que los grupos los especificaron expertos a partir
de la literatura, antes de entrenar ningún modelo.

---

## 2. Reducción de extensión

### Punto de partida y resultado

| | Antes | Ahora | Cambio |
|---|---:|---:|---:|
| Texto corrido (Método + Resultados) | 4.046 | **3.540** | −12,5% |
| Documento completo con tablas y pies | 5.472 | 4.966 | −9,3% |

La cifra neta es de −12,5%, pero no describe bien el trabajo hecho: **se recortaron 758 palabras y
los comentarios de esta ronda añadieron 252**. Contra la versión previa a los comentarios, el
recorte bruto es del 18,7%.

### La diferenciación artículo/carta funcionó, y fue la mayor parte del ahorro

Preguntaban si ayuda separar lo que va en el artículo de lo que va en la respuesta a los revisores.
**Sí, y con diferencia: produjo 445 de las 758 palabras recortadas, el 59%.** La compresión de
estilo aportó las 313 restantes.

| Bloque | En el artículo queda | Se traslada a | Ahorro |
|---|---|---|---:|
| Dimensiones no evaluadas | Una frase con las cuatro direcciones | Discusión (limitaciones) + carta | 165 |
| Exposición al ajuste de hiperparámetros | El hecho, el sitio, el grupo y la consecuencia | Las diez configuraciones y las métricas que guiaron la elección → **Tabla S2 / suplementario** | 100 |
| Brecha entrenamiento–validación | Los valores y su lectura, una sola vez | Se eliminó la duplicación entre §3.1 y §3.4 | 85 |
| Conteo de parámetros por sitio y grupo | El rango y su implicación | Las dieciséis cifras → **Tabla S2** | 45 |
| ANOVA, CV anidada, alcance del estimando | La conclusión metodológica | El desarrollo del argumento → carta | 50 |

La razón por la que esto funciona es que ese material no responde al lector del artículo sino a una
objeción concreta de un revisor. En el artículo basta con que el hecho sea verificable; el
argumento de por qué se hizo así pertenece a la carta.

### Lo que se comprimió sin mover nada

Diecinueve párrafos reescritos más densos, sin pérdida de contenido. Ninguna cifra desapareció:
verifiqué las veintiséis cifras clave del documento contra el texto renderizado y las veintiséis
siguen presentes. Las referencias cruzadas siguen resolviendo, y no hay tablas ni figuras huérfanas.

### Efecto sobre la escritura

| Indicador | Antes | Ahora |
|---|---:|---:|
| Párrafos de más de 160 palabras | 2 | **0** |
| Párrafo más largo | 175 | **160** |
| Oraciones de más de 40 palabras | 6 | 6 |
| Oraciones de más de 50 palabras | 0 | 0 |
| Longitud media de oración | 22,3 | 24,0 |

La media de oración sube dos palabras, efecto esperable de fusionar oraciones al comprimir. Sigue
dentro de la banda de 20 a 26 palabras que es habitual en revistas científicas, y ninguna oración
llega a cincuenta. Los párrafos mejoraron: ya no queda ninguno por encima de 160 palabras.

---

## 3. Lo que crece en el material suplementario

La estrategia desplaza peso, no lo elimina. El suplementario pasa a contener:

- **Tabla S1** — métricas de la partición de entrenamiento externa y brechas por sitio y grupo
- **Tabla S2** — los dieciséis conteos de parámetros y las diez configuraciones comparadas en la
  fase de desarrollo
- Composición de los grupos de 18 y 39 ROIs
- Matrices de confusión al umbral de 0,5
- Exactitud simple, retenida como métrica de auditoría

Conviene que el suplementario esté listo antes del envío, porque el artículo ya remite a él nueve
veces.

---

## 4. Para llegar al 20%: lo que falta y lo que cuesta

Faltan unas 300 palabras. No las recorté por cuenta propia porque las tres opciones que quedan
tienen un costo que el equipo debe valorar.

| Opción | Ahorro | Costo |
|---|---:|---|
| **Mover §2.8 Ética a las declaraciones finales** | 82 | Ninguno de contenido. En la mayoría de revistas las declaraciones no cuentan para el límite. El equipo ya decidió conservarla como subsección numerada, por eso no lo hice |
| **Reducir §3.4 «Convergence and model behavior» a un párrafo y llevar el resto al suplementario** | ~230 | Es la sección que responde a R1.4 y a parte de R2.11. Un revisor que busque ahí la respuesta a su comentario encontraría una remisión |
| **Convertir las cuatro dimensiones de sensibilidad de §2.6 en una tabla** | ~200 de texto corrido | Solo ayuda si la revista cuenta el texto corrido y no las tablas. Habría que confirmarlo en las normas |

La primera es gratis y yo la haría. La tercera depende de un dato que no tengo: **¿el límite de la
revista aplica al texto corrido, al documento completo, o excluye tablas, pies y declaraciones?**
Con ese dato puedo ajustar al número exacto en lugar de a un porcentaje aproximado.
