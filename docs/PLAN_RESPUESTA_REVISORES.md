# Plan para responder los comentarios de los revisores

**Versión:** 3.2 — **definitiva y congelada**
**Fecha:** 30 de julio de 2026
**Estado:** aprobado por el equipo. Campaña de diez corridas aprobada. Documento cerrado tras incorporar las tres correcciones finales de redacción.

Este es el **único documento de planificación vigente** para la respuesta a los revisores. Las versiones previas (v3.0) y el registro intermedio de la deliberación (evaluación de la v3.1) quedaron retiradas del repositorio una vez incorporadas aquí; el resumen de qué se corrigió en el cierre está en la nota de arriba.

**Correcciones incorporadas en el cierre:** (1) se retira «validación externa» como nombre del diseño y se fija la regla léxica exterior/externo (§2.3), que además resolvía una contradicción con §4.4; (2) la respuesta a R2.5 se da por sitio y no en bloque, y se separa el hecho documentado de la inferencia pendiente sobre NYU (§4.1, §4.2, D4); (3) se retira la afirmación de imposibilidad en la justificación para excluir GRU y Transformer (§8.2).

Esta versión integra la v3.1 y la evaluación posterior. **No modifica la campaña de diez corridas**, no abre una segunda ronda de ajuste y no añade experimentos. Los cambios respecto de la v3.1 son de formulación, de ubicación de resultados y de precisión en las declaraciones.

La organización operativa —quién ejecuta qué y en qué momento— queda fuera de este documento y depende de la disponibilidad del equipo. Aquí solo se fija **qué hay que hacer, con qué diseño y con qué reglas de interpretación**, además del orden de dependencias entre bloques (§11).

---

## 1. Propósito y principio rector

Este documento organiza la respuesta a los comentarios de los revisores que dependen del código, de los experimentos o de los resultados. Distingue qué puede responderse con evidencia válida existente, qué requiere nuevas corridas, qué se declina y por qué, qué afirmaciones del manuscrito deben corregirse y qué controles deben cumplirse antes de resometer.

El objetivo no es defender las conclusiones antiguas, sino producir una versión metodológicamente sólida y coherente con la evidencia disponible.

**Principio rector.** Los experimentos realizados con el esquema de validación no aprobado **no constituyen evidencia cuantitativa**. No se recuperan sus métricas, no se combinan con las corridas actuales y no aparecen en tablas ni figuras. Solo pueden mencionarse, sin cifras, para explicar el proceso de desarrollo y por qué se cambió el pipeline.

### Jerarquía en caso de contradicción

1. Artefactos y configuraciones de las corridas definitivas del repositorio.
2. Decisiones metodológicas aprobadas y registradas por el equipo.
3. Plan estadístico congelado y resultados reproducibles derivados de él.
4. Este plan.
5. Reporte técnico histórico, solo como contexto de desarrollo.
6. Manuscrito antiguo, solo como registro de lo que debe responderse o corregirse.

### Por qué se descarta la evidencia histórica

- La validación cruzada original no cumplía el contrato metodológico aprobado después.
- Folds y repeticiones se trataron como observaciones independientes.
- La selección de modelos e hiperparámetros estuvo expuesta a los resultados que después se usaron para reportar desempeño.
- El ANOVA histórico presenta grados de libertad incompatibles entre el texto y la tabla.
- Algunas configuraciones descritas no coinciden con el código histórico, incluidos *batch size*, recorte de gradiente y paciencia.

---

## 2. Estado actual del estudio

### 2.1 Paneles de ROIs

Verificado contra `data/atlas/roi_sets.json`:

- El panel de 12 **está contenido** en el de 18.
- El de 18 añade exactamente seis regiones al de 12: `Precentral_L`, `Precentral_R`, `Frontal_Sup_R`, `Frontal_Sup_Orb_L`, `Frontal_Sup_Orb_R`, `Insula_L`.
- El de 39 fue definido por separado. **No es superconjunto** de 12 ni de 18, pero **tampoco es independiente**: comparte 6 regiones con el panel de 12 (`Frontal_Sup_L`, `Insula_R`, `Cingulum_Ant_L`, `Cingulum_Ant_R`, `Thalamus_L`, `Thalamus_R`) y 8 con el de 18.
- El de 116 corresponde al atlas AAL116 completo.
- Según la información aportada por el equipo, los paneles fueron elegidos por expertos en neurociencia a partir de revisión de literatura y antes de entrenar los modelos.

Queda por tanto **retirada** la afirmación histórica de que los 12 ROIs se obtuvieron mediante una ablación descendente guiada por desempeño. La formulación válida es **paneles anatómicos preespecificados**.

> **Formulación para el manuscrito:** «El panel de 12 ROIs está anidado en el de 18. El panel de 39 ROIs fue construido de forma separada a partir de una revisión más amplia de regiones y redes implicadas, y comparte seis regiones con el panel de 12.»

Antes de crear una tabla nueva, revisar `docs/tabla-consolidada-rois.docx`. La tabla suplementaria final debe incluir índice y nombre AAL116, red o sistema asociado, paneles a los que pertenece cada ROI, justificación y referencias.

### 2.2 Representación de conectividad y arquitectura

La representación `ordered` contiene matrices de correlación de Pearson calculadas en ventanas deslizantes, conservadas en posición fija y apiladas como canales. BrainNetCNN las recibe simultáneamente y no aplica recurrencia, atención ni memoria sobre ellas.

> **Formulación obligatoria:** representación multicanal de conectividad funcional calculada mediante ventanas deslizantes.

No debe describirse como una secuencia procesada temporalmente. **Tampoco debe afirmarse que el reposo carezca de estructura temporal**: la afirmación es arquitectónica, no sustantiva. El modelo actual no analiza las ventanas con un mecanismo secuencial explícito.

### 2.3 Validación definitiva

Validación cruzada estratificada repetida, con un **bucle exterior** de 10 folds × 5 repeticiones y una partición interna del entrenamiento para *early stopping*; semilla 42 y particiones emparejadas dentro de cada sitio; el pliegue exterior se usa solo para producir predicciones OOF; todas las ventanas de un sujeto permanecen en la misma partición; AUC OOF por repetición como métrica primaria; inferencia por sitio mediante remuestreo pareado y estratificado de sujetos.

**Regla léxica.** «Exterior» designa el bucle y los pliegues del procedimiento de validación cruzada. «Externo/a» queda reservado para cohortes independientes, que este estudio **no** tiene. Por tanto no debe usarse «validación externa» como nombre del diseño: en la literatura de modelos de predicción esa expresión significa validación en una cohorte independiente, y usarla aquí entraría en contradicción directa con §4.4.

Sin ANOVA/Tukey sobre folds, sin efecto global entre sitios, sin prueba confirmatoria de no inferioridad.

**Terminología obligatoria:** «50 evaluaciones de pliegues exteriores» o «10 folds × 5 repeticiones». Nunca «50 experimentos independientes» ni «50 evaluaciones externas». No llamar *nested cross-validation* completa al procedimiento: no existe búsqueda de hiperparámetros dentro de cada pliegue exterior.

### 2.4 Configuración BrainNetCNN de referencia

`e2e=4`, `e2n=8`, `dense=8`, `dropout=0.7`, `inter_dropout=0.6`, `leaky=0.33`, `l2_reg=0.05`, tasa `1e-4`, batch 32, máximo 300 épocas, paciencia 25, monitor `val_loss`, `min_delta=1e-5`, validación interna 0.15, semilla 42, FP32, Fisher z desactivado, ventana rectangular.

Ventanas efectivas verificadas contra los 16 `config.json`:

| Sitio | TR | Puntos | Ventana | Paso | Solape | Ventanas |
|---|---:|---:|---:|---:|---:|---:|
| NYU | 2.00 s | 172 | 60 TR = 120 s | 6 TR = 12 s | 90% | 19 |
| Peking | 2.00 s | 232 | 60 TR = 120 s | 6 TR = 12 s | 90% | 29 |
| NeuroIMAGE | 1.96 s | 257 | 61 TR ≈ 120 s | 6 TR ≈ 11.8 s | ≈90% | 33 |
| OHSU | 2.50 s | 74 | 48 TR = 120 s | 5 TR = 12.5 s | ≈90% | 6 |

---

## 3. Evidencia válida disponible

### 3.1 Desempeño de las 16 corridas definitivas

AUC OOF medio de cinco repeticiones:

| Sitio | Sujetos | 12 ROIs | 18 ROIs | 39 ROIs | 116 ROIs |
|---|---:|---:|---:|---:|---:|
| NYU | 177 | 0.590 | 0.600 | 0.528 | 0.539 |
| Peking | 183 | 0.564 | 0.568 | 0.622 | 0.603 |
| NeuroIMAGE | 39 | 0.474 | 0.620 | 0.507 | 0.519 |
| OHSU | 66 | 0.549 | 0.522 | 0.501 | 0.558 |

El máximo puntual corresponde a 18 ROIs en NYU y NeuroIMAGE, 39 en Peking y 116 en OHSU. **El panel de 12 no tiene el máximo puntual en ningún sitio.** Eso no prueba que otro panel sea superior: las estimaciones tienen incertidumbre y los contrastes secundarios son exploratorios.

Contraste principal 12−116:

| Sitio | Δ AUC | IC bootstrap bilateral 95% |
|---|---:|---:|
| NYU | +0.051 | [−0.007, +0.111] |
| Peking | −0.040 | [−0.105, +0.025] |
| OHSU | −0.009 | [−0.099, +0.083] |
| NeuroIMAGE | −0.045 | [−0.200, +0.103] |

Los cuatro intervalos incluyen cero. Como no se aprobó un margen científico de no inferioridad, estos resultados **no permiten declarar equivalencia, no inferioridad, superioridad ni optimalidad**. La conclusión válida es de estimación: las diferencias puntuales son pequeñas o moderadas según el sitio, pero la precisión disponible no permite resolverlas con firmeza.

### 3.2 Relación de los intervalos con AUC = 0.5

En 9 de las 16 combinaciones sitio–panel el IC bilateral puntual de AUC incluye 0.5:

| Sitio | 12 | 18 | 39 | 116 |
|---|---|---|---|---|
| NYU | excluye | excluye | **incluye** | **incluye** |
| Peking | excluye | excluye | excluye | excluye |
| NeuroIMAGE | **incluye** | excluye | **incluye** | **incluye** |
| OHSU | **incluye** | **incluye** | **incluye** | **incluye** |

**Reglas de interpretación:**

- Es un resumen descriptivo de intervalos puntuales, **no una familia de 16 pruebas de significancia**.
- No se usarán las expresiones «significativo» ni «no significativo».
- Incluir 0.5 no demuestra ausencia de señal; indica que, con la muestra y el procedimiento actuales, el intervalo también admite un desempeño compatible con azar.
- Excluir 0.5 tampoco demuestra utilidad clínica ni transporte a otras cohortes.

**Uso admisible de la literatura.** La literatura sobre ADHD-200 puede contextualizar la dificultad del problema y los riesgos de validación, pero **no debe usarse para afirmar que este rango de AUC era «esperado»**. En particular, no deben compararse directamente exactitudes binarias, exactitudes multiclase y AUC como si fueran la misma cantidad: la competencia ADHD-200 reporta exactitud en clasificación de tres vías y las estimaciones con conjuntos retenidos de la literatura ENIGMA provienen de MRI estructural. Tampoco debe afirmarse, sin una auditoría sistemática estudio por estudio, que los desempeños publicados del 95–98% se deban principalmente a errores metodológicos. La afirmación defendible es que la validación cruzada infla la estimación frente a conjuntos retenidos, y esa es una observación documentada de la literatura, no un juicio sobre trabajos concretos.

### 3.3 Brechas entrenamiento–validación

Brechas de AUC verificadas sobre `metrics_train.csv` y `metrics_val.csv`:

| Sitio | 12 | 18 | 39 | 116 |
|---|---:|---:|---:|---:|
| NYU | 0.082 | 0.169 | 0.374 | 0.394 |
| Peking | 0.168 | 0.273 | 0.340 | 0.363 |
| NeuroIMAGE | 0.335 | 0.221 | 0.346 | 0.395 |
| OHSU | 0.107 | 0.180 | 0.297 | 0.357 |

El patrón es monótono en NYU, Peking y OHSU, y no monótono en NeuroIMAGE. El panel de 116 ROIs presenta la mayor brecha en los cuatro sitios.

**Reglas de comunicación.** Es un hallazgo **descriptivo**: no tiene intervalos de confianza, no demuestra que los paneles reducidos generalicen mejor, no convierte al panel compacto en ganador y no sustituye los resultados OOF. Debe mostrarse junto al desempeño externo, para hacer visible que una brecha mayor no se tradujo de manera uniforme en menor AUC OOF.

**Ubicación acordada:** un párrafo conciso en el texto principal, con el patrón, su excepción y su limitación; la tabla completa y las curvas en el suplemento (§10).

> **Redacción para el manuscrito:** «Las brechas entrenamiento–validación de AUC aumentaron con la dimensionalidad en NYU, Peking y OHSU, aunque el patrón no fue monotónico en NeuroIMAGE. El panel de 116 ROIs presentó la mayor brecha en los cuatro sitios. Este diagnóstico describe mayor separación entre ajuste y validación interna, pero no demuestra que los paneles reducidos generalicen mejor, por lo que se interpreta junto con el desempeño OOF. Las curvas y los valores completos se presentan en el material suplementario.»

### 3.4 *Early stopping*

Los artefactos muestran dos cantidades distintas que **no deben confundirse**:

- mediana de `best_epoch = 300` en **12** de 16 corridas;
- mediana de `n_epochs = 300` en **14** de 16 corridas.

**No debe afirmarse que el criterio fue «deliberadamente permisivo»**, salvo que exista una decisión previa documentada que lo sostenga. La descripción correcta es: el máximo de 300 épocas gobernó gran parte de los entrenamientos, se restauraron los pesos de la mejor época observada, y este comportamiento se reporta y se reconoce como limitación.

No se harán nuevas corridas solo para modificar el *early stopping*: eso mezclaría esta revisión con una nueva búsqueda de hiperparámetros.

### 3.5 Redundancia entre ventanas

La similitud mediana entre ventanas adyacentes está entre **0.962 y 0.987** en las 16 corridas, y el código emite entre 1 y 3 avisos metodológicos por corrida. Es esperable con 90% de solape y limita cualquier afirmación fuerte sobre información dinámica.

Este diagnóstico **justifica** los contrastes de §8.1 y §8.3, pero **no predetermina** su resultado.

---

## 4. Declaraciones de transparencia obligatorias

### 4.1 Exposición del ajuste de hiperparámetros

La configuración BrainNetCNN se fijó antes de las 16 corridas definitivas, pero se eligió tras observar resultados durante el desarrollo. No existe un conjunto retenido de todas las decisiones de modelado. Esto interpela directamente a R2.5, que pregunta si los resultados provienen de datos nunca usados en entrenamiento ni ajuste.

**La respuesta debe darse por sitio, no en bloque.** Para los sitios y paneles utilizados durante el ajuste, la respuesta es no: sus datos influyeron en decisiones de modelado. Para los sitios que no fueron consultados durante el desarrollo, la configuración se aplicó de forma congelada, aunque su desempeño sigue estimándose mediante validación cruzada interna y no mediante una cohorte retenida independiente. La nota fechada determinará la respuesta aplicable a cada sitio.

**Antes de redactar**, el equipo debe producir una nota fechada que establezca:

- sitio o sitios observados durante el desarrollo;
- panel o paneles utilizados;
- métrica que guio las decisiones;
- configuraciones comparadas;
- fecha en que se congeló la configuración definitiva.

No usar «estas mismas cohortes» si el ajuste se realizó solo en un sitio. **No llamar «cota superior» a las métricas**: la descripción correcta es sesgo optimista de magnitud desconocida.

> **Formulación base:** «La configuración se fijó antes de las corridas comparativas definitivas, pero fue seleccionada durante una fase de desarrollo con exposición a resultados de [sitio/panel documentado]. Por ello, el desempeño absoluto —y especialmente el del escenario usado durante el desarrollo— puede presentar sesgo optimista. Los resultados de otros sitios, evaluados con la configuración congelada, reducen pero no eliminan las limitaciones derivadas de decisiones previas de diseño.»

Aplicar la misma configuración a los cuatro paneles mejora la comparabilidad, pero **no garantiza neutralidad entre paneles** si el ajuste se realizó con uno de ellos.

### 4.2 Relación entre NYU y el contraste 12−116: regla condicional

**Separar el hecho documentado de la inferencia pendiente.** Son hechos documentados: que **NYU es el único de los cuatro sitios donde Δ(12−116) es positivo** (+0.051; negativo en Peking, NeuroIMAGE y OHSU), y que NYU es el sitio con el que se desarrolló el pipeline. Lo que permanece **condicionado** a la reconstrucción de §4.1 es si el ajuste de BrainNetCNN o de sus hiperparámetros se realizó específicamente observando NYU con 12 ROIs.

No debe escribirse que NYU fue «probablemente» el lugar de ajuste, ni vincular el signo de Δ con el ajuste antes de que la cronología lo confirme.

La redacción final dependerá del resultado de la reconstrucción:

**(a) Si se confirma que el ajuste se realizó observando NYU con 12 ROIs:**

> «NYU fue utilizado durante el desarrollo y ajuste de la configuración con 12 ROIs. Es también el único sitio donde la estimación puntual de Δ(12−116) fue positiva. Esta coincidencia obliga a interpretar el resultado de NYU con especial cautela, porque puede reflejar optimismo asociado a la selección; no demuestra, por sí sola, la existencia ni la magnitud de ese sesgo.»

**(b) Si NYU participó en el desarrollo pero no en el ajuste de BrainNetCNN ni del panel:** los dos hechos se reportan por separado y no se sugiere que el proceso de desarrollo explique el signo de Δ.

**(c) Si el ajuste ocurrió en otro sitio:** los signos se reportan normalmente en Resultados y no se construye una narrativa especial sobre NYU.

**El papel de NYU en el desarrollo se declara en los tres desenlaces**, porque es un hecho documentado. Lo que **no** se declara siempre es su vínculo con el signo de Δ: eso solo se afirma en la rama (a). Los otros sitios pueden presentarse como evaluaciones adicionales con configuración congelada, pero **no como validación en cohorte externa ni como demostración de transporte**: cada sitio se entrenó y evaluó internamente.

### 4.3 Corrección de la narrativa de ablación

Retirar del resumen, introducción, métodos, resultados, discusión, conclusiones y *Highlights*: «descending iterative ablation analysis», «ROIs discovered», «highest discriminative weight» e «identified biomarkers». Sustituir por «paneles anatómicos preespecificados a partir de revisión de literatura y criterio experto».

La carta debe **declarar explícitamente** que se corrigió una descripción documental inexacta de la versión anterior. Una corrección declarada es defendible; una reescritura silenciosa, si el revisor compara versiones, no lo es.

Ventaja sustantiva: si los paneles son a priori, **no hay selección supervisada que repetir dentro de cada fold**, y R2.12 queda resuelto de forma limpia.

### 4.4 Alcance clínico

| Retirar | Usar |
|---|---|
| diagnóstico | clasificación experimental |
| biomarcadores | panel anatómico bajo evaluación |
| validación externa | evaluación interna en múltiples sitios |
| modelado temporal | representación multicanal calculada mediante ventanas |
| óptimo, ganador, suficiente | estimación por sitio, sin declarar ganador |

No afirmar aplicabilidad clínica, utilidad diagnóstica, generalización a otros centros ni superioridad sobre profesionales o instrumentos clínicos. Alcanza al **título**, al resumen y a los *Highlights*.

---

## 5. Estrategia editorial

El manuscrito revisado cambia arquitectura principal, protocolo de validación, número de sitios, resultados, análisis estadístico, explicación de los paneles y alcance de las conclusiones. El equipo **no debe decidir unilateralmente** que basta con una revisión ordinaria ni asumir que debe retirar el artículo.

### 5.1 Consulta al editor y ejecución en paralelo

Se enviará una consulta breve y transparente al editor que: explique que al atender los comentarios sobre validación se detectaron problemas en el análisis original; indique que se rehízo el pipeline y se retirarán todas las métricas históricas; resuma los cambios de arquitectura, cohortes y análisis; y pregunte si prefiere recibirlo como revisión sustancial o mediante retiro y nuevo envío.

**La campaña no queda bloqueada por esa respuesta.** La consulta resuelve la vía de presentación; no cambia las preguntas científicas ni una sola de las diez corridas, que responden solicitudes directas de los revisores. Se ejecutan en paralelo, con cuatro condiciones previas:

1. Congelar el manifiesto exacto de las diez corridas.
2. Registrar que no se modificarán configuraciones a partir de resultados parciales.
3. Comprometerse a reportar resultados favorables, nulos o adversos con las mismas reglas.
4. Mantener pendientes de la respuesta editorial únicamente el formato de envío, la carta definitiva y la organización final del manuscrito.

Esto no equivale a afirmar que ejecutar en paralelo carezca de todo costo: el editor podría solicitar un alcance distinto. La decisión se justifica por la pertinencia científica de las corridas, el plazo disponible y el hecho de que responden peticiones explícitas de los revisores.

> **Redacción para el plan:** «La consulta al editor se enviará al inicio. En paralelo, y después de congelar el manifiesto, se ejecutarán el bloque sin entrenamiento y las diez corridas preespecificadas. Ninguna configuración se adaptará a resultados intermedios. La respuesta editorial determinará la vía y el formato de presentación, no las decisiones científicas ya congeladas.»

### 5.2 Contenido de la carta final

Abrir con la corrección, enumerar lo retirado, enumerar la evidencia nueva y moderar explícitamente el alcance. Declarar el cambio de título y de arquitectura. No presentar los AUC actuales como «el rango esperado».

### 5.3 Contribución central

> «El estudio aporta una evaluación multi-cohorte de paneles anatómicos preespecificados bajo un protocolo que mantiene juntas todas las observaciones de cada sujeto, produce predicciones OOF y cuantifica la incertidumbre por sitio. Su contribución es metodológica y comparativa, no una mejora de exactitud ni una validación clínica.»

**Dependencia con el equipo de literatura.** La tabla comparativa con trabajos recientes pondrá nuestro AUC junto a exactitudes publicadas muy superiores. Debe incluir una columna que distinga **validación cruzada de conjunto retenido**; sin ella, la comparación es engañosa en nuestra contra.

---

## 6. Comentarios que se responden sin nuevas corridas

| Tema | Evidencia actual | Acción |
|---|---|---|
| Validación y fuga (R2.5, R2.6) | 10×5, validación interna, OOF por sujeto, ventanas agrupadas, *fingerprints* emparejados | Reescribir Métodos, aportar auditoría suplementaria y añadir §4.1 |
| Selección de ROIs (R2.12) | Paneles a priori; sin selección supervisada dentro del pipeline | Corregir narrativa y añadir tabla anatómica con referencias |
| Más sitios (R1.2, R2.4) | Cuatro sitios analizados por separado | Reportar evaluación multisitio interna; no transporte entre sitios |
| Comparación de paneles | 16 corridas y estimación 12−116 con IC bilaterales | Reportar sin equivalencia, no inferioridad ni ganador global |
| ANOVA/Tukey (R2.13) | Folds y repeticiones no son réplicas independientes | Declinar y explicar el reemplazo |
| Convergencia y sobreajuste (R1.4, R2.11) | Historias de entrenamiento, `best_epoch`, `n_epochs`, brechas | Párrafo principal más figuras suplementarias |
| Análisis de errores | Predicciones OOF completas por sujeto, sitio y panel | Frecuencia, persistencia, concordancia y covariables |
| Alcance clínico (R2.14a) | Los resultados no soportan uso diagnóstico | Revisar título, resumen, *Highlights* y conclusiones |

### 6.1 Por qué no se añade Tukey

Tukey controla la multiplicidad dentro de un marco que **sigue exigiendo observaciones independientes**. Folds y repeticiones comparten sujetos y conjuntos de entrenamiento; tratarlos como réplicas viola ese supuesto, de modo que el post-hoc heredaría el problema del ANOVA en lugar de resolverlo. Añadirlo daría apariencia de rigor sobre una base inválida.

El sustituto es más exigente, no menos: predicciones OOF a nivel de sujeto, diferencias pareadas y remuestreo estratificado dentro de cada sitio. Este es el único punto en el que se rechaza abiertamente una petición concreta de un revisor, y su redacción debe cuidarse en consecuencia.

### 6.2 Análisis de errores permitido

Frecuencia de error por sujeto, sitio y panel; errores persistentes a lo largo de las cinco predicciones OOF; clase verdadera y confianza media; coincidencia de errores entre 12 y 116 y entre los cuatro paneles; asociación descriptiva con movimiento y variables fenotípicas disponibles; heatmap sujeto × panel como visual exploratorio.

**No excluir sujetos definidos por su propio error ni recalcular las métricas principales tras retirarlos.** El ejercicio exploratorio ya realizado lo confirma: al excluir los sujetos siempre mal clasificados con 12 ROIs, Δ(12−116) **aumenta** de +0.051 a +0.090 en NYU y cambia de signo en Peking, porque el criterio de exclusión está definido sobre uno de los brazos del contraste. Las versiones «sin sujetos difíciles» solo pueden mostrarse como demostración diagnóstica de ese sesgo.

### 6.3 Curvas de aprendizaje por tamaño muestral

Se declinan. Con tamaños de 39 a 183 sujetos y los intervalos observados, el submuestreo añadiría otra capa de varianza y un bloque metodológico amplio sin resolver de forma fiable la pendiente de aprendizaje. Se reconocerá que el escalamiento con más datos no fue evaluado.

---

## 7. Restricción de sitios en los contrastes nuevos

Los contrastes de arquitectura y sensibilidad se restringen a **NYU y Peking**, los sitios con mayor muestra y estimaciones más estables (n=177 y 183; anchos de IC de 0.118 y 0.129, frente a 0.182 en OHSU y 0.303 en NeuroIMAGE).

Esta restricción **debe declararse antes de observar los nuevos resultados**. No implica que OHSU o NeuroIMAGE carezcan de valor: los cuatro sitios permanecen en el análisis principal de paneles y en el contraste `static`.

---

## 8. Campaña experimental: diez corridas

Todas las corridas usan la versión congelada, los mismos sujetos y el mismo `split_fingerprint` que su comparador dentro del sitio. **No se ajustarán hiperparámetros a partir de resultados intermedios.**

### 8.1 BrainNetCNN `static` frente a `ordered` — 4 corridas

**Pregunta.** ¿Cómo cambia el desempeño al sustituir la representación multicanal por una única matriz de Pearson calculada con toda la serie? (R2.2)

**Corridas.** BrainNetCNN, 12 ROIs, representación `static`, en NYU, Peking, NeuroIMAGE y OHSU.

**Constantes.** Configuración de entrenamiento, folds, semilla y `class_weight` del baseline de cada sitio.

**Diferencias inevitables que deben declararse.**

1. `static` estima cada correlación con toda la serie; `ordered` usa ventanas más cortas. La razón puntos totales / puntos por ventana es **2.87** en NYU, **3.87** en Peking, **4.21** en NeuroIMAGE y **1.54** en OHSU.
2. En la implementación actual el número de ventanas actúa como número de canales y modifica la capacidad del modelo. Verificado con el constructor real, con 12 ROIs:

| Configuración | Canales | Parámetros |
|---|---:|---:|
| `static` | 1 | 1 361 |
| OHSU `ordered` | 6 | 1 841 |
| NYU `ordered` | 19 | 3 089 |
| Peking `ordered` | 29 | 4 049 |
| NeuroIMAGE `ordered` | 33 | 4 433 |

Por tanto, este contraste compara **dos procedimientos completos** y no aísla exclusivamente «dinámica frente a estática». **No igualar artificialmente el número de parámetros**: eso introduciría otro modelo y una nueva búsqueda de configuración. La diferencia de capacidad se reporta como limitación.

**Análisis.** Diferencia pareada `ordered − static` por sitio con el mismo procedimiento bootstrap; métricas secundarias descriptivas; sin efecto combinado entre sitios.

### 8.2 BrainNetCNN frente a LSTM-128 — 2 corridas

**Pregunta acotada.** ¿La arquitectura recurrente usada en la versión original reproduce su desempeño cuando se evalúa con el pipeline actual? (R1.3, R2.9)

**Corridas.** NYU y Peking, 12 ROIs, `ordered`, LSTM con `units=128`, `dropout=0`, `bidirectional=False`.

**Protocolo de entrenamiento.** El protocolo actual estandarizado: tasa `1e-4`, batch 32, máximo 300 épocas, paciencia 25, `val_loss`, `min_delta=1e-5`, validación interna 0.15 y los mismos folds externos. **No describirlo como «configuración histórica exacta»**: el reporte y el código históricos no son plenamente consistentes sobre *batch size*, recorte de gradiente y paciencia, de modo que esa configuración no es un objeto recuperable sin ambigüedad.

**Asimetría que debe declararse.** Con 66 aristas de entrada, LSTM-128 tiene aproximadamente **99 969** parámetros y ninguna regularización por defecto. BrainNetCNN tiene **3 089** en NYU y **4 049** en Peking, con dropout 0.7/0.6 y L2 0.05, y su configuración además se ajustó observando resultados (§4.1). El contraste no compara mecanismos en igualdad de condiciones.

> **Interpretación permitida:** «La arquitectura recurrente de la versión original fue reevaluada bajo el protocolo corregido. Como los modelos difieren sustancialmente en capacidad, regularización y mecanismo, el contraste documenta el efecto práctico de la elección dentro de este estudio, pero no demuestra superioridad arquitectónica general.»

**No ajustar la LSTM después de ver los resultados.** Si falla, no aumentar regularización ni cambiar unidades dentro de esta campaña.

**Sobre GRU y Transformer.** R2.9 los menciona por nombre y ambos están implementados en `src/kerasmodels/`. Se declinan porque ampliarían el alcance, la multiplicidad y el número de resultados que deben interpretarse. Con la precisión observada, diferencias modestas probablemente producirían intervalos amplios. La reevaluación de la LSTM ya responde a la alternativa de «otro marco de aprendizaje profundo» solicitada por el revisor, y el estudio no es un contraste de arquitecturas sino una evaluación de una hipótesis anatómica. Se declaran como trabajo futuro.

No debe afirmarse que ninguna cohorte de este tamaño **podría** resolver esas comparaciones: eso depende de la magnitud real de las diferencias, que no se conoce de antemano.

### 8.3 Sensibilidad de longitud y paso — 4 corridas

**Pregunta.** ¿Las conclusiones dependen de la elección de longitud de ventana y paso? (R1.1, R2.10)

**Por qué no se evalúa una ventana de 60 s.** El preprocesamiento ATHENA aplica un pasa-banda cuyo límite inferior es 0.009 Hz, correspondiente a un periodo de ≈111 s. La condición de 60 s queda por debajo de `1/f_mín` y aumentaría el riesgo de fluctuaciones espurias en las correlaciones deslizantes. Evaluarla de forma defendible exigiría controles con datos sustitutos o pruebas de nulidad que quedan fuera del alcance de esta revisión. Se descarta por ese riesgo, **no** porque toda ventana menor sea automáticamente inválida: `1/f_mín` es una recomendación conservadora, no una demostración de invalidez.

**Diseño: se cambia una dimensión cada vez.**

| Condición | Ventana | Paso | Qué cambia | Ventanas NYU / Peking | Parámetros NYU / Peking |
|---|---:|---:|---|---|---|
| Referencia existente | 120 s | 12 s | — | 19 / 29 | 3 089 / 4 049 |
| **A** | 140 s | 12 s | longitud, paso fijo | 18 / 28 | 2 993 / 3 953 |
| **B** | 120 s | 24 s | paso y solape, longitud fija | 10 / 15 | 2 225 / 2 705 |

**Corridas.** BrainNetCNN, 12 ROIs, `ordered`, rectangular, Fisher z desactivado; 2 condiciones × 2 sitios = 4.

Este diseño evita un cambio extremo de paso que reduciría drásticamente el número de canales y la capacidad del modelo. Aun así, **cambiar el paso modifica el número de ventanas** y por tanto la entrada y el número de parámetros. Debe informarse, y **no se afirmará haber aislado causalmente el efecto del solape**.

**Relación con la configuración enviada.** En NYU y Peking, 140 s corresponde a **70 TR**, la longitud de ventana empleada en el manuscrito enviado. El paso, en cambio, es distinto: la configuración histórica usaba 2 TR (≈140/4 s) y la condición A usa 6 TR (140/12 s).

> **Redacción para la respuesta al revisor:** «La condición de 140 s recupera la longitud de 70 TR empleada en el manuscrito enviado, pero conserva el paso actual de 12 s. Por tanto, permite reevaluar la longitud original bajo el pipeline corregido, no reproducir la configuración histórica completa.»

**Controles previos y posteriores.** Ventana y paso efectivos en TR y segundos; número de ventanas; número de parámetros; similitud mediana entre ventanas adyacentes; avisos metodológicos; igualdad de sujetos y `split_fingerprint` con el baseline.

### 8.4 Resumen y regla de parada

| Bloque | Corridas | Sitios |
|---|---:|---|
| `static` frente a `ordered` | 4 | los cuatro |
| LSTM-128 frente a BrainNetCNN | 2 | NYU, Peking |
| Sensibilidad 140/12 s | 2 | NYU, Peking |
| Sensibilidad 120/24 s | 2 | NYU, Peking |
| **Total** | **10** | |

**Campaña cerrada.** No se elegirá una segunda ronda a partir de estos resultados antes de resometer el manuscrito.

---

## 9. Experimentos que no se incluyen

### 9.1 Baseline lineal

No se implementa en esta revisión. El comentario del revisor admite «métodos tradicionales **o** otros marcos de aprendizaje profundo», y la reevaluación de la LSTM satisface la segunda alternativa. Una regresión logística rigurosa no es añadir una capa: exige estandarización ajustada dentro de cada fold, política de regularización y selección de su intensidad sin fuga, integración de artefactos, pruebas y una versión nueva del pipeline.

> **Nota de contingencia:** «Si el editor solicita explícitamente un baseline lineal, se preparará un protocolo específico con penalización preespecificada y estandarización ajustada exclusivamente dentro de cada fold de entrenamiento. La penalización fija evita una búsqueda adicional de ese hiperparámetro, pero no sustituye los controles de fuga, artefactos y pruebas del pipeline.»

Si se solicita, será objeto de un plan breve separado y de una versión de código verificada.

### 9.2 Otros experimentos excluidos

GRU y Transformer; comparación exhaustiva de arquitecturas; `ordered` frente a `permuted`; `static` con 116 ROIs; LOSO o transporte entre sitios; otro atlas; curvas de aprendizaje por submuestreo; nueva búsqueda de hiperparámetros; nuevas variantes de Fisher, *shrinkage* o regularización ya exploradas.

Motivo general: no responden una inquietud prioritaria mejor que los diez experimentos definidos, o requieren un diseño adicional que no cabe con seguridad en esta revisión. **No afirmar que BrainNetCNN sea invariante a cualquier permutación de canales**: no se ha verificado y no es una pregunta necesaria para las afirmaciones actuales.

---

## 10. Figuras y tablas

### 10.1 Texto principal

1. Figura: perfiles de AUC por panel y sitio, con IC bilaterales.
2. Figura: forest plot del contraste 12−116.
3. Tabla: desempeño de las 16 corridas por sitio y panel.
4. Tabla compacta: contrastes nuevos (`ordered − static`, BrainNetCNN − LSTM y sensibilidades frente a la referencia).
5. Párrafo sobre brechas entrenamiento–validación (§3.3), sin tabla ni figura.

### 10.2 Material suplementario

Tabla anatómica de ROIs con referencias; auditoría de comparabilidad y ausencia de fuga; brechas de AUC y *accuracy* completas; curvas de convergencia; ROC OOF; matrices de confusión; análisis de errores y heatmap sujeto × panel; diagnósticos de ventanas y capacidad de los modelos; detalle completo de los contrastes nuevos.

### 10.3 Reglas para ROC y matrices de confusión

- Cada sujeto tiene cinco predicciones OOF. Mostrar **las cinco ROC por repetición y un resumen visual**; no apilar las predicciones como si fueran sujetos independientes.
- Si se muestra una ROC basada en probabilidades promediadas, rotularla como **estimando distinto** y no hacerla pasar por el AUC medio primario.
- Matrices de confusión con umbral fijo 0.5, indicado en el pie.
- No optimizar el umbral usando las mismas observaciones evaluadas.

### 10.4 Curvas de convergencia

Mediana por época, banda intercuartílica, entrenamiento frente a validación interna y número de folds aún activos. Mostrar por separado `best_epoch` y `n_epochs`; no sugerir que alcanzar 300 implique que los pesos de la época 300 fueron los restaurados.

---

## 11. Orden de dependencias

No se fija calendario. El orden lógico es:

**Antes de cualquier corrida**

1. Aprobar las decisiones de §12.
2. Reconstruir y registrar la cronología de ajuste de hiperparámetros (§4.1), de la que depende la rama aplicable de §4.2.
3. Enviar la consulta al editor (§5.1); no bloquea lo demás.
4. Congelar la versión del repositorio y el manifiesto de las diez configuraciones.
5. Asignar responsable y verificador por bloque.

**Independiente de las corridas** (puede avanzar en paralelo desde el inicio)

Matriz comentario–respuesta; corrección de la narrativa de selección de ROIs; figuras y tablas derivadas de las 16 corridas; auditoría de validación; análisis de errores; y redacción del alcance y las limitaciones **antes** de ver resultados nuevos.

**Campaña**

Prueba de humo por tipo de configuración; verificación de la configuración resuelta antes del entrenamiento completo; distribución de corridas sin modificar código; validación de artefactos, sujetos, folds, hashes y predicciones finitas. **No interpretar resultados parciales ni adaptar la campaña.**

**Análisis**

Confirmar comparabilidad con el baseline correspondiente; calcular contrastes pareados e IC por sitio; generar la tabla compacta y el suplemento; reconciliar métricas por dos vías cuando sea viable; registrar resultados favorables, nulos y adversos con el mismo nivel de detalle.

**Redacción y control**

Reescribir título, resumen, *Highlights*, Métodos, Resultados, Discusión y Conclusiones; preparar la respuesta punto por punto y la carta; revisión cruzada por dos integrantes que no hayan generado las mismas tablas; verificar cada afirmación contra un artefacto o una referencia.

---

## 12. Decisiones que debe aprobar el equipo

| ID | Decisión | Recomendación |
|---|---|---|
| D1 | Vía editorial | Consultar al editor; ejecutar en paralelo con manifiesto congelado |
| D2 | Cambio de título, arquitectura y conclusión | Aprobar y declararlo explícitamente |
| D3 | Exposición del ajuste de hiperparámetros | Declaración obligatoria; producir la nota fechada |
| D4 | Relación NYU–ajuste–Δ(12−116) | Declarar el papel de NYU en el desarrollo en todos los casos; aplicar la rama correspondiente de §4.2 y vincular el ajuste con el signo de Δ **únicamente si la cronología lo confirma** |
| D5 | Corrección de la narrativa de ablación | Declaración obligatoria, no corrección silenciosa |
| D6 | Alcance de `static` | 12 ROIs en los cuatro sitios |
| D7 | Alcance de LSTM y sensibilidad | NYU y Peking; alcance exploratorio, con estimaciones principales en la tabla compacta del texto principal y detalle en suplemento |
| D8 | Protocolo de la LSTM | Arquitectura original con protocolo actual estandarizado |
| D9 | Condiciones de sensibilidad | 140/12 s y 120/24 s |
| D10 | Baseline lineal | No implementar; conservar la nota de contingencia |
| D11 | Brecha train−val | Párrafo en texto principal; tabla y curvas en suplemento |
| D12 | Resultados históricos | Excluir todas las métricas; conservar solo contexto |
| D13 | Campaña cerrada | Aprobar diez corridas y prohibir ajuste adaptativo posterior |

---

## 13. Control final antes de resometer

Dos integrantes firman una lista que confirme:

1. No aparece ninguna métrica histórica como evidencia.
2. Cada cifra del manuscrito se reconcilia con un artefacto actual.
3. La representación se describe como multicanal y no como secuencia procesada por BrainNetCNN.
4. No se llama *nested CV* completa al protocolo.
5. No se tratan folds ni repeticiones como muestras independientes.
6. La cronología de ajuste y el papel de NYU están declarados con precisión, se aplicó la rama correcta de §4.2 y el signo de Δ solo se vincula al ajuste si la cronología lo confirma.
6 bis. La respuesta a R2.5 está dada **por sitio** y no en bloque, y no se usa «validación externa» como nombre del diseño en ninguna parte del manuscrito.
6 ter. No queda ninguna afirmación de imposibilidad donde lo defendible es una afirmación sobre riesgo, alcance o precisión.
7. Los paneles se describen como a priori y se corrigió explícitamente la narrativa antigua.
8. Los cuatro sitios se presentan como evaluaciones internas, no como transporte entre centros.
9. No hay afirmaciones de equivalencia, no inferioridad, superioridad global u optimalidad.
10. No hay afirmaciones clínicas ni de biomarcadores, incluidos título y *Highlights*.
11. Los 9/16 intervalos que incluyen 0.5 se describen como diagnóstico puntual, no como 16 pruebas.
12. Las brechas train−val se presentan como descriptivas y junto al desempeño OOF.
13. Los conteos de `best_epoch` y `n_epochs` no se confunden.
14. El contraste `static` reconoce las diferencias en precisión de la correlación y en número de parámetros.
15. La comparación con LSTM reconoce las diferencias de capacidad y regularización.
16. Cada nueva corrida conserva sujetos, folds y configuración no manipulada respecto de su baseline.
17. No se excluyeron sujetos por su error para recalcular desempeño.
18. Resultados favorables y desfavorables se reportan con las mismas reglas.

---

## 14. Criterio de éxito

La revisión será exitosa si responde de forma transparente las inquietudes metodológicas, **aunque las métricas no mejoren y aunque algunos experimentos nuevos favorezcan configuraciones más simples**. No depende de demostrar que 12 ROIs sea mejor.

Conclusión máxima permitida con la evidencia actual:

> «Los paneles reducidos permiten estudiar una hipótesis anatómica con menor dimensionalidad. Las estimaciones puntuales difieren en magnitud y dirección entre sitios y presentan incertidumbre considerable; este diseño no permite decidir si esas diferencias reflejan heterogeneidad real o variabilidad de muestreo. Los datos no establecen que 12 ROIs sea equivalente, no inferior, superior u óptimo frente a 116. El trabajo aporta una comparación multi-cohorte con validación por sujeto, incertidumbre explícita y una delimitación transparente de sus condiciones de generalización.»

La primera frase deberá revisarse si los nuevos contrastes modifican su pertinencia, pero nunca sustituirse por una declaración causal o clínica que los datos no soporten.

---

## 15. Referencias

- Brown, M. R. G., et al. (2012). ADHD-200 Global Competition: diagnosing ADHD using personal characteristic data can outperform resting state fMRI measurements. *Frontiers in Systems Neuroscience*, 6, 69. https://pmc.ncbi.nlm.nih.gov/articles/PMC3460316/
- Leonardi, N., & Van De Ville, D. (2015). On spurious and real fluctuations of dynamic functional connectivity during rest. *NeuroImage*, 104, 430–436. https://doi.org/10.1016/j.neuroimage.2014.09.007
- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91. https://pubmed.ncbi.nlm.nih.gov/16504092/
- Zalesky, A., & Breakspear, M. (2015). Towards a statistical test for functional connectivity dynamics. *NeuroImage*, 114, 466–470. https://doi.org/10.1016/j.neuroimage.2015.03.047
- Zhang-James, Y., Razavi, A. S., Hoogman, M., Franke, B., & Faraone, S. V. Machine Learning and MRI-based Diagnostic Models for ADHD: Are We There Yet? *Journal of Attention Disorders*. https://doi.org/10.1177/10870547221146256

---

**Artefactos usados para la verificación numérica de este documento:** `results/runs/*/*/config.json`, `metrics_train.csv`, `metrics_val.csv`; `analysis/roi_comparison/outputs/tables/`; `data/atlas/roi_sets.json`; `src/kerasmodels/` para los conteos de parámetros con el constructor real.
