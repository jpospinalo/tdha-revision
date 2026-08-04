# Baseline de ML, condiciones de ventana, §2.7 y coherencia de la sección 2

**Fecha:** 3 de agosto de 2026
**Documento:** `Manuscript_Methods_Results_EN.docx`
**Estado:** los cambios de redacción están aplicados. Las dos preguntas sobre experimentos nuevos
quedan con recomendación, no ejecutadas: requieren decisión del equipo y una enmienda fechada al
plan antes de correr nada.

---

## 1. ¿Vale la pena hacer un baseline de ML? **Sí. Es el experimento con mejor relación valor-costo que queda.**

### Por qué

**Es la objeción más previsible que puede recibir el artículo.** Un trabajo de aprendizaje profundo
sobre muestras de 39 a 183 sujetos, con AUC de validación entre 47 y 62 puntos, invita
inevitablemente a la pregunta «¿hacía falta una red?». Y el texto actual la invita de forma
explícita, porque nombra el baseline entre lo que no se hizo. Nombrar una ausencia sin cerrarla es
peor que no nombrarla.

**El resultado es informativo en las dos direcciones**, que es la prueba de si un experimento vale
la pena:

- Si el modelo lineal empata con BrainNetCNN, eso **refuerza la tesis del propio artículo**: a estos
  tamaños muestrales la arquitectura no es la restricción activa, y la conclusión de cautela pasa de
  ser una advertencia a ser un hallazgo respaldado.
- Si BrainNetCNN lo supera con claridad, el artículo tendría por primera vez una justificación
  empírica para usar aprendizaje profundo, que hoy no tiene.

**El costo es bajo y la parte cara ya está construida.** Lo laborioso de este estudio no es entrenar
modelos, sino la infraestructura: particiones congeladas, huella de partición, predicciones
out-of-fold, bootstrap pareado. Un baseline lineal reutiliza todo eso y corre en CPU en minutos.

**Y mi propia objeción en el manuscrito ya no se sostiene.** El texto decía que hacerlo con rigor
«requeriría estandarización ajustada dentro de cada pliegue y una política de regularización
preespecificada». Eso era cierto cuando no existía el andamiaje; hoy es un `Pipeline` de
scikit-learn dentro del bucle de pliegues que ya está escrito. Mantener esa frase sería defender una
ausencia con un argumento que dejó de ser verdadero.

### Especificación que propongo, para congelar antes de correr

| Elemento | Valor |
|---|---|
| Características | Triángulo inferior vectorizado de la matriz de conectividad estática: 66 rasgos con 12 ROIs, 6.670 con 116 |
| Modelo | Regresión logística con regularización L2 |
| Hiperparámetro | C sobre una rejilla fija {10⁻³, 10⁻², 10⁻¹, 1, 10}, elegida por AUC en la partición de validación interna y **solo** en ella |
| Estandarización | Ajustada dentro de cada pliegue, nunca sobre el conjunto completo |
| Particiones | Las existentes. Misma semilla y misma huella de partición, para que la comparación sea pareada |
| Sitios y grupos | Los cuatro sitios, con 12 y 116 ROIs: ocho corridas |
| Incertidumbre | Bootstrap pareado estratificado por clase, 10.000 remuestreos. Con un modelo lineal el costo es despreciable |
| Reporte | Como quinta dimensión de sensibilidad, no como comparación primaria |

### La condición que no es negociable

Este es un análisis nuevo, decidido después de ver resultados. No es lo mismo que preespecificar, y
conviene decirlo sin adornos. La mitigación es la que el equipo ya aplicó con las diez corridas de
sensibilidad: **enmienda fechada al plan, con la especificación completa escrita antes de ejecutar,
y compromiso explícito de reportar el resultado en cualquier dirección.** Si se corre primero y se
decide después si entra, el baseline pasa de fortalecer el artículo a ser exactamente el tipo de
flexibilidad que esta revisión lleva meses cerrando.

### Mientras tanto

La frase defensiva quedó reemplazada por un enunciado positivo de alcance:

> The four dimensions above define the scope of the evaluation. Traditional machine learning
> baselines, alternative sequence architectures, learning curves by sample size, and atlases other
> than AAL116 lie outside it and are taken up in the Discussion.

Si se hace el baseline, esa frase se ajusta para retirarlo de la lista.

---

## 2. ¿Completar la Tabla 5 con OHSU y NeuroIMAGE? **No, y para OHSU hay una razón técnica que zanja el asunto.**

Verifiqué la aritmética de ventanas con la fórmula `(volúmenes − ventana) ÷ paso + 1`, y reproduce
exactamente los ocho valores publicados en las Tablas 2 y 5: 19, 29, 33 y 6 en la configuración de
referencia, y 18/10 y 28/15 en las condiciones A y B de NYU y Peking. La proyección a los dos sitios
faltantes es por tanto fiable:

| Sitio | Volúmenes | Referencia | Condición A (140 s) | Condición B (paso 24 s) |
|---|---:|---:|---:|---:|
| NYU | 172 | 19 | 18 | 10 |
| Peking | 232 | 29 | 28 | 15 |
| NeuroIMAGE | 257 | 33 | 32 | **17** |
| OHSU | 74 | 6 | **4** | **3** |

**OHSU es degenerado.** Con 74 volúmenes, la condición B dejaría **tres ventanas** por sujeto y la A
dejaría cuatro. Con tres ventanas la condición «con ventanas» es prácticamente indistinguible de la
conectividad estática, de modo que el experimento no mediría el efecto del paso: mediría la
desaparición del propio esquema de ventanas. Sería una comparación confundida por construcción, y un
revisor atento lo vería.

**NeuroIMAGE es viable pero no informativo.** La condición A cambiaría 33 ventanas por 32: un 3% de
la entrada, un efecto nulo por diseño. La B sí es una manipulación real, 33 frente a 17, pero con
n=39 y un intervalo primario de 0,303 de ancho, cualquier resultado vendría con una barra de treinta
puntos de AUC. Añadiría dos intervalos anchos que incluyen el cero, es decir, ruido con apariencia
de dato.

**Y hay un costo que no es técnico.** La restricción a NYU y Peking se declaró *antes* de correr las
condiciones nuevas. Levantarla ahora, después de ver los resultados, es precisamente la clase de
flexibilidad post hoc que el artículo declara no haber ejercido. Se ganaría simetría visual en una
tabla y se perdería una afirmación de procedimiento que hoy es cierta.

**Lo que sí hice**, porque mejora el argumento sin correr nada: la justificación de la restricción
era solo estadística —los intervalos de OHSU y NeuroIMAGE son más anchos— y ahora incorpora la razón
técnica, que es más fuerte y verificable:

> OHSU is in any case unsuitable for the windowing conditions: its 74 volumes yield 6 windows in the
> reference configuration and would leave 4 under condition A and 3 under condition B, too few for
> the manipulation to be separable from the loss of windowing itself.

Cuesta 35 palabras y convierte un juicio de precisión en una restricción del dato.

---

## 3. §2.7 reescrita

De 462 a 413 palabras, con la estructura del lector y la del revisor separadas como en §2.6.

**Escritura defensiva retirada.** Tres formulaciones se enunciaban por negación, definiendo el
método por lo que no es:

| Antes | Ahora |
|---|---|
| «…**nor is it** a full nested cross-validation, because no hyperparameter search is performed within each outer fold» | Retirada. No aportaba al lector: nadie había afirmado que lo fuera |
| «Aggregating by repetition **avoids the problem of** fold-wise AUC values, which are extremely discrete» | «Aggregating by repetition rather than by fold **is what makes the estimate stable**, since an individual fold holds between three and eighteen subjects» |
| «Analysis of variance and post-hoc tests **were not applied**… **would violate** the assumption» | Movida al bloque del revisor, en una sola frase |

**Un bloque marcado para el revisor.** Igual que «*Exposure to hyperparameter tuning*» en §2.6, ahora
hay «**Scope of the estimates**», que reúne las tres limitaciones del diseño: los intervalos son
condicionales a las predicciones y particiones existentes, el estimando es el pipeline completo y no
un modelo final, y sin margen de no inferioridad el análisis es estimativo. La ausencia de ANOVA
queda ahí, donde corresponde, y no interrumpe la exposición del método.

**El cuerpo para el lector** quedó en cuatro párrafos que se leen como un procedimiento: qué métrica,
por qué esa métrica, cómo se cuantifica la incertidumbre y cómo se leen los intervalos.

---

## 4. Revisión de ilación de la sección 2

Leí la sección completa de corrido. Dos defectos reales, ambos introducidos por el reordenamiento de
la ronda anterior, y ambos corregidos.

### 4.1 Las tablas estaban numeradas fuera de orden

Al intercambiar §2.3 y §2.4, la tabla de ventanas se quedó con el número 3 y la de composición del
grupo de 12 ROIs con el número 2, pero el orden de lectura pasó a ser el inverso. El texto mencionaba
**Tabla 1, Tabla 3, Tabla 2, Tabla 4**. Es un defecto que las editoriales marcan en producción.

Corregido: la tabla de ventanas es ahora la **Tabla 2** y la de composición del grupo compacto la
**Tabla 3**. Verifiqué que el orden de primera mención es 1-2-3-4-5-6-7, que no hay pies duplicados y
que Resultados no citaba ninguna de las dos.

### 4.2 §2.3 describía la extracción de forma imprecisa

Decía: «The same extraction was applied to each of the ROI groups described in Section 2.4». Pero la
señal se extrae **una sola vez** con el atlas AAL116; los grupos reducidos son un subconjunto de esas
116 series, no cuatro extracciones distintas. Además, tal como estaba, la frase creaba un bucle:
§2.3 dependía de §2.4 para una operación que en realidad no la necesita.

Corregido:

> The 116 regional time series obtained in this way were **then restricted to** each of the ROI
> groups described in Section 2.4.

### 4.3 Lo que sí encadena bien

| Transición | Estado |
|---|---|
| §2.1 → §2.2 | La muestra y sus diferencias de adquisición preceden al pipeline que las procesa |
| §2.2 → §2.3 | §2.2 termina en el filtrado y el suavizado; §2.3 abre con «extracted from the **filtered and smoothed** functional derivative». El empalme es literal |
| §2.3 → §2.4 | Única referencia hacia adelante de la sección, y ahora es una restricción de columnas, no una dependencia de procedimiento |
| §2.4 → §2.5 | Los grupos quedan definidos justo antes de las arquitecturas que los consumen |
| §2.5 → §2.6 | El conteo de parámetros de §2.6 depende de la arquitectura de §2.5. El orden es el correcto |
| §2.6 → §2.7 | El protocolo produce las predicciones out-of-fold sobre las que §2.7 calcula |
| §2.6 → §2.1 | La ponderación de clase remite al argumento de no agrupar sitios, que vive en §2.1 |

### 4.4 Dos dependencias que se resolverán al escribir §2.5

No son errores hoy, pero conviene tenerlas presentes:

- **«With 66 input edges»**, en la tercera dimensión de sensibilidad. 66 es el triángulo inferior de
  una matriz de 12 × 12, pero el lector no puede deducirlo hasta que §2.5 explique cómo entra la
  conectividad al modelo.
- **«instantiating the architecture»**, en el pie de la Tabla 5, y **«the number of model
  parameters»**, en §2.6. Ambos suponen una arquitectura ya descrita.

Sigue pendiente lo que ya señalé: **§2.5 debe reintroducir el término «multichannel»**, porque §3.3.2
y el pie de la Figura 2 lo usan y hoy no tiene antecedente.

---

## 5. Estado del documento

| Indicador | Valor |
|---|---:|
| Texto corrido | 3.610 palabras, −10,8% sobre la línea base de 4.046 |
| Párrafos de más de 160 palabras | 0 |
| Oraciones de más de 50 palabras | 0 |
| Referencias cruzadas rotas | 0 |
| Tablas fuera de orden | 0 |
| Cifras clave verificadas | todas |

El texto corrido subió 45 palabras respecto de la medición anterior por la justificación técnica de
la restricción de OHSU. Es un gasto que recomiendo asumir: sustituye un argumento de precisión por
uno de diseño, y es de los pocos sitios donde añadir palabras hace el artículo más difícil de
objetar.
