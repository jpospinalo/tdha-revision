# Plan para resolver el problema de los modelos sin entrenar

Antes de cambiar inicialización, activaciones o calentamiento, hay que responder
una pregunta que todavía no está respondida. Este documento propone cómo hacerlo.

---

## El problema, en una frase

En el piloto, 18 de 50 pliegues devolvieron un modelo con 51,76 % de accuracy **en
entrenamiento**, es decir, esencialmente sin entrenar. `restore_best_weights=True`
restauró los pesos de la época 1 porque ahí estaba el mínimo de la pérdida sobre la
partición interna.

## Por qué no basta con aplicar una solución

Hay dos explicaciones posibles, y **exigen intervenciones opuestas**:

**H1 — Ruido.** La partición interna tiene 24 sujetos. Con tan pocas muestras la
`val_loss` es tan variable que su mínimo cae en la época 1 por azar, aunque el modelo
sí estuviera mejorando. Si esto es cierto, hay que impedir que se elija una época
temprana: `start_from_epoch`, una partición interna mayor, o suavizar la curva.

**H2 — Señal.** La `val_loss` interna nunca baja de su valor inicial porque el modelo
sobreajusta desde las primeras épocas y no generaliza en ningún momento. Si esto es
cierto, la época 1 **es** genuinamente la mejor según el criterio, y forzar
`start_from_epoch=20` seleccionaría un modelo peor, más sobreajustado. La intervención
correcta sería otra: reducir capacidad, regularizar, o aceptar y reportar que esta
configuración no supera el azar.

### Lo que ya sabemos apunta a H2

| Evidencia | Valor |
|---|---|
| Correlación entre `best_epoch` y accuracy de **entrenamiento** | r = **+0,944** |
| Correlación entre `best_epoch` y accuracy de **validación** | r = −0,049 |
| Accuracy de validación en pliegues con `best_epoch` = 1 | 52,43 % |
| Accuracy de validación en pliegues con `best_epoch` > 10 | 50,33 % |
| Accuracy de entrenamiento en pliegues con `best_epoch` > 20 | 89,16 % |

Entrenar más eleva el entrenamiento de forma casi perfecta y **no mueve la
validación**. Eso es la firma de H2: el modelo memoriza y no aprende nada
transferible.

Contexto que lo hace plausible: con 136 sujetos de ajuste, incluso la configuración
más frugal tiene **735 parámetros por sujeto**.

| Grupo | Características | Parámetros | Parámetros por sujeto |
|---|---|---|---|
| 12 ROIs | 66 | 99.969 | 735 |
| 18 ROIs | 153 | 144.513 | 1.063 |
| 39 ROIs | 741 | 445.569 | 3.276 |
| 116 ROIs | 6.670 | 3.481.217 | 25.597 |

### Pero no está demostrado

La distinción se resuelve mirando **la forma de la curva de `inner_val_loss` por
época**: si baja y luego sube, hay una época buena y el problema es de selección (H1);
si sube desde el principio, no la hay (H2).

Esa curva está en `history.csv`, que dejamos de guardar por defecto. **El primer paso
del plan es recuperarla.**

---

## Paso 1 · Diagnóstico (una corrida, ~10 min)

```bash
python run_experiment.py --site NYU --roi-set 12 --model lstm \
    --patience 25 --save-history --save-folds --tag diag
```

Sin tocar ningún otro parámetro: se trata de observar el mismo modelo que falló, no
uno distinto. `--patience 25` solo abarata la corrida.

Qué mirar en `history.csv`:

1. **Forma media de `inner_val_loss`.** Promediar la curva sobre los 50 pliegues.
   Si tiene un mínimo interior claro → H1. Si es monótona creciente desde la época 1
   → H2.
2. **Dispersión entre pliegues.** Si las 50 curvas son muy distintas entre sí, el
   ruido domina y H1 pesa más.
3. **`loss` de entrenamiento.** Confirma en qué época empieza la memorización.

Con eso, la pregunta queda cerrada y las decisiones siguientes dejan de ser a ciegas.

---

## Paso 2 · Intervenciones, según el resultado

### Si es H1 — problema de selección

Ordenadas de menor a mayor intrusión:

| Opción | Qué hace | Costo |
|---|---|---|
| `--start-from-epoch 10–20` | No vigila la parada hasta pasada esa época. Es el argumento oficial de Keras para un *warm-up* en el que no se espera mejora | ninguno |
| `--inner-val-frac 0.20–0.25` | Partición interna de 32–40 sujetos en lugar de 24; menos ruido en la curva | 8–16 sujetos menos de ajuste |
| Vigilar `val_loss` suavizada | Reduce el efecto de un mínimo aislado | requiere un callback propio |

### Si es H2 — el modelo no generaliza

Aquí las opciones que mencionas tienen sentidos muy distintos:

**Inicialización.** Keras ya usa lo estándar: `glorot_uniform` en el núcleo de
entrada, `orthogonal` en el recurrente y ceros en el sesgo. La ortogonal es
precisamente la recomendada para redes recurrentes. La única mejora documentada que
falta es **inicializar el sesgo de la puerta de olvido en 1**: Jozefowicz et al. (2015)
mostraron que cierra la brecha entre LSTM y GRU, porque un sesgo positivo empuja la
sigmoide cerca de 1 y el estado de celda se preserva desde el principio, evitando un
gradiente que se desvanece. Es un cambio pequeño y bien fundado. **Pero afecta a la
optimización, no a la generalización**: ayudaría si el modelo no lograra ajustar el
entrenamiento, y aquí lo ajusta hasta el 93 %.

**Activación.** La salida es sigmoide con entropía cruzada binaria, que es lo correcto
para clasificación binaria. Dentro de la LSTM, `tanh` y `sigmoid` son las que activan
el camino rápido de cuDNN; cambiarlas lo desactivaría y multiplicaría el tiempo. No
hay razón para tocarlas.

**Calentamiento de la tasa de aprendizaje.** Sirve cuando el entrenamiento diverge o
es inestable al inicio, típicamente con lotes grandes. Aquí el lote es 8 y `lr=1e-4`;
el problema no es inestabilidad. Es la intervención con menos base en este caso.

**Lo que sí atacaría H2**, por orden de lo que yo probaría:

| Opción | Razón |
|---|---|
| Reducir capacidad: `--model-arg units=32` o `16` | 735 parámetros por sujeto es mucho. Con 32 unidades bajan a ~50 |
| Regularizar: `--model-arg dropout=0.3` | Ya está expuesto en el registro; no requiere código nuevo |
| `--clipnorm 1.0` | Además resolvería la discrepancia con la Tabla 1 del manuscrito |
| Reducir la longitud de entrada | 66 características por ventana con 136 sujetos es mucha dimensionalidad |

---

## Paso 3 · Cómo elegir sin sesgar el resultado

Esto es lo más delicado del plan.

Cualquier búsqueda de hiperparámetros que se decida mirando el **pliegue externo**
reintroduce exactamente el sesgo que quitamos al arreglar la selección de época, un
nivel más arriba. Si se prueban seis configuraciones y se reporta la mejor, el
estimado final está inflado.

Dos formas limpias:

**A · Decidir con la partición interna.** Comparar configuraciones por su
`inner_val_loss` mínima media, que ya se calcula sobre datos de entrenamiento, y
reportar solo la ganadora sobre el pliegue externo. Es coherente con el protocolo
actual y no requiere código nuevo.

**B · Fijar por argumento y no buscar.** Elegir la configuración por razones
metodológicas —capacidad proporcional a la muestra, `clipnorm` para coincidir con la
Tabla 1— declararla de antemano y correrla una sola vez. Es lo más defendible ante un
revisor, aunque probablemente no dé el mejor número.

Lo que **no** haría es probar variantes mirando el accuracy de validación y quedarse
con la mejor. Es lo que produjo el 76,59 % que no pudimos rastrear.

---

## Paso 4 · Qué hacer si el modelo no supera el azar

Es un desenlace posible y conviene tenerlo pensado, no improvisarlo.

Con 177 sujetos, 12 ROIs y una LSTM, puede que la señal simplemente no dé para más.
Si tras el Paso 2 la validación sigue en torno al 53 %, eso **es** el resultado, y hay
formas honestas de presentarlo:

- La comparación entre grupos de ROIs sigue siendo válida aunque todos estén cerca del
  azar: el argumento del artículo es sobre desempeño *relativo* y sobreajuste, no sobre
  un accuracy absoluto alto.
- La brecha entre entrenamiento y validación —89 % frente a 50 %— es en sí misma un
  resultado sobre la capacidad de estos modelos con muestras de este tamaño, y encaja
  con la crítica que el propio artículo hace a los trabajos que reportan 98 %.
- Los revisores no piden un accuracy alto; piden rigor metodológico y comparaciones
  robustas.

Conviene hablarlo con los coautores antes de tener el número, no después.

---

## Resumen del plan

1. Una corrida con `--save-history` para distinguir H1 de H2. Diez minutos.
2. Según el resultado, intervenir sobre la **selección** (H1) o sobre la **capacidad**
   (H2). Inicialización, activaciones y calentamiento no atacan ninguno de los dos.
3. Elegir entre configuraciones usando la partición interna, nunca el pliegue externo.
4. Tener acordado de antemano cómo se reporta si el desempeño no supera el azar.

---

## Fuentes

- [Keras · LSTM layer](https://keras.io/api/layers/recurrent_layers/lstm/) — valores
  por defecto de `kernel_initializer`, `recurrent_initializer` y `bias_initializer`.
- [Keras · Layer weight initializers](https://keras.io/api/layers/initializers/) —
  definición de Glorot uniforme y ortogonal.
- [Keras · EarlyStopping](https://keras.io/api/callbacks/early_stopping/) —
  `start_from_epoch` como periodo de calentamiento en el que no se espera mejora;
  `baseline` y `restore_best_weights`.
- [Jozefowicz, Zaremba & Sutskever (2015), *An Empirical Exploration of Recurrent
  Network Architectures*](https://proceedings.mlr.press/v37/jozefowicz15.pdf) —
  sesgo de la puerta de olvido inicializado en 1.
- [Varoquaux (2018), *Cross-validation failure: Small sample sizes lead to large error
  bars*](https://www.sciencedirect.com/science/article/abs/pii/S1053811917305311) —
  variabilidad de la validación cruzada con muestras pequeñas en neuroimagen.
- [Vabalas et al. (2019), *Machine learning algorithm validation with a limited sample
  size*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6837442/) — la validación cruzada
  simple sesga al alza con muestras pequeñas; la anidada no.
