# Revisión de eficiencia

Análisis del costo computacional del pipeline y de las palancas disponibles para
reducirlo, pensando en un escenario de corridas encoladas.

---

## Dónde está el costo

El enventanado **no** es el problema. Medido sobre NYU:

| Grupo | Construcción de secuencias | Parámetros de la LSTM-128 |
|---|---|---|
| 12 ROIs | 0,1 s | 99.969 |
| 39 ROIs | 0,9 s | 445.569 |
| 116 ROIs | 7,5 s | 3.481.217 |

Y se construye **una sola vez por corrida**, antes del bucle de validación cruzada;
los 50 pliegues reutilizan la misma matriz.

El costo real está en el entrenamiento. Con la configuración por defecto sobre NYU:

- El conjunto de ajuste tiene 136 sujetos, la partición interna 24, la validación 17.
- Con `batch_size=8` son 17 pasos por época.
- Con `epochs=150` y `patience=100`, el early stopping casi nunca se dispara antes
  del final: en la práctica cada pliegue corre las 150 épocas.
- Total: **127.500 pasos de entrenamiento por corrida**.

---

## Cambios aplicados

Ninguno altera los resultados.

### 1. Construcción de secuencias en float32 y por lotes

Antes se calculaba en float64 y con todos los sujetos a la vez, lo que exigía un
arreglo intermedio de unos 600 MB con 116 ROIs. Ahora se calcula en float32 en lotes
de 32 sujetos.

| | Antes | Ahora |
|---|---|---|
| 116 ROIs | 28,0 s | **7,5 s** |
| 39 ROIs | 3,0 s | 0,9 s |
| 12 ROIs | 0,6 s | 0,1 s |

La precisión no se resiente: la diferencia contra los tensores del proyecto original
pasa de 8 × 10⁻⁷ a 1 × 10⁻⁶, ambas por debajo de la resolución de float32, que es el
tipo en que se almacenan las secuencias. Los seis casos de verificación siguen
coincidiendo.

### 2. Caché de señales y de secuencias

`load_bold` y `build_sequences_cached` guardan su resultado en memoria. Al encadenar
corridas en un mismo proceso, la segunda construcción con la misma configuración de
datos cuesta **0 s** en lugar de 7,5 s.

La caché de secuencias guarda una sola entrada, porque el tensor de 116 ROIs ocupa
cerca de 500 MB. Por eso conviene ordenar la cola agrupando por configuración de
datos, que es lo que hace `run_queue.py`.

### 3. Ejecutor de colas

`run_queue.py` ejecuta varias corridas en un solo proceso. Evita pagar N veces el
arranque de TensorFlow (10–20 s), la lectura de las señales y la construcción de las
secuencias.

```bash
python run_queue.py --site NYU --roi-set 12 18 39 116 --model lstm
python run_queue.py --file cola.txt
```

Una corrida que falle no detiene la cola, y las configuraciones ya ejecutadas se
saltan sin recalcular.

Para una cola de 8 corridas sobre NYU con dos arquitecturas, el ahorro es de unos
2–3 minutos de sobrecarga. Es modesto frente al entrenamiento, pero gratis.

### 4. Capa de salida en float32 explícito

Las cuatro arquitecturas declaran `dtype="float32"` en la capa de salida. Con la
configuración actual no cambia nada —todo es float32— pero deja el comportamiento
fijado si en el futuro alguien experimenta con precisión reducida.

---

## Palancas que sí cambian los resultados

Estas **no** se aplicaron, porque son decisiones metodológicas y no optimizaciones.
Se documentan para que el equipo decida con los números a la vista.

### `patience=100` con `epochs=150`

Es, con diferencia, la palanca más grande. Una paciencia de 100 sobre un máximo de
150 épocas equivale a no tener early stopping: prácticamente todos los pliegues
llegan al final.

Reducir la paciencia a 20 o 30 haría que muchos pliegues se detuvieran antes.
El ahorro depende de la curva de convergencia de cada modelo, que se puede leer en
`history.csv` de las corridas ya hechas: si la mediana de `best_epoch` es, por
ejemplo, 40, entonces las 110 épocas restantes de cada pliegue son cómputo perdido.

**Recomendación:** correr primero una configuración con los valores actuales, mirar
la distribución de `best_epoch` en `metrics_val.csv`, y decidir la paciencia con ese
dato. No conviene ajustarla a ojo.

### `batch_size=8`

Con 136 sujetos de ajuste, un lote de 8 da 17 pasos por época:

| Lote | Pasos/época | Pasos por corrida |
|---|---|---|
| 8 | 17 | 127.500 |
| 16 | 9 | 67.500 |
| 32 | 5 | 37.500 |
| 64 | 3 | 22.500 |

En GPU, lotes pequeños desaprovechan el paralelismo: buena parte del tiempo se va en
lanzar núcleos en lugar de calcular. Pasar a 32 reduce los pasos a la cuarta parte.

Pero el tamaño de lote afecta la dinámica de optimización, y la Tabla 1 del
manuscrito declara 32 mientras el código usaba 8 — esa discrepancia hay que
resolverla de todos modos. Si se decide 32, conviene que sea por coherencia con lo
publicado y no por velocidad, y relanzar todas las configuraciones con el mismo
valor.

### Precisión mixta

Se evaluó y **se descartó**. `mixed_float16` acelera los modelos grandes en GPU con
núcleos tensoriales, y con 116 ROIs es donde más se notaría. Pero introduce un riesgo
mayor que el beneficio: si unas corridas la usan y otras no, la tabla final mezcla
configuraciones numéricas distintas, y con 50 repeticiones por configuración esa
diferencia es difícil de distinguir del ruido de partición.

Todas las corridas usan float32. Si en algún momento se adopta la precisión mixta,
debe ser para el conjunto completo de experimentos y relanzando todo.

### Número de repeticiones

`10 × 5 = 50` es lo que da la distribución sobre la que se apoya el argumento del
artículo. Reducirlo abarataría todo proporcionalmente, pero debilitaría justamente la
parte metodológica que diferencia este trabajo. No es una palanca recomendable.

---

## Orden sugerido para la cola

Agrupar por configuración de datos aprovecha la caché:

```
NYU · 12 ROIs · lstm, gru, cnn1d, transformer
NYU · 18 ROIs · lstm
NYU · 39 ROIs · lstm
NYU · 116 ROIs · lstm          ← el más caro, dejar para el final
```

Las corridas de 116 ROIs son las que dominan el tiempo total: 35 veces más parámetros
que las de 12 sobre la misma muestra. Si el tiempo de GPU es limitado, conviene
completar primero todo lo demás.

---

## Lo que no se puede optimizar sin cambiar el modelo

Con 116 ROIs, la entrada tiene 6.670 características y la matriz de entrada de la
LSTM es de 6.670 × 512. Esa matriz **es** el modelo: 3,4 de los 3,5 millones de
parámetros. No hay forma de abaratarla sin reducir la dimensionalidad de entrada,
que es precisamente la variable bajo estudio.
