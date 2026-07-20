# Revisión del notebook `tdha.ipynb` y de los datos de entrada

Fecha: 20 de julio de 2026
Alcance: notebook de la carpeta del proyecto (38 celdas), `src/kerasmodels/lstm.py`, y los
tensores de `data/fc-pearson-seq/` del repositorio.

Los hallazgos están ordenados por gravedad. Los del bloque A impiden ejecutar los
experimentos planeados; los del bloque B comprometen la validez de los resultados; los
del bloque C son de reproducibilidad y de higiene.

---

## A. Problemas en los datos

> **Corrección (20/07/2026).** Una versión anterior de este documento afirmaba que los
> tensores `*_X12.joblib` de Peking, NeuroIMAGE y OHSU estaban corruptos, y que no existía
> el tensor de 116 ROIs. **Ambas afirmaciones eran incorrectas.** Los archivos de ~130
> bytes son punteros de Git LFS, no archivos truncados; el contenido real existe y está
> íntegro. El tensor de 116 ROIs también existe, comprimido en varias partes. Los puntos
> A1 y A5 se corrigen abajo. Los hallazgos A2, A3 y A4 se mantienen y quedaron reforzados
> con la verificación adicional.

### A1. Los datos están completos — el repositorio usa Git LFS

Los archivos de 12 ROIs de los tres sitios adicionales se almacenan mediante Git LFS. Un
clon sin `git-lfs` instalado descarga punteros de texto de ~130 bytes en lugar del
contenido, lo que puede confundirse con archivos corruptos.

Tamaños reales, leídos de los propios punteros:

| Archivo | Almacenamiento | Tamaño real |
|---|---|---|
| `NYU_X12.joblib` | directo | 5.30 MB |
| `Peking_X12.joblib` | LFS | 8.64 MB |
| `NeuroIMAGE_X12.joblib` | LFS | 2.11 MB |
| `OHSU_X12.joblib` | LFS | 0.11 MB |

**Implicación práctica para Colab:** el notebook clona el repositorio con `!git clone`.
Si el entorno no tiene `git-lfs` inicializado, `joblib.load` recibirá el puntero de texto
y fallará con un error de deserialización poco descriptivo. Conviene añadir al inicio del
notebook:

```python
!git lfs install && git lfs pull
```

y una verificación explícita de la forma del tensor después de cargarlo, para que un
fallo de descarga no se confunda con un problema de los datos.

### A2. Revisar la corrida de OHSU con 12 ROIs antes de usarla

`results/rois_all_sites/OHSU_12_lstm128_{train,val}.csv` contiene 50 pliegues con una
anomalía clara: el accuracy de validación (73.57 %) supera al de entrenamiento (61.68 %)
y los máximos de validación llegan al 100 %.

Con los datos ya verificados, la explicación es el punto A3: las secuencias de OHSU tienen
**3 ventanas**, no 52. Con secuencias tan cortas y 66 sujetos, los pliegues de validación
son de ~7 sujetos y las métricas se vuelven inestables. No es un archivo inválido, pero
tampoco es comparable con las corridas de NYU.

### A3. La longitud de secuencia varía de 3 a 94 ventanas entre sitios

Con la ventana de 70 TR y paso de 2 TR fijada para NYU:

| Sitio | Sujetos | Clases (0/1) | Ventanas | TRs implícitos |
|---|---|---|---|---|
| NYU | 177 | 87 / 90 | 52 | ~172 |
| Peking | 183 | 109 / 74 | 82 | ~232 |
| NeuroIMAGE | 39 | 22 / 17 | 94 | ~256 |
| OHSU | 66 | 38 / 28 | **3** | ~74 |

Verificado por dos vías independientes: las formas leídas directamente de los tensores de
18 ROIs —(177, 52, 18, 18), (183, 82, 18, 18), (39, 94, 18, 18) y (66, 3, 18, 18)— y el
tamaño en bytes de los punteros LFS de los tensores de 12 ROIs, que dan exactamente el
mismo número de ventanas por sitio.

OHSU produce **3 ventanas por sujeto**. Con adquisiciones de unos 74 volúmenes, una
ventana de 70 TR prácticamente no cabe: las tres ventanas se solapan casi por completo y
no hay dinámica temporal que modelar. Aplicar ahí los parámetros de enventanado de NYU no
tiene sentido metodológico.

Esto no es solo un obstáculo: es material para el artículo. Los revisores piden
justamente análisis de sensibilidad al tamaño de ventana (R1.1, R2.10) y generalización
multi-sitio (R1.2, R2.4). El hecho de que los parámetros óptimos en NYU sean
inaplicables en OHSU es un resultado reportable, siempre que se documente en vez de
ocultarse tras un accuracy.

Consideraciones de diseño que hay que resolver antes de correr: NeuroIMAGE tiene 39
sujetos —una validación cruzada de 10 pliegues deja ~4 sujetos por pliegue—, y Peking
está desbalanceado 60/40, lo que exige reportar métricas robustas al desbalance y no solo
accuracy.

### A4. El grupo de 39 ROIs se generó con un paso de enventanado distinto

`X39.joblib` tiene forma **(177, 26, 39, 39)**: 26 ventanas, no 52. Con ventana de 70 TR
sobre 172 TRs, 26 ventanas corresponden a un **paso de 4 TR**, no de 2.

Los demás grupos tienen 52 ventanas (paso 2). Es decir, en la Tabla 2 del manuscrito la
fila de 39 ROIs difiere de las otras en el número de regiones **y** en los parámetros de
enventanado, pese a que el archivo se llama `39-rois-52-seq`. Es un segundo factor
confundido, además del cambio de arquitectura ya detectado en la fila de 116 ROIs.

### A5. El tensor de 116 ROIs existe, pero en un ZIP dividido en partes

No hay `X116.joblib` en `data/fc-pearson-seq/`, pero sí está el archivo comprimido en tres
partes: `X116.zip`, `X116.z01` y `X116.z02`, junto a `y116.joblib`. También existe
`data/fcm-fisher/X116.joblib`, que corresponde a matrices de Fisher promediadas y no a
las secuencias dinámicas.

Esto significa que el notebook, tal como está, **no puede cargar el grupo de 116 ROIs**:
la ruta que construye es `{site}_X{n_rois}.joblib`, que no existe para 116 ni para 39. El
script nuevo debe contemplar la descompresión previa del ZIP multiparte, y conviene
verificar que el contenido tenga 52 ventanas y no 26 como ocurre con `X39`.

---

## B. Problemas en el código que afectan la validez

### B1. La época del modelo se elige con el mismo pliegue que luego se reporta

```python
model_checkpoint_callback = ModelCheckpoint(..., monitor='val_accuracy', mode='max')
early_stopping_callback   = EarlyStopping(monitor='val_loss', restore_best_weights=True)
```

Ambos callbacks observan el pliegue de validación, se guarda la época que **maximiza el
accuracy de validación**, y esa misma métrica es la que se reporta. Es selección de modelo
sobre el conjunto de evaluación: las métricas de validación quedan sesgadas al alza de
forma sistemática, y el sesgo es mayor cuanto más ruidoso es el pliegue (es decir, mayor
en los sitios pequeños).

Esto responde de forma desfavorable al comentario 5 del revisor 2 (*"were the reported
results obtained on data that were never used during training or hyperparameter
tuning?"*). Tal como está, la respuesta honesta es que no.

Opciones: separar un subconjunto interno de entrenamiento para la selección de época
(validación anidada), o fijar un número de épocas sin early stopping. Los valores
absolutos bajarán, pero bajarán en todos los grupos por igual y la comparación entre ellos
—que es lo que sostiene la tesis— se conserva.

### B2. Sin semilla: las particiones son distintas en cada ejecución y entre grupos

```python
skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)   # sin random_state
```

Tampoco hay `tf.random.set_seed` ni semilla de NumPy. Dos consecuencias:

1. Los resultados no son reproducibles ni por los propios autores.
2. Cada grupo de ROIs se evaluó sobre **particiones diferentes**, de modo que parte de la
   diferencia entre grupos es ruido de partición y no efecto del número de regiones.

Fijar la semilla y usar las mismas 50 particiones en todos los grupos convierte la
comparación en pareada, lo que aumenta la potencia estadística de forma considerable —
justo lo que hace falta para que los contrastes de Tukey alcancen significancia.

Nota estadística asociada: si los cuatro grupos comparten sujetos y particiones, las
observaciones entre grupos están correlacionadas y el ANOVA de una vía deja de cumplir el
supuesto de independencia. Lo apropiado sería un ANOVA de medidas repetidas o un modelo
mixto.

### B3. Estado acumulado entre ejecuciones

`train_results` y `val_results` se inicializan en la celda 8, la misma que fija `n_rois` y
`site`. El bucle de la celda 21 hace `append` sin limpiar. Si se ejecuta el bucle dos
veces sin volver a correr la celda 8, los resultados de dos configuraciones quedan
mezclados en el mismo CSV; si se corre la celda 8 a mitad de camino, se pierden.

Esto explica de forma natural los archivos con conteos anómalos que hay en el
repositorio: `18-rois-52-seq` con 10 filas, `18-rois-con1d-seq-train2` con 20, y en
general la mezcla de corridas de 10, 20 y 50 pliegues que hizo imposible rastrear varias
filas de la Tabla 2.

### B4. El notebook depende del orden de ejecución y de edición manual

`site` y `n_rois` se editan a mano (celdas 8 y 9), y varias celdas son restos de
exploración (3–7, 11–18, 22–25). Para el plan de esta semana —4 sitios × 4 grupos de ROIs,
más los controles de ROIs aleatorios— son más de veinte ejecuciones manuales encadenadas.
Con el estado acumulado del punto B3, la probabilidad de repetir exactamente el problema
que estamos tratando de corregir es alta.

---

## C. Reproducibilidad e higiene

- **No se guarda el `history` del entrenamiento.** Sin él no hay curvas de loss y accuracy
  por época, que son lo que piden R1.4 y R2.11. Guardarlas en esta corrida evita tener que
  reentrenar después.
- **No se exportan los índices de las particiones.** `train_indices` y `val_indices` se
  acumulan y se meten en un `DataFrame` (celda 26) que nunca se escribe a disco. Sin ellos
  no se puede auditar la partición ni demostrar que no hay fuga de datos.
- **`predictions.append(best_model.predict(X))`** predice sobre los 177 sujetos completos,
  incluidos los de entrenamiento, en cada uno de los 50 pliegues. No se exporta y solo
  consume tiempo. Para las matrices de confusión y las curvas ROC lo que hace falta son las
  predicciones sobre `X_val`.
- **`best_model.evaluate` se llama cuatro veces por pliegue**, dos de ellas redundantes.
- **`tmp_keras_checkpoint/` no se crea** con `os.makedirs(..., exist_ok=True)`.
- **`trang_sup` usa `np.triu_indices`**, es decir el triángulo **superior**, mientras el
  manuscrito y el Reporte de Experimentos describen el triángulo inferior. Por la simetría
  de la matriz de correlación el resultado numérico es equivalente, pero conviene alinear
  texto y código.
- **`batch_size=8`** en el notebook frente a los 32 que declara la Tabla 1 del manuscrito;
  y `build_model` no aplica el recorte de gradiente de 1.0 que esa tabla declara.
- El AUC se calcula pero no se imprime en el resumen por pliegue.
- El desempaquetado manual de los nueve valores que devuelve `evaluate` es frágil ante
  cualquier cambio en la lista de métricas.

---

## Orden sugerido de trabajo

1. **Asegurar el acceso a los datos en Colab**: `git lfs install && git lfs pull` tras el
   clone, descompresión del ZIP multiparte de 116 ROIs, y una verificación de forma
   después de cada carga. Los datos están completos; el riesgo es de descarga, no de
   integridad.
2. **Decidir el esquema de enventanado para el análisis multi-sitio**, dado que OHSU solo
   admite 3 ventanas con los parámetros actuales. Es también la ocasión de producir el
   análisis de sensibilidad que piden los revisores.
3. **Regenerar `X39` con paso de 2 TR** para que el grupo de 39 ROIs sea comparable con
   los demás, y confirmar el número de ventanas de `X116` tras descomprimirlo.
4. **Reescribir el bucle experimental como script parametrizado**, con semilla fija,
   particiones compartidas entre grupos, selección de época sin fuga, y registro de
   historiales, predicciones de validación e índices de partición.
5. Recién entonces lanzar las corridas.
