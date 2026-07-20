# Auditoría de métricas — Manuscript.docx vs. repositorio TDHA-fMRI

Fecha: 20 de julio de 2026
Fuente: repositorio público https://github.com/jpospinalo/TDHA-fMRI

Alcance de la revisión:
- Rama `main`, carpeta `results/`: 56 archivos CSV (estado actual, último commit
  8ccde31 del 17/07/2026).
- **Historial completo de `main`**: 47 commits que tocaron `results/`, con 90 versiones
  únicas de CSV. Es decir, 34 versiones fueron sobrescritas y ya no son visibles en la
  carpeta actual. Varias cifras del manuscrito provienen de esas versiones antiguas.
- Ramas `preprocc` y `proc-save-matrices`: no contienen resultados.
- `src/kerasmodels/lstm.py` y `tdha.ipynb`.

Objetivo: determinar el origen de las cifras reportadas en el manuscrito y resolver la
discrepancia detectada entre el artículo y el Reporte de Experimentos.

---

## Resumen

De las cifras principales del manuscrito, **solo una fila de la Tabla 2 es reproducible**
a partir de los resultados versionados en el repositorio. El accuracy que encabeza el
resumen y la introducción (76.59 %) no aparece en ningún archivo de resultados, y el
manuscrito se contradice a sí mismo respecto de esa cifra.

---

## 1. El 76.59 % no es trazable y contradice la Tabla 2

El resumen afirma "an average accuracy of 76.59 %" y la introducción "validation
ACC = 76.59 % and training ACC = 89.82 %".

Se recorrieron las 90 versiones de CSV existentes en todo el historial de `main` —no solo
los 56 archivos actuales—, todas sus columnas numéricas (loss, accuracy, precision,
recall, auc y los cuatro conteos de la matriz de confusión), calculando media y máximo
sobre el total de filas y sobre los primeros 10 y 20 pliegues. **Ningún valor coincide
con 76.59 %** con tolerancia de ±0.06 puntos.

Al mismo tiempo, la Tabla 2 del propio manuscrito reporta para 12 ROIs un ACC de
validación de **68.10 ± 8.49**. Es decir, el número del resumen contradice la tabla del
mismo artículo por más de 8 puntos porcentuales.

Esta es la inconsistencia más urgente: es visible sin acceso al repositorio, con solo
leer el artículo.

## 2. El 89.82 % es un AUC, no un accuracy, y proviene de otra configuración

El valor 89.82 aparece una sola vez en todo el repositorio: es la media de la columna
`auc` de `results/rois/18/18-rois-52-seq-train.csv`.

Ese archivo corresponde a **18 ROIs**, no a 12, y contiene **10 pliegues**, no 50. La
introducción lo presenta como "training ACC" de la configuración principal. Todo indica
un error de transcripción entre columnas y entre configuraciones.

## 3. Tabla 2 mezcla arquitecturas distintas bajo el rótulo de una sola

Este es el hallazgo de mayor impacto para la tesis del artículo.

| ROIs | Manuscrito (train / val) | Origen identificado | Arquitectura real |
|---|---|---|---|
| 12 | 73.26 / 68.10 | `architectures/lstm/{train,val}_lstm_128.csv` (17/07/2026, n=50) | **LSTM-128** |
| 18 | 81.05 / 68.26 | no identificado | — |
| 39 | 84.37 / 66.46 | no identificado | — |
| 116 | 86.73 / 62.28 | `rois/116-con1d-lstm118-{train,val}.csv`, **versión del 27/10/2025** (n=50) | **CNN-1D + LSTM-118** |

Verificación realizada sobre `results/rois/`: se revisaron los 18 CSV actuales y las 40
versiones históricas de esa carpeta, comparando las **seis** cifras de cada fila de la
Tabla 2 de forma simultánea (ACC con su desviación, precisión, recall, especificidad
—calculada como TN/(TN+FP)— y ROC-AUC).

Resultado: de las cuatro filas de la Tabla 2, **solo la de 116 ROIs proviene de
`results/rois/`**. La de 12 ROIs proviene de `results/architectures/lstm/`. Las de 18 y
39 no aparecen en ninguna de las dos carpetas.

Coincidencias exactas encontradas, en las seis cifras a la vez:

| Fila Tabla 2 | ACC±SD | Prec | Rec | SP | AUC | Archivo |
|---|---|---|---|---|---|---|
| 12 train | 73.26±16.56 | 72.66 | 81.90 | 64.33 | 78.70 | `architectures/lstm/train_lstm_128.csv` |
| 12 val | 68.10±8.49 | 66.99 | 76.22 | 59.69 | 63.75 | `architectures/lstm/val_lstm_128.csv` |
| 116 train | 86.73±14.16 | 86.38 | 92.81 | 80.42 | 95.14 | `rois/116-con1d-lstm118-train.csv` (27/10/2025) |
| 116 val | 62.28±9.66 | 61.66 | 73.78 | 50.47 | 56.51 | `rois/116-con1d-lstm118-val.csv` (27/10/2025) |

La coincidencia es inequívoca. Y el nombre del archivo indica que la corrida de 116 ROIs
**no usó la arquitectura LSTM-128 del artículo**, sino un híbrido convolucional 1D
seguido de una LSTM de 118 unidades.

Implicación: en la comparación central del artículo, entre la configuración de 12 ROIs y
la de 116, cambian simultáneamente el número de regiones **y la arquitectura del modelo**.
El sobreajuste observado en 116 ROIs no puede atribuirse limpiamente al número de ROIs.
Esto afecta directamente la interpretación de la Tabla 2, de la Figura 3 y del ANOVA, y
es verificable por cualquier revisor con acceso al repositorio.

Además, esa versión del archivo fue sobrescrita posteriormente: la versión actual de
`116-con1d-lstm118-*.csv` tiene 10 pliegues y valores distintos (68.83 / 70.95). Quien
consulte hoy el repositorio no encontrará las cifras del artículo.

Las filas de 18 y 39 ROIs no se pudieron identificar ni en el estado actual ni en las 90
versiones históricas de los CSV. Los valores más próximos no coinciden en ninguna de las
seis cifras:

| | Manuscrito (train) | Candidato más cercano en `results/rois/` |
|---|---|---|
| 18 | 81.05 / 80.98 / 85.19 / 76.78 / 86.35 | `18-rois-con1d-seq-train`: 80.79 / 80.97 / 81.95 / 79.58 / 86.39 |
| 39 | 84.37 / 83.72 / 87.09 / 81.55 / 89.41 | `39-rois-52-seq-train`: 84.58 / 83.93 / 88.20 / 80.84 / 93.76 |

Un detalle relevante: `results/rois/12/12-rois-52-seq-*.csv` sí contiene una corrida de
50 pliegues con 12 ROIs (76.06 train / 72.44 validación), pero **no es la que aparece en
el artículo**. La fila publicada usa la corrida de `architectures/lstm`, que da 4 puntos
menos en validación.

## 4. Tabla 3 (ANOVA): tres inconsistencias

**a) Error aritmético.** La tabla reporta SS = 0.1164 con 3 grados de libertad y
MS = 0.0338. Pero 0.1164 / 3 = 0.0388. El valor de F reportado (4.3067) es consistente
con 0.0388, no con 0.0338: se trata de una errata en la columna MS.

**b) Grados de libertad contradictorios.** La Tabla 3 indica 245 grados de libertad
residuales (N = 249 observaciones), mientras que el texto del artículo y el Reporte de
Experimentos dicen F(3,196), que implicaría N = 200, es decir 4 grupos × 50 repeticiones.
Ninguna de las dos cifras es compatible con la otra, y 249 no es divisible entre 4 grupos
de igual tamaño.

Vale la pena notar que F = 4.3067 solo es aritméticamente consistente con la versión de
245 grados de libertad; con 196 el estadístico daría alrededor de 3.40. Esto sugiere que
el error está en el texto, no en la tabla, y que el análisis se corrió sobre un conjunto
de observaciones distinto al que describe el artículo.

**c) No reproducible.** Se probaron todas las combinaciones posibles de los archivos de
validación disponibles para los cuatro grupos de ROIs. Ninguna reproduce F = 4.3067 con
p = 0.0057. La combinación más cercana (F = 4.3547, p = 0.0056) mezcla tres archivos
CNN-1D con uno LSTM, lo que no sería un contraste válido entre grupos de ROIs.

## 5. Tabla 1 (parámetros) vs. código

| Parámetro | Tabla 1 | Código | Estado |
|---|---|---|---|
| Optimizador | Adam | Adam | ok |
| Learning rate | 1e-4 | `Adam(1e-4)` | ok |
| Épocas | 150 | 150 | ok |
| Early stopping | patience 100 | patience 100 | ok |
| Pérdida | Binary cross-entropy | binary_crossentropy | ok |
| Inicialización | Ortogonal (LSTM) / Glorot (densas) | valores por defecto de Keras | ok (coinciden con los defaults) |
| **Batch size** | **32** | **`batch_size=8`** | **discrepante** |
| **Gradient clipping** | **1.0** | **no implementado** | **discrepante** |

`build_model` construye `Adam(1e-4)` sin `clipnorm` ni `clipvalue`, de modo que el
recorte de gradiente declarado en la Tabla 1 no ocurre. Ambas discrepancias son
verificables por cualquier revisor que abra el repositorio público.

## 6. Estado de los datos para los experimentos pendientes

- `data/fc-pearson-seq/` contiene secuencias para los cuatro sitios (NYU, Peking, OHSU,
  NeuroIMAGE) **solo para 12 y 18 ROIs**. Para 39 y 116 ROIs únicamente existe NYU. Para
  replicar la comparación completa entre grupos en otros sitios hay que extraer esas
  señales primero.
- `results/rois_all_sites/` contiene por ahora únicamente OHSU con 12 y 18 ROIs.
- En esos resultados de OHSU el accuracy de validación supera al de entrenamiento
  (12 ROIs: 73.57 val vs 61.68 train; 18 ROIs: 75.33 vs 74.55), con máximos de validación
  de 100 %. Conviene revisar esa corrida antes de usarla: el patrón sugiere un problema
  en la partición o en el tamaño de muestra del sitio.

---

## Acciones sugeridas

Por orden de urgencia:

1. **Regenerar la Tabla 2 completa con una sola arquitectura.** Correr los cuatro grupos
   de ROIs con LSTM-128 y 50 repeticiones, versionando los CSV. Mientras la fila de 116
   ROIs provenga de un modelo CNN-1D+LSTM-118, la comparación no sostiene la afirmación
   sobre el efecto del número de regiones.
2. Localizar la corrida de la que salió el 76.59 % o, si no existe, corregir el resumen,
   la introducción, la discusión y las conclusiones para que reflejen la Tabla 2.
3. Recalcular el ANOVA sobre esos resultados homogéneos, dejando explícito qué
   observaciones lo componen, y corregir la columna MS y los grados de libertad.
4. Corregir la Tabla 1 (batch size) y decidir si se implementa el recorte de gradiente o
   se elimina esa fila.
5. Adoptar una convención de nombres y no sobrescribir archivos de resultados: varias
   cifras del artículo solo existen en versiones antiguas del historial de git.
