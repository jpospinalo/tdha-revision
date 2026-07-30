# Plan de análisis estadístico: comparación de 12, 18, 39 y 116 ROIs
**Versión del plan:** 5.6 — alcance de reconciliación precisado  
**Repositorio verificado:** `tdha-revision-main(18).zip`  
**Actualización posterior:** `results/README.md` actualizado y verificado por el equipo  
**Sitios:** NYU, Peking, NeuroIMAGE y OHSU  
**Estado:** listo para congelar tras aprobar D1–D5
## 1. Cambio metodológico principal
La versión anterior proponía describir el desempeño mediante AUC out-of-fold (OOF) por repetición, pero realizar la inferencia mediante AUC por pliegue. Esa combinación no es adecuada:
- utiliza dos unidades estadísticas diferentes;
- contradice la convención OOF documentada por el proyecto;
- produce AUC extremadamente discretos y variables en los pliegues pequeños;
- hace que un margen de no inferioridad de 0.05 tenga muy poca posibilidad de ser informativo.
La versión 5.6 elimina el AUC por pliegue y la prueba de Nadeau–Bengio como análisis principal. Descripción e inferencia utilizarán la misma unidad: **el promedio de los cinco AUC OOF calculados por repetición**. También fija de antemano cómo comunicar una diferencia favorable a 12 ROIs sin convertir un resultado exploratorio en una afirmación de superioridad.
La revisión de factibilidad del diseño anterior produjo:
| Sitio | Sujetos aproximados por pliegue | SD de diferencias por pliegue | Semi-amplitud corregida |
|---|---:|---:|---:|
| NYU | 18 | 0.195 | 0.119 |
| Peking | 18 | 0.143 | 0.087 |
| OHSU | 7 | 0.356 | 0.216 |
| NeuroIMAGE | 4 | 0.453 | 0.275 |
La última columna indica la precisión aproximada bajo la inferencia por pliegue. Si \(\Delta=0\), un margen de 0.05 no superaría ese límite en ningún sitio. No es una imposibilidad matemática absoluta —una diferencia suficientemente positiva podría superar el criterio—, pero demuestra que esa unidad produce una prueba con muy poca capacidad para responder la pregunta planteada, especialmente bajo una regla que exige éxito en los cuatro sitios.
## 2. Transparencia sobre la revisión de factibilidad
Durante la revisión del plan se calcularon varianzas de las diferencias por pliegue y quedaron visibles las diferencias medias observadas entre 12 y 116 ROIs.
Por tanto:
- este análisis ya no puede presentarse como una preinscripción prospectiva ciega a los resultados;
- el plan debe registrarse como **análisis bloqueado después de una revisión de factibilidad con exposición a resultados**;
- cualquier afirmación confirmatoria definitiva requerirá una cohorte o conjunto de datos externo que no haya participado en estas decisiones;
- los resultados actuales podrán proporcionar estimaciones, intervalos y evidencia de apoyo, pero deberán comunicarse con esta limitación.
No se modificará el margen para acomodarlo a las diferencias observadas.
## 3. Objetivo y estimando principal
El objetivo es estimar cuánto cambia la discriminación al usar 12 ROIs en lugar de 116 ROIs dentro de cada sitio.
Para cada repetición \(r\), sitio \(s\) y configuración \(q\), se calculará un AUC usando las predicciones OOF de todos los sujetos:
\[
A_{sqr}=\mathrm{AUC}(y_{is},p_{isqr})
\]
Cada sujeto aparece exactamente una vez en validación externa dentro de cada repetición. El estimando principal será la media de los cinco AUC:
\[
\bar A_{sq}=\frac{1}{R}\sum_{r=1}^{R}A_{sqr},
\qquad R=5
\]
y el efecto principal:
\[
\Delta_s=\bar A_{s,12}-\bar A_{s,116}
\]
Un valor positivo favorece 12 ROIs.
Este estimando coincide con la convención OOF publicada en `results/README.md`: primero se calcula AUC por repetición y después se promedian los AUC. No se promedian probabilidades antes del AUC.
La probabilidad media por sujeto podrá conservarse para diagnóstico de estabilidad y errores, pero su AUC se identificaría como desempeño de un ensamble de repeticiones y no será el resultado principal.
## 4. Decisiones que debe aprobar el equipo
### D1. Métricas
Propuesta:
- **Primaria:** media de los cinco AUC OOF por repetición.
- **Secundarias:** balanced accuracy, F1-macro, sensibilidad y especificidad.
Las secundarias seguirán la misma regla: calcular la métrica sobre todos los sujetos dentro de cada repetición y promediar las cinco estimaciones. Utilizarán un umbral fijo de 0.5. Accuracy, log-loss y Brier no aparecerán en las tablas principales; podrán calcularse después si existe una pregunta específica.
La tabla principal mostrará estimaciones puntuales de las secundarias para no hacerse ilegible; sus intervalos bootstrap se exportarán en una tabla suplementaria. Sensibilidad, especificidad y balanced accuracy son medidas condicionadas por clase; F1-macro también depende de la composición observada de la muestra. El bootstrap estratificado mantendrá los conteos de clase observados y esta condición se declarará.
### D2. Margen práctico
El equipo debe definir si existe una pérdida máxima de AUC científicamente aceptable, \(\delta\). El margen representa relevancia práctica y debe definirse independientemente de la precisión observada [Lakens, 2017](https://pubmed.ncbi.nlm.nih.gov/28736600/).
El orden obligatorio será:
1. justificar \(\delta\) por relevancia científica, no por los resultados;
2. registrar la justificación;
3. calcular la precisión alcanzable para el estimando definido en la sección 3 —la media de los cinco AUC OOF por repetición— mediante el bootstrap pareado sobre sujetos;
4. si la precisión es insuficiente, mantener \(\delta\) y declarar que los datos no pueden responder la no inferioridad;
5. no ampliar \(\delta\) para lograr una conclusión favorable.
Hasta que este proceso se complete, `noninferiority_margin` permanecerá en `null`.
Con la precisión observada en NYU, la semi-amplitud unilateral es aproximadamente 0.049–0.050. Por tanto, si se aprobara \(\delta=0.05\), la incertidumbre consumiría casi todo el margen y la no inferioridad solo sería compatible cuando la diferencia puntual fuera aproximadamente cero o positiva. Esto hace que el criterio sea muy exigente en la práctica, pero **no convierte formalmente la prueba de no inferioridad en una prueba de superioridad**: la primera contrasta el límite \(-\delta\), mientras la segunda contrasta cero.
La revisión de factibilidad ya expuesta produjo, de forma aproximada, un sitio compatible con no inferioridad para \(\delta=0.05\) —NYU— y tres no compatibles —Peking, OHSU y NeuroIMAGE—. Estos valores se registran como información de factibilidad conocida antes de congelar el plan; no se utilizarán para escoger ni modificar el margen.
### D3. Tipo de conclusión
Recomendación para los datos actuales:
> Utilizar estimación por sitio como análisis principal, no una única prueba confirmatoria global.
Si se aprueba un margen, se podrá informar si cada intervalo es compatible con no inferioridad, pero como evidencia de apoyo y no como una preinscripción prospectiva.
No se recomienda restringir retrospectivamente el análisis principal a NYU y Peking. NeuroIMAGE y OHSU deben mantenerse porque muestran la limitación real de precisión y transportabilidad.
### D4. Incertidumbre
La inferencia principal utilizará un bootstrap pareado y estratificado por clase sobre sujetos. La sección 7 especifica el procedimiento.
### D5. Pipeline o ensamble
Se recomienda aprobar explícitamente:
> El objetivo principal es el desempeño medio del pipeline bajo las cinco repeticiones de validación cruzada, no el desempeño de un ensamble que promedia cinco probabilidades.
Promediar probabilidades antes de calcular AUC cambia el estimando y modificó \(\Delta\) hasta aproximadamente ±0.022 según el sitio. Si más adelante se desea desplegar un ensamble, deberá definirse cómo se entrenará con todos los datos y analizarse por separado.
La media de AUC OOF tampoco es el desempeño observado de un único modelo final entrenado con todos los sujetos: estima el comportamiento del procedimiento de entrenamiento mediante validación cruzada. El desempeño de un modelo final desplegado solo puede confirmarse en datos externos no usados para entrenarlo.
## 5. Estado de los datos
La versión 18 contiene las 16 combinaciones requeridas:
| Sitio | 12 | 18 | 39 | 116 |
|---|---:|---:|---:|---:|
| NYU | disponible | disponible | disponible | disponible |
| Peking | disponible | disponible | disponible | disponible |
| NeuroIMAGE | disponible | disponible | disponible | disponible |
| OHSU | disponible | disponible | disponible | disponible |
Se verificó:
- esquema 4 y validación 10×5 en las 16 corridas;
- 50 filas de `metrics_val.csv` por corrida;
- una predicción externa por sujeto y repetición;
- mismos sujetos, etiquetas y pliegues entre tamaños dentro de cada sitio;
- igualdad dentro de sitio de arquitectura, representación, entrenamiento, `split_fingerprint`, `bold_hash`, `data_code_hash` y `runner_code_hash`.
Conteos:
| Sitio | Sujetos | Predicciones por corrida |
|---|---:|---:|
| NYU | 177 | 885 |
| Peking | 183 | 915 |
| NeuroIMAGE | 39 | 195 |
| OHSU | 66 | 330 |
`class_weight=true` únicamente en Peking; en los otros tres sitios es `false`. Esto no afecta el pareamiento dentro de sitio, pero impide tratar los cuatro efectos como réplicas plenamente intercambiables. No se realizará una estimación combinada entre sitios.
La corrida Peking–116 debe registrarse con su identificador real:
```text
Peking_rois116_w60s6_brainnetcnn_240732d1
```
No debe renombrarse para añadir `control_baseline_v13`. La configuración se validará mediante `config.json`, no mediante el nombre.
`peking_dummy.txt` no forma parte del contrato de resultados, se ignorará y no invalida la corrida.
## 6. Construcción de la base analítica
### 6.1 Fuentes
| Archivo | Uso |
|---|---|
| `config.json` | Configuración, hashes y procedencia |
| `predictions_val.csv` | Probabilidades OOF por sujeto |
| `folds.csv` | Verificación de sujetos y particiones |
| `metrics_val.csv` | Comprobación de integridad y resumen histórico |
| `metrics_train.csv` y `history.csv` | Diagnóstico de entrenamiento, no inferencia principal |
`results/archive/` quedará excluida y `results/runs/` será de solo lectura.
### 6.2 Manifiesto
`run_manifest.csv` contendrá exactamente una corrida por sitio y tamaño:
```text
site,roi_set,run_id,relative_path,include,rationale
```
No se seleccionarán corridas por fecha, tag o coincidencia de nombres.
### 6.3 Tabla principal por sujeto
Se generará `subject_scores.csv` con una fila por sujeto y columnas separadas para las cinco predicciones:
```text
site
roi_set
subject_id
y_true
y_prob_r1
y_prob_r2
y_prob_r3
y_prob_r4
y_prob_r5
y_prob_mean
y_prob_sd
n_positive_predictions
```
Controles:
- exactamente cinco predicciones por sujeto y configuración;
- misma etiqueta entre repeticiones y tamaños;
- probabilidades finitas dentro de \([0,1]\);
- mismos sujetos en 12, 18, 39 y 116;
- todas las predicciones son externas;
- ausencia de duplicados.
### 6.4 Métricas por repetición y reconciliación
`metrics_by_repeat.csv` contendrá por sitio, configuración y repetición las cinco métricas de D1: AUC, balanced accuracy, F1-macro, sensibilidad y especificidad. También contendrá accuracy con función exclusiva de auditoría; accuracy no formará parte de la inferencia ni de las tablas principales.
En la versión 18 inicialmente revisada, `results/README.md` declaraba un alcance limitado a 12, 18 y 39 ROIs, no contenía la columna de 116 y afirmaba incorrectamente que no existían esos resultados. En una actualización documental posterior, el equipo informó haber aplicado y verificado todos los cambios requeridos. Esta sección conserva las cifras y modificaciones como registro de auditoría.
Las cifras de referencia para añadir la columna `116 ROIs`, verificadas directamente desde `predictions_val.csv` con la convención AUC / balanced accuracy / accuracy OOF, son:
| Sitio | Corrida de 116 | 116 ROIs |
|---|---|---:|
| NYU | `NYU_rois116_w60s6_brainnetcnn_control_baseline_v13_160b89cd` | 53.93 / 53.09 / 53.11 |
| NeuroIMAGE | `NeuroIMAGE_rois116_w61s6_brainnetcnn_control_baseline_v13_669d72bd` | 51.93 / 50.32 / 51.79 |
| OHSU | `OHSU_rois116_w48s5_brainnetcnn_control_baseline_v13_f82f17b4` | 55.83 / 54.15 / 54.24 |
| Peking | `Peking_rois116_w60s6_brainnetcnn_240732d1` | 60.33 / 58.09 / 59.02 |
La actualización registrada:
- amplió el alcance a conjuntos de 12, 18, 39 y 116 ROIs;
- eliminó la frase "No hay resultados de 116 ROIs en el compendio";
- actualizó las referencias a "tres corridas activas" para incluir las cuatro configuraciones;
- registró los identificadores reales de las cuatro corridas de 116 sin renombrarlos;
- retiró "Ejecutar 116 ROIs como baseline diagnóstico" de las próximas corridas y lo registró como completado en los cuatro sitios;
- revisó la conclusión prudente incluyendo 116: los AUC puntuales más altos son 18 ROIs en NYU y NeuroIMAGE, 116 en OHSU y 39 en Peking. Esta es una descripción de estimaciones puntuales, no evidencia de superioridad entre tamaños;
- conservó `config.json` y `predictions_val.csv` como fuentes de verdad;
- documentó que, dentro de cada sitio, las cuatro corridas comparten `split_fingerprint`, `bold_hash`, `data_code_hash`, `runner_code_hash`, arquitectura y `folds.csv`, respaldando el pareamiento sujeto a sujeto;
- advirtió que Peking–116 no lleva la etiqueta `control_baseline_v13`, no debe renombrarse y que `peking_dummy.txt` queda fuera del contrato de siete artefactos;
- añadió una remisión al plan estadístico para impedir que la tabla de estimaciones puntuales se interprete como conclusión inferencial.
La cobertura documental real es:
| Métrica | Corridas con valor publicado | Uso en la reconciliación |
|---|---:|---|
| AUC | 16/16 | reconciliación obligatoria de la media |
| Balanced accuracy | 16/16 | control documental adicional |
| Accuracy | 16/16 | control documental adicional; no es métrica de D1 |
| F1-macro | 8/16 | no se usará como control externo por su cobertura incompleta |
| Sensibilidad | 0/16 | autoconsistencia únicamente |
| Especificidad | 0/16 | autoconsistencia únicamente |
El README publica las medias de cinco repeticiones, no los cinco valores individuales. Por tanto, la reconciliación externa comparará únicamente las medias disponibles: AUC para las 16 corridas y, como controles adicionales, balanced accuracy y accuracy. La comparación se hará en porcentaje y con el redondeo a dos decimales utilizado por el README. No se exigirá que un valor por repetición coincida con un dato que el README no publica.
Los cinco valores individuales de cada métrica y sus medias sin redondear se verificarán por autoconsistencia recalculándolos desde `predictions_val.csv`. Esta comprobación cubrirá las 16 corridas y todas las métricas de D1; accuracy se mantendrá solo como control adicional.
No se ampliará el README con F1-macro, sensibilidad o especificidad únicamente para aumentar la cobertura de reconciliación. El README es un registro breve; las salidas completas corresponderán al análisis.
Las cuatro ternas de 116 fueron confirmadas por dos implementaciones computacionales independientes: la implementación habitual de métricas y un recálculo por rangos de Mann–Whitney para AUC y conteo directo de TP/TN/FP/FN para las métricas de clasificación. La coincidencia dígito por dígito respalda la corrección aritmética y la reproducibilidad de las cifras. No constituye validación con datos estadísticamente independientes, porque ambos cálculos usan las mismas predicciones OOF. Se mantendrá el control de autoconsistencia: cinco AUC OOF por repetición, su media y el recálculo directo desde las predicciones deben coincidir.
La tabla principal no mostrará una segunda columna de "AUC ensamble", porque el estimando elegido ya coincide con la convención del README y añadirla reintroduciría dos respuestas a preguntas distintas.
No se construirá `metrics_by_fold.csv` para inferencia y no se usarán AUC de pliegues pequeños.
## 7. Incertidumbre y comparación 12 frente a 116
### 7.1 Bootstrap primario
Para cada sitio:
1. Ordenar los sujetos de forma ascendente por `subject_id` y separar control y TDAH.
2. Remuestrear con reemplazo dentro de cada clase.
3. Usar exactamente los mismos índices remuestreados para todas las configuraciones ROI, las cinco repeticiones y las métricas comparadas.
4. Dentro de cada repetición, calcular AUC de 12 y AUC de 116.
5. Promediar los cinco AUC de cada configuración y calcular \(\Delta_s\).
6. Repetir 10.000 veces reiniciando un generador PCG64 con semilla 42 al comenzar cada sitio, de modo que cambiar el orden de procesamiento de los sitios no cambie sus remuestreos. No se combinarán inferencias entre sitios, por lo que no se requiere independencia entre sus secuencias bootstrap.
El remuestreo estratificado garantiza presencia de ambas clases y el remuestreo pareado conserva la correlación entre las dos configuraciones.
Se reportarán para el promedio de AUC por repetición:
- AUC por configuración con intervalo bilateral del 95%;
- \(\Delta_s\) con intervalo bilateral del 95%;
- si se aprueba \(\delta\), límite inferior unilateral del 95%.
La no inferioridad sería compatible con los datos cuando:
\[
L_{95\%,\,unilateral}(\Delta_s)>-\delta
\]
Se utilizará un intervalo percentil como método base. Los cuantiles se calcularán con interpolación lineal (`numpy.quantile(..., method="linear")`). La semilla, generador, alcance de la semilla, orden de sujetos, iteraciones y regla de cuantiles quedarán registrados. Esta especificación es importante porque el límite unilateral de NYU está muy cerca de cero y pequeñas diferencias de implementación pueden cambiar su signo sin alterar la conclusión de no inferioridad respecto de \(-0.05\).
`sklearn.metrics.roc_auc_score` será la referencia de corrección. Una implementación por rangos de Mann–Whitney podrá añadirse solo si un benchmark demuestra una mejora necesaria y si reproduce exactamente a scikit-learn en fixtures con empates y remuestreos duplicados. No se sustituirá una función validada únicamente para ahorrar segundos.
### 7.2 Alcance de la incertidumbre
El bootstrap remuestrea sujetos manteniendo fijas las predicciones cross-fitted ya obtenidas y las cinco particiones de validación que están incorporadas en ellas. Por tanto:
- cuantifica incertidumbre entre sujetos para estas predicciones;
- no vuelve a sortear particiones ni representa nuevos entrenamientos del pipeline;
- el promedio de los cinco AUC integra la variación observada entre esas cinco repeticiones en la estimación puntual, pero el intervalo no incorpora como componente adicional la incertidumbre de generar otras particiones o reentrenamientos;
- no debe interpretarse como variabilidad total del algoritmo en nuevas cohortes.
La inferencia sobre validación cruzada es compleja porque los entrenamientos se solapan y no existe un estimador universalmente insesgado de su varianza [Bengio y Grandvalet, 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html). Los intervalos convencionales de validación cruzada también pueden tener cobertura insuficiente [Bates, Hastie y Tibshirani, 2024](https://pubmed.ncbi.nlm.nih.gov/39308484/).
El uso de una AUC cross-validada sobre predicciones OOF, en vez de AUC diminutas por pliegue, tiene precedente en métodos desarrollados específicamente para inferencia de CV-AUC [LeDell, Petersen y van der Laan, 2015](https://pubmed.ncbi.nlm.nih.gov/26279737/). Ese trabajo utiliza curvas de influencia; no debe citarse como si validara automáticamente el bootstrap condicional propuesto aquí.
Una inferencia completa sobre la variabilidad de reentrenamiento requeriría volver a ejecutar el pipeline dentro de un remuestreo externo o validación anidada. Eso queda fuera de este análisis y se reservará para una futura confirmación independiente.
### 7.3 Diagnóstico de precisión
Una implementación computacional separada del bootstrap, ejecutada sobre las mismas predicciones OOF con 10.000 remuestreos y semilla 42, produjo estas semi-amplitudes unilaterales aproximadas:
| Sitio | Semi-amplitud unilateral bootstrap |
|---|---:|
| NYU | 0.050 |
| Peking | 0.054 |
| OHSU | 0.075 |
| NeuroIMAGE | 0.128 |
Por tanto, con una diferencia verdadera cercana a cero, \(\delta=0.05\) solo estaría aproximadamente al nivel de precisión de NYU y seguiría siendo insuficiente para una regla conjunta de cuatro sitios. Esto refuerza la recomendación de usar estimación por sitio y no una prueba global.
La revisión de factibilidad con \(\delta=0.05\) obtuvo límites inferiores unilaterales aproximados de +0.002 en NYU, −0.094 en Peking, −0.084 en OHSU y −0.17 en NeuroIMAGE. Solo NYU supera \(-0.05\). El límite de NYU próximo a cero no se presentará como superioridad: su intervalo bilateral del 95% (aproximadamente −0.007 a +0.112) todavía contiene cero y no se había preespecificado una prueba confirmatoria unilateral de superioridad.
La implementación reproducirá una tabla de precisión con:
- error estándar bootstrap de \(\Delta_s\);
- amplitud de los intervalos;
- margen científicamente aprobado;
- indicación de si la precisión permite una conclusión.
El diagnóstico no propondrá un margen alternativo. Si el intervalo es demasiado ancho, la conclusión será `inconclusa por precisión insuficiente`.
## 8. Comparación secundaria de 12, 18, 39 y 116
El contraste 12 frente a 116 aparecerá únicamente en la tabla principal. No se repetirá dentro de una familia secundaria ni se ajustará con Holm.
Los cinco contrastes secundarios serán:
- 12–18;
- 12–39;
- 18–39;
- 18–116;
- 39–116.
Se estimarán con la misma media de métricas OOF por repetición y el mismo bootstrap pareado. Se reportarán diferencias e intervalos del 95%, sin declaraciones de significancia. De este modo no se necesita añadir una familia de valores p ni una corrección múltiple.
## 9. Análisis de errores
Con umbral 0.5, comparar 12 frente a 116 por sitio, clase y repetición:
- ambos correctos;
- correcto solo con 12;
- correcto solo con 116;
- ambos incorrectos.
La tabla agregará los cinco resultados externos de cada sujeto, sin convertir la probabilidad media en una predicción de ensamble. También se resumirá por sujeto cuántas de sus cinco clasificaciones fueron correctas con cada configuración.
Se mostrarán sensibilidad, especificidad y diferencias de probabilidad. No se eliminarán sujetos difíciles.
`y_prob_sd` y `n_positive_predictions` se usarán para describir estabilidad entre repeticiones. Se marcarán como inestables, solo con finalidad descriptiva, los sujetos cuyas cinco clasificaciones no coincidan; esto no afectará su inclusión ni la métrica principal.
No se incorporarán edad, sexo o movimiento hasta disponer de una fuente completa, versionada y aprobada para los cuatro sitios.
## 10. Tablas y figuras
### Tabla 1. Auditoría
Las 16 corridas, parámetros críticos, hashes y estado.
### Tabla 2. Desempeño
| Sitio | ROIs | N | AUC (IC 95%) | Balanced accuracy | F1-macro | Sensibilidad | Especificidad |
|---|---:|---:|---:|---:|---:|---:|---:|
Todos los valores se calcularán desde `subject_scores.csv`.
La tabla principal mostrará intervalos solo para AUC. Los intervalos bootstrap de balanced accuracy, F1-macro, sensibilidad y especificidad se entregarán en `secondary_metric_intervals.csv` para no sobrecargar la tabla.
### Tabla 3. Comparación principal
| Sitio | AUC 12 | AUC 116 | Δ 12−116 (IC 95%) | Límite unilateral | Margen | Interpretación |
|---|---:|---:|---:|---:|---:|---|
Si no se aprueba un margen, las columnas de límite y margen se omitirán.
### Tabla 4. Contrastes secundarios
Los cinco contrastes exploratorios con diferencias e intervalos.
### Figura 1. AUC por configuración y sitio
- cuatro paneles, uno por sitio;
- eje X: 12, 18, 39 y 116;
- punto: media de los cinco AUC OOF por repetición;
- barras: intervalo bootstrap del 95%;
- misma escala vertical en todos los paneles.
Los cinco AUC por repetición podrán mostrarse como puntos grises pequeños para estabilidad, sin confundirse con la estimación principal.
### Figura 2. Forest plot 12 frente a 116
- \(\Delta_s\) por sitio;
- intervalo bootstrap del 95%;
- línea vertical en cero;
- si existe \(\delta\), línea en \(-\delta\);
- sin efecto combinado entre sitios.
Las figuras mostrarán estimaciones e incertidumbre en lugar de barras simples que oculten la distribución [Weissgerber et al., 2015](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128).
## 11. Organización del repositorio
```text
analysis/
├── README.md
│
├── roi_comparison/
│   ├── README.md
│   ├── analysis_plan.md
│   ├── roi_comparison.ipynb
│   │
│   ├── config/
│   │   ├── analysis_config.json
│   │   └── run_manifest.csv
│   │
│   ├── scripts/
│   │   ├── build_analysis_dataset.py
│   │   └── run_statistical_analysis.py
│   │
│   ├── outputs/
│   │   ├── data/
│   │   │   ├── subject_scores.csv
│   │   │   └── metrics_by_repeat.csv
│   │   ├── tables/
│   │   ├── figures/
│   │   └── analysis_manifest.json
│   │
│   └── tests/
│       └── test_analysis.py
│
└── loso/
    └── README.md
```
La auditoría se integrará en `build_analysis_dataset.py`; no habrá un `validate_analysis.py` separado.
El notebook mostrará el proceso y los resultados, pero importará las funciones de los scripts. No duplicará estadísticas.
## 12. Configuración
`analysis_config.json` contendrá:
```json
{
  "analysis_schema_version": 1,
  "primary_metric": "mean_repeat_oof_auc",
  "repeat_aggregation": "metric_then_mean",
  "secondary_metrics": [
    "balanced_accuracy",
    "f1_macro",
    "sensitivity",
    "specificity"
  ],
  "classification_threshold": 0.5,
  "positive_label": 1,
  "noninferiority_margin": null,
  "bootstrap_iterations": 10000,
  "bootstrap_seed": 42,
  "bootstrap_rng": "numpy_pcg64",
  "bootstrap_seed_scope": "reset_per_site",
  "bootstrap_subject_order": "subject_id_ascending",
  "bootstrap_quantile_method": "linear",
  "bootstrap_method": "paired_stratified_percentile"
}
```
Los hiperparámetros experimentales no se copiarán aquí; se leerán de cada `config.json`.
## 13. Controles mínimos
La implementación se aprobará si:
1. valida las 16 corridas y el pareamiento por sujeto;
2. detiene el proceso ante una repetición incompleta o etiqueta contradictoria;
3. construye una sola fila por sujeto, sitio y tamaño;
4. reproduce para las 16 corridas la media de los cinco AUC OOF publicada en `results/README.md` y, como controles adicionales, las medias publicadas de balanced accuracy y accuracy; no intenta reconciliar valores individuales por repetición contra el README;
5. comprueba para las 16 corridas que el recálculo directo desde `predictions_val.csv`, los cinco valores por repetición y su media sin redondear son internamente consistentes para todas las métricas de D1; comprueba también accuracy como auditoría y contrasta las cifras de 116 con el registro de la sección 6.4;
6. demuestra mediante tests que el bootstrap es pareado, estratificado y reproducible;
7. demuestra mediante un fixture que primero calcula la métrica por repetición y después promedia, sin promediar probabilidades;
8. mantiene siempre el signo 12 menos 116;
9. no lee `archive/` ni modifica `results/` durante el análisis;
10. produce los mismos resultados desde scripts y desde un notebook reiniciado;
11. produce para cada sitio los mismos remuestreos aunque cambie el orden en que se procesan los sitios;
12. registra que el plan se cerró después de exposición a resultados.
## 14. Secuencia
1. Aprobar D1, D3, D4 y D5.
2. Justificar científicamente D2 o decidir trabajar solo con estimación.
3. Actualizar y revisar `results/README.md` según la sección 6.4 — **cumplido y verificado por el equipo antes de congelar**.
4. Congelar el plan, el README de resultados, la configuración y el manifiesto.
5. Construir y validar `subject_scores.csv`.
6. Ejecutar el diagnóstico de precisión sin modificar \(\delta\).
7. Generar tablas, figuras y análisis de errores.
8. Ejecutar tests y reiniciar el notebook.
9. Revisar la redacción y registrar explícitamente la exposición previa a resultados.
## 15. Reglas de comunicación
- **Estimación principal:** diferencia de AUC y su intervalo por sitio.
- **Compatible con no inferioridad:** solo si existe un margen científicamente aprobado y el límite inferior lo supera.
- **Inconcluso:** el intervalo cruza el margen.
- **Diferencia puntual favorable a 12:** si \(\hat\Delta_s>0\) pero el intervalo bilateral del 95% contiene cero, se informará que la estimación puntual favorece 12 ROIs, pero que los datos siguen siendo compatibles con ausencia de diferencia y con efectos en ambas direcciones.
- **Intervalo bilateral completamente positivo:** se describirá como evidencia exploratoria favorable a 12 ROIs, no como conclusión confirmatoria de superioridad, debido a la exposición previa a los resultados.
- **Límite unilateral por encima de cero con intervalo bilateral que cruza cero:** no se titulará ni resumirá como superioridad. Podrá registrarse únicamente como resultado exploratorio unilateral, señalando que esa prueba de superioridad no fue preespecificada.
- **No inferioridad y superioridad son conclusiones distintas:** superar \(-\delta\) no implica superar cero.
- **Otra configuración presenta la estimación puntual más alta:** se informará el orden descriptivo por sitio y la diferencia con su intervalo dentro de los contrastes secundarios. No se utilizarán "mejor", "óptima" ni "ganadora" sin un criterio de selección preespecificado y evidencia confirmatoria. Si el patrón varía entre sitios, se destacará esa heterogeneidad. El resultado podrá motivar una futura comparación confirmatoria, pero no reemplazará ni reinterpretará retrospectivamente el contraste principal 12 frente a 116.
- **No significativo:** no equivale a igualdad.
- **Equivalente:** no se utilizará; este plan no incluye TOST.
- **Confirmatorio:** no se utilizará para los datos actuales debido a la exposición previa a resultados.
La conclusión deberá presentar magnitud e incertidumbre, por ejemplo:
> La diferencia AUC 12−116 fue \(\Delta_s\) (IC 95% [...]) en cada sitio; la precisión disponible y las diferencias entre sitios limitan la generalización.
Si no existe un margen aprobado o los intervalos son amplios, se evitarán palabras como "similar", "igual" o "equivalente".
No afirmar que 12 ROIs contiene la misma información biológica que 116 ni generalizar fuera de estas cohortes sin validación externa.
En particular, que 18 ROIs supere puntualmente a 12 en varios sitios no contradice por sí solo la pregunta de no inferioridad 12 frente a 116. Las diferencias AUC puntuales 18−12 ya conocidas antes de congelar el plan son aproximadamente +0.0095 en NYU, +0.0038 en Peking, +0.1460 en NeuroIMAGE y −0.0270 en OHSU. Además, 18 presenta el máximo puntual entre los cuatro tamaños solo en NYU y NeuroIMAGE; el máximo corresponde a 39 en Peking y a 116 en OHSU. Estas observaciones se identificarán como resultados conocidos durante la planificación.
Este patrón tampoco permite concluir que 18 sea el punto óptimo del equilibrio entre parsimonia y desempeño: ese balance no fue definido como estimando primario y los máximos puntuales cambian entre sitios. Podrá describirse como una hipótesis para un estudio posterior.
## 16. Preguntas para aprobación
1. ¿Se aprueba la media de los cinco AUC OOF por repetición como unidad primaria?
2. ¿Se aprueba AUC como métrica primaria y las cuatro métricas secundarias?
3. ¿Existe un margen científicamente defendible o se trabajará solo con estimación?
4. ¿Se aprueba retirar la prueba global y presentar los cuatro sitios por separado?
5. ¿Se aprueba el bootstrap pareado estratificado y su interpretación condicional?
6. ¿Se aprueba documentar el análisis actual como posterior a exposición de resultados?
7. ¿Se aprueba reservar el promedio de probabilidades únicamente para diagnóstico y no usar su AUC como resultado principal?
8. ¿Se aprueba la regla de comunicación para resultados puntualmente favorables y para límites unilaterales por encima de cero?
9. ¿Se aprueba comunicar los máximos de configuraciones secundarias como ordenamientos descriptivos por sitio, sin denominarlos óptimos ni alterar el contraste principal?
## Referencias
1. Lakens D. Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses. *Social Psychological and Personality Science*. 2017. [PubMed](https://pubmed.ncbi.nlm.nih.gov/28736600/).
2. Bengio Y, Grandvalet Y. No Unbiased Estimator of the Variance of K-Fold Cross-Validation. *Journal of Machine Learning Research*. 2004;5:1089–1105. [Artículo](https://www.jmlr.org/papers/v5/grandvalet04a.html).
3. Bates S, Hastie T, Tibshirani R. Cross-validation: what does it estimate and how well does it do it? *Journal of the American Statistical Association*. 2024;119:1434–1445. [PubMed](https://pubmed.ncbi.nlm.nih.gov/39308484/).
4. LeDell E, Petersen M, van der Laan M. Computationally efficient confidence intervals for cross-validated area under the ROC curve estimates. *Electronic Journal of Statistics*. 2015;9:1583–1607. [PubMed](https://pubmed.ncbi.nlm.nih.gov/26279737/).
5. Weissgerber TL, Milic NM, Winham SJ, Garovic VD. Beyond Bar and Line Graphs: Time for a New Data Presentation Paradigm. *PLOS Biology*. 2015. [Artículo](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128).
