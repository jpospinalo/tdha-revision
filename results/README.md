# Registro breve de experimentación

Última actualización: 2026-07-27  
Alcance actual: NYU, 12 ROIs, BrainNetCNN, clasificación control/TDAH.

## Para qué sirve este archivo

Este documento evita repetir experimentos y orienta las siguientes corridas. No sustituye
`config.json`, `resumen.md` ni los CSV de cada corrida: esos artefactos siguen siendo la
fuente de verdad. Algunas corridas resumidas aquí todavía pueden estar almacenadas como ZIP
fuera de este checkout; el `run_id` o el `config_hash` permite identificarlas.

Las cifras de la tabla son métricas **OOF por repetición**: en cada una de las cinco
repeticiones se agruparon las predicciones externas de los diez folds antes de calcular la
métrica. Son preferibles al promedio de métricas de folds pequeños. Diferencias de unas
décimas no deben interpretarse como mejora sin una comparación pareada.

## Referencia vigente

Configuración que debe mantenerse fija al evaluar un cambio:

- NYU, 12 ROIs, BrainNetCNN y representación `ordered`.
- Ventana 120 s (60 TR), paso 12 s (6 TR), rectangular y sin Fisher z.
- `e2e=4`, `e2n=8`, `dense=8`, `dropout=0.7`, `leaky=0.33`,
  `l2_reg=0.05`, `inter_dropout=0.6`.
- `lr=1e-4`, batch 32, máximo 300 épocas, paciencia 25 y monitor `val_loss`.
- Validación 10 × 5, `inner_val_frac=0.15`, semilla 42 y sin `class_weight`.

Corrida de referencia:
`NYU_rois12_w60s6_brainnetcnn_control_baseline_repeat_300_clean_a88f2eb7`.

Resultado OOF: accuracy 56.95 ± 1.89 %, AUC 58.98 ± 2.48 %, F1-macro
56.91 ± 1.93 % y balanced accuracy 56.98 ± 1.89 %.

La repetición independiente de la misma configuración produjo accuracy 57.18 % y AUC
58.99 %. La concordancia es buena: la referencia es reproducible y no conviene seguir
repitiéndola salvo como control de una modificación del entorno.

## Qué se probó

| Cambio frente al control | Evidencia | Accuracy/AUC OOF | Decisión actual |
|---|---|---:|---|
| Máximo 150 épocas | `f7ada452` | 54.80 / 57.75 | Inferior; conservar 300 épocas. |
| Paso 24 s, 80 % de solapamiento, 150 épocas | `8e831252` | 52.77 / 52.64 | Inferior al control de 150 épocas; no repetir. |
| `ordered_scaled`, 150 épocas | `6645839e` | 53.90 / 55.33 | No mejoró frente a `ordered` con 150 épocas. |
| Batch 16, 300 épocas | `a9781609` | 56.50 / 59.31 | Resultado mixto, esencialmente empatado y más costoso; no es prioridad. |
| `l2_reg=0.01` | `ee291ab8` | 53.22 / 56.46 | Inferior; mantener 0.05. |
| `inter_dropout=0.3` | `ed0eed10` | 55.71 / 58.58 | Sin mejora consistente; mantener 0.6. |
| Monitor `val_bce`, sin warm-up | `7714baf4` | 54.01 / 55.56 | Inferior; mantener `val_loss`. |
| `val_bce`, warm-up 150, paciencia 25 | `80329a25` | 56.27 / 58.40 | No supera la referencia. |
| `val_bce`, warm-up 150, paciencia 75 | `410c2892` | 56.72 / 58.82 | No supera la referencia y aumenta el costo. |
| Batch 16 y máximo 500 épocas | `4c133d74` | 55.03 / 57.76 | No mejoró al batch 16/300; no repetir. |

También se evaluaron previamente Fisher z, conectividad `shrunk`, `static` y `mean`. No se
observó una ventaja consistente sobre Pearson `ordered`. No deben repetirse exactamente
las mismas configuraciones salvo que exista una hipótesis nueva o sea necesario reconstruir
una comparación formal con la versión vigente del código.

Interpretación prudente: estas decisiones se refieren a NYU, 12 ROIs y el protocolo actual.
"No priorizar" no significa que una técnica sea inútil en otros sitios, atlas o modelos.

## Análisis de errores ya realizado

En la repetición limpia del control, doce sujetos quedaron mal clasificados en sus cinco
predicciones OOF:

- Controles: `NYU-10004`, `NYU-10093`, `NYU-10110`, `NYU-3518345`,
  `NYU-3650634`, `NYU-4562206`.
- TDAH: `NYU-10050`, `NYU-10107`, `NYU-10118`, `NYU-10129`,
  `NYU-3174224`, `NYU-3653737`.

Son candidatos para inspeccionar movimiento, fenotipo, calidad de señal y posibles casos
atípicos. **No deben eliminarse por haber sido difíciles de clasificar**: hacerlo después de
ver las predicciones sesgaría la estimación de desempeño.

## Qué conviene probar ahora

Prioridad sugerida, siempre cambiando un solo factor respecto a la referencia:

1. **Ventana gaussiana:** misma ventana 120 s/paso 12 s, con
   `WINDOW_SHAPE="gaussian"` y `GAUSSIAN_SIGMA=None`. Es la prueba pendiente más directa
   para enriquecer la estimación dinámica sin cambiar datos, arquitectura ni validación.
2. **Ventana algo más larga:** `WINDOW_SECONDS=140`, `STEP_SECONDS=14`,
   `WINDOW_TR=STEP_TR=None` y forma rectangular. Solo si la gaussiana no ayuda; compara
   estabilidad de la correlación frente al número de ventanas.
3. **Capacidad de BrainNetCNN, barrido mínimo:** antes de correr, recuperar los
   `config.json` de las primeras corridas sin etiquetas claras. Probar únicamente anchos no
   cubiertos, uno por vez; por ejemplo una corrida con `e2e=8` y otra, separada, con
   `e2n=16`, manteniendo todos los demás valores de la referencia. Evitar una cuadrícula
   grande.
4. Repetir una configuración nueva únicamente si supera de forma coherente a la referencia;
   la réplica debe conservar el mismo `config_hash` y usar otro `TAG`.

No son experimentos de optimización: `DETERMINISTIC`, `MIXED_PRECISION` y el tipo de
máquina. Son controles de ejecución. Tampoco se debe buscar una semilla favorable, activar
`class_weight` en NYU sin una hipótesis específica, elegir el mejor fold ni ajustar decisiones
con `outer_val`.

## Regla para aceptar una mejora

- Misma semilla, particiones, sujetos y versión de datos que la referencia.
- Un solo cambio metodológico por comparación.
- `preflight`, prueba de humo y validación final sin fallos.
- Comparar principalmente AUC, balanced accuracy y F1-macro OOF por repetición.
- Usar la comparación pareada/corregida del proyecto; no decidir por el mejor fold ni por
  una diferencia marginal en una sola métrica.
- Confirmar una mejora prometedora con una repetición antes de convertirla en referencia.

## Cómo actualizar este registro

Al terminar una corrida aprobada, añadir una fila con: cambio único, `run_id` o
`config_hash`, métricas OOF principales y decisión (`prometedora`, `no mejora`,
`inconclusa`). Si cambian varios factores, marcarla como **inconclusa/confundida** y no
atribuir el resultado a uno de ellos.
