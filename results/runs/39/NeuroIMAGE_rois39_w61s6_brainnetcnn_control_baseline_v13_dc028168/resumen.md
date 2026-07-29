# NeuroIMAGE_rois39_w61s6_brainnetcnn_control_baseline_v13_dc028168

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-29 16:20:37.

## Configuración

- **Sitio**: NeuroIMAGE · **ROIs**: 39 (n=39) · **Sujetos**: 39 (control/TDAH: 22/17)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 61 TR / 119.56 s · paso 6 TR / 11.76 s · solape 90% · 33 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 54.83 ± 27.31 %
- **F1-macro**: 47.95 ± 28.82 %
- **AUC**: 50.00 ± 35.75 %
- **Balanced acc.**: 53.33 ± 27.66 %
- **Brecha train−val (accuracy)**: 24.40 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site NeuroIMAGE --roi-set 39 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag control_baseline_v13 --verbose
```
