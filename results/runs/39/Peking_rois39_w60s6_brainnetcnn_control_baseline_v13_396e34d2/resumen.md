# Peking_rois39_w60s6_brainnetcnn_control_baseline_v13_396e34d2

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-29 13:57:16.

## Configuración

- **Sitio**: Peking · **ROIs**: 39 (n=39) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 29 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: sí
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 61.16 ± 11.55 %
- **F1-macro**: 58.86 ± 12.04 %
- **AUC**: 62.79 ± 13.77 %
- **Balanced acc.**: 60.44 ± 11.35 %
- **Brecha train−val (accuracy)**: 30.60 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site Peking --roi-set 39 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --class-weight --tag control_baseline_v13 --verbose
```
