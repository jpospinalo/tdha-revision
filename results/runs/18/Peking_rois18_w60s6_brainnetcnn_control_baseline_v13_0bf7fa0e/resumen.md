# Peking_rois18_w60s6_brainnetcnn_control_baseline_v13_0bf7fa0e

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-29 13:45:18.

## Configuración

- **Sitio**: Peking · **ROIs**: 18 (n=18) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 29 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: sí
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 55.36 ± 12.23 %
- **F1-macro**: 52.62 ± 13.17 %
- **AUC**: 56.18 ± 14.46 %
- **Balanced acc.**: 54.04 ± 12.73 %
- **Brecha train−val (accuracy)**: 20.39 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site Peking --roi-set 18 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --class-weight --tag control_baseline_v13 --verbose
```
