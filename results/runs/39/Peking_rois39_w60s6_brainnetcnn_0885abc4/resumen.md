# Peking_rois39_w60s6_brainnetcnn_0885abc4

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-28 20:25:26.

## Configuración

- **Sitio**: Peking · **ROIs**: 39 (n=39) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.8, leaky=0.33, l2_reg=0.15, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 29 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: sí
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 57.66 ± 12.62 %
- **F1-macro**: 55.02 ± 13.10 %
- **AUC**: 61.69 ± 14.04 %
- **Balanced acc.**: 57.42 ± 11.49 %
- **Brecha train−val (accuracy)**: 32.66 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site Peking --roi-set 39 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.8 leaky=0.33 l2_reg=0.15 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --class-weight --verbose
```
