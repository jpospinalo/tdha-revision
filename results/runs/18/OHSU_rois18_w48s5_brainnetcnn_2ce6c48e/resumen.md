# OHSU_rois18_w48s5_brainnetcnn_2ce6c48e

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-28 20:16:15.

## Configuración

- **Sitio**: OHSU · **ROIs**: 18 (n=18) · **Sujetos**: 66 (control/TDAH: 38/28)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 48 TR / 120.0 s · paso 5 TR / 12.5 s · solape 90% · 6 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 52.24 ± 16.62 %
- **F1-macro**: 47.52 ± 17.92 %
- **AUC**: 54.25 ± 23.91 %
- **Balanced acc.**: 51.00 ± 16.59 %
- **Brecha train−val (accuracy)**: 14.35 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site OHSU --roi-set 18 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --verbose
```
