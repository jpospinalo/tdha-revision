# Peking_rois12_w60s12_brainnetcnn_rev32_window120_step24_f2be42f3

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-31 01:58:51.

## Configuración

- **Sitio**: Peking · **ROIs**: 12 (n=12) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 12 TR / 24.0 s · solape 80% · 15 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 60.00 ± 11.24 %
- **F1-macro**: 52.73 ± 13.20 %
- **AUC**: 56.54 ± 14.83 %
- **Balanced acc.**: 55.61 ± 11.07 %
- **Brecha train−val (accuracy)**: 6.53 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site Peking --roi-set 12 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 24 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag rev32_window120_step24 --verbose
```
