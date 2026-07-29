# OHSU_rois12_w48s5_brainnetcnn_control_baseline_v13_1a7c37ce

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-28 16:41:07.

## Configuración

- **Sitio**: OHSU · **ROIs**: 12 (n=12) · **Sujetos**: 66 (control/TDAH: 38/28)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 48 TR / 120.0 s · paso 5 TR / 12.5 s · solape 90% · 6 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 54.67 ± 18.08 %
- **F1-macro**: 48.21 ± 19.91 %
- **AUC**: 55.06 ± 23.81 %
- **Balanced acc.**: 52.75 ± 17.89 %
- **Brecha train−val (accuracy)**: 6.56 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site OHSU --roi-set 12 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag control_baseline_v13 --verbose
```
