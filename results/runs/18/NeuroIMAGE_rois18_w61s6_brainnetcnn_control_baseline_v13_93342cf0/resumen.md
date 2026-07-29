# NeuroIMAGE_rois18_w61s6_brainnetcnn_control_baseline_v13_93342cf0

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-28 20:04:03.

## Configuración

- **Sitio**: NeuroIMAGE · **ROIs**: 18 (n=18) · **Sujetos**: 39 (control/TDAH: 22/17)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 61 TR / 119.56 s · paso 6 TR / 11.76 s · solape 90% · 33 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 57.83 ± 23.23 %
- **F1-macro**: 51.12 ± 25.91 %
- **AUC**: 65.83 ± 30.03 %
- **Balanced acc.**: 58.17 ± 23.90 %
- **Brecha train−val (accuracy)**: 20.22 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site NeuroIMAGE --roi-set 18 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag control_baseline_v13 --verbose
```
