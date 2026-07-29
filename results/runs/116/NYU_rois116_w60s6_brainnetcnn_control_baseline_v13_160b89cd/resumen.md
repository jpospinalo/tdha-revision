# NYU_rois116_w60s6_brainnetcnn_control_baseline_v13_160b89cd

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-29 17:09:44.

## Configuración

- **Sitio**: NYU · **ROIs**: 116 (n=116) · **Sujetos**: 177 (control/TDAH: 87/90)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 19 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 52.99 ± 11.52 %
- **F1-macro**: 51.37 ± 12.16 %
- **AUC**: 53.05 ± 13.93 %
- **Balanced acc.**: 52.93 ± 11.50 %
- **Brecha train−val (accuracy)**: 33.36 pp
- **Época elegida (mediana)**: 296

## Reproducir

```
run_experiment.py --site NYU --roi-set 116 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag control_baseline_v13 --overwrite --verbose
```
