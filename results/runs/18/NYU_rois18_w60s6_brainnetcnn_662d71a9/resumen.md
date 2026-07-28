# NYU_rois18_w60s6_brainnetcnn_662d71a9

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-27 22:15:06.

## Configuración

- **Sitio**: NYU · **ROIs**: 18 (n=18) · **Sujetos**: 177 (control/TDAH: 87/90)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 19 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 56.93 ± 11.68 %
- **F1-macro**: 55.44 ± 12.38 %
- **AUC**: 59.20 ± 14.49 %
- **Balanced acc.**: 56.89 ± 11.71 %
- **Brecha train−val (accuracy)**: 11.95 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site NYU --roi-set 18 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --verbose
```
