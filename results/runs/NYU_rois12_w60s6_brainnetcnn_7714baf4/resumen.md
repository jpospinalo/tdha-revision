# NYU_rois12_w60s6_brainnetcnn_7714baf4

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-26 22:29:32.

## Configuración

- **Sitio**: NYU · **ROIs**: 12 (n=12) · **Sujetos**: 177 (control/TDAH: 87/90)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 19 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_bce, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 54.06 ± 9.80 %
- **F1-macro**: 50.46 ± 12.19 %
- **AUC**: 55.99 ± 11.69 %
- **Balanced acc.**: 53.97 ± 9.68 %
- **Brecha train−val (accuracy)**: -0.71 pp
- **Época elegida (mediana)**: 18

## Reproducir

```
run_experiment.py --site NYU --roi-set 12 --model brainnetcnn --representation ordered --window-seconds 120 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_bce --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --verbose
```
