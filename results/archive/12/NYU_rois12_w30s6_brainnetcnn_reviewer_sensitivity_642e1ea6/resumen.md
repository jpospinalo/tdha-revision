# NYU_rois12_w30s6_brainnetcnn_reviewer_sensitivity_642e1ea6

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-05 21:16:17.

## Configuración

- **Sitio**: NYU · **ROIs**: 12 (n=12) · **Sujetos**: 177 (control/TDAH: 87/90)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 30 TR / 60.0 s · paso 6 TR / 12.0 s · solape 80% · 24 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 54.24 ± 12.46 %
- **F1-macro**: 52.68 ± 12.86 %
- **AUC**: 55.85 ± 14.75 %
- **Balanced acc.**: 54.18 ± 12.41 %
- **Brecha train−val (accuracy)**: 12.48 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site NYU --roi-set 12 --model brainnetcnn --representation ordered --window-seconds 60 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag reviewer_sensitivity
```
