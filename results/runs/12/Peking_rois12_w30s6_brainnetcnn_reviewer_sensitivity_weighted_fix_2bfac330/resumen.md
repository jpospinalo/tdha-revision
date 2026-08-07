# Peking_rois12_w30s6_brainnetcnn_reviewer_sensitivity_weighted_fix_2bfac330

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-07 13:30:17.

## Configuración

- **Sitio**: Peking · **ROIs**: 12 (n=12) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: ordered · **Ventana**: 30 TR / 60.0 s · paso 6 TR / 12.0 s · solape 80% · 34 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: sí
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 56.73 ± 9.84 %
- **F1-macro**: 54.30 ± 10.53 %
- **AUC**: 57.65 ± 12.75 %
- **Balanced acc.**: 55.86 ± 9.98 %
- **Brecha train−val (accuracy)**: 15.19 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site Peking --roi-set 12 --model brainnetcnn --representation ordered --window-seconds 60 --step-seconds 12 --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --class-weight --tag reviewer_sensitivity_weighted_fix
```
