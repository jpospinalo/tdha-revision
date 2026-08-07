# Peking_rois12_w30s6_gru_reviewer_sensitivity_weighted_fix_2e23b0b9

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-07 13:41:04.

## Configuración

- **Sitio**: Peking · **ROIs**: 12 (n=12) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `gru` — units=151, dropout=0.0, bidirectional=False
- **Representación**: ordered · **Ventana**: 30 TR / 60.0 s · paso 6 TR / 12.0 s · solape 80% · 34 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: sí
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 52.26 ± 10.07 %
- **F1-macro**: 49.11 ± 10.58 %
- **AUC**: 50.28 ± 12.48 %
- **Balanced acc.**: 50.99 ± 9.84 %
- **Brecha train−val (accuracy)**: 10.84 pp
- **Época elegida (mediana)**: 8

## Reproducir

```
run_experiment.py --site Peking --roi-set 12 --model gru --representation ordered --window-seconds 60 --step-seconds 12 --model-arg units=151 dropout=0.0 bidirectional=False --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --class-weight --tag reviewer_sensitivity_weighted_fix
```
