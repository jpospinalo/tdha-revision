# Peking_rois12_w60s6_gru_reviewer_sensitivity_60c51708

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-05 20:47:44.

## Configuración

- **Sitio**: Peking · **ROIs**: 12 (n=12) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `gru` — units=151, dropout=0.0, bidirectional=False
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 29 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 58.16 ± 6.57 %
- **F1-macro**: 45.02 ± 9.90 %
- **AUC**: 55.02 ± 12.83 %
- **Balanced acc.**: 51.38 ± 6.56 %
- **Brecha train−val (accuracy)**: 9.11 pp
- **Época elegida (mediana)**: 30

## Reproducir

```
run_experiment.py --site Peking --roi-set 12 --model gru --representation ordered --window-seconds 120 --step-seconds 12 --model-arg units=151 dropout=0.0 bidirectional=False --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag reviewer_sensitivity
```
