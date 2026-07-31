# Peking_rois12_w60s6_lstm_rev32_lstm128_ordered_2ead6445

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-31 00:05:55.

## Configuración

- **Sitio**: Peking · **ROIs**: 12 (n=12) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `lstm` — units=128, dropout=0.0, bidirectional=False
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 29 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 59.25 ± 7.72 %
- **F1-macro**: 47.62 ± 11.07 %
- **AUC**: 54.99 ± 14.36 %
- **Balanced acc.**: 52.96 ± 7.62 %
- **Brecha train−val (accuracy)**: 8.90 pp
- **Época elegida (mediana)**: 22

## Reproducir

```
run_experiment.py --site Peking --roi-set 12 --model lstm --representation ordered --window-seconds 120 --step-seconds 12 --model-arg units=128 dropout=0.0 bidirectional=False --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag rev32_lstm128_ordered --verbose
```
