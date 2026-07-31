# NYU_rois12_w60s6_lstm_rev32_lstm128_ordered_0209fc13

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-31 00:50:54.

## Configuración

- **Sitio**: NYU · **ROIs**: 12 (n=12) · **Sujetos**: 177 (control/TDAH: 87/90)
- **Modelo**: `lstm` — units=128, dropout=0.0, bidirectional=False
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 19 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 52.37 ± 9.46 %
- **F1-macro**: 49.16 ± 10.82 %
- **AUC**: 53.54 ± 12.24 %
- **Balanced acc.**: 51.96 ± 9.42 %
- **Brecha train−val (accuracy)**: 7.23 pp
- **Época elegida (mediana)**: 12

## Reproducir

```
run_experiment.py --site NYU --roi-set 12 --model lstm --representation ordered --window-seconds 120 --step-seconds 12 --model-arg units=128 dropout=0.0 bidirectional=False --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag rev32_lstm128_ordered --verbose
```
