# Peking_rois12_w60s6_deepsets_reviewer_sensitivity_weighted_fix_fbe99635

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-07 13:26:49.

## Configuración

- **Sitio**: Peking · **ROIs**: 12 (n=12) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `deepsets` — units=64, hidden=64, dropout=0.0, pooling=mean
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 29 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: sí
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 55.63 ± 9.63 %
- **F1-macro**: 49.12 ± 12.62 %
- **AUC**: 54.48 ± 13.75 %
- **Balanced acc.**: 53.87 ± 8.81 %
- **Brecha train−val (accuracy)**: 10.58 pp
- **Época elegida (mediana)**: 65

## Reproducir

```
run_experiment.py --site Peking --roi-set 12 --model deepsets --representation ordered --window-seconds 120 --step-seconds 12 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --class-weight --tag reviewer_sensitivity_weighted_fix
```
