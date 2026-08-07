# Peking_rois12_static_lstm_reviewer_sensitivity_weighted_fix_f0640423

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-07 13:13:43.

## Configuración

- **Sitio**: Peking · **ROIs**: 12 (n=12) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `lstm` — units=128, dropout=0.0, bidirectional=False
- **Representación**: static · **Ventana**: estática — una matriz sobre toda la serie, sin ventanas
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: sí
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 53.74 ± 12.07 %
- **F1-macro**: 47.88 ± 13.58 %
- **AUC**: 53.68 ± 14.32 %
- **Balanced acc.**: 52.33 ± 10.87 %
- **Brecha train−val (accuracy)**: 6.68 pp
- **Época elegida (mediana)**: 70

## Reproducir

```
run_experiment.py --site Peking --roi-set 12 --model lstm --representation static --model-arg units=128 dropout=0.0 bidirectional=False --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --class-weight --tag reviewer_sensitivity_weighted_fix
```
