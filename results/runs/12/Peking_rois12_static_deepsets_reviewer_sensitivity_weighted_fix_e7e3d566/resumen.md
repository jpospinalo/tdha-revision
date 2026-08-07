# Peking_rois12_static_deepsets_reviewer_sensitivity_weighted_fix_e7e3d566

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-07 13:23:28.

## Configuración

- **Sitio**: Peking · **ROIs**: 12 (n=12) · **Sujetos**: 183 (control/TDAH: 109/74)
- **Modelo**: `deepsets` — units=64, hidden=64, dropout=0.0, pooling=mean
- **Representación**: static · **Ventana**: estática — una matriz sobre toda la serie, sin ventanas
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: sí
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 53.92 ± 11.39 %
- **F1-macro**: 47.82 ± 12.91 %
- **AUC**: 55.03 ± 13.17 %
- **Balanced acc.**: 51.98 ± 10.03 %
- **Brecha train−val (accuracy)**: 12.86 pp
- **Época elegida (mediana)**: 70

## Reproducir

```
run_experiment.py --site Peking --roi-set 12 --model deepsets --representation static --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --class-weight --tag reviewer_sensitivity_weighted_fix
```
