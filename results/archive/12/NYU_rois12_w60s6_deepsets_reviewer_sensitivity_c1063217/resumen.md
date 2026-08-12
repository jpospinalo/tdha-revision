# NYU_rois12_w60s6_deepsets_reviewer_sensitivity_c1063217

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-05 21:08:35.

## Configuración

- **Sitio**: NYU · **ROIs**: 12 (n=12) · **Sujetos**: 177 (control/TDAH: 87/90)
- **Modelo**: `deepsets` — units=64, hidden=64, dropout=0.0, pooling=mean
- **Representación**: ordered · **Ventana**: 60 TR / 120.0 s · paso 6 TR / 12.0 s · solape 90% · 19 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 51.29 ± 10.58 %
- **F1-macro**: 46.58 ± 13.08 %
- **AUC**: 53.02 ± 13.36 %
- **Balanced acc.**: 51.24 ± 10.47 %
- **Brecha train−val (accuracy)**: 10.47 pp
- **Época elegida (mediana)**: 23

## Reproducir

```
run_experiment.py --site NYU --roi-set 12 --model deepsets --representation ordered --window-seconds 120 --step-seconds 12 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag reviewer_sensitivity
```
