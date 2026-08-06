# NYU_rois12_w30s6_gru_reviewer_sensitivity_945b2b57

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-05 21:39:30.

## Configuración

- **Sitio**: NYU · **ROIs**: 12 (n=12) · **Sujetos**: 177 (control/TDAH: 87/90)
- **Modelo**: `gru` — units=151, dropout=0.0, bidirectional=False
- **Representación**: ordered · **Ventana**: 30 TR / 60.0 s · paso 6 TR / 12.0 s · solape 80% · 24 ventanas · rectangular
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 50.18 ± 10.05 %
- **F1-macro**: 47.41 ± 10.78 %
- **AUC**: 51.38 ± 14.72 %
- **Balanced acc.**: 49.93 ± 10.12 %
- **Brecha train−val (accuracy)**: 6.00 pp
- **Época elegida (mediana)**: 2

## Reproducir

```
run_experiment.py --site NYU --roi-set 12 --model gru --representation ordered --window-seconds 60 --step-seconds 12 --model-arg units=151 dropout=0.0 bidirectional=False --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag reviewer_sensitivity
```
