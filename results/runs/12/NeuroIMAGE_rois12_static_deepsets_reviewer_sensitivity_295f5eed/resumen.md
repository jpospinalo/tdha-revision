# NeuroIMAGE_rois12_static_deepsets_reviewer_sensitivity_295f5eed

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-05 21:01:36.

## Configuración

- **Sitio**: NeuroIMAGE · **ROIs**: 12 (n=12) · **Sujetos**: 39 (control/TDAH: 22/17)
- **Modelo**: `deepsets` — units=64, hidden=64, dropout=0.0, pooling=mean
- **Representación**: static · **Ventana**: estática — una matriz sobre toda la serie, sin ventanas
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 50.33 ± 20.27 %
- **F1-macro**: 38.91 ± 20.98 %
- **AUC**: 47.17 ± 36.60 %
- **Balanced acc.**: 50.00 ± 19.20 %
- **Brecha train−val (accuracy)**: 7.53 pp
- **Época elegida (mediana)**: 1

## Reproducir

```
run_experiment.py --site NeuroIMAGE --roi-set 12 --model deepsets --representation static --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag reviewer_sensitivity
```
