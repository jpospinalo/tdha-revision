# NYU_rois12_static_lstm_reviewer_sensitivity_6686b406

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-08-05 20:33:08.

## Configuración

- **Sitio**: NYU · **ROIs**: 12 (n=12) · **Sujetos**: 177 (control/TDAH: 87/90)
- **Modelo**: `lstm` — units=128, dropout=0.0, bidirectional=False
- **Representación**: static · **Ventana**: estática — una matriz sobre toda la serie, sin ventanas
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 52.30 ± 7.69 %
- **F1-macro**: 45.52 ± 11.44 %
- **AUC**: 55.60 ± 12.57 %
- **Balanced acc.**: 51.86 ± 7.79 %
- **Brecha train−val (accuracy)**: 5.42 pp
- **Época elegida (mediana)**: 42

## Reproducir

```
run_experiment.py --site NYU --roi-set 12 --model lstm --representation static --model-arg units=128 dropout=0.0 bidirectional=False --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag reviewer_sensitivity
```
