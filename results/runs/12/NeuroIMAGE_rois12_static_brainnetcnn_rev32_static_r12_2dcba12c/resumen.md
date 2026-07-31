# NeuroIMAGE_rois12_static_brainnetcnn_rev32_static_r12_2dcba12c

Vista legible generada desde `config.json` (la fuente de verdad). Timestamp: 2026-07-31 01:03:24.

## Configuración

- **Sitio**: NeuroIMAGE · **ROIs**: 12 (n=12) · **Sujetos**: 39 (control/TDAH: 22/17)
- **Modelo**: `brainnetcnn` — e2e=4, e2n=8, dense=8, dropout=0.7, leaky=0.33, l2_reg=0.05, inter_dropout=0.6
- **Representación**: static · **Ventana**: estática — una matriz sobre toda la serie, sin ventanas
- **Fisher z**: no · **Precisión mixta**: no
- **Validación**: 10×5 = 50 evaluaciones externas · semilla 42 · class_weight: no
- **Entrenamiento**: lr=0.0001, batch=32, epochs=300, patience=25, monitor=val_loss, min_delta=1e-05

## Resultados — validación externa (media ± sd sobre 50 pliegues)

- **Accuracy**: 51.00 ± 18.10 %
- **F1-macro**: 41.25 ± 19.21 %
- **AUC**: 55.33 ± 33.49 %
- **Balanced acc.**: 50.33 ± 18.21 %
- **Brecha train−val (accuracy)**: 1.76 pp
- **Época elegida (mediana)**: 300

## Reproducir

```
run_experiment.py --site NeuroIMAGE --roi-set 12 --model brainnetcnn --representation static --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 --lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 --start-from-epoch 0 --early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 --n-splits 10 --n-repeats 5 --tag rev32_static_r12 --verbose
```
