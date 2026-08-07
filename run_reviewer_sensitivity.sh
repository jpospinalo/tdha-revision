#!/usr/bin/env bash
# Experimentos de sensibilidad adicionales, en respuesta a comentarios de revisores.
# Protocolo completo (n_repeats=5, n_splits=10, seed=42) -- igual que el resto del estudio.
#
# Notas de diseño (verificadas con Keras antes de correr, no a ojo):
#   - GRU usa units=151 (no 128) para igualar aproximadamente el conteo de parámetros de
#     LSTM(128 unidades): LSTM=99,969 parámetros, GRU(151)=99,359 (<1% de diferencia).
#     Con units=128 GRU quedaba en 75,393 (75% de LSTM), lo que habría hecho la comparación
#     de arquitectura injusta desde el inicio.
#   - LSTM-estático responde literalmente el comentario "SFC+LSTM vs dFC+LSTM", pero un
#     modelo recurrente alimentado con una sola ventana no ejercita ninguna compuerta
#     temporal -- es una comparación limitada por diseño, no por parámetros. Por eso se
#     complementa con DeepSets-estático vs DeepSets-windowed: misma familia de arquitectura,
#     bien definida en ambos regímenes (con una sola ventana, DeepSets se reduce exactamente
#     a un MLP, sin nada forzado), y es la prueba arquitectónicamente más limpia de si la
#     conectividad dinámica aporta algo.
#
# Uso: correr desde la raíz del repositorio, con el venv activado:
#   cd /ruta/al/repo
#   source .venv/bin/activate
#   bash run_reviewer_sensitivity.sh
#
# Cada bloque tarda unos pocos minutos en una máquina normal (sin GPU). Total estimado: ~35-70 min.
# Los resultados quedan en results/runs/12/<sitio>_..._<tag>_<hash>/, con la misma estructura
# (config.json, metrics_train.csv, metrics_val.csv, predictions_val.csv, resumen.md) que las
# corridas existentes, y son directamente comparables porque usan el mismo split_fingerprint.
#
# ADVERTENCIA (corrección class_weight de Peking, 2026-08-07): las 14 corridas de
# este script ya existen en results/runs/12/ (8 sin afectar + 6 Peking corregidas
# por separado bajo el tag reviewer_sensitivity_weighted_fix). NO ejecutar este
# script completo de nuevo -- volvería a entrenar las 8 condiciones no afectadas
# sin necesidad. Para reproducir solo la corrección, correr únicamente los seis
# comandos Peking marcados "(CORREGIDO: ...)" abajo, uno por uno. No agregar
# --overwrite.

set -euo pipefail

COMMON="--lr 0.0001 --batch-size 32 --epochs 300 --patience 25 --inner-val-frac 0.15 \
--early-stopping-monitor val_loss --early-stopping-min-delta 1e-05 --seed 42 \
--n-splits 10 --n-repeats 5"

TAG="reviewer_sensitivity"

# Peking uses class_weight=True by the prespecified site policy (see Gate G2,
# docs/finalization/f1_gates.md). The six Peking calls below originally
# omitted --class-weight, in violation of that policy. They are corrected
# here and tagged separately so old and corrected runs never collide under
# the same run_id pattern.
# The earlier Peking reviewer_sensitivity runs without weighting are retained
# for provenance but are superseded and must not be used in canonical
# manuscript sensitivity analyses.
TAG_PEKING_FIX="reviewer_sensitivity_weighted_fix"

echo "=== [1/14] LSTM estático, NYU  (comentario: SFC+LSTM vs dFC+LSTM) ==="
python3 src/run_experiment.py --site NYU --roi-set 12 --model lstm --representation static \
  --model-arg units=128 dropout=0.0 bidirectional=False $COMMON --tag $TAG

echo "=== [2/14] LSTM estático, Peking (CORREGIDO: class_weight por política del sitio) ==="
python3 src/run_experiment.py --site Peking --roi-set 12 --model lstm --representation static \
  --model-arg units=128 dropout=0.0 bidirectional=False $COMMON --class-weight --tag "$TAG_PEKING_FIX"

echo "=== [3/14] GRU ventaneado 120/12, NYU  (comentario: por qué LSTM, probar GRU; units=151 para igualar parámetros con LSTM) ==="
python3 src/run_experiment.py --site NYU --roi-set 12 --model gru --representation ordered \
  --window-seconds 120 --step-seconds 12 --model-arg units=151 dropout=0.0 bidirectional=False \
  $COMMON --tag $TAG

echo "=== [4/14] GRU ventaneado 120/12, Peking (CORREGIDO: class_weight por política del sitio) ==="
python3 src/run_experiment.py --site Peking --roi-set 12 --model gru --representation ordered \
  --window-seconds 120 --step-seconds 12 --model-arg units=151 dropout=0.0 bidirectional=False \
  $COMMON --class-weight --tag "$TAG_PEKING_FIX"

echo "=== [5/14] DeepSets estático (baseline denso), NYU ==="
python3 src/run_experiment.py --site NYU --roi-set 12 --model deepsets --representation static \
  $COMMON --tag $TAG

echo "=== [6/14] DeepSets estático, Peking (CORREGIDO: class_weight por política del sitio) ==="
python3 src/run_experiment.py --site Peking --roi-set 12 --model deepsets --representation static \
  $COMMON --class-weight --tag "$TAG_PEKING_FIX"

echo "=== [7/14] DeepSets estático, NeuroIMAGE ==="
python3 src/run_experiment.py --site NeuroIMAGE --roi-set 12 --model deepsets --representation static \
  $COMMON --tag $TAG

echo "=== [8/14] DeepSets estático, OHSU ==="
python3 src/run_experiment.py --site OHSU --roi-set 12 --model deepsets --representation static \
  $COMMON --tag $TAG

echo "=== [9/14] DeepSets ventaneado 120/12, NYU  (par limpio con [5], para SFC vs dFC dentro de la misma arquitectura) ==="
python3 src/run_experiment.py --site NYU --roi-set 12 --model deepsets --representation ordered \
  --window-seconds 120 --step-seconds 12 $COMMON --tag $TAG

echo "=== [10/14] DeepSets ventaneado 120/12, Peking (CORREGIDO: class_weight por política del sitio) ==="
python3 src/run_experiment.py --site Peking --roi-set 12 --model deepsets --representation ordered \
  --window-seconds 120 --step-seconds 12 $COMMON --class-weight --tag "$TAG_PEKING_FIX"

echo "=== [11/14] BrainNetCNN ventana 60/12, NYU  (comentario: ablación de tamaño de ventana) ==="
python3 src/run_experiment.py --site NYU --roi-set 12 --model brainnetcnn --representation ordered \
  --window-seconds 60 --step-seconds 12 \
  --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 \
  $COMMON --tag $TAG

echo "=== [12/14] BrainNetCNN ventana 60/12, Peking (CORREGIDO: class_weight por política del sitio) ==="
python3 src/run_experiment.py --site Peking --roi-set 12 --model brainnetcnn --representation ordered \
  --window-seconds 60 --step-seconds 12 \
  --model-arg e2e=4 e2n=8 dense=8 dropout=0.7 leaky=0.33 l2_reg=0.05 inter_dropout=0.6 \
  $COMMON --class-weight --tag "$TAG_PEKING_FIX"

echo "=== [13/14] GRU ventana 60/12, NYU  (units=151, igual que en [3]/[4]) ==="
python3 src/run_experiment.py --site NYU --roi-set 12 --model gru --representation ordered \
  --window-seconds 60 --step-seconds 12 --model-arg units=151 dropout=0.0 bidirectional=False \
  $COMMON --tag $TAG

echo "=== [14/14] GRU ventana 60/12, Peking (CORREGIDO: class_weight por política del sitio) ==="
python3 src/run_experiment.py --site Peking --roi-set 12 --model gru --representation ordered \
  --window-seconds 60 --step-seconds 12 --model-arg units=151 dropout=0.0 bidirectional=False \
  $COMMON --class-weight --tag "$TAG_PEKING_FIX"

echo
echo "Listo. Resultados en results/runs/12/*_${TAG}_*/"
