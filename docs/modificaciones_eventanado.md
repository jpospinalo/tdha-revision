# Integración del enventanado v2

Comprobaciones recomendadas antes de ejecutar experimentos:

```bash
python src/verify_setup.py
python src/run_experiment.py --site NYU --roi-set 12 --dry-run
python src/run_experiment.py --site NYU --roi-set 12 \
  --window-seconds 100 --overlap 0.75 --dry-run
python src/run_experiment.py --site NYU --roi-set 12 \
  --representation static --dry-run
```

La ejecución histórica sin argumentos temporales conserva `window=70` y `step=2`.
El enventanado físico se especifica en segundos con `--window-seconds` y
`--step-seconds`/`--overlap`. Los pesos de clase se calculan por pliegue
exclusivamente con `fit_idx`.
