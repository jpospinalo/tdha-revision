# Integración del enventanado v2

Copiar los archivos conservando la estructura:

- `src/data.py`
- `src/run_experiment.py`
- `tests/test_data.py`
- `tests/test_run_experiment.py`

Antes de ejecutar experimentos:

```bash
pytest -q tests/test_data.py tests/test_run_experiment.py
python src/run_experiment.py --site NYU --roi-set 12 --dry-run
python src/run_experiment.py --site NYU --roi-set 12 \
  --window-seconds 100 --overlap 0.75 --dry-run
python src/run_experiment.py --site NYU --roi-set 12 \
  --representation static --dry-run
```

La ejecución histórica sin argumentos temporales conserva `window=70` y `step=2`.
Los pesos de clase se calculan por pliegue exclusivamente con `fit_idx`.
