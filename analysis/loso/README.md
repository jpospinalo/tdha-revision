# `analysis/loso/` (reservado, no implementado)

Este directorio está reservado para un futuro análisis leave-site-out (LOSO):
evaluar la transferencia del clasificador entrenando en unos sitios y
validando en el sitio excluido, en vez de la validación cruzada interna por
sitio usada en `results/runs/`.

**No implica que LOSO se vaya a ejecutar.** La decisión vigente para el
envío actual del manuscrito es no implementar LOSO en ninguna variante (ver
`docs/finalization/limitations_handoff.md` §1 y
`docs/Concepto_LOSO_armonizacion_multisitio.md`, histórico). Este directorio
solo fija el contrato de dónde iría ese trabajo *si* se decide hacerlo en una
fase posterior, para que no se mezcle con el pipeline actual.

No hay código ni resultados aquí todavía. Ninguna parte de
`analysis/roi_comparison/` depende de este directorio ni asume que exista.

## Contrato (si una fase futura implementa LOSO)

```text
src/run_loso.py          entrenamiento (no existe todavía)
        ↓
results/loso/            corridas LOSO (no existe todavía)
        ↓
analysis/loso/            estadística, tablas, figuras y manifests derivados
                          de esas corridas — este directorio
```

- `src/`: entrenamiento. Un nuevo runner (`run_loso.py`), separado de
  `run_experiment.py`, con su propio cargador multisitio y partición
  `LeaveOneGroupOut`/`GroupKFold` (ninguno de los dos existe hoy en el
  pipeline).
- `results/loso/`: corridas LOSO, con la misma disciplina de `config.json`/
  procedencia/hashes que `results/runs/`.
- `analysis/loso/`: solo estadística derivada sobre esas corridas ya
  almacenadas — igual que `analysis/roi_comparison/` respecto de
  `results/runs/`.

### Prohibido dentro de `analysis/loso/`

- entrenar modelos;
- seleccionar hiperparámetros;
- ejecutar GPU;
- modificar resultados;
- duplicar la lógica del runner.

### No crear todavía

```text
src/run_loso.py
results/loso/
analysis/loso/config/
analysis/loso/scripts/
analysis/loso/outputs/
```

Cualquiera de estos requiere una decisión explícita del equipo, no solo la
existencia de este README.
