# `analysis/loso/` (reservado, no implementado todavía)

Este directorio está reservado para el análisis leave-site-out (LOSO):
evaluar la transferencia del clasificador entrenando en unos sitios y
validando en el sitio excluido, en vez de la validación cruzada interna por
sitio usada en `results/runs/`.

**Decisión del equipo (2026-08-07): LOSO se implementará.** Queda por decidir
si entra al cuerpo del paper o como material complementario del envío —
ninguna de las dos cosas está resuelta todavía. Hasta que esa decisión de
alcance se cierre y arranque la implementación, este directorio sigue vacío:
fija el contrato de dónde va ese trabajo, no ejecuta nada por sí mismo. Ver
`docs/finalization/limitations_handoff.md` §1 (marcado parcialmente
superseded por esta decisión) y `docs/Concepto_LOSO_armonizacion_multisitio.md`
(histórico, no es la especificación de esta implementación).

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

La decisión de implementar LOSO ya está tomada, pero eso no autoriza por sí
solo a crear estos archivos: falta resolver el alcance (paper vs.
suplemento) y el diseño concreto (cargador multisitio, partición
`LeaveOneGroupOut`/`GroupKFold`, tratamiento de BrainNetCNN windowed —ver la
limitación de capacidad variable entre sitios en
`docs/finalization/limitations_handoff.md` §2— antes de escribir código.
