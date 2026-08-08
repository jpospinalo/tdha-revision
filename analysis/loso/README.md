# `analysis/loso/` — campaña `loso_static_v1` (implementada, 2026-08-07)

Análisis leave-site-out (LOSO): evaluar la transferencia del clasificador
entrenando en tres sitios y validando en el sitio excluido (rotando los
cuatro), en vez de la validación cruzada interna por sitio usada en
`results/runs/`. Implementado según
`PLAN_FINAL_LOSO_STATIC_V1_IA_REVISADO.md`.

**Estado: 48/48 corridas formales completas y auditadas.** Ver
`outputs/LOSO_STATIC_V1_REPORT.md` para el reporte completo (diseño
congelado, cohorte, entorno, AUC con 95% CI por las 16 condiciones, los 3
contrastes preespecificados x 4 sitios, dispersión entre semillas y estado de
QA).

**Esto NO decide si LOSO entra al cuerpo del paper o como material
complementario.** Esa decisión se toma aparte, después de revisar
científicamente estos resultados — no se ha tocado ningún artefacto del
manuscrito (`README.md`, `docs/paper_reference_configuration.md`,
`docs/paper_environment.md`, `docs/manuscrito_revisado/**`) para esta
campaña.

## Diseño (resumen; el detalle completo está en el plan y en
`results/loso/_design/loso_static_v1_design.json`)

```text
4 sitios held-out (NYU, Peking, NeuroIMAGE, OHSU)
x 2 ROI sets (12, 116)
x {BrainNetCNN (5 seeds: 42-46), regresión logística L2 (determinista)}
= 48 corridas formales

Representación: estática únicamente (fisher_z=False, constant_policy="zero")
Split: un único split fit/inner_val/test por sitio held-out, idéntico entre
       ROI sets, seeds y modelo (StratifiedShuffleSplit test_size=0.15,
       random_state=42, estratificado por sitio x diagnóstico)
Sin harmonización, sin class_weight/site_weighting/sample_weight,
sin ajuste de hiperparámetros
```

Caveat obligatorio: la configuración de BrainNetCNN se desarrolló/fijó
históricamente usando NYU antes de la evaluación multisitio. La rotación con
NYU held-out es una "development-site held-out re-evaluation", no una
evaluación en un sitio totalmente ajeno al desarrollo del modelo.

## Contrato (vigente)

```text
src/run_loso.py, src/run_loso_campaign.py   entrenamiento (aislado de
                                             run_experiment.py, sin
                                             modificarlo)
        ↓
results/loso/<run_id>/                      48 corridas formales, misma
                                             disciplina de config.json/
                                             procedencia/hashes que
                                             results/runs/
        ↓
analysis/loso/                               estadística, manifests y reporte
                                             derivados de esas corridas —
                                             este directorio
```

- `config/loso_analysis_config.json`: especificación de análisis congelada
  ANTES de ver resultados reales (bootstrap 10,000 iteraciones PCG64 seed 42,
  percentil 95% sin ajustar, metric-then-mean para BrainNetCNN).
- `scripts/analyze_loso_static.py`: lee únicamente corridas formales ya
  almacenadas bajo `results/loso/<run_id>/`; no entrena, no selecciona
  hiperparámetros, no ejecuta GPU, no modifica esos resultados.
- `tests/test_loso_static.py`: 30 pruebas (T1-T30, más una de regresión T31)
  contra fixtures sintéticas/toy — nunca contra `results/loso/` real.
- `outputs/`: `loso_manifest.json`, `loso_predictions_long.csv` (5580 filas),
  `loso_metrics_by_run.csv` (48), `loso_metrics_summary.csv` (16),
  `loso_contrasts.csv` (12), `loso_bootstrap_manifest.json`,
  `LOSO_STATIC_V1_REPORT.md`.

### Prohibido dentro de `analysis/loso/`

- entrenar modelos;
- seleccionar hiperparámetros;
- ejecutar GPU;
- modificar `results/loso/`;
- duplicar la lógica del runner.

## Extensiones futuras (fuera de alcance de `loso_static_v1`)

Windowed BrainNetCNN LOSO, DeepSets/LSTM/GRU LOSO, ROI 18/39, harmonización,
domain adaptation, site balancing, sensibilidad a class weighting,
búsqueda de hiperparámetros — cada una requeriría su propio `campaign_id` y
su propio plan, no una extensión silenciosa de este.
