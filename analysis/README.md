# `analysis/`

Módulos de análisis estadístico sobre las corridas ya existentes bajo
`results/`. Este directorio no entrena modelos ni genera nuevas corridas: solo
lee artefactos ya producidos por `src/run_experiment.py` y los analiza.

## Contenido

- **`roi_comparison/`** — análisis comparativo del desempeño de BrainNetCNN
  entre los cuatro tamaños de conjunto de ROIs (12, 18, 39, 116) en los cuatro
  sitios (NYU, Peking, NeuroIMAGE, OHSU). Implementa el plan estadístico
  congelado en `roi_comparison/analysis_plan.md` (versión 5.6). Ver
  `roi_comparison/README.md` para instrucciones de uso.

- **`loso/`** — análisis leave-site-out. Campaña `loso_static_v1` (4 sitios
  held-out x 2 ROI sets x {BrainNetCNN x 5 seeds, regresión logística} = 48
  corridas formales, conectividad estática, sin harmonización ni ponderación
  de clase/sitio) implementada y auditada el 2026-08-07. Ver `loso/README.md`
  para el estado y `loso/outputs/LOSO_STATIC_V1_REPORT.md` para los
  resultados. Todavía no forma parte del manuscrito — esa decisión se toma
  aparte, después de la revisión científica de estos resultados.

## Alcance

`results/` es la fuente de datos de este módulo y permanece de solo lectura:
nada bajo `analysis/` modifica corridas, métricas o el `README.md` de
`results/`.
