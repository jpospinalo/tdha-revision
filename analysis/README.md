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

- **`loso/`** — reservado para un futuro análisis leave-site-out. No está
  implementado todavía; ver `loso/README.md`.

## Alcance

`results/` es la fuente de datos de este módulo y permanece de solo lectura:
nada bajo `analysis/` modifica corridas, métricas o el `README.md` de
`results/`.
