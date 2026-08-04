# NeuroIMAGE_rois18_static_logreg_baseline_bc5c92b2

Baseline de regresión logística (L2, C=1.0, sin búsqueda de hiperparámetro). Conectividad estática, 153 características.

AUC OOF por repetición: 0.5401, 0.5428, 0.5481, 0.5989, 0.5936

AUC OOF media: 0.5647

Guardarraíl verificado contra: NeuroIMAGE_rois18_w61s6_brainnetcnn_control_baseline_v13_93342cf0 [representación del comparador: ordered]

**Confusión declarada:** el comparador es la corrida `ordered` (dinámica), no `static`. Este contraste cambia representación y arquitectura a la vez y no aísla el factor arquitectura — mismo aviso que la dimensión «signal representation» del manuscrito (§2.6). Interpretar junto con las corridas de roi_set=12, donde sí existe comparador `static` y el contraste es de un solo factor.

split_fingerprint: bc5c92b29b429f9d

Enmienda: docs/PLAN_RESPUESTA_REVISORES.md §9.1, 3 de agosto de 2026.
