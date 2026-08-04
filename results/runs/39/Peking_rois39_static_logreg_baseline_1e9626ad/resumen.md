# Peking_rois39_static_logreg_baseline_1e9626ad

Baseline de regresión logística (L2, C=1.0, sin búsqueda de hiperparámetro). Conectividad estática, 741 características.

AUC OOF por repetición: 0.6265, 0.6318, 0.6552, 0.6080, 0.5974

AUC OOF media: 0.6238

Guardarraíl verificado contra: Peking_rois39_w60s6_brainnetcnn_control_baseline_v13_396e34d2 [representación del comparador: ordered]

**Confusión declarada:** el comparador es la corrida `ordered` (dinámica), no `static`. Este contraste cambia representación y arquitectura a la vez y no aísla el factor arquitectura — mismo aviso que la dimensión «signal representation» del manuscrito (§2.6). Interpretar junto con las corridas de roi_set=12, donde sí existe comparador `static` y el contraste es de un solo factor.

split_fingerprint: 1e9626ad3839ff46

Enmienda: docs/PLAN_RESPUESTA_REVISORES.md §9.1, 3 de agosto de 2026.
