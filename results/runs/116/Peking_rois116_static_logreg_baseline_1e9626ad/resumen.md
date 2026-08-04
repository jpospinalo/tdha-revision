# Peking_rois116_static_logreg_baseline_1e9626ad

Baseline de regresión logística (L2, C=1.0, sin búsqueda de hiperparámetro). Conectividad estática, 6670 características.

AUC OOF por repetición: 0.5733, 0.6226, 0.5924, 0.6000, 0.5731

AUC OOF media: 0.5923

Guardarraíl verificado contra: Peking_rois116_w60s6_brainnetcnn_240732d1 [representación del comparador: ordered]

**Confusión declarada:** el comparador es la corrida `ordered` (dinámica), no `static`. Este contraste cambia representación y arquitectura a la vez y no aísla el factor arquitectura — mismo aviso que la dimensión «signal representation» del manuscrito (§2.6). Interpretar junto con las corridas de roi_set=12, donde sí existe comparador `static` y el contraste es de un solo factor.

split_fingerprint: 1e9626ad3839ff46

Enmienda: docs/PLAN_RESPUESTA_REVISORES.md §9.1, 3 de agosto de 2026.
