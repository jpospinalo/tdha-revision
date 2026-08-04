# OHSU_rois18_static_logreg_baseline_a59dca47

Baseline de regresión logística (L2, C=1.0, sin búsqueda de hiperparámetro). Conectividad estática, 153 características.

AUC OOF por repetición: 0.6363, 0.6429, 0.5930, 0.6391, 0.6316

AUC OOF media: 0.6286

Guardarraíl verificado contra: OHSU_rois18_w48s5_brainnetcnn_2ce6c48e [representación del comparador: ordered]

**Confusión declarada:** el comparador es la corrida `ordered` (dinámica), no `static`. Este contraste cambia representación y arquitectura a la vez y no aísla el factor arquitectura — mismo aviso que la dimensión «signal representation» del manuscrito (§2.6). Interpretar junto con las corridas de roi_set=12, donde sí existe comparador `static` y el contraste es de un solo factor.

split_fingerprint: a59dca47e72dc24d

Enmienda: docs/PLAN_RESPUESTA_REVISORES.md §9.1, 3 de agosto de 2026.
