# OHSU_rois39_static_logreg_baseline_a59dca47

Baseline de regresión logística (L2, C=1.0, sin búsqueda de hiperparámetro). Conectividad estática, 741 características.

AUC OOF por repetición: 0.4436, 0.3271, 0.3675, 0.3224, 0.3543

AUC OOF media: 0.3630

Guardarraíl verificado contra: OHSU_rois39_w48s5_brainnetcnn_control_baseline_v13_299719fe [representación del comparador: ordered]

**Confusión declarada:** el comparador es la corrida `ordered` (dinámica), no `static`. Este contraste cambia representación y arquitectura a la vez y no aísla el factor arquitectura — mismo aviso que la dimensión «signal representation» del manuscrito (§2.6). Interpretar junto con las corridas de roi_set=12, donde sí existe comparador `static` y el contraste es de un solo factor.

split_fingerprint: a59dca47e72dc24d

Enmienda: docs/PLAN_RESPUESTA_REVISORES.md §9.1, 3 de agosto de 2026.
