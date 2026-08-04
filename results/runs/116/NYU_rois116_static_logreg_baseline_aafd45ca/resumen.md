# NYU_rois116_static_logreg_baseline_aafd45ca

Baseline de regresión logística (L2, C=1.0, sin búsqueda de hiperparámetro). Conectividad estática, 6670 características.

AUC OOF por repetición: 0.5791, 0.5649, 0.5478, 0.5411, 0.5831

AUC OOF media: 0.5632

Guardarraíl verificado contra: NYU_rois116_w60s6_brainnetcnn_control_baseline_v13_160b89cd [representación del comparador: ordered]

**Confusión declarada:** el comparador es la corrida `ordered` (dinámica), no `static`. Este contraste cambia representación y arquitectura a la vez y no aísla el factor arquitectura — mismo aviso que la dimensión «signal representation» del manuscrito (§2.6). Interpretar junto con las corridas de roi_set=12, donde sí existe comparador `static` y el contraste es de un solo factor.

split_fingerprint: aafd45ca73662139

Enmienda: docs/PLAN_RESPUESTA_REVISORES.md §9.1, 3 de agosto de 2026.
