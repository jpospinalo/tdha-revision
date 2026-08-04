# NYU_rois18_static_logreg_baseline_aafd45ca

Baseline de regresión logística (L2, C=1.0, sin búsqueda de hiperparámetro). Conectividad estática, 153 características.

AUC OOF por repetición: 0.5604, 0.5681, 0.5692, 0.5845, 0.5222

AUC OOF media: 0.5609

Guardarraíl verificado contra: NYU_rois18_w60s6_brainnetcnn_control_baseline_v13_662d71a9 [representación del comparador: ordered]

**Confusión declarada:** el comparador es la corrida `ordered` (dinámica), no `static`. Este contraste cambia representación y arquitectura a la vez y no aísla el factor arquitectura — mismo aviso que la dimensión «signal representation» del manuscrito (§2.6). Interpretar junto con las corridas de roi_set=12, donde sí existe comparador `static` y el contraste es de un solo factor.

split_fingerprint: aafd45ca73662139

Enmienda: docs/PLAN_RESPUESTA_REVISORES.md §9.1, 3 de agosto de 2026.
