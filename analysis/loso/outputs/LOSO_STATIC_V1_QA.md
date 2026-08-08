# LOSO_STATIC_V1_QA

Tabla de auditoría A-X (microcierre v31->v32, Secciones 31-38). Los gates A-U se generan y verifican dentro de este mismo proceso, ANTES de cualquier promoción de outputs — si esta tabla existe con A-U en PASS es porque esos gates realmente se ejecutaron y pasaron sobre las 48 corridas formales reales. Los gates V/W/X dependen de pasos externos al proceso (hashes de campaña cruda/repositorio histórico, suite de tests en un entorno con TensorFlow) y quedan PENDING hasta ejecutar `analyze_loso_static.py --finalize-qa` con los logs reales de esos pasos — nunca se declaran PASS sin evidencia.

| Gate | Description | Expected | Observed | Status |
|:---|:---|:---|:---|:---|
| A | campaign_id == loso_static_v1 y formal is True | 48/48 | 48/48 | PASS |
| B | único training_source_git_sha en 48/48 | 1 valor único | 428cbc18f9b7e099d56bed91acd2fbc4f18ee6e8 | PASS |
| C | único environment_signature de entrenamiento en 48/48 | 1 valor único | 0951e380a4901bc2 | PASS |
| D | nombre de directorio == run_id | 48/48 | 48/48 | PASS |
| E | 48 run_id/identity_hash únicos; 40 BrainNetCNN + 8 logistic | 48 únicas, 40/8 | 48 run_id, 48 hash, 40 BNN, 8 logreg | PASS |
| F | design.json y splits.csv: campaign_id/participant_count/1860 filas/rotation_sizes | campaign_id=loso_static_v1, 465 participantes, 1860 filas, 4x465 | coincide | PASS |
| G | rotation_split_fingerprint == design[...][held_out_site] | 48/48 | 48/48 | PASS |
| H | split_membership.csv == bloque design por subject_key (465/465) | 48/48 | 48/48 | PASS |
| I | no leakage: held-out 100% test; fit/inner_val/test disjuntos; unión=465 | 48/48 | 48/48 | PASS |
| J | no weighting/harmonization; static; fisher_z=False; constant_policy=zero | 48/48 | 48/48 | PASS |
| K | roi_set∈{12,116}; model∈{brainnetcnn,logreg}; feature counts; arch BNN congelada | 48/48 | 48/48 | PASS |
| L | predictions_test.csv: schema, [0,1], site==held_out, ambas clases, columnas consistentes | 48/48 | 48/48 | PASS |
| M | AUC/balanced_accuracy/f1_macro/recall/specificity recalculados == metrics_test.csv | 48/48 (tol 1e-09) | 48/48 | PASS |
| N | 40 BNN: history.csv + convergence coherentes; best/restored monitor ~iguales | 40/40 | 40/40 | PASS |
| O | 8 logistic: arch/model_seed null; hyperparams == LOGREG_CONFIG congelado | 8/8 | 8/8 | PASS |
| P | runner/data/model code hash actual[:16] == prefijo histórico en 48/48 | 48/48 | 48/48 | PASS |
| Q | feature_matrix_sha256: prefijos16 == design; NO se recomputan full hashes | 48/48 | 48/48 | PASS |
| R | Prediction completeness: BrainNetCNN=4650, logistic=930, total=5580 | 4650/930/5580 | 4650/930/5580 | PASS |
| S | Metrics-summary completeness: 16 filas | 16 | 16 | PASS |
| T | Contrast completeness: 12 filas | 12 | 12 | PASS |
| U | Scientific regression U1-U5 vs tag pre-closeout (loso-static-v1-complete) y estado v31 auditado (PRE_FIX_HEAD) | 48/48; 16/16; 12/12; 16/16; 8/8 | 48/48; 16/16; 12/12; 16/16; 8/8 | PASS |
| V | Raw LOSO integrity (sha256sum -c sobre resultados/loso/ congelados antes del microcierre) | ALL OK | 378/378 OK | PASS |
| W | Historical repository integrity (sha256sum -c sobre src/data/results-runs/results-archive/roi_comparison/docs/READMEs/requirements) | ALL OK | 554/554 OK | PASS |
| X | Complete LOSO test-suite certification (unittest, entorno con TensorFlow/Keras) | failures=0, errors=0 | tests_run=86, failures=0, errors=0, skipped=0 | PASS |

`loso_provenance_manifest_file_sha256`: `3f33b748acaea7aadb9aaa3590c57dafc9524e28837f0f393f44f931ce6d47de`
