# LOSO_STATIC_V1_QA

Tabla de auditoría de cierre (`fix/loso-static-v1-analysis-closeout`). Generada únicamente después de que los Gates A-Q (Sección 27, CP2 PASS) pasaron sobre las 48 corridas formales reales. No reentrena, no cambia splits, no cambia AUC/CI/contrastes primarios.

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

Adicionalmente (verificado en CP14-CP20, fuera de esta tabla de gates por-corrida): X test suite (35 tests históricos + tests de auditoría nuevos), V raw LOSO hash protection, W historical repo hash protection — ver `git diff`/`sha256sum -c` registrados en el commit de cierre.

`loso_provenance_manifest_file_sha256`: `115567964e1c2bdc4e0eb07872ab45d444ca98e14996c3509d83d08f526b29ff`
