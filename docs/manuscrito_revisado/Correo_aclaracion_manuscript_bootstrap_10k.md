Asunto: Aclaración necesaria — discrepancia interna en manuscript_bootstrap_10k.csv (n y AUC no coinciden con los hashes de predicciones citados en el mismo archivo)

Hola equipo,

Estamos cerrando la revisión editorial de Manuscript_Methods_Results_English_Working_v9_9.docx y necesito su ayuda para resolver una inconsistencia antes de tocar el documento.

Recibimos un archivo `manuscript_bootstrap_10k.csv` con n_subjects = 166 (NYU), 165 (Peking) y 35 (NeuroIMAGE), y valores de AUC de referencia distintos a los que tiene actualmente el manuscrito (63.6%, 63.6% y 52.0%, respectivamente, para el panel de 12 ROIs).

Al revisar ese archivo encontramos que cada fila incluye la columna `ref_predictions_sha256`, que identifica el archivo exacto de predicciones crudas usado para calcular el AUC de referencia. Verificamos esos mismos hashes contra los archivos de predicciones almacenados en el pipeline (`results/runs/12/.../predictions_val.csv`) y recalculamos el AUC directamente desde ahí. Resultado:

| Sitio | Hash de predicciones citado | Sujetos únicos reales en ese archivo | AUC recalculado desde ese archivo | n y AUC que aparecen en la fila del CSV recibido |
|---|---|---|---|---|
| NYU | b4af0f10... | 177 | 59.05% | 166 / 63.60% |
| Peking | a82502af... | 183 | 56.37% | 165 / 63.64% |
| NeuroIMAGE | 72058547... | 39 | 47.38% | 35 / 52.04% |

En los tres casos, el hash que el propio archivo cita como su fuente corresponde matemáticamente a 177/183/39 sujetos y a los AUC que ya tiene el manuscrito actual — no a los valores que aparecen en esa misma fila del CSV recibido. Esto no parece ser una versión alternativa válida del análisis, sino una inconsistencia interna del archivo: probablemente el script que lo generó dejó los campos de `n_subjects` y AUC de una versión anterior sin recalcular al actualizar las columnas de hash.

¿Podrían revisar el paso del pipeline que produce `manuscript_bootstrap_10k.csv` y confirmar cuál de los dos conjuntos de valores es el correcto? Si 177/183/39/66 sigue siendo la muestra final válida, no necesitamos ningún cambio. Si en realidad hay una muestra distinta (166/165/35) con su propio cálculo, necesitaríamos un archivo regenerado donde los hashes de predicciones correspondan efectivamente a esa muestra.

Mientras tanto, dejamos el documento oficial sin modificar, con los valores actuales (n=177/183/39/66), que son los que sí están respaldados por los datos verificables del pipeline.

Gracias,
Juan
