# Procedencia del fenotípico ADHD-200 usado para la demografía (Gate G3)

**No se versiona el archivo fuente.** Este documento registra su procedencia, hash y las
reglas de decodificación, para que la tabla derivada (`demographics_by_site_dx.csv`) sea
reproducible sin redistribuir el fenotípico bruto.

---

## 1. Fuente

| Campo | Valor |
|---|---|
| Nombre del archivo | `adhd200_preprocessed_phenotypics.tsv` |
| Origen | Consolidado del fenotípico de ADHD-200 (ADHD-200 Consortium, 2011), aportado por el equipo; no distribuido en este repositorio |
| Mecanismo de acceso | Registro y aceptación de los términos de uso de ADHD-200/INDI (International Neuroimaging Data-sharing Initiative) |
| Filas | 973 sujetos, 8 sitios |
| SHA-256 | `7a37195f0260b04246b833ff4b8050afc4756b8a8f1622feca52944189f5a898` |
| Columnas | `ScanDir ID`, `Site`, `Gender`, `Age`, `Handedness`, `DX`, `Secondary Dx`, `ADHD Measure`, `ADHD Index`, `Inattentive`, `Hyper/Impulsive`, `IQ Measure`, `Verbal IQ`, `Performance IQ`, `Full2 IQ`, `Full4 IQ`, `Med Status`, `QC_Athena`, `QC_NIAK` |

## 2. Codificación (verificada contra la clave oficial del consorcio)

Fuente: *ADHD-200 Phenotypic Key*, ADHD-200 Consortium / NITRC ([fcon_1000.projects.nitrc.org](http://fcon_1000.projects.nitrc.org/indi/adhd200/general/ADHD-200_PhenotypicKey.pdf)).

| Variable | Código | Significado |
|---|---|---|
| `Site` | 1 | Peking University |
| | 4 | NeuroIMAGE Sample |
| | 5 | New York University Child Study Center |
| | 6 | Oregon Health & Science University |
| `Gender` | 0 | Female |
| | 1 | Male |
| `DX` | 0 | Typically Developing (control) |
| | 1 | ADHD-Combined |
| | 2 | ADHD-Hyperactive/Impulsive |
| | 3 | ADHD-Inattentive |

**Regla de binarización usada en el manuscrito:** `DX == 0` → *Control*; `DX ∈ {1,2,3}` → *ADHD* (se colapsan los tres subtipos, como en Table 1).

La codificación de `Gender` y `Site` se verificó de dos formas independientes antes de usarla:
1. Contra la clave oficial del consorcio (arriba).
2. Cruzando el `ScanDir ID` de cada sujeto del fenotípico, filtrado por `Site`, contra el `subject_id` (`{SITIO}-{ScanDirID}`) de las predicciones OOF ya almacenadas (`results/runs/12/*_static_logreg_baseline_*/predictions_val.csv`). Los cuatro sitios cruzan exactamente 1:1.

## 3. Resultado del cruce (Gate G3)

| Sitio | n cohorte (OOF) | n cruzado en fenotípico | Coincidencia |
|---|---|---|---|
| NYU | 177 | 177 | 100% |
| Peking | 183 | 183 | 100% |
| NeuroIMAGE | 39 | 39 | 100% |
| OHSU | 66 | 66 | 100% |
| **Total** | **465** | **465** | **100%** |

Ningún sujeto de la cohorte de análisis queda sin `Age`/`Gender`. **G3 = PASS**, verificado sobre el archivo real (no por inferencia).

La ruptura Control/ADHD obtenida por `DX==0` reproduce exactamente los recuentos ya documentados en el manuscrito y en el plan (177=87+90 · 183=109+74 · 39=22+17 · 66=38+28), lo que confirma independientemente que la binarización usada aquí es la misma que la del artículo.

## 4. Selección de los 465

Los 465 sujetos son exactamente los que aparecen en `predictions_val.csv` de las corridas baseline logreg estáticas oficiales por sitio (`results/runs/12/{SITE}_rois12_static_logreg_baseline_*`), que a su vez replican la cohorte de Table 1. No se aplicó ningún filtro adicional sobre el fenotípico: todos los sujetos de la cohorte de análisis tenían registro fenotípico completo.

## 5. No imputación

No hay valores faltantes de `Age` o `Gender` en los 465 sujetos cruzados (`age_missing=0`, `gender_missing=0` en las cuatro combinaciones sitio × diagnóstico). No fue necesario imputar.

## 6. Reproducibilidad

El script `analysis/finalization/build_demographics.py` recibe la ruta local del archivo
fuente (no incluida en el repositorio), verifica su SHA-256 contra el valor registrado
arriba antes de usarlo, y escribe `analysis/finalization/demographics_by_site_dx.csv`.
Si el hash no coincide, el script se detiene sin generar la tabla.
