# Fase 3 — Corrección de terminología ROI (Methods/Results)

**Gate G1 cerrado como G1-A** (confirmación directa del equipo, 2026-08-07): los paneles de ROI se seleccionaron antes de los experimentos, con base en las redes funcionales que el manuscrito ya describe en §2.2 (DMN, ECN, SN, DAN, FST). Ver `docs/finalization/f1_gates.md` §1.1.

## Cambios aplicados

Solo terminología. Ningún valor numérico, tabla o figura fue tocado — verificado (108 números decimales en el texto antes y después del cambio, idéntico conteo).

| Párrafo | Antes | Después |
|---|---|---|
| 19 (Methods, definición de paneles) | "...four fixed panels (116, 39, 18, and 12 ROIs) **derived by progressive ablation during earlier NYU development** were evaluated." | "...four fixed panels (116, 39, 18, and 12 ROIs) **were prespecified, informed by prior literature and expert neuroanatomical judgment regarding the functional networks implicated in ADHD (Figure 1)**." |
| 26 (Methods, Experimental Design) | "The ROI panels **and** reference BrainNetCNN configuration were selected during NYU development..." (una sola proposición, paneles e hiperparámetros fusionados) | Dividido en dos proposiciones independientes: los paneles fueron prespecificados por literatura y no se seleccionaron por desempeño; la configuración de BrainNetCNN sí se desarrolló en NYU. |
| 45 (Results, encabezado §3.1) | "Performance of the **Ablation-Derived** ROI Panels" | "Performance of the **Prespecified** ROI Panels" |
| 46 (Results, intro Table 4) | "...evaluated with the fixed, **ablation-derived** panels." | "...evaluated with the fixed, **prespecified** panels." |
| 55 (Results, §3.2) | "Because NYU was the development site for the **panel and training** configuration..." (misma fusión que en el párrafo 26) | Dividido: NYU es sitio de desarrollo del **entrenamiento del modelo**; los paneles se prespecificaron por literatura y no se desarrollaron con el desempeño de NYU. |

## Por qué se dividió la Fase 1.2 en dos proposiciones

El plan advertía explícitamente: *"No inferir que paneles a priori implica hiperparámetros a priori"* — y, simétricamente, tampoco al revés. Los párrafos 26 y 55 originales fusionaban ambas afirmaciones en una sola oración ("the panel and training configuration... selected during NYU development"), lo cual ya no es correcto una vez que G1 se resuelve como G1-A: los paneles no se seleccionaron en NYU, solo la configuración de BrainNetCNN. Separarlas evita reintroducir la misma ambigüedad que motivó el gate.

## Verificación de alcance

Términos prohibidos por el plan (§3.1), buscados en el documento completo tras el cambio: `ablation`, `discovered`, `discriminative weight`, `identified biomarker` — **ninguno presente**.

## Verificación estructural

- 85 párrafos (sin cambio; ninguna edición añadió o quitó párrafos).
- 5 tablas, 4 imágenes intactas.
- 108 números decimales en el cuerpo del texto, idéntico antes y después.
- Copia previa al cambio: `docs/finalization/f3_terminologia/PREVIO_antes_de_F3.docx`.

**Checkpoint F3: re-certificado.** La auditoría numérica (Table 4 16/16, Table 5 32/32, ya certificada en PASS) no se ve afectada porque ningún valor cambió; solo terminología descriptiva sobre la procedencia de los paneles.
