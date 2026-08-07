# Paquete de traspaso — Limitations / Discussion (Fase 4-bis)

**Destinatario:** el coautor que tenga asignadas Discussion, Limitations, Abstract e Introduction. Ninguna de esas secciones existe hoy en `Manuscript_Methods_Results_English_Working_v9_9.docx` (solo contiene Methods, Results y References); este documento no las escribe, entrega el contenido ya acordado para que quien las redacte no tenga que empezar desde cero ni renegociar puntos ya cerrados con el equipo.

**Base:** acuerdos alcanzados en las rondas de discusión LOSO/multisitio y en la revisión del plan de implementación (agosto de 2026), verificados contra el código y los datos del repositorio en el HEAD `4c15d39` y posteriores.

---

## 1. Transporte cross-site

> *The present study evaluates within-site discrimination and the stability of methodological conclusions across heterogeneous acquisition sites. Each model was trained and evaluated within the same site; the study therefore does not estimate transport of a single trained model to an unseen acquisition site.*

Por qué: no se implementó ningún esquema LOSO ni de entrenamiento multisitio (decisión cerrada del plan, §1). Todo lo reportado es rendimiento intra-sitio.

## 2. Capacidad variable del modelo entre sitios

> *Because the number of windows differed across sites, the windowed BrainNetCNN input dimensionality and parameter count also varied across sites. Site-level performance differences therefore reflect complete site-specific pipelines rather than acquisition or sampling differences alone.*

Respaldo verificado (construcción literal del modelo, no inferencia): 3.089 · 4.049 · 4.433 · 1.841 parámetros para NYU · Peking · NeuroIMAGE · OHSU respectivamente, por la capa `UpperTriToMatrix` de BrainNetCNN, que trata el número de ventanas como dimensión de canal de entrada.

## 3. NYU como sitio de desarrollo, no validación independiente

> *NYU served as the development site for the ROI panels and BrainNetCNN configuration. Results at NYU therefore represent an internal re-evaluation rather than independent validation. No site constitutes external validation, because each site is trained and evaluated internally.*

Ya está parcialmente en Results §3.2 (párrafo 55 del documento actual: *"Because NYU was the development site for the panel and training configuration, its favorable 12-ROI estimate represents an internal re-evaluation rather than independent validation."*). Este punto debe mantenerse consistente entre Results y Discussion/Limitations, no introducir una formulación distinta.

## 4. Menor precisión en NeuroIMAGE y OHSU

> *Confidence intervals were widest at NeuroIMAGE (n=39) and OHSU (n=66), the two smallest cohorts. Point estimates at these sites should be interpreted with the corresponding precision, not as more or less favorable conclusions than the wider intervals support.*

Ya reflejado en Results (párrafo 49: nueve de dieciséis intervalos incluyen AUC=0.5, con los cuatro de OHSU y tres de NeuroIMAGE entre ellos). Limitations debe explicitarlo como limitación de tamaño de muestra, no repetir el resultado.

## 5. Bootstrap condicionado a predicciones almacenadas

> *All confidence intervals reported here are computed from stored out-of-fold predictions using participant-level, class-stratified bootstrap resampling (10,000 resamples, seed 42); no model was retrained to produce these intervals. Uncertainty in model training itself (e.g., initialization, optimization stochasticity) is not captured by this procedure.*

Aplica a Table 4, Table 5 y Figure 2-3. Es una limitación metodológica genuina que Results no declara explícitamente (Results describe el método, pero no discute qué fuente de incertidumbre queda fuera).

## 6. `class_weight` específico de Peking

**Resultado del Gate G2: PASS.** La política (`class_weight=True` solo en Peking) está documentada en `docs/guia-experimentacion-colaborativa.md` desde el 2026-07-26, dos días antes de la corrida oficial de Peking (2026-07-28), y se deriva de una característica fija y observable de los datos (desbalance de clases: DX=0 61, DX=3 17, DX=1 7), no del desempeño del modelo. Verificación completa en `docs/finalization/f1_gates.md`.

**Consecuencia para Limitations:** no se ejecutó sensibilidad de `class_weight`, porque el gate pasó. El texto de Limitations debe decir:

> *For Peking, class weighting was applied to address a class imbalance documented prior to model training; this policy was not tuned post hoc. No sensitivity analysis removing this weighting was performed, as the chronology-based prespecification criterion was satisfied.*

No usar lenguaje que sugiera que la política es incierta o que quedó sin resolver: quedó resuelta, con evidencia documental fechada.

## 7. Movimiento residual

**Resultado del Gate G4: métrica disponible solo para NYU** (`NYU_motion.csv`, aportado por el equipo). No existe un archivo equivalente y comparable para Peking, NeuroIMAGE u OHSU en este repositorio. Por la regla del plan (§1.4), no se construye una tabla parcial de movimiento.

> *Motion-related quality metrics were available for NYU but not for Peking, NeuroIMAGE, or OHSU in a directly comparable form. Residual motion confounding cannot be ruled out at the three sites lacking this metric.*

Si el equipo localiza una métrica comparable para los otros tres sitios, este punto debe actualizarse antes de enviar; mientras tanto, se declara como lo que es: información faltante, no evidencia de ausencia de movimiento.

## 8. Atlas AAL116 y paneles fijos; diferencias de TR y duración

> *All analyses used the AAL116 anatomical atlas and four fixed ROI panels (12, 18, 39, 116 regions); no data-driven parcellation or ROI selection was performed within the reported analysis. Site-specific differences in TR and acquisition duration were accommodated through site-specific windowing parameters (see Table 3), which is itself a source of cross-site heterogeneity in the resulting representations.*

## 9. Procedencia de los paneles ROI — depende del cierre de G1

**G1 sigue abierto** (ver `docs/finalization/f1_gates.md`, §1.1). El texto final de Limitations sobre este punto depende de qué salida se active:

- Si **G1-A** (evidencia suficiente de definición a priori): no se necesita lenguaje de limitación adicional más allá del §9 general de no-optimización retroactiva.
- Si **G1-C** (procedencia no reconstruible): Limitations debe incluir explícitamente:

  > *Four fixed ROI panels were evaluated and were not reselected during the multisite evaluation. Because contemporaneous records did not permit complete reconstruction of the historical derivation of the reduced panels, we do not interpret their evaluation at NYU as independent confirmation of ROI selection.*

**No redactar este punto de forma definitiva hasta que G1 se resuelva.** Este documento entrega ambas variantes para que quien escriba Limitations no tenga que esperar a una segunda entrega.

## 10. Lenguaje a evitar en Discussion/Limitations/Conclusions

Por consistencia con Methods/Results, evitar: `external validation`, `generalizes across sites`, `optimal`, `non-inferior`, `equivalent`, `biomarker`, `ablation-derived`/`derived by progressive ablation` (salvo que G1 determine lo contrario), y cualquier formulación equivalente a «más riguroso que la mayoría de los estudios comparables» sin una comparación explícita y acotada.

---

## 11. Matriz claim → evidence, acotada a Results (§3.1–§3.4)

No incluye Discussion/Limitations porque esas secciones no existen en el documento actual. Cubre únicamente las afirmaciones interpretativas ya presentes en Results.

| # | Afirmación (Results) | Ubicación | Respaldo | Intervalo / valor |
|---|---|---|---|---|
| 1 | AUC fuera de pliegue limitado en los 4 sitios y paneles, rango 47.4%–62.2% | §3.1, párr. 48 | Table 4 / `descriptive_performance.csv` | 16 combinaciones, ver tabla |
| 2 | El panel de 12 ROI no tuvo el mayor punto estimado en ningún sitio | §3.1, párr. 48 | Table 4 | — |
| 3 | 9/16 intervalos incluyen AUC=0.5; Peking es el único sitio cuyos 4 intervalos lo excluyen | §3.1, párr. 49 | Table 4, Figure 2 | — |
| 4 | Δ(116−12 ROI): −5.1pp NYU, +4.0 Peking, +4.5 NeuroIMAGE, +0.9 OHSU, los 4 IC incluyen cero | §3.2, párr. 54 | Table 5 / `primary_12_vs_116.csv` | [−11.1,+0.7]·[−2.5,+10.5]·[−10.3,+20.0]·[−8.3,+9.9] |
| 5 | NYU es sitio de desarrollo; su estimación favorable es re-evaluación interna, no validación independiente | §3.2, párr. 55 | — (declaración metodológica) | — |
| 6 | 39 ROI vs 12 ROI en NYU: −6.2pp, IC excluye cero; 18 ROI en NeuroIMAGE: +14.6pp, IC excluye cero | §3.2, párr. 56 | Table 5 | IC excluye cero en ambos casos |
| 7 | Estático vs ventaneado (referencia BrainNetCNN): −4.1 NYU, −3.5 Peking, −3.4 OHSU, +5.4 NeuroIMAGE, los 4 IC incluyen cero | §3.3, párr. 59 | Table 5, Figure 3 / `figure4_v6_audit.csv` | 4 IC incluyen cero |
| 8 | DeepSets estático vs ventaneado: +1.1pp NYU, +0.6pp Peking, ambos IC incluyen cero | §3.3, párr. 59 | Table 5, Figure 3 | [−1.6,+3.9]·[−1.9,+2.9] |
| 9 | LSTM estático vs ventaneado: +1.9pp NYU, −0.5pp Peking, ambos IC incluyen cero | §3.3, párr. 59 | Table 5, Figure 3 | [−1.0,+4.9]·[−3.6,+2.7] |
| 10 | LSTM-128 vs BrainNetCNN: −6.5pp NYU (IC excluye cero), −0.1pp Peking (IC incluye cero) | §3.3, párr. 60 | Table 5, Figure 3 | [−10.9,−2.0]·[−4.7,+4.4] |
| 11 | GRU-151 vs BrainNetCNN: −7.1pp NYU (IC excluye cero), −1.8pp Peking (IC incluye cero); no difiere materialmente de LSTM | §3.3, párr. 60 | Table 5, Figure 3 | [−11.9,−2.3]·[−6.3,+2.8] |
| 12 | Ventana 140s/12s vs 120s/12s: −3.3pp NYU (límite superior cerca de cero); ventana 120s/24s: −4.7pp NYU, IC excluye cero; Peking <1pp en ambos, IC incluye cero | §3.3, párr. 61 | Table 5, Figure 3 | [−6.5,+0.1]·[−8.6,−0.8] |
| 13 | Ventana 60s/12s BrainNetCNN: −4.6pp NYU (IC excluye cero), +1.2pp Peking (IC incluye cero); GRU 60s vs 120s: −1.2pp NYU (IC incluye cero), −3.5pp Peking (IC excluye cero) | §3.3, párr. 61 | Table 5, Figure 3 | [−8.8,−0.6]·[−2.9,+5.3]·[−4.3,+1.8]·[−6.7,−0.2] |
| 14 | Regresión logística vs BrainNetCNN (12 ROI estático): +2.5 NYU, +6.3 Peking, +10.4 NeuroIMAGE, −1.9 OHSU, los 4 IC incluyen cero; excepción en 39 ROI OHSU: −13.8pp, IC excluye cero | §3.3, párr. 62 | Table 5 / `manuscript_bootstrap_10k.csv` | ver tabla; excepción [−23.3,−4.7] |
| 15 | DeepSets vs BrainNetCNN (12 ROI estático): −1.9 NYU, +3.8 Peking, −4.2 NeuroIMAGE, +2.6 OHSU, los 4 IC incluyen cero | §3.3, párr. 62 | Table 5 / `algorithm_comparison_deepsets_audit.csv` | [−6.6,+2.8]·[−0.9,+8.6]·[−16.0,+7.3]·[−5.3,+10.6] |
| 16 | Patrón de convergencia: pérdida de entrenamiento decrece, validación interna decrece más lento, sin repunte tardío claro; en 39/116 ROI la relación se invierte tras ~1/3 del entrenamiento (memorización) | §3.4, párr. 67 | Figure 4 (curvas ROC), `history.csv` por corrida | — (descriptivo, sin IC) |
| 17 | Early stopping raramente se activó en la configuración de referencia (44–50/50 pliegues llegan al techo de épocas); no fue uniforme entre configuraciones (19–28/50 en 116 ROI; LSTM se activa en todos los pliegues) | §3.4, párr. 68 | `metrics_train.csv`/`folds.csv` por corrida | — (descriptivo) |

**Nota de alcance:** esta matriz no incluye afirmaciones de Discussion/Limitations porque esas secciones no existen en el documento actual. Si se agregan en el futuro, deberán construir su propia matriz de trazabilidad siguiendo este mismo formato.
