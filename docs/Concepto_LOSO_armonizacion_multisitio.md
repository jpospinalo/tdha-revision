# Concepto: LOSO vs. armonización + pooling multi-sitio para ampliar la comparación

Fecha: 2026-08-06

## Pregunta

El equipo evalúa dos experimentos para ampliar el rango de comparación del artículo y responder al comentario de los jurados sobre generalización entre sitios (ya se pasó de 1 sitio, NYU, a 4: NYU, Peking, NeuroIMAGE, OHSU): (1) leave-one-site-out (LOSO) y (2) combinar los cuatro sitios en una sola muestra con armonización estadística de efectos de sitio. La idea inicial es incluir ambos en el manuscrito.

## Recomendación

Sí a ambos, pero no son intercambiables ni deben presentarse como dos chequeos redundantes de lo mismo — responden preguntas distintas y conviene secuenciarlos:

1. **LOSO primero.** Es la prueba más directa y más exigida por revisores de generalización entre sitios, no requiere resolver el problema de armonización, y reutiliza la infraestructura ya existente (solo hay que ensamblar entrenamiento sobre 3 sitios y prueba sobre el cuarto, rotando).
2. **Armonización + pooling después**, una vez resuelto el ajuste por longitud/TR que el equipo ya identificó como pendiente. Hacerlo antes arriesga enmascarar un problema estructural real (ventanas de OHSU) como si fuera solo un efecto de sitio corregible estadísticamente.

Encajan bien como **dos análisis complementarios con etiquetas distintas** en el manuscrito: LOSO responde "¿generaliza el modelo a un sitio nunca visto?"; el pooling armonizado responde "¿aporta combinar sitios más poder estadístico, corrigiendo el sesgo de sitio?". Presentarlos así es más defendible ante revisores que como "doble verificación de robustez".

## Por qué LOSO es apropiado y suficiente para el comentario del jurado

- LOSO es el protocolo estándar para evaluar generalización cross-sitio en fMRI multi-sitio: se entrena con todos los sitios menos uno y se prueba en el excluido, rotando hasta que cada sitio sirve una vez de prueba. Es metodológicamente necesario aquí porque un k-fold aleatorio filtraría información de sitio entre folds (el modelo "vería" la firma de cada sitio durante el entrenamiento).
- Es una práctica ya usada específicamente sobre ADHD-200 multi-sitio, incluyendo trabajos que reportan exactamente AUC/accuracy bajo LOSO como métrica principal de generalización.

## Por qué la armonización + pooling necesita más cuidado en este dataset específico

Tres riesgos concretos, no genéricos:

1. **Fuga de información (leakage) si se ajusta con la muestra completa.** La literatura reciente documenta explícitamente que ComBat (y variantes) puede producir mejoras de rendimiento infladas por fuga cuando se ajusta con toda la muestra antes de partir en folds, y que el problema se agrava con desbalance de clases entre sitios — que es exactamente la situación aquí (Peking usa `class_weight`, los demás sitios no, según ya está documentado en el manuscrito). La corrección estándar es ajustar los parámetros de armonización **solo con los datos de entrenamiento de cada fold**, nunca con la muestra combinada completa.
2. **Tamaño y heterogeneidad por sitio.** La estabilidad de la estimación de armonización decrece cuando el tamaño de muestra por sitio es chico y cuando un sitio está muy alejado de la distribución de los demás. NeuroIMAGE (n=39) y sobre todo OHSU son los eslabones más frágiles: no es que la armonización sea inválida con esos tamaños, pero sí que hay que reportar esa limitación explícitamente, no asumir que corrige el sitio de igual calidad en los cuatro.
3. **El problema de longitud de escaneo no lo resuelve la armonización.** OHSU tiene 74 volúmenes (TR=2.50 s) contra 172–257 en los otros tres sitios, lo que con la ventana de 120 s/paso 12 s da solo 6 ventanas en OHSU frente a 19–33 en el resto (ya documentado en la Tabla 1 del manuscrito). ComBat corrige la media/varianza de las características de conectividad ya calculadas, pero no corrige que esas 6 ventanas de OHSU se estimaron sobre una serie temporal mucho más corta y más ruidosa. Armonizar sobre esa asimetría estructural sin resolverla antes arriesga producir una comparabilidad artificial, no real. Esto coincide con lo que el equipo ya identificó ("ajustes por longitud... se considerará después") — mi sugerencia es tratarlo como prerrequisito de la armonización, no como un detalle posterior.

## Nota práctica de arquitectura

Para cualquier experimento que combine sitios (LOSO o pooling), los modelos secuenciales (LSTM/GRU) asumen implícitamente una longitud de ventana fija, lo que choca directamente con la asimetría de OHSU. El modelo DeepSets que ya está implementado y validado esta sesión (pooling invariante al orden y al número de ventanas) es candidato natural para estos experimentos cross-sitio precisamente porque no asume un número fijo de ventanas por sujeto — evita tener que truncar o rellenar artificialmente las series de OHSU para igualarlas a las de los demás sitios.

## En resumen para el equipo

- Adelante con LOSO ahora: responde directamente al comentario del jurado, bajo riesgo metodológico, reutiliza infraestructura existente.
- Posponer la armonización + pooling hasta resolver el ajuste por longitud de ventana; cuando se haga, ajustar los parámetros de armonización solo dentro de cada fold de entrenamiento (nunca con la muestra completa) y reportar explícitamente la fragilidad en NeuroIMAGE/OHSU.
- Presentar ambos en el manuscrito como respuestas a preguntas distintas (generalización vs. poder combinado), no como una duplicación de robustez.

## Fuentes

- [Leave-one-site-out validation en fMRI multi-sitio para diagnóstico — Frontiers in Medicine, 2026](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1839975/full)
- [Separated Channel Attention CNN para ADHD en dataset multi-sitio rs-fMRI (LOSO) — PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7517519/)
- [Yu et al. (2018), "Statistical harmonization corrects site effects in functional connectivity measurements from multi-site fMRI data" — Human Brain Mapping](https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.24241) (ya citado en el manuscrito actual)
- [Riemannian geometry of functional connectivity matrices for multi-site ADHD data harmonization — PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9171428/) (aplica armonización + LOSO conjuntamente sobre ADHD-200)
- [Impact of Leakage on Data Harmonization in Machine Learning Pipelines in Class Imbalance Across Sites — arXiv](https://arxiv.org/html/2410.19643)
- [Efficacy of MRI data harmonization in the age of machine learning: a multicenter study across 36 datasets — Scientific Data / Nature](https://www.nature.com/articles/s41597-023-02421-7)
- [Sample size requirement for achieving multisite harmonization using structural brain MRI features — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1053811922008898)
