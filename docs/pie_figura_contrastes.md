# Pies de figura — contrastes de la campaña de 10 corridas

Las tres figuras comparten la misma guía de estilo y **no llevan título ni notas incrustadas**:
todo el detalle metodológico va en estos pies.

| Archivo | Qué muestra | Uso sugerido |
|---|---|---|
| `estimation_clean.png` | Magnitud e incertidumbre en dos paneles (Gardner–Altman) | Texto principal |
| `forest_styled_v3.png` | Δ AUC con IC, un panel por contraste | Alternativa compacta al anterior |
| `perfiles_styled_v3.png` | AUC por condición con IC y las cinco repeticiones | Suplemento |

**Elementos comunes de estilo:** monocromo (la posición codifica el efecto, no el color); sin marcadores
dicotómicos de significancia, sustituidos por el intervalo; retícula tenue; ejes sin marco;
orden de condiciones fijo entre sitios para permitir la lectura horizontal.

---

## Figura A — `estimation_clean.png`


## Versión completa (suplemento)

> **Figura X. Desempeño de cada condición y su diferencia frente a la configuración de referencia, por sitio.**
> Izquierda: AUC OOF de cada condición (media de las cinco repeticiones), con intervalo de confianza bilateral del 95%.
> Derecha: diferencia pareada por sujeto frente a la configuración de referencia del mismo sitio (Δ = condición − referencia), con su intervalo de confianza bilateral del 95%.
> La referencia es BrainNetCNN sobre la representación multicanal `ordered` con ventana de 120 s y paso de 12 s, 12 ROIs; se indica con la línea punteada del panel izquierdo.
> Las condiciones son: `static` (una única matriz de Pearson sobre toda la serie), LSTM-128 (arquitectura recurrente de la versión original del estudio) y dos alternativas de enventanado (140 s / 12 s y 120 s / 24 s).
> Los contrastes de arquitectura y enventanado se restringieron a NYU y Peking, los sitios con mayor tamaño muestral, por una decisión preespecificada de precisión.
> Todos los intervalos provienen de un bootstrap pareado y estratificado por sujeto (N = 1000 remuestreos), condicionado a las particiones ya existentes.
> Los contrastes son de estimación: no se declara significancia estadística ni se aplica corrección por comparaciones múltiples; el intervalo comunica la precisión disponible.
> Nótese que los intervalos absolutos del panel izquierdo se solapan entre sí más de lo que sugieren las diferencias del panel derecho: el contraste es pareado a nivel de sujeto y por tanto más preciso que la comparación de dos intervalos independientes.
> *n* indica el número de sujetos por sitio. La línea vertical del panel izquierdo en 0.5 corresponde al desempeño esperado por azar.

## Versión breve (texto principal, si el espacio es crítico)

> **Figura X.** AUC OOF por condición (izquierda) y diferencia pareada por sujeto frente a la referencia BrainNetCNN / `ordered` 120 s–12 s, 12 ROIs (derecha), por sitio. Barras: IC bilateral 95% por bootstrap pareado y estratificado por sujeto (N = 1000). Línea punteada: referencia del sitio. Contrastes de estimación, sin declaración de significancia. Los contrastes de arquitectura y enventanado se restringieron a NYU y Peking por precisión, según decisión preespecificada.

---

---

## Figura B — `forest_styled_v3.png`

> **Figura X. Diferencia de AUC de cada condición frente a la referencia, por sitio.**
> Cada panel corresponde a una condición: `static` (una única matriz de Pearson sobre toda la serie), LSTM-128 (arquitectura recurrente de la versión original) y dos alternativas de enventanado (140 s / 12 s y 120 s / 24 s).
> Δ = condición − referencia; valores negativos indican desempeño por debajo de la configuración de referencia (BrainNetCNN, `ordered`, 120 s / 12 s, 12 ROIs).
> Barras: intervalo de confianza bilateral del 95% obtenido por bootstrap pareado y estratificado por sujeto (N = 1000 remuestreos), condicionado a las particiones existentes.
> Las filas mantienen la misma posición en los cuatro paneles para permitir la lectura horizontal por sitio; se rotulan una sola vez, en el panel izquierdo.
> Los contrastes de arquitectura y enventanado se restringieron a NYU y Peking, los sitios con mayor tamaño muestral, por una decisión preespecificada de precisión; las filas correspondientes a NeuroIMAGE y OHSU quedan vacías en esos paneles.
> Contrastes de estimación: no se declara significancia ni se corrige por comparaciones múltiples.

---

## Figura C — `perfiles_styled_v3.png`

> **Figura X. AUC OOF por condición y sitio.**
> Punto: media de las cinco AUC por repetición. Barras: intervalo de confianza bilateral del 95% por bootstrap estratificado por sujeto (N = 1000 remuestreos). Puntos grises: las cinco repeticiones individuales.
> Línea discontinua: desempeño esperado por azar (AUC = 0.5).
> El ancho de cada panel es proporcional al número de condiciones evaluadas en ese sitio; los contrastes de arquitectura y enventanado se restringieron a NYU y Peking por una decisión preespecificada de precisión.
> «referencia» corresponde a BrainNetCNN sobre la representación multicanal `ordered` con ventana de 120 s y paso de 12 s, 12 ROIs.
> Nótese que los intervalos absolutos aquí mostrados son más anchos que las diferencias pareadas de la Figura B: el contraste pareado por sujeto es más preciso que la comparación de dos intervalos independientes.

---

## Elementos retirados de las imágenes y trasladados a estos pies

| Elemento | Motivo |
|---|---|
| Título y subtítulo incrustados | El título va en el pie del manuscrito, no en el archivo |
| «(IC bootstrap bilateral 95%)» repetido en ambos ejes | Se declara una sola vez aquí |
| «Referencia = BrainNetCNN, ordered, 120 s / 12 s, 12 ROIs» | Al pie |
| Valor numérico de la referencia junto a cada sitio | Redundante con la línea punteada; los valores están en la tabla de resultados |
| Rótulos «azar» y «sin diferencia» sobre las líneas guía | El eje ya nombra la cantidad; el pie explica ambas líneas |
| Nota de tres líneas sobre bootstrap y solape de intervalos | Al pie, donde puede leerse con calma |

Quedan en la imagen únicamente: nombre del sitio, *n*, nombre de la condición, los puntos con sus intervalos, las dos líneas guía y los dos rótulos de eje.
