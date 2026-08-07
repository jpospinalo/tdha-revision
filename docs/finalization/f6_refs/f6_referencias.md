# Fase 6 — Verificación de referencias

**Alcance:** las tres citas marcadas por el propio manuscrito como pendientes de verificación bibliográfica (Hale et al. 2014, Reimann et al. 2024, Singh et al. 2024), más una auditoría de correspondencia cita↔referencia sobre el documento completo.

---

## 1. Contexto de las tres citas

Las tres aparecen juntas en Methods §2.2 (párrafo sobre la asignación de los paneles reducidos a sistemas funcionales):

> *"...five functional systems implicated in ADHD: [DMN, ECN, SN, DAN, FST] (Damiani et al., 2021; Francx et al., 2015; **Hale et al., 2014; Singh et al., 2024**). These systems support self-referential, executive, salience, attentional, and motor processes relevant to ADHD (Blomberg et al., 2022; Koirala et al., 2024; Parlatini et al., 2023; **Reimann et al., 2024**; Sutcubasi et al., 2020)."*

## 2. Resultado de la verificación

| Cita | Resultado | Evidencia |
|---|---|---|
| **Hale et al. (2014)** | **Verificada.** Existe, y respalda la afirmación. | Hale, T. S., Kane, A. M., Kaminsky, O., Tung, K. L., Wiley, J. F., McGough, J. J., Loo, S. K., & Kaplan, J. T. (2014). *Visual network asymmetry and default mode network function in ADHD: An fMRI study.* Frontiers in Psychiatry, 5, Article 81. https://doi.org/10.3389/fpsyt.2014.00081 — estudio fMRI sobre función de la red por defecto (DMN) en TDAH; respalda directamente la afirmación sobre sistemas funcionales implicados en TDAH. |
| **Reimann et al. (2024)** | **No verificable.** | Búsqueda con múltiples formulaciones (autor + año + TDAH + redes atencional/ejecutiva/saliencia/frontostriatal-talámica, con y sin restricción a TDAH) no localizó ninguna publicación real que coincida. |
| **Singh et al. (2024)** | **No verificable.** | Misma búsqueda exhaustiva; ningún resultado corresponde a una publicación real con ese autor, año y contenido. |

El propio manuscrito ya marcaba las tres como *"pending bibliographic verification... not resolved as of this revision"* — la verificación confirma que dos de ellas, en efecto, no superan la verificación.

## 3. Acción aplicada (conforme a la regla del plan: *"si no puede verificarse, retirar la cita o sustituirla por una fuente verificable"*)

No se sustituyeron por fuentes alternativas, porque no fue necesario: cada una acompañaba a otras 3–4 citas que ya respaldan la misma afirmación de forma independiente.

1. **Hale et al. (2014):** se reemplazó el marcador `[Reference pending...]` en References por la referencia completa y verificada.
2. **Reimann et al. (2024):** se retiró la cita del cuerpo (Methods §2.2) y se eliminó su entrada de References. La afirmación que acompañaba queda respaldada por Blomberg (2022), Koirala (2024), Parlatini (2023) y Sutcubasi (2020).
3. **Singh et al. (2024):** se retiró la cita del cuerpo y se eliminó su entrada de References. La afirmación queda respaldada por Damiani (2021), Francx (2015) y Hale (2014).

Ningún valor numérico, tabla o figura fue tocado. El cambio se limita a un párrafo de Methods §2.2 y a tres entradas de References.

## 4. Auditoría de correspondencia cita ↔ referencia (documento completo, post-edición)

- **12 referencias** en la lista final.
- Toda cita en el cuerpo del texto aparece en References: verificado, incluidas `ADHD-200 Consortium (2011)` y `Leonardi & Van De Ville (2015)` (ambas confirmadas presentes en el cuerpo pese a que la detección automática por regex las marcó inicialmente como posibles huérfanas, por el guion y el "&"; se verificaron manualmente).
- Toda referencia de la lista aparece citada en el cuerpo: verificado, 0 huérfanas.
- Sin duplicados.
- Sin referencias añadidas para rellenar: la lista se redujo de 14 a 12 entradas; no se añadió ninguna referencia nueva no citada.

## 5. Estado del archivo

- `docs/manuscrito_revisado/Manuscript_Methods_Results_English_Working_v9_9.docx`: 85 párrafos (antes 87; se eliminaron los dos párrafos de referencia retirados), 5 tablas intactas, 4 imágenes intactas — estructura verificada tras el cambio.
- Copia de respaldo previa al cambio: `docs/finalization/f6_refs/PREVIO_antes_de_F6.docx`.

**Checkpoint F6: cerrado.** No quedan citas sin resolver en el documento.
