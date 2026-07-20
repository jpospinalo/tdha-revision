"""
Carga de señales BOLD y construcción de secuencias de conectividad dinámica.

Idea central
------------
El repositorio versiona **solo las señales BOLD** (unos 41 MB para los cuatro
sitios). Los tensores de conectividad —que ocupan más de 460 MB— se derivan aquí
cada vez que hacen falta.

Esto tiene tres consecuencias prácticas:

1. El enventanado deja de estar horneado en un archivo binario y pasa a ser un
   parámetro explícito. El análisis de sensibilidad a la ventana y al paso, que
   piden los revisores, se hace cambiando un número.
2. Es imposible que un tensor quede desincronizado de los parámetros que lo
   generaron. En el repositorio anterior, ``X39.joblib`` tenía 26 ventanas porque
   se había generado con paso 4, pero el nombre del archivo decía otra cosa y el
   artículo lo comparó con grupos de 52 ventanas.
3. El repositorio cabe sin Git LFS y se clona en segundos.

Formato de las señales
----------------------
``data/bold/{sitio}.joblib`` contiene un diccionario con:

    subjects   lista de identificadores, longitud n
    bold       (n, 116, T) float32 — serie temporal media por ROI del atlas AAL116
    labels     (n,) int — 0 control, 1 TDAH
    roi_names  lista de 116 nombres, en el orden del eje 1 de ``bold``

``T`` varía por sitio: NYU 172, Peking 232, NeuroIMAGE 257, OHSU 74.
"""

import json
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BOLD_DIR = REPO_ROOT / "data" / "bold"
ATLAS_DIR = REPO_ROOT / "data" / "atlas"

SITES = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]


# --------------------------------------------------------------------------- #

def load_bold(site, bold_dir=None):
    """Carga las señales BOLD de un sitio.

    Returns
    -------
    dict
        Con las claves ``subjects``, ``bold``, ``labels`` y ``roi_names``.
    """
    path = Path(bold_dir or BOLD_DIR) / f"{site}.joblib"
    if not path.exists():
        disponibles = sorted(p.stem for p in Path(bold_dir or BOLD_DIR).glob("*.joblib"))
        raise SystemExit(
            f"\nERROR: no existen señales BOLD para el sitio '{site}' en {path.parent}\n"
            f"  Disponibles: {disponibles or '(ninguno)'}\n"
        )
    d = joblib.load(path)
    d["bold"] = np.asarray(d["bold"], dtype="float32")
    d["labels"] = np.asarray(d["labels"], dtype="int32")
    return d


def load_roi_sets(atlas_dir=None):
    """Subconjuntos de ROIs definidos en ``data/atlas/roi_sets.json``.

    Returns
    -------
    dict
        Nombre del subconjunto -> ``{"indices": [...], "names": [...], ...}``.
        Los índices son base 0 sobre el eje de ROIs del atlas AAL116.
    """
    path = Path(atlas_dir or ATLAS_DIR) / "roi_sets.json"
    return json.loads(path.read_text(encoding="utf-8"))


def roi_indices(roi_set, atlas_dir=None):
    """Índices del subconjunto pedido, validados contra el atlas."""
    sets = load_roi_sets(atlas_dir)
    key = str(roi_set)
    if key not in sets:
        raise SystemExit(
            f"\nERROR: subconjunto de ROIs '{roi_set}' desconocido.\n"
            f"  Disponibles: {', '.join(sorted(sets, key=lambda s: int(s)))}\n"
        )
    return np.asarray(sets[key]["indices"], dtype=int)


# --------------------------------------------------------------------------- #

def n_windows(n_timepoints, window, step):
    """Número de ventanas que caben en una serie de longitud dada."""
    if window > n_timepoints:
        raise SystemExit(
            f"\nERROR: la ventana de {window} TR no cabe en series de "
            f"{n_timepoints} TR.\n"
            f"  OHSU tiene 74 TR por sujeto: con ventana 70 solo caben 3 ventanas, "
            f"y con ventanas mayores, ninguna.\n"
        )
    return (n_timepoints - window) // step + 1


def build_sequences(bold, indices, window, step):
    """Secuencia de matrices de conectividad por sujeto.

    Para cada ventana temporal se calcula la matriz de correlación de Pearson
    entre las señales de los ROIs seleccionados.

    Parameters
    ----------
    bold : ndarray (n, n_rois_atlas, T)
        Señales BOLD.
    indices : array de int
        ROIs a conservar, en índices base 0 sobre el atlas.
    window, step : int
        Longitud de la ventana y desplazamiento entre ventanas, en TR.

    Returns
    -------
    ndarray (n, n_windows, r, r) float32

    Notes
    -----
    Implementación vectorizada: se estandariza cada ventana y se obtiene la
    correlación como ``Z @ Z.T / (window - 1)``. Es equivalente a llamar a
    ``np.corrcoef`` ventana por ventana, y bastante más rápido para 116 ROIs.
    """
    sig = np.asarray(bold, dtype="float64")[:, indices, :]
    n, r, T = sig.shape
    nw = n_windows(T, window, step)

    # (n, nw, r, window) mediante vistas, sin copiar la señal.
    idx = np.arange(window)[None, :] + (np.arange(nw) * step)[:, None]
    win = sig[:, :, idx]                       # (n, r, nw, window)
    win = np.transpose(win, (0, 2, 1, 3))      # (n, nw, r, window)

    win = win - win.mean(axis=-1, keepdims=True)
    sd = win.std(axis=-1, ddof=1, keepdims=True)
    # ROIs constantes dentro de una ventana: evita dividir por cero. Su correlación
    # queda indefinida y se fija en 0, que es lo que hace np.corrcoef salvo por el NaN.
    constante = sd < 1e-12
    win = np.divide(win, np.where(constante, 1.0, sd))
    win[np.broadcast_to(constante, win.shape)] = 0.0

    fc = win @ np.swapaxes(win, -1, -2) / (window - 1)
    np.clip(fc, -1.0, 1.0, out=fc)
    return fc.astype("float32")


def upper_triangle(x):
    """(n, w, r, r) -> (n, w, r*(r-1)/2), triángulo superior sin diagonal.

    La matriz de correlación es simétrica, así que es numéricamente equivalente
    al triángulo inferior que describe el manuscrito.
    """
    r = x.shape[-1]
    iu = np.triu_indices(r, k=1)
    return x[:, :, iu[0], iu[1]]
