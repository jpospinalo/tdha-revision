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

_bold_cache = {}
_seq_cache = {}


def load_bold(site, bold_dir=None):
    """Carga las señales BOLD de un sitio.

    El resultado queda en memoria: al encolar varias corridas sobre el mismo
    sitio en un solo proceso, solo se lee de disco la primera vez.

    Returns
    -------
    dict
        Con las claves ``subjects``, ``bold``, ``labels`` y ``roi_names``.
    """
    clave = (str(bold_dir or BOLD_DIR), site)
    if clave in _bold_cache:
        return _bold_cache[clave]
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
    _bold_cache[clave] = d
    return d


def build_sequences_cached(site, bold, indices, window, step, roi_set=None):
    """``build_flat_sequences`` con memoria de la última construcción.

    Devuelve las secuencias ya vectorizadas, listas para el modelo.

    Pensado para colas de experimentos: correr cuatro arquitecturas sobre el
    mismo sitio, subconjunto de ROIs y enventanado construye las secuencias una
    sola vez en lugar de cuatro. Con 116 ROIs eso ahorra unos 8 s por corrida.

    La caché guarda **una sola** entrada: el tensor de 116 ROIs ocupa cerca de
    500 MB y conservar varios agotaría la memoria de Colab. Por eso conviene
    ordenar la cola agrupando por configuración de datos.
    """
    clave = (site, str(roi_set), int(window), int(step), len(indices))
    if _seq_cache.get("clave") == clave:
        return _seq_cache["valor"]
    # Liberar ANTES de construir: si no, el tensor viejo y el nuevo coexisten y el
    # pico de memoria se duplica.
    _seq_cache.clear()
    valor = build_flat_sequences(bold, indices, window, step)
    _seq_cache.update(clave=clave, valor=valor)
    return valor


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


def build_sequences(bold, indices, window, step, chunk=32):
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
    chunk : int
        Sujetos procesados por lote. Acota la memoria intermedia sin afectar al
        resultado. Con 116 ROIs, hacerlo de una sola vez exigiría un arreglo
        intermedio de unos 600 MB.

    Returns
    -------
    ndarray (n, n_windows, r, r) float32

    Notes
    -----
    Implementación vectorizada: se estandariza cada ventana y se obtiene la
    correlación como ``Z @ Z.T / (window - 1)``. Es equivalente a llamar a
    ``np.corrcoef`` ventana por ventana y bastante más rápido.

    Se calcula en float32. Frente a float64, la diferencia contra los tensores
    del proyecto original pasa de 8e-7 a 1e-6 —ambas por debajo de la precisión
    de float32, que es el tipo en que se almacenan— y el cálculo es unas cuatro
    veces más rápido con 116 ROIs.
    """
    sig_all = np.asarray(bold, dtype="float32")[:, indices, :]
    n, r, T = sig_all.shape
    nw = n_windows(T, window, step)

    # Índices de ventana: (nw, window). Genera las ventanas por indexado.
    widx = np.arange(window)[None, :] + (np.arange(nw) * step)[:, None]
    out = np.empty((n, nw, r, r), dtype="float32")

    for s in range(0, n, chunk):
        w = np.transpose(sig_all[s:s + chunk][:, :, widx], (0, 2, 1, 3))
        w = w - w.mean(axis=-1, keepdims=True)
        sd = w.std(axis=-1, ddof=1, keepdims=True)
        # ROIs constantes dentro de una ventana: evita dividir por cero. Su
        # correlación queda indefinida y se fija en 0, que es lo que hace
        # np.corrcoef salvo por el NaN.
        constante = sd < 1e-12
        w = np.divide(w, np.where(constante, np.float32(1.0), sd))
        w[np.broadcast_to(constante, w.shape)] = 0.0
        fc = w @ np.swapaxes(w, -1, -2) / np.float32(window - 1)
        np.clip(fc, -1.0, 1.0, out=fc)
        out[s:s + chunk] = fc

    return out


def _chunk_for(n_rois):
    """Sujetos por lote. Bloques menores con muchos ROIs: el arreglo intermedio
    crece con el cuadrado del número de regiones y con 116 ROIs medimos que 16
    es algo más rápido que 32 o 64."""
    return 16 if n_rois >= 100 else (32 if n_rois >= 40 else 64)


def build_flat_sequences(bold, indices, window, step, chunk=None):
    """Secuencias de conectividad ya vectorizadas: (n, n_windows, r*(r-1)/2).

    Equivale a ``upper_triangle(build_sequences(...))`` pero extrae el triángulo
    dentro del bucle por lotes, sin materializar nunca el tensor simétrico completo.

    Con 116 ROIs eso importa: la matriz completa ocupa 495 MB de los cuales la mitad
    es redundante por simetría, y mantener a la vez el tensor y su triángulo llevaba
    el pico a unos 770 MB. Así el pico queda en los ~246 MB del resultado.

    Es la forma que consume el modelo, de modo que es la que conviene usar en el
    camino normal; ``build_sequences`` se conserva para inspección y verificación.
    """
    sig_all = np.asarray(bold, dtype="float32")[:, indices, :]
    n, r, T = sig_all.shape
    chunk = chunk or _chunk_for(r)
    nw = n_windows(T, window, step)
    iu = np.triu_indices(r, k=1)

    widx = np.arange(window)[None, :] + (np.arange(nw) * step)[:, None]
    out = np.empty((n, nw, len(iu[0])), dtype="float32")

    for s in range(0, n, chunk):
        w = np.transpose(sig_all[s:s + chunk][:, :, widx], (0, 2, 1, 3))
        w = w - w.mean(axis=-1, keepdims=True)
        sd = w.std(axis=-1, ddof=1, keepdims=True)
        constante = sd < 1e-12
        w = np.divide(w, np.where(constante, np.float32(1.0), sd))
        w[np.broadcast_to(constante, w.shape)] = 0.0
        fc = w @ np.swapaxes(w, -1, -2) / np.float32(window - 1)
        np.clip(fc, -1.0, 1.0, out=fc)
        out[s:s + chunk] = fc[:, :, iu[0], iu[1]]

    return out


def upper_triangle(x):
    """(n, w, r, r) -> (n, w, r*(r-1)/2), triángulo superior sin diagonal.

    La matriz de correlación es simétrica, así que es numéricamente equivalente
    al triángulo inferior que describe el manuscrito.
    """
    r = x.shape[-1]
    iu = np.triu_indices(r, k=1)
    return x[:, :, iu[0], iu[1]]