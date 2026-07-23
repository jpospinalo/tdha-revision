"""
Carga de señales BOLD y construcción reproducible de conectividad funcional.

Esta versión mantiene compatibilidad con la API histórica del repositorio:

- ``load_bold``
- ``load_roi_sets`` / ``roi_indices``
- ``n_windows``
- ``build_sequences``
- ``build_flat_sequences``
- ``build_sequences_cached``
- ``upper_triangle``

La ruta histórica (Pearson, ventana rectangular, sin Fisher z) conserva el
mismo cálculo vectorizado en ``float32``. Además, se incorporan validaciones,
parámetros temporales expresables en segundos, conectividad estática, ventana
gaussiana ponderada, Fisher z opcional, diagnósticos del enventanado y
representaciones para ablaciones temporales.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
BOLD_DIR = REPO_ROOT / "data" / "bold"
ATLAS_DIR = REPO_ROOT / "data" / "atlas"
SITES = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]

# TR documentado para los sitios utilizados en este proyecto.
SITE_TR_SECONDS: dict[str, float] = {
    "NYU": 2.0,
    "Peking": 2.0,
    "NeuroIMAGE": 1.96,
    "OHSU": 2.5,
}

FloatArray = np.ndarray
WindowShape = Literal["rectangular", "gaussian"]
ConstantPolicy = Literal["zero", "raise"]
ConnectivityMode = Literal["dynamic", "static"]
SummaryStatistic = Literal["mean", "std"]

_CONSTANT_EPS = np.float32(1e-12)
_FISHER_LIMIT = np.nextafter(np.float32(1.0), np.float32(0.0))

# Se conserva una sola secuencia para evitar duplicar tensores grandes.
_bold_cache: dict[tuple[str, str], dict[str, Any]] = {}
_seq_cache: dict[str, Any] = {}


@dataclass(frozen=True)
class WindowSpec:
    """Especificación temporal resuelta en TR y segundos.

    ``requested_*`` conserva lo solicitado por el usuario; ``effective_*``
    refleja la discretización finalmente aplicada después del redondeo a TR.
    """

    tr_seconds: float
    window_tr: int
    step_tr: int
    window_seconds: float
    step_seconds: float
    requested_window_seconds: float | None
    requested_step_seconds: float | None
    requested_overlap: float | None
    effective_overlap: float
    shape: WindowShape = "rectangular"
    fisher_z: bool = False
    gaussian_sigma: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clear_caches() -> None:
    """Vacía las cachés de señales y secuencias."""

    _bold_cache.clear()
    _seq_cache.clear()


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} debe ser un entero; se recibió {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} debe ser >= {minimum}; se recibió {value}.")
    return value


def _round_positive_to_int(value: float) -> int:
    """Redondeo convencional para valores positivos (0.5 hacia arriba)."""

    return int(np.floor(float(value) + 0.5))


def _require_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} debe ser numérico; se recibió {value!r}.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} debe ser finito y > 0; se recibió {value!r}.")
    return value


def validate_indices(indices: Iterable[int], n_rois: int) -> np.ndarray:
    """Valida y devuelve índices ROI unidimensionales, únicos y base 0."""

    n_rois = _require_int("n_rois", n_rois, minimum=2)
    idx = np.asarray(indices)
    if idx.ndim != 1:
        raise ValueError(f"indices debe ser un arreglo 1D; forma recibida: {idx.shape}.")
    if idx.size < 2:
        raise ValueError("Se requieren al menos dos ROIs para calcular conectividad.")
    if not np.issubdtype(idx.dtype, np.integer):
        if not np.all(np.isfinite(idx)) or not np.all(idx == np.floor(idx)):
            raise TypeError("Todos los índices ROI deben ser enteros.")
    idx = idx.astype(np.int64, copy=False)
    if np.unique(idx).size != idx.size:
        raise ValueError("indices contiene ROIs duplicados.")
    if idx.min() < 0 or idx.max() >= n_rois:
        raise IndexError(
            f"Índices ROI fuera de rango [0, {n_rois - 1}]: "
            f"mínimo={idx.min()}, máximo={idx.max()}."
        )
    return idx


def validate_bold_array(
    bold: Any,
    *,
    check_finite: bool = True,
    minimum_rois: int = 2,
    minimum_timepoints: int = 2,
) -> np.ndarray:
    """Valida un tensor ``(sujetos, ROIs, tiempo)`` sin cambiar sus valores."""

    arr = np.asarray(bold)
    if arr.ndim != 3:
        raise ValueError(
            "bold debe tener forma (n_sujetos, n_rois, n_timepoints); "
            f"forma recibida: {arr.shape}."
        )
    if arr.shape[0] < 1:
        raise ValueError("bold no contiene sujetos.")
    if arr.shape[1] < minimum_rois:
        raise ValueError(f"bold debe contener al menos {minimum_rois} ROIs.")
    if arr.shape[2] < minimum_timepoints:
        raise ValueError(
            f"bold debe contener al menos {minimum_timepoints} puntos temporales."
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"bold debe ser numérico; dtype recibido: {arr.dtype}.")
    if check_finite and not np.isfinite(arr).all():
        bad = int(arr.size - np.isfinite(arr).sum())
        raise ValueError(f"bold contiene {bad} valores NaN o infinitos.")
    return arr


def validate_bold_payload(payload: Mapping[str, Any], site: str | None = None) -> None:
    """Valida el contrato de ``data/bold/{sitio}.joblib``.

    El formato esperado contiene ``subjects``, ``bold``, ``labels`` y
    ``roi_names``. La función no modifica el diccionario.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("El archivo BOLD debe contener un diccionario.")

    required = {"subjects", "bold", "labels", "roi_names"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Faltan claves obligatorias en el archivo BOLD: {sorted(missing)}.")

    bold = validate_bold_array(payload["bold"], check_finite=True)
    n_subjects, n_rois, _ = bold.shape

    subjects = list(payload["subjects"])
    labels = np.asarray(payload["labels"])
    roi_names = list(payload["roi_names"])

    if len(subjects) != n_subjects:
        raise ValueError(
            f"subjects tiene {len(subjects)} elementos, pero bold tiene {n_subjects} sujetos."
        )
    if len({str(subject) for subject in subjects}) != n_subjects:
        raise ValueError("subjects contiene identificadores duplicados.")
    if labels.ndim != 1 or labels.shape[0] != n_subjects:
        raise ValueError(
            f"labels debe tener forma ({n_subjects},); forma recibida: {labels.shape}."
        )
    if not np.isin(labels, (0, 1)).all():
        values = np.unique(labels).tolist()
        raise ValueError(f"labels solo puede contener 0 y 1; valores encontrados: {values}.")
    if len(roi_names) != n_rois:
        raise ValueError(
            f"roi_names tiene {len(roi_names)} nombres, pero bold tiene {n_rois} ROIs."
        )

    if site is not None and site not in SITES:
        raise ValueError(f"Sitio desconocido {site!r}. Disponibles: {SITES}.")


def load_bold(site: str, bold_dir: str | Path | None = None) -> dict[str, Any]:
    """Carga y valida las señales BOLD de un sitio.

    El resultado queda en memoria. Los arreglos cargados se marcan como solo
    lectura para prevenir modificaciones accidentales que invaliden la caché.
    """

    if site not in SITES:
        raise SystemExit(
            f"\nERROR: sitio '{site}' desconocido. Disponibles: {', '.join(SITES)}.\n"
        )

    directory = Path(bold_dir or BOLD_DIR)
    key = (str(directory.resolve()), site)
    if key in _bold_cache:
        return _bold_cache[key]

    path = directory / f"{site}.joblib"
    if not path.exists():
        available = sorted(p.stem for p in directory.glob("*.joblib"))
        raise SystemExit(
            f"\nERROR: no existen señales BOLD para el sitio '{site}' en {directory}\n"
            f" Disponibles: {available or '(ninguno)'}\n"
        )

    raw = joblib.load(path)
    validate_bold_payload(raw, site=site)

    data = dict(raw)
    data["bold"] = np.asarray(raw["bold"], dtype=np.float32)
    data["labels"] = np.asarray(raw["labels"], dtype=np.int32)
    data["subjects"] = list(raw["subjects"])
    data["roi_names"] = list(raw["roi_names"])
    data["bold"].setflags(write=False)
    data["labels"].setflags(write=False)

    _bold_cache[key] = data
    return data


def load_roi_sets(atlas_dir: str | Path | None = None) -> dict[str, Any]:
    """Carga los subconjuntos ROI definidos en ``roi_sets.json``."""

    path = Path(atlas_dir or ATLAS_DIR) / "roi_sets.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de subconjuntos ROI: {path}.")
    sets = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sets, dict) or not sets:
        raise ValueError(f"{path} debe contener un objeto JSON no vacío.")
    return sets


def roi_indices(roi_set: str | int, atlas_dir: str | Path | None = None) -> np.ndarray:
    """Devuelve los índices base 0 del subconjunto solicitado."""

    sets = load_roi_sets(atlas_dir)
    key = str(roi_set)
    if key not in sets:
        def _sort_key(value: str) -> tuple[int, str]:
            return (0, f"{int(value):012d}") if value.isdigit() else (1, value)

        available = ", ".join(sorted(sets, key=_sort_key))
        raise SystemExit(
            f"\nERROR: subconjunto de ROIs '{roi_set}' desconocido.\n"
            f" Disponibles: {available}\n"
        )

    entry = sets[key]
    if not isinstance(entry, Mapping) or "indices" not in entry:
        raise ValueError(f"El subconjunto ROI {key!r} no contiene la clave 'indices'.")

    idx = np.asarray(entry["indices"])
    if idx.ndim != 1 or idx.size < 2:
        raise ValueError(f"El subconjunto ROI {key!r} debe contener al menos dos índices.")
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"Los índices del subconjunto ROI {key!r} deben ser enteros.")
    idx = idx.astype(np.int64, copy=False)
    if np.unique(idx).size != idx.size or idx.min() < 0:
        raise ValueError(f"El subconjunto ROI {key!r} contiene índices inválidos o duplicados.")
    return idx


def resolve_window_spec(
    *,
    tr_seconds: float,
    window_tr: int | None = None,
    window_seconds: float | None = None,
    step_tr: int | None = None,
    step_seconds: float | None = None,
    overlap: float | None = None,
    shape: WindowShape = "rectangular",
    fisher_z: bool = False,
    gaussian_sigma: float | None = None,
) -> WindowSpec:
    """Resuelve una ventana definida en TR o segundos.

    Debe especificarse exactamente una longitud (``window_tr`` o
    ``window_seconds``) y, como máximo, una definición del desplazamiento
    (``step_tr``, ``step_seconds`` u ``overlap``). Si no se especifica paso,
    se utiliza una ventana no solapada.
    """

    tr = _require_positive_float("tr_seconds", tr_seconds)
    if (window_tr is None) == (window_seconds is None):
        raise ValueError("Especifique exactamente uno entre window_tr y window_seconds.")

    requested_window_seconds: float | None = None
    if window_seconds is not None:
        requested_window_seconds = _require_positive_float("window_seconds", window_seconds)
        resolved_window = max(2, _round_positive_to_int(requested_window_seconds / tr))
    else:
        resolved_window = _require_int("window_tr", window_tr, minimum=2)

    provided_steps = sum(value is not None for value in (step_tr, step_seconds, overlap))
    if provided_steps > 1:
        raise ValueError(
            "Especifique como máximo uno entre step_tr, step_seconds y overlap."
        )

    requested_step_seconds: float | None = None
    requested_overlap: float | None = None
    if step_tr is not None:
        resolved_step = _require_int("step_tr", step_tr, minimum=1)
    elif step_seconds is not None:
        requested_step_seconds = _require_positive_float("step_seconds", step_seconds)
        resolved_step = max(1, _round_positive_to_int(requested_step_seconds / tr))
    elif overlap is not None:
        if isinstance(overlap, bool) or not isinstance(overlap, (int, float, np.number)):
            raise TypeError("overlap debe ser numérico en el intervalo [0, 1).")
        requested_overlap = float(overlap)
        if not np.isfinite(requested_overlap) or not 0 <= requested_overlap < 1:
            raise ValueError("overlap debe pertenecer al intervalo [0, 1).")
        resolved_step = max(1, _round_positive_to_int(resolved_window * (1.0 - requested_overlap)))
    else:
        resolved_step = resolved_window

    if resolved_step > resolved_window:
        # Se permite separación entre ventanas, pero ya no existe solapamiento.
        effective_overlap = 0.0
    else:
        effective_overlap = 1.0 - resolved_step / resolved_window

    if shape not in ("rectangular", "gaussian"):
        raise ValueError("shape debe ser 'rectangular' o 'gaussian'.")

    sigma: float | None = None
    if shape == "gaussian":
        sigma = (
            _require_positive_float("gaussian_sigma", gaussian_sigma)
            if gaussian_sigma is not None
            else resolved_window / 6.0
        )
    elif gaussian_sigma is not None:
        raise ValueError("gaussian_sigma solo puede usarse con shape='gaussian'.")

    return WindowSpec(
        tr_seconds=tr,
        window_tr=resolved_window,
        step_tr=resolved_step,
        window_seconds=resolved_window * tr,
        step_seconds=resolved_step * tr,
        requested_window_seconds=requested_window_seconds,
        requested_step_seconds=requested_step_seconds,
        requested_overlap=requested_overlap,
        effective_overlap=effective_overlap,
        shape=shape,
        fisher_z=bool(fisher_z),
        gaussian_sigma=sigma,
    )


def n_windows(n_timepoints: int, window: int, step: int) -> int:
    """Número de ventanas completas que caben en la serie."""

    n_timepoints = _require_int("n_timepoints", n_timepoints, minimum=2)
    window = _require_int("window", window, minimum=2)
    step = _require_int("step", step, minimum=1)
    if window > n_timepoints:
        raise ValueError(
            f"La ventana de {window} TR no cabe en una serie de {n_timepoints} TR."
        )
    return (n_timepoints - window) // step + 1


def window_indices(n_timepoints: int, window: int, step: int) -> np.ndarray:
    """Matriz ``(n_ventanas, window)`` con los índices temporales utilizados."""

    nw = n_windows(n_timepoints, window, step)
    return (
        np.arange(window, dtype=np.int64)[None, :]
        + (np.arange(nw, dtype=np.int64) * step)[:, None]
    )


def _chunk_for(n_rois: int) -> int:
    """Selecciona un lote razonable según el número de ROIs."""

    return 16 if n_rois >= 100 else (32 if n_rois >= 40 else 64)


def _validate_build_inputs(
    bold: Any,
    indices: Iterable[int],
    *,
    window: int | None = None,
    step: int | None = None,
    chunk: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    arr = validate_bold_array(bold, check_finite=False)
    idx = validate_indices(indices, arr.shape[1])
    selected = np.asarray(arr, dtype=np.float32)[:, idx, :]
    if not np.isfinite(selected).all():
        raise ValueError("Las señales seleccionadas contienen NaN o valores infinitos.")

    if window is not None and step is not None:
        n_windows(selected.shape[-1], window, step)
    selected_chunk = _chunk_for(idx.size) if chunk is None else _require_int(
        "chunk", chunk, minimum=1
    )
    return selected, idx, selected_chunk


def _gaussian_weights(window: int, sigma: float | None) -> np.ndarray:
    sigma_value = window / 6.0 if sigma is None else _require_positive_float(
        "gaussian_sigma", sigma
    )
    center = (window - 1) / 2.0
    x = np.arange(window, dtype=np.float64) - center
    weights = np.exp(-0.5 * (x / sigma_value) ** 2)
    weights /= weights.sum()
    return weights.astype(np.float32)


def _apply_fisher_z(values: np.ndarray) -> np.ndarray:
    np.clip(values, -_FISHER_LIMIT, _FISHER_LIMIT, out=values)
    np.arctanh(values, out=values)
    return values


def _correlation_windows(
    windows: np.ndarray,
    *,
    shape: WindowShape,
    gaussian_sigma: float | None,
    constant_policy: ConstantPolicy,
) -> tuple[np.ndarray, int]:
    """Calcula correlaciones para ``(lote, ventanas, ROIs, tiempo)``."""

    if constant_policy not in ("zero", "raise"):
        raise ValueError("constant_policy debe ser 'zero' o 'raise'.")

    if shape == "rectangular":
        if gaussian_sigma is not None:
            raise ValueError(
                "gaussian_sigma solo puede usarse con window_shape='gaussian'."
            )
        # Esta rama conserva exactamente el cálculo histórico.
        centered = windows - windows.mean(axis=-1, keepdims=True)
        sd = centered.std(axis=-1, ddof=1, keepdims=True)
        constant = sd < _CONSTANT_EPS
        constant_count = int(constant.sum())
        if constant_count and constant_policy == "raise":
            raise ValueError(
                f"Se encontraron {constant_count} combinaciones ROI-ventana constantes."
            )
        standardized = np.divide(
            centered,
            np.where(constant, np.float32(1.0), sd),
        )
        standardized[np.broadcast_to(constant, standardized.shape)] = 0.0
        fc = (
            standardized @ np.swapaxes(standardized, -1, -2)
        ) / np.float32(windows.shape[-1] - 1)
    elif shape == "gaussian":
        weights = _gaussian_weights(windows.shape[-1], gaussian_sigma)
        weight_view = weights.reshape((1,) * (windows.ndim - 1) + (-1,))
        mean = np.sum(windows * weight_view, axis=-1, keepdims=True)
        centered = windows - mean
        weighted = centered * np.sqrt(weight_view)
        ss = np.sum(weighted * weighted, axis=-1, keepdims=True)
        constant = ss < _CONSTANT_EPS
        constant_count = int(constant.sum())
        if constant_count and constant_policy == "raise":
            raise ValueError(
                f"Se encontraron {constant_count} combinaciones ROI-ventana constantes."
            )
        denom = np.sqrt(ss @ np.swapaxes(ss, -1, -2))
        numerator = weighted @ np.swapaxes(weighted, -1, -2)
        fc = np.divide(
            numerator,
            np.where(denom < _CONSTANT_EPS, np.float32(1.0), denom),
        )
        invalid = constant | np.swapaxes(constant, -1, -2)
        fc[invalid] = 0.0
    else:
        raise ValueError("shape debe ser 'rectangular' o 'gaussian'.")

    np.clip(fc, -1.0, 1.0, out=fc)
    return np.asarray(fc, dtype=np.float32), constant_count


def build_sequences(
    bold: Any,
    indices: Iterable[int],
    window: int,
    step: int,
    chunk: int | None = 32,
    *,
    window_shape: WindowShape = "rectangular",
    gaussian_sigma: float | None = None,
    fisher_z: bool = False,
    constant_policy: ConstantPolicy = "zero",
) -> np.ndarray:
    """Construye matrices dinámicas ``(n, n_ventanas, r, r)``.

    Con los valores predeterminados opcionales, la salida es compatible con la
    implementación histórica del repositorio.
    """

    window = _require_int("window", window, minimum=2)
    step = _require_int("step", step, minimum=1)
    sig_all, _, chunk_size = _validate_build_inputs(
        bold, indices, window=window, step=step, chunk=chunk
    )
    n, r, n_timepoints = sig_all.shape
    widx = window_indices(n_timepoints, window, step)
    out = np.empty((n, widx.shape[0], r, r), dtype=np.float32)

    for start in range(0, n, chunk_size):
        windows = np.transpose(
            sig_all[start : start + chunk_size][:, :, widx],
            (0, 2, 1, 3),
        )
        fc, _ = _correlation_windows(
            windows,
            shape=window_shape,
            gaussian_sigma=gaussian_sigma,
            constant_policy=constant_policy,
        )
        if fisher_z:
            _apply_fisher_z(fc)
            diagonal = np.arange(r)
            fc[:, :, diagonal, diagonal] = 0.0
        out[start : start + chunk_size] = fc
    return out


def build_flat_sequences(
    bold: Any,
    indices: Iterable[int],
    window: int,
    step: int,
    chunk: int | None = None,
    *,
    window_shape: WindowShape = "rectangular",
    gaussian_sigma: float | None = None,
    fisher_z: bool = False,
    constant_policy: ConstantPolicy = "zero",
) -> np.ndarray:
    """Construye secuencias vectorizadas ``(n, ventanas, r*(r-1)/2)``.

    Extrae el triángulo superior dentro de cada lote para no conservar matrices
    simétricas completas en memoria.
    """

    window = _require_int("window", window, minimum=2)
    step = _require_int("step", step, minimum=1)
    sig_all, idx, chunk_size = _validate_build_inputs(
        bold, indices, window=window, step=step, chunk=chunk
    )
    n, r, n_timepoints = sig_all.shape
    widx = window_indices(n_timepoints, window, step)
    iu = np.triu_indices(r, k=1)
    out = np.empty((n, widx.shape[0], len(iu[0])), dtype=np.float32)

    for start in range(0, n, chunk_size):
        windows = np.transpose(
            sig_all[start : start + chunk_size][:, :, widx],
            (0, 2, 1, 3),
        )
        fc, _ = _correlation_windows(
            windows,
            shape=window_shape,
            gaussian_sigma=gaussian_sigma,
            constant_policy=constant_policy,
        )
        flat = fc[:, :, iu[0], iu[1]]
        if fisher_z:
            _apply_fisher_z(flat)
        out[start : start + chunk_size] = flat
    return out


def build_static_connectivity(
    bold: Any,
    indices: Iterable[int],
    chunk: int | None = None,
    *,
    fisher_z: bool = False,
    constant_policy: ConstantPolicy = "zero",
) -> np.ndarray:
    """Conectividad estática ``(n, 1, r, r)`` sobre toda la serie."""

    arr = validate_bold_array(bold, check_finite=False)
    return build_sequences(
        arr,
        indices,
        window=arr.shape[-1],
        step=arr.shape[-1],
        chunk=chunk,
        fisher_z=fisher_z,
        constant_policy=constant_policy,
    )


def build_flat_static_connectivity(
    bold: Any,
    indices: Iterable[int],
    chunk: int | None = None,
    *,
    fisher_z: bool = False,
    constant_policy: ConstantPolicy = "zero",
) -> np.ndarray:
    """Conectividad estática vectorizada ``(n, 1, r*(r-1)/2)``."""

    arr = validate_bold_array(bold, check_finite=False)
    return build_flat_sequences(
        arr,
        indices,
        window=arr.shape[-1],
        step=arr.shape[-1],
        chunk=chunk,
        fisher_z=fisher_z,
        constant_policy=constant_policy,
    )


def build_flat_partial_connectivity(
    bold: Any,
    indices: Iterable[int],
    chunk: int | None = None,
    *,
    fisher_z: bool = False,
    constant_policy: ConstantPolicy = "zero",
) -> np.ndarray:
    """Correlación parcial regularizada (Ledoit-Wolf) por sujeto, ``(n, 1, r*(r-1)/2)``.

    Estima la covarianza con shrinkage de Ledoit-Wolf sobre toda la serie de cada
    sujeto, obtiene la matriz de precisión y la convierte en correlación parcial. El
    shrinkage garantiza una matriz bien condicionada aunque haya menos puntos temporales
    que ROIs (p. ej. OHSU: 74 TR, 116 ROIs). El cálculo es por sujeto, así que no
    introduce dependencia entre sujetos ni fuga entre particiones.
    """

    from sklearn.covariance import LedoitWolf  # dependencia perezosa

    if constant_policy not in ("zero", "raise"):
        raise ValueError("constant_policy debe ser 'zero' o 'raise'.")

    arr = validate_bold_array(bold, check_finite=False)
    idx = validate_indices(indices, arr.shape[1])
    selected = np.asarray(arr, dtype=np.float64)[:, idx, :]
    if not np.isfinite(selected).all():
        raise ValueError("Las señales seleccionadas contienen NaN o valores infinitos.")

    n, r, _ = selected.shape
    iu = np.triu_indices(r, k=1)
    out = np.empty((n, 1, len(iu[0])), dtype=np.float32)
    constant_total = 0

    for subject in range(n):
        series = selected[subject].T  # (tiempo, ROI)
        constant = series.std(axis=0, ddof=1) < float(_CONSTANT_EPS)
        constant_total += int(constant.sum())
        precision = LedoitWolf().fit(series).precision_
        scale = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
        partial = -precision / np.outer(scale, scale)
        vec = partial[iu[0], iu[1]].astype(np.float64)
        if constant.any():  # una conexión con un ROI constante no está definida
            vec[constant[iu[0]] | constant[iu[1]]] = 0.0
        np.clip(vec, -1.0, 1.0, out=vec)
        if fisher_z:
            np.clip(vec, -float(_FISHER_LIMIT), float(_FISHER_LIMIT), out=vec)
            np.arctanh(vec, out=vec)
        out[subject, 0] = vec.astype(np.float32)

    if constant_total and constant_policy == "raise":
        raise ValueError(
            f"Se encontraron {constant_total} ROIs constantes al estimar la precisión."
        )
    return out


def hybrid_summary(sequences: Any, static: Any) -> np.ndarray:
    """Conectividad estática combinada con estadísticos invariantes al orden.

    Concatena, por conexión: la conectividad estática, y la media, la desviación estándar
    y el cambio medio absoluto entre ventanas consecutivas de la secuencia dinámica.
    Salida ``(n, 1, 4*r*(r-1)/2)``. No asume que el orden de las ventanas informe.
    """

    seq = np.asarray(sequences, dtype=np.float32)
    st = np.asarray(static, dtype=np.float32)
    if seq.ndim != 3:
        raise ValueError("sequences debe tener forma (sujetos, ventanas, características).")
    if st.ndim != 3 or st.shape[0] != seq.shape[0] or st.shape[1] != 1:
        raise ValueError("static debe tener forma (sujetos, 1, características).")
    if st.shape[2] != seq.shape[2]:
        raise ValueError("static y sequences deben compartir el número de características.")

    mean = seq.mean(axis=1)
    std = seq.std(axis=1, ddof=0)
    if seq.shape[1] > 1:
        delta = np.abs(np.diff(seq, axis=1)).mean(axis=1)
    else:
        delta = np.zeros_like(mean)
    combined = np.concatenate([st[:, 0, :], mean, std, delta], axis=-1)
    return combined[:, None, :].astype(np.float32)


def build_flat_multiview(
    bold: Any,
    indices: Iterable[int],
    chunk: int | None = None,
    *,
    fisher_z: bool = False,
    constant_policy: ConstantPolicy = "zero",
) -> np.ndarray:
    """Apila varias matrices de conectividad como canales: ``(n, k, r*(r-1)/2)``.

    Cada "ventana" es una vista distinta de la misma serie completa: canal 0 la
    correlación de Pearson y canal 1 la correlación parcial regularizada (Ledoit-Wolf).
    Pensada para BrainNetCNN, cuya capa de reconstrucción trata el eje de ventanas como
    canales de entrada, de modo que los filtros aprenden de ambas matrices a la vez.
    Todas las vistas comparten el número de características (mismo r), así que se apilan
    sin normalización adicional (ambas están en [-1, 1]).
    """

    pearson = build_flat_static_connectivity(
        bold, indices, chunk=chunk, fisher_z=fisher_z, constant_policy=constant_policy
    )
    partial = build_flat_partial_connectivity(
        bold, indices, chunk=chunk, fisher_z=fisher_z, constant_policy=constant_policy
    )
    return np.concatenate([pearson, partial], axis=1).astype(np.float32)


def build_connectivity(
    bold: Any,
    indices: Iterable[int],
    *,
    mode: ConnectivityMode = "dynamic",
    window: int | None = None,
    step: int | None = None,
    flat: bool = True,
    chunk: int | None = None,
    window_shape: WindowShape = "rectangular",
    gaussian_sigma: float | None = None,
    fisher_z: bool = False,
    constant_policy: ConstantPolicy = "zero",
) -> np.ndarray:
    """Fachada común para conectividad dinámica o estática."""

    if mode == "static":
        if window is not None or step is not None:
            raise ValueError("window y step no se usan cuando mode='static'.")
        function = build_flat_static_connectivity if flat else build_static_connectivity
        return function(
            bold,
            indices,
            chunk=chunk,
            fisher_z=fisher_z,
            constant_policy=constant_policy,
        )

    if mode != "dynamic":
        raise ValueError("mode debe ser 'dynamic' o 'static'.")
    if window is None or step is None:
        raise ValueError("window y step son obligatorios cuando mode='dynamic'.")

    function = build_flat_sequences if flat else build_sequences
    return function(
        bold,
        indices,
        window,
        step,
        chunk=chunk,
        window_shape=window_shape,
        gaussian_sigma=gaussian_sigma,
        fisher_z=fisher_z,
        constant_policy=constant_policy,
    )


def _indices_digest(indices: np.ndarray) -> str:
    stable = np.ascontiguousarray(indices.astype(np.int64, copy=False))
    return hashlib.sha256(stable.tobytes()).hexdigest()


def build_sequences_cached(
    site: str,
    bold: Any,
    indices: Iterable[int],
    window: int,
    step: int,
    roi_set: str | int | None = None,
    *,
    mode: ConnectivityMode = "dynamic",
    window_shape: WindowShape = "rectangular",
    gaussian_sigma: float | None = None,
    fisher_z: bool = False,
    constant_policy: ConstantPolicy = "zero",
) -> np.ndarray:
    """Versión vectorizada con memoria de una sola construcción.

    La clave incluye los índices concretos y todos los parámetros que cambian
    el tensor, evitando colisiones entre subconjuntos de igual tamaño.
    """

    arr = validate_bold_array(bold, check_finite=False)
    idx = validate_indices(indices, arr.shape[1])
    key = (
        site,
        str(roi_set),
        id(arr),
        arr.shape,
        arr.dtype.str,
        _indices_digest(idx),
        mode,
        int(window),
        int(step),
        window_shape,
        None if gaussian_sigma is None else float(gaussian_sigma),
        bool(fisher_z),
        constant_policy,
    )
    if _seq_cache.get("key") == key:
        return _seq_cache["value"]

    _seq_cache.clear()
    if mode == "static":
        value = build_flat_static_connectivity(
            arr,
            idx,
            fisher_z=fisher_z,
            constant_policy=constant_policy,
        )
    elif mode == "dynamic":
        value = build_flat_sequences(
            arr,
            idx,
            window,
            step,
            window_shape=window_shape,
            gaussian_sigma=gaussian_sigma,
            fisher_z=fisher_z,
            constant_policy=constant_policy,
        )
    else:
        raise ValueError("mode debe ser 'dynamic' o 'static'.")

    _seq_cache.update(key=key, value=value)
    return value


def upper_triangle(x: Any) -> np.ndarray:
    """Convierte ``(..., r, r)`` al triángulo superior sin diagonal."""

    arr = np.asarray(x)
    if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError(
            "x debe terminar en dos dimensiones cuadradas; "
            f"forma recibida: {arr.shape}."
        )
    iu = np.triu_indices(arr.shape[-1], k=1)
    return arr[..., iu[0], iu[1]]


def windowing_diagnostics(
    n_timepoints: int,
    window: int,
    step: int,
    *,
    tr_seconds: float | None = None,
    sequences: np.ndarray | None = None,
) -> dict[str, Any]:
    """Resume cobertura, redundancia y duración de una configuración."""

    n_timepoints = _require_int("n_timepoints", n_timepoints, minimum=2)
    window = _require_int("window", window, minimum=2)
    step = _require_int("step", step, minimum=1)
    idx = window_indices(n_timepoints, window, step)
    coverage = np.bincount(idx.ravel(), minlength=n_timepoints)
    last_used = int(idx[-1, -1])

    diagnostics: dict[str, Any] = {
        "n_timepoints": n_timepoints,
        "window_tr": window,
        "step_tr": step,
        "n_windows": int(idx.shape[0]),
        "effective_overlap": max(0.0, 1.0 - step / window),
        "unused_timepoints": int(n_timepoints - last_used - 1),
        "window_fraction_of_scan": float(window / n_timepoints),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "coverage_mean": float(coverage.mean()),
    }

    if tr_seconds is not None:
        tr = _require_positive_float("tr_seconds", tr_seconds)
        diagnostics.update(
            window_seconds=float(window * tr),
            step_seconds=float(step * tr),
            scan_seconds=float(n_timepoints * tr),
        )

    if sequences is not None:
        seq = np.asarray(sequences)
        if seq.ndim != 3:
            raise ValueError(
                "sequences debe tener forma (sujetos, ventanas, características)."
            )
        if seq.shape[1] != idx.shape[0]:
            raise ValueError(
                f"sequences contiene {seq.shape[1]} ventanas; se esperaban {idx.shape[0]}."
            )
        if seq.shape[1] < 2:
            diagnostics["median_adjacent_similarity"] = None
        else:
            left = seq[:, :-1].astype(np.float64, copy=False)
            right = seq[:, 1:].astype(np.float64, copy=False)
            left -= left.mean(axis=-1, keepdims=True)
            right -= right.mean(axis=-1, keepdims=True)
            denom = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
            similarity = np.divide(
                np.sum(left * right, axis=-1),
                denom,
                out=np.full_like(denom, np.nan, dtype=np.float64),
                where=denom > 0,
            )
            finite = similarity[np.isfinite(similarity)]
            diagnostics["median_adjacent_similarity"] = (
                float(np.median(finite)) if finite.size else None
            )
    return diagnostics


def methodological_warnings(diagnostics: Mapping[str, Any]) -> list[str]:
    """Genera advertencias descriptivas sin impedir la ejecución."""

    warnings: list[str] = []
    n_w = diagnostics.get("n_windows")
    if isinstance(n_w, (int, np.integer)) and n_w < 8:
        warnings.append(f"La configuración produce solo {n_w} ventanas por sujeto.")
    fraction = diagnostics.get("window_fraction_of_scan")
    if isinstance(fraction, (int, float, np.number)) and fraction > 0.75:
        warnings.append(
            f"La ventana cubre {float(fraction):.1%} del escaneo completo."
        )
    overlap = diagnostics.get("effective_overlap")
    if isinstance(overlap, (int, float, np.number)) and overlap > 0.95:
        warnings.append(f"El solapamiento efectivo es {float(overlap):.1%}.")
    similarity = diagnostics.get("median_adjacent_similarity")
    if isinstance(similarity, (int, float, np.number)) and similarity > 0.98:
        warnings.append(
            "La similitud mediana entre ventanas consecutivas supera 0.98 "
            f"({float(similarity):.4f})."
        )
    unused = diagnostics.get("unused_timepoints")
    if isinstance(unused, (int, np.integer)) and unused > 0:
        warnings.append(f"Quedan {unused} puntos temporales sin utilizar al final.")
    return warnings


def _subject_seed(subject_id: Any, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{subject_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def permute_windows(
    sequences: Any,
    *,
    subject_ids: Sequence[Any] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Permuta ventanas dentro de cada sujeto de forma reproducible."""

    seq = np.asarray(sequences)
    if seq.ndim != 3:
        raise ValueError("sequences debe tener forma (sujetos, ventanas, características).")
    n_subjects, n_w, _ = seq.shape
    ids = list(range(n_subjects)) if subject_ids is None else list(subject_ids)
    if len(ids) != n_subjects:
        raise ValueError(
            f"subject_ids tiene {len(ids)} elementos; se esperaban {n_subjects}."
        )
    if len({str(value) for value in ids}) != n_subjects:
        raise ValueError("subject_ids debe contener identificadores únicos.")
    seed = _require_int("seed", seed, minimum=0)

    out = np.empty_like(seq)
    for subject_index, subject_id in enumerate(ids):
        rng = np.random.default_rng(_subject_seed(subject_id, seed))
        out[subject_index] = seq[subject_index, rng.permutation(n_w)]
    return out


def summarize_windows(
    sequences: Any,
    statistics: Sequence[SummaryStatistic] = ("mean", "std"),
    *,
    keep_time_axis: bool = True,
) -> np.ndarray:
    """Resume una secuencia por conexión sin mezclar sujetos.

    Con ``statistics=('mean', 'std')`` concatena ambos resúmenes en el eje de
    características. ``ddof=0`` evita valores no finitos cuando existe una sola
    ventana, como en conectividad estática.
    """

    seq = np.asarray(sequences)
    if seq.ndim != 3:
        raise ValueError("sequences debe tener forma (sujetos, ventanas, características).")
    if not statistics:
        raise ValueError("Debe solicitarse al menos una estadística.")

    pieces: list[np.ndarray] = []
    for statistic in statistics:
        if statistic == "mean":
            pieces.append(seq.mean(axis=1))
        elif statistic == "std":
            pieces.append(seq.std(axis=1, ddof=0))
        else:
            raise ValueError(f"Estadística desconocida: {statistic!r}.")

    result = np.concatenate(pieces, axis=-1).astype(np.float32, copy=False)
    return result[:, None, :] if keep_time_axis else result


__all__ = [
    "ATLAS_DIR",
    "BOLD_DIR",
    "REPO_ROOT",
    "SITES",
    "SITE_TR_SECONDS",
    "WindowSpec",
    "build_connectivity",
    "build_flat_multiview",
    "build_flat_partial_connectivity",
    "build_flat_sequences",
    "build_flat_static_connectivity",
    "build_sequences",
    "build_sequences_cached",
    "build_static_connectivity",
    "clear_caches",
    "hybrid_summary",
    "load_bold",
    "load_roi_sets",
    "methodological_warnings",
    "n_windows",
    "permute_windows",
    "resolve_window_spec",
    "roi_indices",
    "summarize_windows",
    "upper_triangle",
    "validate_bold_array",
    "validate_bold_payload",
    "validate_indices",
    "window_indices",
    "windowing_diagnostics",
]
