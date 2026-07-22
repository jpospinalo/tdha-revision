"""
Registro de arquitecturas para los experimentos de clasificación TDAH.

Motivación
----------
Los experimentos varían tres cosas de forma independiente: el conjunto de datos
(sitio y grupo de ROIs), la arquitectura, y los hiperparámetros de entrenamiento.
Este paquete se ocupa **solo de la arquitectura**. La compilación del modelo
—optimizador, tasa de aprendizaje, función de pérdida y métricas— la hace
``run_experiment.py``, de modo que cambiar de LSTM a GRU no arrastra cambios en la
configuración de entrenamiento y las comparaciones entre arquitecturas son limpias.

Contrato
--------
Toda función registrada debe:

1. Aceptar ``n_windows`` y ``n_features`` como primeros argumentos posicionales.
2. Aceptar el resto de sus hiperparámetros como argumentos con valor por defecto.
3. Recibir entradas de forma ``(lote, n_windows, n_features)``: la secuencia de
   vectores de conectividad de cada sujeto.
4. Devolver un ``keras.Model`` **sin compilar** con una salida sigmoide de dimensión 1.

Los valores por defecto de cada función quedan registrados en el ``config.json`` de
cada corrida, así que sirven de documentación ejecutable de la configuración usada.

Añadir una arquitectura
-----------------------
Crear un módulo nuevo en esta carpeta y registrarlo::

    # kerasmodels/mi_modelo.py
    from . import register

    @register("mi_modelo")
    def build(n_windows, n_features, units=64):
        import keras
        from keras import layers
        inp = layers.Input(shape=(n_windows, n_features))
        ...
        return keras.Model(inp, out, name="mi_modelo")

y añadirlo a la lista de importaciones del final de este archivo. Queda disponible
de inmediato como ``--model mi_modelo``, sin tocar ningún otro archivo.

Uso desde la línea de comandos::

    python run_experiment.py --list-models
    python run_experiment.py --site NYU --rois 12 --model gru --model-arg units=64

Compatibilidad
--------------
``kerasmodels.lstm.build_model`` y ``kerasmodels.lstm.METRICS`` se conservan con su
firma original para no romper los notebooks que ya los usan. El código nuevo debería
usar el registro.
"""

import inspect

__all__ = ["REGISTRY", "register", "build", "validate_args", "defaults", "available"]

#: Nombre de arquitectura -> función constructora. Poblado por el decorador.
REGISTRY = {}


def register(name):
    """Decorador que da de alta una arquitectura en el registro.

    Parameters
    ----------
    name : str
        Identificador que se usará en ``--model``.
    """
    def deco(fn):
        if name in REGISTRY:
            raise RuntimeError(f"la arquitectura '{name}' ya está registrada")
        REGISTRY[name] = fn
        return fn
    return deco


def available():
    """Nombres de arquitectura disponibles, ordenados."""
    return sorted(REGISTRY)


def defaults(name):
    """Hiperparámetros por defecto de una arquitectura.

    Se guardan en el ``config.json`` de cada corrida para que la configuración
    quede documentada aunque no se haya pasado explícitamente.
    """
    _assert_known(name)
    return {
        k: v.default
        for k, v in inspect.signature(REGISTRY[name]).parameters.items()
        if k not in ("n_windows", "n_features")
        and v.default is not inspect.Parameter.empty
    }


def validate_args(name, kwargs):
    """Valida los hiperparámetros de arquitectura.

    Se llama antes de cargar los datos para que un nombre mal escrito falle de
    inmediato y no después de varios minutos de preparación.

    Raises
    ------
    SystemExit
        Si la arquitectura no existe o algún hiperparámetro no le corresponde.
    """
    _assert_known(name)
    valid = set(inspect.signature(REGISTRY[name]).parameters) - {"n_windows", "n_features"}
    unknown = set(kwargs) - valid
    if unknown:
        raise SystemExit(
            f"ERROR: parámetros no válidos para '{name}': {sorted(unknown)}. "
            f"Acepta: {sorted(valid)}"
        )


def build(name, n_windows, n_features, **kwargs):
    """Construye una arquitectura registrada, sin compilar.

    Parameters
    ----------
    name : str
        Nombre registrado, p. ej. ``"lstm"``.
    n_windows, n_features : int
        Forma de la secuencia de entrada.
    **kwargs
        Hiperparámetros propios de la arquitectura.

    Returns
    -------
    keras.Model
        Modelo sin compilar. La compilación es responsabilidad de quien entrena.
    """
    validate_args(name, kwargs)
    return REGISTRY[name](n_windows, n_features, **kwargs)


def _assert_known(name):
    if name not in REGISTRY:
        raise SystemExit(
            f"ERROR: arquitectura '{name}' desconocida. "
            f"Disponibles: {', '.join(available()) or '(ninguna)'}"
        )


# Las importaciones van al final: los submódulos hacen `from . import register`,
# que necesita que el decorador ya exista. Añadir aquí cada módulo nuevo.
from . import lstm      # noqa: E402,F401
from . import gru       # noqa: E402,F401
from . import cnn1d     # noqa: E402,F401
from . import transformer  # noqa: E402,F401
from . import deepsets  # noqa: E402,F401
