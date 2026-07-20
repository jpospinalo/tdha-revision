"""
Arquitectura CNN unidimensional sobre el eje temporal.

Por qué 1D y no 2D
------------------
La convolución se aplica **a lo largo de las ventanas temporales**, tratando el
vector de conectividad de cada ventana como los canales de entrada. No se convoluciona
sobre la disposición de los ROIs dentro de la matriz de conectividad, porque ese orden
es arbitrario: proviene de la numeración del atlas AAL116 y no de ninguna relación
anatómica de vecindad. Una convolución 2D sobre la matriz aprendería patrones
asociados a esa numeración, es decir, artefactos.

Ese razonamiento es el mismo que aparece en el Reporte de Experimentos, y conviene
que quede escrito aquí porque es la justificación metodológica de la arquitectura.

Relación con ``cnn.py``
-----------------------
``kerasmodels/cnn.py`` implementa una CNN **2D** sobre matrices estáticas de 18×18,
con la forma de entrada fija en el propio código. No es compatible con el flujo de
secuencias de ``run_experiment.py`` y por eso no está registrada. Se conserva sin
cambios para no romper el código que la use.
"""

from . import register


@register("cnn1d")
def build(n_windows, n_features, filters=64, kernel_size=10, n_blocks=1,
          dropout=0.0, pooling="max"):
    """CNN 1D sobre la secuencia de vectores de conectividad.

    Parameters
    ----------
    n_windows, n_features : int
        Forma de la secuencia de entrada.
    filters : int
        Filtros de la primera capa convolucional. Con ``n_blocks`` > 1 el número se
        duplica en cada bloque sucesivo (64, 128, 256…).
    kernel_size : int
        Extensión temporal del filtro, en número de ventanas. El valor 10 replica la
        configuración base del Reporte de Experimentos.
    n_blocks : int
        Número de bloques convolucionales apilados.
    dropout : float
        Dropout antes de la capa de salida.
    pooling : {"max", "avg"}
        Agregación temporal antes de la capa densa. ``max`` responde a la ventana más
        discriminativa; ``avg`` promedia toda la secuencia.

    Returns
    -------
    keras.Model
        Modelo sin compilar.
    """
    import keras
    from keras import layers

    if pooling not in ("max", "avg"):
        raise ValueError(f"pooling debe ser 'max' o 'avg', se recibió {pooling!r}")
    if kernel_size > n_windows:
        raise ValueError(
            f"kernel_size={kernel_size} supera las {n_windows} ventanas disponibles. "
            f"Con secuencias cortas (p. ej. OHSU tiene 3 ventanas) hay que reducirlo."
        )

    inp = layers.Input(shape=(n_windows, n_features))
    x = inp
    for b in range(n_blocks):
        x = layers.Conv1D(filters * (2 ** b), kernel_size,
                          activation="relu", padding="same")(x)
    x = (layers.GlobalMaxPooling1D() if pooling == "max"
         else layers.GlobalAveragePooling1D())(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    # dtype="float32" explícito: con precisión mixta (mixed_float16) la sigmoide
    # y la pérdida deben calcularse en float32 para no perder estabilidad numérica.
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name=f"cnn1d{filters}k{kernel_size}b{n_blocks}")
