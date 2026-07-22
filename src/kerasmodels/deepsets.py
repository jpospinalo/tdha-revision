"""
Modelo invariante al orden (DeepSets) sobre la secuencia de conectividad.

Baseline honesto frente a las arquitecturas secuenciales. Trata las ventanas como un
**conjunto**, no como una secuencia: aplica el mismo MLP a cada ventana (la función φ
de DeepSets), agrega con un *pooling* simétrico (media o máximo) y clasifica con un
segundo MLP (la función ρ). No usa recurrencia ni codificación posicional, de modo que
su salida no cambia si se permutan las ventanas de un sujeto.

Es la arquitectura adecuada cuando el orden temporal de las ventanas en reposo no aporta
señal discriminativa —hipótesis que se contrasta con la representación ``permuted``—. Si
``deepsets`` iguala a la LSTM, la dependencia temporal no estaba aportando nada y las
recurrentes solo estaban ajustando ruido. Por eso es la comparación natural: aísla lo que
gana un modelo por leer la secuencia frente a resumir el conjunto de ventanas.

Con una sola ventana (conectividad estática) el *pooling* es la identidad y el modelo se
reduce a un MLP sobre el vector de conectividad, lo que lo hace comparable en ese caso.
"""

from . import register


@register("deepsets")
def build(n_windows, n_features, units=64, hidden=64, dropout=0.0, pooling="mean"):
    """MLP por ventana + pooling simétrico, invariante al orden.

    Parameters
    ----------
    n_windows, n_features : int
        Forma de la secuencia de entrada.
    units : int
        Unidades del MLP compartido que se aplica a cada ventana (φ).
    hidden : int
        Unidades del MLP posterior al pooling (ρ).
    dropout : float
        Dropout antes de la capa de salida.
    pooling : {"mean", "max"}
        Agregación simétrica sobre las ventanas. ``mean`` resume el conjunto; ``max``
        responde a la ventana más discriminativa. Ambas son invariantes al orden.

    Returns
    -------
    keras.Model
        Modelo sin compilar.
    """
    import keras
    from keras import layers

    if pooling not in ("mean", "max"):
        raise ValueError(f"pooling debe ser 'mean' o 'max', se recibió {pooling!r}")

    inp = layers.Input(shape=(n_windows, n_features))
    # Dense actúa sobre la última dimensión, así que comparte pesos entre ventanas (φ).
    x = layers.Dense(units, activation="relu")(inp)
    x = layers.Dense(units, activation="relu")(x)
    x = (layers.GlobalAveragePooling1D() if pooling == "mean"
         else layers.GlobalMaxPooling1D())(x)
    x = layers.Dense(hidden, activation="relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    # dtype="float32" explícito: con precisión mixta (mixed_float16) la sigmoide
    # y la pérdida deben calcularse en float32 para no perder estabilidad numérica.
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name=f"deepsets{units}_{pooling}")
