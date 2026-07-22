"""
Codificador Transformer sobre la secuencia de conectividad.

Pedido en el comentario 9 del revisor 2. Es la arquitectura del registro con mayor
riesgo de sobreajuste: la autoatención tiene muchos parámetros y la muestra es de
177 sujetos. Los valores por defecto son deliberadamente pequeños —un solo bloque,
d_model=64— y conviene no aumentarlos sin una razón, porque un Transformer
sobredimensionado que rinda mal no informa sobre la arquitectura sino sobre el
tamaño de la muestra.

Se usa codificación posicional **aprendida** (una capa Embedding sobre el índice de
ventana) en lugar de sinusoidal: las secuencias son cortas y de longitud fija dentro
de cada sitio, así que no hace falta que la codificación extrapole.

El orden de las ventanas en reposo es en buena medida intercambiable, así que la
codificación posicional puede memorizar ruido específico de posición con muestras
pequeñas. Con ``positional=False`` se omite y el modelo se comporta como un modelo de
conjuntos (atención + *pooling* por promedio), invariante al orden. Conviene contrastar
ambas variantes; si la representación ``permuted`` iguala a ``ordered``, la variante sin
posición es la adecuada.
"""

from . import register


@register("transformer")
def build(n_windows, n_features, d_model=64, num_heads=4, ff_dim=128,
          num_blocks=1, dropout=0.1, positional=True):
    """Codificador Transformer con agregación por promedio.

    Parameters
    ----------
    n_windows, n_features : int
        Forma de la secuencia de entrada.
    d_model : int
        Dimensión interna. Una capa densa proyecta las ``n_features`` de entrada a
        este tamaño antes de la atención, lo que además evita que el coste dependa
        del número de ROIs.
    num_heads : int
        Cabezas de atención. Debe dividir a ``d_model``.
    ff_dim : int
        Dimensión oculta de la red de avance de cada bloque.
    num_blocks : int
        Bloques apilados.
    dropout : float
        Dropout en la atención, la red de avance y antes de la salida.
    positional : bool
        Si es True (por defecto), suma una codificación posicional aprendida y el
        modelo es sensible al orden de las ventanas. Con False se omite y el modelo
        es invariante al orden (atención sobre un conjunto de ventanas).

    Returns
    -------
    keras.Model
        Modelo sin compilar.
    """
    import keras
    from keras import layers, ops

    if d_model % num_heads:
        raise ValueError(f"d_model={d_model} debe ser divisible entre num_heads={num_heads}")

    inp = layers.Input(shape=(n_windows, n_features))
    x = layers.Dense(d_model)(inp)
    if positional:
        pos = layers.Embedding(input_dim=n_windows, output_dim=d_model)(
            ops.arange(0, n_windows, dtype="int32")
        )
        x = x + pos

    for _ in range(num_blocks):
        # Pre-normalización: más estable que post-norm con pocos datos.
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        h = layers.MultiHeadAttention(num_heads=num_heads,
                                      key_dim=d_model // num_heads,
                                      dropout=dropout)(h, h)
        x = x + h
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        h = layers.Dense(ff_dim, activation="relu")(h)
        h = layers.Dropout(dropout)(h)
        h = layers.Dense(d_model)(h)
        x = x + h

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    # dtype="float32" explícito: con precisión mixta (mixed_float16) la sigmoide
    # y la pérdida deben calcularse en float32 para no perder estabilidad numérica.
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    suffix = "" if positional else "_nopos"
    return keras.Model(inp, out,
                       name=f"transformer_d{d_model}h{num_heads}b{num_blocks}{suffix}")
