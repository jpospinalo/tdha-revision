"""
BrainNetCNN: convolución sobre la matriz de conectividad respetando su topología.

Por qué esta y no una Conv2D estándar
-------------------------------------
Una convolución 2D corriente sobre una matriz de conectividad aprende patrones ligados
al orden de las filas y columnas, que proviene de la numeración arbitraria del atlas
AAL y no de ninguna vecindad anatómica. BrainNetCNN (Kawahara et al., 2017) evita ese
artefacto usando filtros con forma de cruz —fila + columna completas— en lugar de
kernels locales. Sus tres capas respetan la estructura de grafo de la conectividad:

- edge-to-edge (E2E): combina, para cada arista (i, j), toda su fila i y su columna j.
- edge-to-node (E2N): colapsa las aristas de cada nodo en una característica de nodo.
- node-to-graph (N2G): colapsa los nodos en una característica global del sujeto.

Integración con el pipeline
---------------------------
El modelo recibe la representación vectorizada estándar ``(lote, n_ventanas,
n_features)`` —el triángulo superior de la matriz— y **reconstruye la matriz simétrica
r×r dentro del propio grafo de cómputo**, de modo que no hace falta cambiar ``data.py``
ni ``run_experiment.py``. Está pensada para ``--representation static`` (una matriz por
sujeto); con una representación dinámica trata las ``n_ventanas`` como canales de entrada.

La reconstrucción usa TensorFlow directamente, así que esta arquitectura requiere el
backend de TensorFlow (el que se usa en Colab).
"""

import numpy as np

from . import register


def _infer_n_rois(n_features: int) -> int:
    """Recupera r a partir de n_features = r*(r-1)/2 del triángulo superior."""

    r = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
    if r * (r - 1) // 2 != n_features:
        raise ValueError(
            f"n_features={n_features} no corresponde al triángulo superior de una "
            "matriz cuadrada; BrainNetCNN necesita la conectividad completa."
        )
    return r


def _make_upper_to_matrix_layer(n_rois: int):
    """Capa que reconstruye ``(lote, n_ventanas, F)`` -> ``(lote, r, r, n_ventanas)``."""

    import keras
    import tensorflow as tf

    n_features = n_rois * (n_rois - 1) // 2
    iu = np.triu_indices(n_rois, k=1)
    k_of = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu[0], iu[1]))}
    gather = np.full(n_rois * n_rois, n_features, dtype=np.int64)  # n_features -> cero
    for i in range(n_rois):
        for j in range(n_rois):
            if i < j:
                gather[i * n_rois + j] = k_of[(i, j)]
            elif i > j:
                gather[i * n_rois + j] = k_of[(j, i)]

    class UpperTriToMatrix(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.n_rois = n_rois
            self.n_features = n_features
            self.gather = tf.constant(gather, dtype=tf.int64)

        def call(self, x):
            shape = tf.shape(x)
            batch, windows = shape[0], shape[1]
            flat = tf.reshape(x, (-1, self.n_features))            # (B*W, F)
            zeros = tf.zeros((tf.shape(flat)[0], 1), dtype=flat.dtype)
            augmented = tf.concat([flat, zeros], axis=-1)          # (B*W, F+1)
            mat = tf.gather(augmented, self.gather, axis=-1)       # (B*W, r*r)
            target = tf.stack([batch, windows, self.n_rois, self.n_rois])
            mat = tf.reshape(mat, target)                          # (B, W, r, r)
            return tf.transpose(mat, (0, 2, 3, 1))                 # (B, r, r, W)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.n_rois, self.n_rois, input_shape[1])

    return UpperTriToMatrix()


@register("brainnetcnn")
def build(n_windows, n_features, e2e=32, e2n=64, dense=96, dropout=0.5, leaky=0.33,
          l2_reg=0.0, inter_dropout=0.0, batchnorm=False):
    """BrainNetCNN sobre la matriz de conectividad reconstruida.

    Parameters
    ----------
    n_windows, n_features : int
        Forma de la secuencia de entrada. ``n_features`` debe ser el triángulo superior
        de una matriz cuadrada; con ``--representation static`` es una sola matriz.
    e2e, e2n, dense : int
        Filtros de las capas edge-to-edge, edge-to-node y node-to-graph. Los valores por
        defecto (32/64/96) siguen al artículo original, pensado para muestras grandes;
        con n≈177 conviene reducirlos mucho (p. ej. 4/8/8) para no sobreajustar.
    dropout : float
        Dropout antes de la capa de salida.
    leaky : float
        Pendiente negativa de las LeakyReLU.
    l2_reg : float
        Regularización L2 sobre los pesos de todas las convoluciones y la salida.
        0.0 desactiva. Estabiliza el entrenamiento en muestras pequeñas.
    inter_dropout : float
        Dropout aplicado entre bloques (tras E2E y tras E2N), además del final.
        0.0 desactiva.
    batchnorm : bool
        Si es True, añade BatchNormalization tras cada convolución, antes de la
        activación. Estabiliza el entrenamiento con capacidad reducida y regularización
        fuerte, y evita el colapso a una sola clase en muestras pequeñas.

    Returns
    -------
    keras.Model
        Modelo sin compilar.
    """
    import keras
    from keras import layers, ops, regularizers

    n_rois = _infer_n_rois(n_features)
    reg = regularizers.l2(l2_reg) if l2_reg else None

    def activate(z):
        # BatchNorm opcional antes de la activación (variante para muestra pequeña).
        if batchnorm:
            z = layers.BatchNormalization()(z)
        return layers.LeakyReLU(negative_slope=leaky)(z)

    inp = layers.Input(shape=(n_windows, n_features))
    x = _make_upper_to_matrix_layer(n_rois)(inp)          # (lote, r, r, n_windows)

    # edge-to-edge: filtro en cruz = fila (1×r) + columna (r×1), sumados por difusión.
    row = layers.Conv2D(e2e, (1, n_rois), padding="valid", kernel_regularizer=reg)(x)
    col = layers.Conv2D(e2e, (n_rois, 1), padding="valid", kernel_regularizer=reg)(x)
    x = layers.Lambda(
        lambda t: ops.add(t[0], t[1]),
        output_shape=(n_rois, n_rois, e2e),
        name="edge2edge",
    )([row, col])
    x = activate(x)
    if inter_dropout:
        x = layers.Dropout(inter_dropout)(x)

    # edge-to-node: colapsa las aristas de cada nodo -> característica de nodo.
    x = layers.Conv2D(e2n, (1, n_rois), padding="valid", kernel_regularizer=reg)(x)
    x = activate(x)
    if inter_dropout:
        x = layers.Dropout(inter_dropout)(x)

    # node-to-graph: colapsa los nodos -> característica global del sujeto.
    x = layers.Conv2D(dense, (n_rois, 1), padding="valid", kernel_regularizer=reg)(x)
    x = activate(x)

    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    # dtype="float32" explícito: con precisión mixta (mixed_float16) la sigmoide
    # y la pérdida deben calcularse en float32 para no perder estabilidad numérica.
    out = layers.Dense(1, activation="sigmoid", dtype="float32",
                       kernel_regularizer=reg)(x)
    return keras.Model(inp, out, name=f"brainnetcnn_e{e2e}n{e2n}")
