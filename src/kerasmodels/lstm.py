"""
Arquitectura LSTM.

Es la configuración base del artículo: una sola capa recurrente sobre la secuencia
de vectores de conectividad, seguida de una neurona sigmoide. Con 177 sujetos, una
arquitectura de este tamaño es deliberada: el argumento del trabajo es que basta una
red frugal sobre un subconjunto reducido de ROIs.

Número de parámetros de la capa recurrente: ``4 · ((n_features + units) · units + units)``.
Con 128 unidades eso va de unos 100 mil parámetros con 12 ROIs (66 características) a
unos 3,5 millones con 116 ROIs (6.670 características), sobre la misma muestra. Esa
diferencia de escala es parte de lo que se está midiendo al comparar grupos de ROIs.
"""

from . import register


@register("lstm")
def build(n_windows, n_features, units=128, dropout=0.0, bidirectional=False):
    """LSTM de una capa.

    Parameters
    ----------
    n_windows, n_features : int
        Forma de la secuencia de entrada.
    units : int
        Unidades de la capa recurrente. El artículo usa 128.
    dropout : float
        Dropout sobre las entradas de la capa recurrente. 0.0 = sin regularización,
        que es la configuración del artículo.
    bidirectional : bool
        Si es True, envuelve la capa en ``Bidirectional``. Duplica los parámetros y
        deja de ser causal, cosa irrelevante aquí porque la secuencia completa está
        disponible de antemano.

    Returns
    -------
    keras.Model
        Modelo sin compilar.
    """
    import keras
    from keras import layers

    core = layers.LSTM(units, dropout=dropout)
    rnn = layers.Bidirectional(core) if bidirectional else core
    inp = layers.Input(shape=(n_windows, n_features))
    out = layers.Dense(1, activation="sigmoid")(rnn(inp))
    return keras.Model(inp, out, name=f"lstm{units}{'_bi' if bidirectional else ''}")


# --------------------------------------------------------------------------- #
# API heredada
#
# Se conserva por compatibilidad con los notebooks que ya importan
# `from kerasmodels.lstm import build_model, METRICS`. A diferencia de `build`,
# devuelve el modelo YA COMPILADO, así que la configuración de entrenamiento queda
# fijada dentro de la arquitectura. El código nuevo debería usar el registro.
# --------------------------------------------------------------------------- #

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "auc",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
]


def build_model(n_windows, n_features, units=128, lr=1e-4):
    """Versión compilada de la LSTM. Obsoleta: preferir ``build``."""
    import keras

    model = build(n_windows, n_features, units=units)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=list(METRICS),
    )
    return model
