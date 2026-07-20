"""
Arquitectura GRU.

Alternativa recurrente a la LSTM, pedida en el comentario 9 del revisor 2. La celda
GRU tiene dos compuertas en lugar de tres y carece de estado de celda separado, de
modo que usa aproximadamente tres cuartas partes de los parámetros de una LSTM con
el mismo número de unidades. Con muestras pequeñas eso suele traducirse en menos
sobreajuste, así que es una comparación informativa y no solo un requisito del
revisor.
"""

from . import register


@register("gru")
def build(n_windows, n_features, units=128, dropout=0.0, bidirectional=False):
    """GRU de una capa.

    Parameters
    ----------
    n_windows, n_features : int
        Forma de la secuencia de entrada.
    units : int
        Unidades de la capa recurrente. Usar el mismo valor que la LSTM para que la
        comparación entre arquitecturas sea interpretable.
    dropout : float
        Dropout sobre las entradas de la capa recurrente.
    bidirectional : bool
        Envuelve la capa en ``Bidirectional``.

    Returns
    -------
    keras.Model
        Modelo sin compilar.
    """
    import keras
    from keras import layers

    core = layers.GRU(units, dropout=dropout)
    rnn = layers.Bidirectional(core) if bidirectional else core
    inp = layers.Input(shape=(n_windows, n_features))
    out = layers.Dense(1, activation="sigmoid")(rnn(inp))
    return keras.Model(inp, out, name=f"gru{units}{'_bi' if bidirectional else ''}")
