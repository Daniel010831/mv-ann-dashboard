# src/models_seq.py
from tensorflow.keras import layers, models, optimizers, regularizers

def build_lstm(seq_len: int, n_features: int,
               units: int = 64, dropout: float = 0.2, l2: float = 1e-5) -> models.Model:
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(units, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

def build_gru(seq_len: int, n_features: int,
              units: int = 64, dropout: float = 0.2, l2: float = 1e-5) -> models.Model:
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.GRU(units, return_sequences=False,
                   kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model
