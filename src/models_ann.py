# src/models_ann.py
from tensorflow.keras import layers, models, optimizers, regularizers

def build_dense_ann(input_dim: int,
                    hidden_units: list = [64, 32, 16],
                    dropout: float = 0.1,
                    l2: float = 1e-5) -> models.Model:
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for units in hidden_units:
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        if dropout and dropout > 0.0:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return model
