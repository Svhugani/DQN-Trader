import tensorflow as tf
from tensorflow.keras import layers, models


def standard_model(input_dim: int, output_dim: int, hidden_layers: int, hidden_dim):
    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=(input_dim,), name='input_layer'))

    for i in range(hidden_layers):
        model.add(layers.Dense(hidden_dim, activation='relu', name=f'hidden_layer{i}'))

    model.add(layers.Dense(output_dim, activation='linear', name='actions_layer'))
    model.compile(loss='mse', optimizer='adam')
    return model
