import tensorflow as tf
from tensorflow import keras


class TypicalNNHiddenLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super(TypicalNNHiddenLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
        

class DNN(keras.Model):

    def __init__(
        self,
        num_units_in_hidden_layers,
        num_units_in_output_layer,
        name='dnn',
        **kwargs
    ):
        super(DNN, self).__init__(name=name, **kwargs)
        self.layer_1 = TypicalNNHiddenLayer(num_units_in_hidden_layers[0])
        self.layer_2 = TypicalNNHiddenLayer(num_units_in_hidden_layers[1])
        self.layer_3 = TypicalNNHiddenLayer(num_units_in_hidden_layers[2])
        self.layer_out = keras.layers.Dense(num_units_in_output_layer)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_out(x)
        return x