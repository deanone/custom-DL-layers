import tensorflow as tf
from tensorflow import keras


class GCNLayer(keras.layers.Layer):
    def __init__(self, F=32, A_norm):
        super(GCNLayer, self).__init__()
        self.F = F
        self.A_norm = A_norm
        self.N = A_norm.shape[0]

    def build(self, input_shape):
        self.theta = self.add_weight(
            shape=(input_shape[-1], self.F),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(N,), initializer="random_normal", trainable=True
        )

    def call(self, X):
        return tf.nn.relu(tf.matmul(tf.matmul(self.A_norm, X), self.theta) + self.b)