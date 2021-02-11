import tensorflow as tf
from tensorflow import keras


class GCNLayer(keras.layers.Layer):
    """

    The class implements the Graph Convolutional Layer proposed by Kipf and Welling.
    This class subclasses the keras.layers.Layer base class.

    """
    def __init__(self, C = 32, A_norm):
        """
        Constructor.
        :param C: the number of channels/filter of the layer
        :type C: int
        :param A_norm: the normalized adjacency matrix of the graph
        :type A_norm: numpy.ndarray

        """
        super(GCNLayer, self).__init__()
        self.C = C
        self.A_norm = A_norm
        self.N = A_norm.shape[0]    #   number of nodes in the graph

    def build(self, input_shape):
        self.theta = self.add_weight(
            shape = (input_shape[-1], self.C),
            initializer = "random_normal",
            trainable = True,
        )
        self.b = self.add_weight(shape = (N,), initializer = "random_normal", trainable = True)

    def call(self, X):
        """
        The call function that implements the propagation rule (i.e. forward pass rule) of the GCN.
        :param X: the input node features matrix
        :type X: numpy.ndarray

        """
        return tf.nn.relu(tf.matmul(tf.matmul(self.A_norm, X), self.theta) + self.b)


class GCN(keras.Model):
    """

    The class implements a Graph Convolutional Network composed by GCN layers.
    This class subclasses the keras.Model base class.

    """

    def __init__(
        self,
        num_units_in_hidden_layers,
        num_units_in_output_layer,
        name='gcn',
        **kwargs
    ):
        super(GCN, self).__init__(name=name, **kwargs)
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