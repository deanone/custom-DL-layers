import tensorflow as tf
from tensorflow import keras


class GCNLayer(keras.layers.Layer):
    """

    The class implements the Graph Convolutional Layer proposed by Kipf and Welling.
    This class subclasses the keras.layers.Layer base class.

    """
    def __init__(self, C, A_norm, activation_type):
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
        self.activation_type = activation_type

    def build(self, input_shape):
        self.theta = self.add_weight(
            shape = (input_shape[-1], self.C),
            initializer = "random_normal",
            trainable = True,
        )
        self.b = self.add_weight(shape = (self.N,), initializer = "random_normal", trainable = True)

    def call(self, X):
        """
        The call function that implements the propagation rule (i.e. forward pass rule) of the GCN.
        :param X: the input node features matrix
        :type X: numpy.ndarray

        """

        self.A_norm = self.A_norm.astype(X.dtype)
        self.theta = self.theta.astype(X.dtype)
        self.b = self.b.astype(X.dtype)

        if self.activation_type == 'relu':
            return tf.nn.relu(tf.matmul(tf.matmul(self.A_norm, X), self.theta) + self.b)
        elif self.activation_type == 'softmax':
            return tf.nn.softmax(tf.matmul(tf.matmul(self.A_norm, X), self.theta) + self.b)


class GCN(keras.Model):
    """

    The class implements a Graph Convolutional Network composed by GCN layers.
    This class subclasses the keras.Model base class.

    """

    def __init__(
        self,
        num_units_in_hidden_layers,
        num_units_in_output_layer,
        A_norm,
        name='gcn',
        **kwargs
    ):
        super(GCN, self).__init__(name=name, **kwargs)
        self.layer_1 = GCNLayer(num_units_in_hidden_layers[0], A_norm, 'relu')
        self.layer_out = GCNLayer(num_units_in_output_layer, A_norm, 'softmax')

    def call(self, X):
        H = self.layer_1(X)
        Z = self.layer_out(H)
        return Z