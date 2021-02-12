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

    def call(self, X):
        """
        The call function that implements the propagation rule (i.e. forward pass rule) of the GCN.
        :param X: the input node features matrix
        :type X: numpy.ndarray

        """

        if self.activation_type == 'relu':
            prod_0 = tf.matmul(self.A_norm, X)
            prod_1 = tf.matmul(prod_0, self.theta)
            res = tf.nn.relu(prod_1)
            return res
        elif self.activation_type == 'softmax':
            prod_0 = tf.matmul(self.A_norm, X)
            prod_1 = tf.matmul(prod_0, self.theta)
            res = tf.nn.softmax(prod_1)
            return res


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
        self.input_layer = GCNLayer(num_units_in_hidden_layers[0], A_norm, 'relu')
        self.out = GCNLayer(num_units_in_output_layer, A_norm, 'softmax')

    def call(self, X):
        H = self.input_layer(X)
        Z = self.out(H)
        return Z