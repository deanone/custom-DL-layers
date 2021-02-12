import tensorflow as tf
from tensorflow import keras


loss_tracker = keras.metrics.Mean(name='softmax cross-entropy with masking loss')
metric_tracker = keras.metrics.Mean(name='accuracy with masking')


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
            initializer = tf.keras.initializers.GlorotNormal(),
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
        train_mask,
        test_mask,
        name='gcn',
        **kwargs
    ):
        super(GCN, self).__init__(name=name, **kwargs)
        self.num_units_in_hidden_layers = num_units_in_hidden_layers
        self.num_units_in_output_layer = num_units_in_output_layer
        self.A_norm = A_norm
        self.train_mask = train_mask
        self.test_mask = test_mask

        self.input_layer = GCNLayer(self.num_units_in_hidden_layers[0], self.A_norm, 'relu')
        self.out = GCNLayer(self.num_units_in_output_layer, self.A_norm, 'softmax')


    def call(self, X):
        H = self.input_layer(X)
        Z = self.out(H)
        return Z


    def train_step(self, data):
        x, y = data
        mask = self.train_mask
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Softmax cross-entropy loss with masking
            masked_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
            mask = tf.cast(mask, dtype=tf.float32)
            mask /= tf.reduce_mean(mask)
            masked_loss *= mask
            #masked_loss = tf.reduce_mean(masked_loss)
            loss = masked_loss

            # Accuracy with masking
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
            accuracy_all = tf.cast(correct_prediction, tf.float32)
            accuracy_all *= mask
            metric = accuracy_all

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        metric_tracker.update_state(metric)

        return {'softmax cross-entropy with masking loss': loss_tracker.result(), 'accuracy with masking': metric_tracker.result()}


    @property
    def metrics(self):
        return [loss_tracker, metric_tracker]


    def test_step(self, data):
        x, y = data

        # Cumpote predictions
        y_pred = self(x, training=False)

        # Softmax cross-entropy loss with masking
        mask = self.test_mask
        masked_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        masked_loss *= mask
        loss = masked_loss

        # Accuracy with masking
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        accuracy_all *= mask
        metric = accuracy_all

        loss_tracker.update_state(loss)
        metric_tracker.update_state(metric)

        return {'softmax cross-entropy with masking loss': loss_tracker.result(), 'accuracy with masking': metric_tracker.result()}