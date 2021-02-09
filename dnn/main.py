import numpy as np
import tensorflow as tf
from custom_dnn import DNN


def main():
	num_units_in_hidden_layers = [8, 16, 16]
	num_units_in_output_layer = 1

	# Generate random data
	np.random.seed(42)
	x_train = np.random.randn(1000, 5)
	y_train = np.random.randn(1000, 1)
	x_test = np.random.randn(200, 5)
	y_test = np.random.randn(200, 1)

	# Set up, train and evaluate the custom DNN model
	dnn_custom = DNN(num_units_in_hidden_layers, num_units_in_output_layer)
	dnn_custom.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	dnn_custom.fit(x_train, y_train, epochs=5, batch_size=1)
	evaluation_results_custom = dnn_custom.evaluate(x_test, y_test)
	print('Test MAE (custom DNN): ', evaluation_results_custom[1])


if __name__ == '__main__':
	main()