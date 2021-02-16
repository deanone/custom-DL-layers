import tensorflow as tf
from preprocessing import *
from custom_gcn import GCN


def ohe_y(y):
	y_ohe = np.zeros((len(y), 2))
	for i in range(len(y)):
		if y[i] == 0:
			y_ohe[i] = np.array([1, 0])
		else:
			y_ohe[i] = np.array([0, 1])
	return y_ohe


def main():
	print('Creating graph...')
	graph_filename = 'data/zachary/zachary'
	g = create_graph_zackary(graph_filename)
	N = g.number_of_nodes()

	print('Creating normalized adjacency matrix...')
	A_norm = create_A_norm(g)

	# Nodes feature matrix
	F = np.eye(N)

	# Real labels - found from original Zachary's paper
	y = ohe_y(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

	# Split data
	F, y_train_masked, train_mask, y_test_masked, test_mask = create_train_test_data(F, y, 0.2)

	# Set up, train and evaluate the custom DNN model
	print('Setting up GCN...')
	num_units_in_hidden_layers = [32, 32]
	num_units_in_output_layer = 2
	gcn = GCN(num_units_in_hidden_layers, num_units_in_output_layer, A_norm, train_mask, test_mask)
	gcn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2))

	print('Training GCN...')
	gcn.fit(F, y_train_masked, epochs=200, batch_size=F.shape[0])	#	we take into account the whole dataset (i.e. node features) in each iteration - i.e. Batch Gradient Descent	
	
	print('Evaluating GCN...')
	evaluation_results = gcn.evaluate(F, y_test_masked, batch_size=F.shape[0])
	
	print('Test Accuracy (%): ', round(evaluation_results[1] * 100.0, 3))


if __name__ == '__main__':
	main()