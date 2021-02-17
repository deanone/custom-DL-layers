import tensorflow as tf
from preprocessing import *
from gcn import GCN


def main():
	tf.random.set_seed(42)

	print('Creating graph...')
	graph_filename = 'data/zachary/zachary'
	g = graph_zackary(graph_filename)
	N = g.number_of_nodes()

	print('Creating normalized adjacency matrix...')
	A_norm = normalized_adjacency(g)

	# Nodes feature matrix
	F = np.eye(N)

	# Real labels - found from original Zachary's paper
	y = ohe_label_zackary(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

	# Split data
	F, y_train_masked, train_mask, y_test_masked, test_mask = split_train_test_data(F, y, 0.2)

	# Set up, train and evaluate the custom DNN model
	print('Setting up GCN...')
	num_units_in_hidden_layers = [32, 32]
	num_units_in_output_layer = 2
	gcn_model = GCN(num_units_in_hidden_layers, num_units_in_output_layer, A_norm, train_mask, test_mask)
	gcn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2))

	print('Training GCN...')
	gcn_model.fit(F, y_train_masked, epochs=10, batch_size=F.shape[0])	#	we take into account the whole dataset (i.e. node features) in each iteration - i.e. Batch Gradient Descent	
	
	print('Evaluating GCN...')
	evaluation_results_train = gcn_model.evaluate(F, y_train_masked, batch_size=F.shape[0])
	evaluation_results_test = gcn_model.evaluate(F, y_test_masked, batch_size=F.shape[0])

	print('Train Accuracy (%): ', round(evaluation_results_train[1] * 100.0, 3))
	print('Test Accuracy (%): ', round(evaluation_results_test[1] * 100.0, 3))


if __name__ == '__main__':
	main()