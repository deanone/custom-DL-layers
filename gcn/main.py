import tensorflow as tf
from preprocessing import *
from custom_gcn import GCN


def main():

	print('Creating graph...')
	graph_filename = 'data/citeseer/citeseer.cites'
	g = create_graph_citeseer(graph_filename)

	print('Creating normalized adjacency matrix...')
	A_norm = create_A_norm(g)

	print('Loading nodes feature matrix...')
	features_filename = 'data/citeseer/citeseer.content'
	F, y = create_feature_matrix_citeseer(features_filename, list(g.nodes))

	# Split data
	print('Splitting data into training and test subsets...')
	F, y_train_masked, y_test_masked, train_mask, test_mask = create_train_test_data(F, y, 1000)

	# Set up, train and evaluate the custom DNN model
	print('Setting up GCN...')
	num_units_in_hidden_layers = [32]
	num_units_in_output_layer = 6
	gcn = GCN(num_units_in_hidden_layers, num_units_in_output_layer, A_norm, y_train_masked, train_mask)
	gcn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

	print('Training GCN...')
	gcn.fit(F, y_train_masked, epochs=100, batch_size=F.shape[0])	#	we take into account the whole dataset (i.e. node features) in each iteration - i.e. Batch Gradient Descent	
	
	print('Evaluating GCN...')
	evaluation_results_custom = gcn.evaluate(F, y_test_masked, batch_size=F.shape[0])
	
	print('Test Accuracy: ', round(evaluation_results_custom[1] * 100.0, 3))


if __name__ == '__main__':
	main()