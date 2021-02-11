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
	F, y_train_masked, y_test_masked = create_train_test_data(F, y, 1000)

	# Set up, train and evaluate the custom DNN model
	print('Setting up GCN...')
	num_units_in_hidden_layers = [32]
	num_units_in_output_layer = 6
	gcn = GCN(num_units_in_hidden_layers, num_units_in_output_layer, A_norm)
	gcn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

	print('Training GCN...')
	gcn.fit(F, y_train_masked, epochs=5, batch_size=100)
	

	#evaluation_results_custom = dnn_custom.evaluate(x_test, y_test)
	#print('Test MAE (custom DNN): ', evaluation_results_custom[1])



if __name__ == '__main__':
	main()