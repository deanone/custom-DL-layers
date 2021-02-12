import numpy as np
import networkx as nx
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_train_test_data(F, y, num_test_samples):
	N = F.shape[0]	#	number of nodes in the graph
	indices = np.arange(N)
	
	X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(F, y, indices, test_size=num_test_samples, random_state=42)
	
	y_train_masked = np.zeros(y.shape)
	y_train_masked[idx_train] = y_train

	y_test_masked = np.zeros(y.shape)
	y_test_masked[idx_test] = y_test

	return F, y_train_masked, y_test_masked


def create_graph_citeseer(graph_filename):
	f = open(graph_filename)
	lines = f.readlines()
	f.close()

	nodes = []
	edges = []
	for line in lines:
		line = line.strip('\n')
		line = line.split('\t')
		node_a = line[0]
		node_b = line[1]
		if node_a not in nodes:
			nodes.append(node_a)
		if node_b not in nodes:
			nodes.append(node_b)
		edge = (node_a, node_b)
		if edge not in edges:
			edges.append(edge)
	g = nx.Graph()
	g.add_nodes_from(nodes)
	g.add_edges_from(edges)
	return g


def str_to_ndarray_label_citeseer(label_str, num_of_classes):
	label_ndarray = np.zeros(num_of_classes)
	if label_str == 'Agents':
		label_ndarray[0] = 1
	elif label_str == 'AI':
		label_ndarray[1] = 1
	elif label_str == 'DB':
		label_ndarray[2] = 1
	elif label_str == 'IR':
		label_ndarray[3] = 1
	elif label_str == 'ML':
		label_ndarray[4] = 1
	elif label_str == 'HCI':
		label_ndarray[5] = 1
	return label_ndarray


def create_feature_matrix_citeseer(features_filename, nodes):
	num_of_words = 3703
	num_of_classes = 6

	f = open(features_filename)
	lines = f.readlines()
	f.close()

	features_dict = {}
	for node in nodes:
		features_dict[node] = np.zeros(num_of_words)

	labels_dict = {}
	for node in nodes:
		labels_dict[node] = np.zeros(num_of_classes)

	for line in lines:
		line = line.strip('\n')
		line = line.split('\t')
		node_id = line[0]
		node_label = str_to_ndarray_label_citeseer(line[-1], num_of_classes)
		line = line[1:]
		line = line[:-1]
		line = [int(x) for x in line]
		if node_id in nodes:
			features_dict[node_id] = line
			labels_dict[node_id] = node_label

	F = np.array(list(features_dict.values()))
	F = F.astype('float32')
	y = np.array(list(labels_dict.values()))
	y = y.astype('float32')
	return F, y


def create_A_norm(g):
	# Adjacency matrix
	A = nx.adjacency_matrix(g)
	A = A.toarray()

	# Degree matrix
	D = np.diag(np.sum(A, axis=1))
	
	# Normalized degree matrix 
	D_frac_pow = fractional_matrix_power(D, -0.5)

	# Normalized A matrix
	A_norm = D_frac_pow.dot(A).dot(D_frac_pow)
	A_norm = A_norm.astype('float32')

	return A_norm
