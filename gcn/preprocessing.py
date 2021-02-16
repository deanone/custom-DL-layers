import numpy as np
import networkx as nx
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_graph_citeseer(graph_filename):
	"""

	:param graph_filename: the name of the file in which the graph is stored
	:type graph_filename: str
	:return: the generated graph object
	:rtype: networkx.classes.graph.Graph

	"""
	f = open(graph_filename)
	lines = f.readlines()
	f.close()

	nodes = []	#	list of integer indexes
	edges = []	#	list of tuples of integer indexes
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


def create_graph_zackary(graph_filename):
	"""
	
	:param graph_filename: the name of the file in which the graph is stored
	:type graph_filename: str
	:return: the generated graph object
	:rtype: networkx.classes.graph.Graph

	"""

	f = open(graph_filename)
	lines = f.readlines()
	f.close()

	nodes = []	#	list of integer indexes
	edges = []	#	list of tuples of integer indexes
	for i, line in enumerate(lines):
		if (i == 0) or (i == 1):
			continue
		else:
			line = line.strip('\n')
			line = line.split(' ')
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


def create_zeros_ones_mask(input_array, l):
	"""

	It creates a mask of length l, which has 1s in the indexes of the input array and 0s elsewere.
	:param input_array: the input array from which the mask is generated
	:type input_array: numpy.ndarray
	:param l: the required length of the generated mask
	:type l: int 
	:return: the generated mask of length l
	:rtype: numpy.ndarray

	"""

	zeros_ones_mask = np.zeros(l)
	zeros_ones_mask[input_array] = 1
	return np.array(zeros_ones_mask, dtype=np.bool)


def create_train_test_data(F, y, num_test_samples):
	"""

	It splits the data into train and test subsets along with their corresponding train and test masks.
	:param F: the data samples
	:type F: numpy.ndarray
	:param y: the data labels
	:type y: numpy.ndarray
	:param num_test_samples: the absolute number or percentage of samples to be considered for testing
	:return: the data samples
	:rtype: numpy.ndarray
	:return: the training labels
	:rtype: numpy.ndarray
	:return: the training mask
	:rtype: numpy.ndarray
	:return: the test labels
	:rtype: numpy.ndarray
	:return: the test mask
	:rtype: numpy.ndarray

	"""

	N = F.shape[0]	#	number of nodes in the graph
	indices = np.arange(N)
	
	X_train, X_test, y_train, y_test, train_mask, test_mask = train_test_split(F, y, indices, test_size=num_test_samples, random_state=42)
	
	y_train_masked = np.zeros(y.shape)
	y_train_masked[train_mask] = y_train

	y_test_masked = np.zeros(y.shape)
	y_test_masked[test_mask] = y_test

	train_mask = create_zeros_ones_mask(train_mask, y.shape[0])
	test_mask = create_zeros_ones_mask(test_mask, y.shape[0])

	return F, y_train_masked, train_mask, y_test_masked, test_mask


def ohe_label_citeseer(label, n_class=6):
	'''

	It one-hot-encodes a label of one data point of the citeseer dataset.
	:param label: a label of one data point
	:type labels: numpy.ndarray
	:param n_class: the number of classes in the citeseer dataset
	:type n_class: int
	:return: the one-hot-encoded vector that corresponds to the input label
	:rtype: numpy.ndarray

	'''

	label_ohe = np.zeros(n_class)
	if label == 'Agents':
		label_ohe[0] = 1
	elif label == 'AI':
		label_ohe[1] = 1
	elif label == 'DB':
		label_ohe[2] = 1
	elif label == 'IR':
		label_ohe[3] = 1
	elif label == 'ML':
		label_ohe[4] = 1
	elif label == 'HCI':
		label_ohe[5] = 1
	return label_ohe


def create_feature_matrix_citeseer(features_filename, nodes):
	num_of_words = 3703
	n_class = 6

	f = open(features_filename)
	lines = f.readlines()
	f.close()

	features_dict = {}
	for node in nodes:
		features_dict[node] = np.zeros(num_of_words)

	labels_dict = {}
	for node in nodes:
		labels_dict[node] = np.zeros(n_class)

	for line in lines:
		line = line.strip('\n')
		line = line.split('\t')
		node_id = line[0]
		node_label = ohe_label_citeseer(line[-1], n_class)
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
