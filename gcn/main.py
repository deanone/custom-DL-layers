from preprocessing import *

def main():
	graph_filename = 'data/citeseer/citeseer.cites'
	features_filename = 'data/citeseer/citeseer.content'
	g = create_graph_citeseer(graph_filename)
	A_norm = create_A_norm(g)
	F, y = create_feature_matrix_citeseer(features_filename, list(g.nodes))
	res = A_norm.dot(F)
	print('A normalized: ', A_norm.shape)
	print('Feature matrix: ', F.shape)
	print('Their product: ', res.shape)


if __name__ == '__main__':
	main()