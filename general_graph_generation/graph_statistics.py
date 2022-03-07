import numpy as np 

def graph_degree(graph):
	return np.sum(graph) / graph.shape[0]

def graph_statistics(graphs):
	degree = []
	num_graphs = len(graphs)
	for index in range(num_graphs):
		graph = graphs[index]
		degree.append(np.array([graph_degree(graph)]))
	dict = {'degree': degree}
	return dict