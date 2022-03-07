import numpy as np
import random
from mmd import *
from stats import *
import networkx as nx

def Erdos_Renyi(num_graphs, min_size, max_size, prob = 0.5):
	assert min_size <= max_size
	graphs = []
	for index in range(num_graphs):
		size = random.randint(min_size, max_size + 1)
		adj = np.zeros((size, size))
		for i in range(size):
			for j in range(size):
				if i < j:
					value = random.random()
					if value <= prob:
						adj[i, j] = 1
						adj[j, i] = 1
		graph = nx.from_numpy_matrix(adj)
		graphs.append(graph)
	return graphs

er_graphs_1 = Erdos_Renyi(100, 10, 20, 0.9)
er_graphs_2 = Erdos_Renyi(150, 10, 20, 0.9)
er_graphs_3 = Erdos_Renyi(150, 10, 20, 0.9)

# Degree statistics
print("Degree stats (1-1):", degree_stats(er_graphs_1, er_graphs_1))
print("Degree stats (1-2):", degree_stats(er_graphs_1, er_graphs_2))
print("Degree stats (1-3):", degree_stats(er_graphs_1, er_graphs_3))
print("Degree stats (2-3):", degree_stats(er_graphs_2, er_graphs_3))

# Clustering statistics
print("Clustering stats (1-1):", clustering_stats(er_graphs_1, er_graphs_1))
print("Clustering stats (1-2):", clustering_stats(er_graphs_1, er_graphs_2))
print("Clustering stats (1-3):", clustering_stats(er_graphs_1, er_graphs_3))
print("Clustering stats (2-3):", clustering_stats(er_graphs_2, er_graphs_3))

# Orbit statistics
orca_path = 'orca'
print("Orbit stats (1-1):", orbit_stats_all(er_graphs_1, er_graphs_1, orca_path))
print("Orbit stats (1-2):", orbit_stats_all(er_graphs_1, er_graphs_2, orca_path))
print("Orbit stats (1-3):", orbit_stats_all(er_graphs_1, er_graphs_3, orca_path))
print("Orbit stats (2-3):", orbit_stats_all(er_graphs_2, er_graphs_3, orca_path))

# Spectral statistics
print("Spectral stats (1-1):", spectral_stats(er_graphs_1, er_graphs_1))
print("Spectral stats (1-2):", spectral_stats(er_graphs_1, er_graphs_2))
print("Spectral stats (1-3):", spectral_stats(er_graphs_1, er_graphs_3))
print("Spectral stats (2-3):", spectral_stats(er_graphs_2, er_graphs_3))
