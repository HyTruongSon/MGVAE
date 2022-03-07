import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def read_adj(file_name):
    file = open(file_name, 'r')
    num_graphs = int(file.readline())
    all_adj = []
    for idx in range(num_graphs):
        num_nodes = int(file.readline())
        A = np.zeros((num_nodes, num_nodes))
        for u in range(num_nodes):
            vect = [float(element) for element in file.readline().strip().split(' ')]
            A[u, :] = vect[:]
        degree = np.sum(A, axis = 0)
        adj = A[degree > 0, :]
        adj = adj[:, degree > 0]
        all_adj.append(nx.from_numpy_matrix(adj))
    file.close()
    return all_adj

def visualize(graph, file_name):
    plt.clf()
    nx.draw(graph, node_color = 'r', node_size = 50)
    plt.savefig(file_name)

# dataset = 'citeseer'
dataset = 'mrf_mlp'
folder = 'visualization'

train_set = read_adj(folder + '/' + dataset + '.train_set')
gener_set = read_adj(folder + '/' + dataset + '.gener_set')

for idx in range(min(100, len(train_set))):
    visualize(train_set[idx], folder + '/' + dataset + '.train_set.' + str(idx) + '.png')

for idx in range(min(100, len(gener_set))):
    visualize(gener_set[idx], folder + '/' + dataset + '.gener_set.' + str(idx) + '.png')

