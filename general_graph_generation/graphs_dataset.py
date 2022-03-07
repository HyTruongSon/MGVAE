import torch
import numpy as np
import json
from torch.utils.data import Dataset

import networkx as nx
from utils import *
from data import *
from create_graphs import *

class graphs_dataset(Dataset):
    def __init__(self, graphs, max_num_nodes = 20):
        # Parameters
        self.graphs = graphs
        self.max_num_nodes = max_num_nodes
        self.num_graphs = len(self.graphs)


    '''
    def __init__(self, args, split = 'train', percent = 80, max_num_nodes = 20):
        # Parameters
        self.args = args
        self.split = split
        self.percent = percent

        # Data creation
        self.graphs = create(args)
        num_samples = int(len(self.graphs) * percent / 100)
        if self.split == 'train':
            self.graphs = self.graphs[:num_samples]
        else:
            self.graphs = self.graphs[len(self.graphs)-num_samples:]

        # Number of samples
        self.num_graphs = len(self.graphs)
        print('Number of samples:', self.num_graphs)

        # Maximum number of nodes
        self.max_num_nodes = 0
        self.max_num_edges = 0
        for idx in range(self.num_graphs):
            A = nx.adjacency_matrix(self.graphs[idx]).todense()
            num_nodes = A.shape[0]
            num_edges = np.sum(A)
            if num_nodes > self.max_num_nodes:
                self.max_num_nodes = num_nodes
            if num_edges > self.max_num_edges:
                self.max_num_edges = num_edges
        if self.max_num_nodes < max_num_nodes:
            self.max_num_nodes = max_num_nodes
        print('Maximum number of nodes:', self.max_num_nodes)
        print('Maximum number of edges:', self.max_num_edges)
    '''

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        # The sample
        graph = self.graphs[idx]
        A = nx.adjacency_matrix(graph).todense()
        num_nodes = A.shape[0]
        assert num_nodes <= self.max_num_nodes

        # Adjacency
        adj = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj[:num_nodes, :num_nodes] = A[:, :]
        
        # Add the identity
        # adj += np.eye(self.max_num_nodes)

        # Node degree
        node_feat = np.sum(adj, axis = 0).reshape(self.max_num_nodes, 1)
        
        # To torch tensors
        num_nodes = torch.LongTensor(np.array([int(num_nodes)]))
        node_feat = torch.Tensor(node_feat)
        adj = torch.Tensor(adj)
        
        # Sample
        sample = { 
                'num_nodes': num_nodes,
                'node_feat': node_feat,
                'adj': adj
        }
        return sample

    def pad_zeros_features(self, features):
        if features.size(0) == self.max_num_nodes:
            return features
        num_node_features = features.size(1)
        zeros = torch.zeros(self.max_num_nodes - features.size(0), num_node_features)
        return torch.cat((features, zeros), dim = 0)

    def pad_zeros_adj(self, adj):
        if adj.size(0) == self.max_num_nodes:
            return adj
        N = adj.size(0)
        zeros = torch.zeros(self.max_num_nodes - N, N)
        adj = torch.cat((adj, zeros), dim = 0)
        zeros = torch.zeros(self.max_num_nodes, self.max_num_nodes - N)
        adj = torch.cat((adj, zeros), dim = 1)
        return adj
