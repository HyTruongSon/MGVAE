import torch
import numpy as np
import json
from torch.utils.data import Dataset

from ogb.graphproppred import GraphPropPredDataset

class molhiv_dataset(Dataset):
    def __init__(self, dataset, split = 'train', max_num_atoms = None):
        self.dataset = dataset
        self.split = split
        split_idx = self.dataset.get_idx_split()
        self.split_idx = split_idx[self.split]
        self.num_molecules = len(self.split_idx)

        if max_num_atoms is None:
            self.max_num_atoms = 0
            for idx in range(len(self.dataset)):
                graph_obj, label = self.dataset[idx]
                num_atoms = graph_obj['num_nodes']
                if num_atoms > self.max_num_atoms:
                    self.max_num_atoms = num_atoms
        else:
            self.max_num_atoms = max_num_atoms

        print('Number of samples:', self.num_molecules)
        print('Maximum number of atoms:', self.max_num_atoms)

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        real_idx = self.split_idx[idx]
        graph_obj, label = self.dataset[real_idx]

        num_atoms = graph_obj['num_nodes']
        assert num_atoms <= self.max_num_atoms
        
        edge_index = graph_obj['edge_index']
        edge_feat = graph_obj['edge_feat']
        node_feat = graph_obj['node_feat']

        # Adjacency
        adj = np.zeros((self.max_num_atoms, self.max_num_atoms))
        adj[edge_index[0, :], edge_index[1, :]] = 1.0
         
        # Edge feature
        edge = np.zeros((self.max_num_atoms, self.max_num_atoms, edge_feat.shape[1]))
        for e in range(edge_index.shape[1]):
            u = int(edge_index[0, e])
            v = int(edge_index[1, e])
            edge[u, v, :] = edge_feat[e, :]

        # To torch tensors
        num_atoms = torch.LongTensor(np.array([int(num_atoms)]))
        edge_feat = torch.Tensor(edge)
        node_feat = self.pad_zeros_features(torch.Tensor(node_feat))
        adj = torch.Tensor(adj)
        
        # Learning target
        label = torch.from_numpy(np.array([label]))
        
        # Sample
        sample = { 
                'num_atoms': num_atoms,
                'edge_feat': edge_feat,
                'node_feat': node_feat,
                'adj': adj,
                'label': label
        }
        return sample

    def pad_zeros_features(self, features):
        if features.size(0) == self.max_num_atoms:
            return features
        num_atomic_features = features.size(1)
        zeros = torch.zeros(self.max_num_atoms - features.size(0), num_atomic_features)
        return torch.cat((features, zeros), dim = 0)

    def pad_zeros_adj(self, adj):
        if adj.size(0) == self.max_num_atoms:
            return adj
        N = adj.size(0)
        zeros = torch.zeros(self.max_num_atoms - N, N)
        adj = torch.cat((adj, zeros), dim = 0)
        zeros = torch.zeros(self.max_num_atoms, self.max_num_atoms - N)
        adj = torch.cat((adj, zeros), dim = 1)
        return adj
