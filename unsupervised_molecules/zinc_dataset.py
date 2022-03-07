import torch
import numpy as np
import json
from torch.utils.data import Dataset

from ogb.lsc import PCQM4MDataset
from ogb.utils import smiles2graph

from scipy.sparse import csgraph

def read_index(file_name):
    file = open(file_name, 'r')
    num_molecules = int(file.readline())
    res = []
    for sample in range(num_molecules):
        value = int(file.readline())
        res.append(value)
    file.close()
    res = np.array(res)
    return res

def read_smiles(file_name):
    file = open(file_name, 'r')
    num_molecules = int(file.readline())
    res = []
    for sample in range(num_molecules):
        value = file.readline()
        res.append(value)
    file.close()
    res = np.array(res)
    return res

def read_target(file_name):
    file = open(file_name, 'r')
    num_molecules = int(file.readline())
    res = []
    for sample in range(num_molecules):
        value = float(file.readline())
        res.append(value)
    file.close()
    res = np.array(res)
    return res

class zinc_dataset(Dataset):
    def __init__(self, directory, on_the_fly = False, max_num_atoms = 38):
        self.directory = directory
        self.smiles_fn = self.directory + '/smiles'

        # Read the smiles
        self.smiles = read_smiles(self.smiles_fn)
        self.num_molecules = len(self.smiles)
        print('Done reading smiles')
        
        # Read the target
        self.target_names = ['logp', 'qed', 'sas']
        self.targets = []
        for name in self.target_names:
            target = read_target(self.directory + '/' + name)
            self.targets.append(target)
            assert target.shape[0] == len(self.smiles)
        print('Done reading targets')

        # Convert smiles into graph
        self.on_the_fly = on_the_fly
        if on_the_fly == False:
            if max_num_atoms is None:
                self.max_num_atoms = 0
            else:
                self.max_num_atoms = max_num_atoms
            self.graph = []
            for sample in range(self.num_molecules):
                graph_obj = smiles2graph(self.smiles[sample])
                self.graph.append(graph_obj)
                if sample % 1000 == 0:
                    print('Done smiles conversion to graphs:', sample)
                num_atoms = graph_obj['num_nodes']
                if max_num_atoms is None:
                    if num_atoms > self.max_num_atoms:
                        self.max_num_atoms = num_atoms
                else:
                    assert num_atoms <= self.max_num_atoms
            print('Complete smiles conversion')
        else:
            assert max_num_atoms is not None
            self.max_num_atoms = max_num_atoms
        print('Maximum number of atoms:', self.max_num_atoms)

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        if self.on_the_fly == True:
            graph_obj = smiles2graph(self.smiles[idx])
        else:
            graph_obj = self.graph[idx]
        num_atoms = graph_obj['num_nodes']
        assert num_atoms <= self.max_num_atoms
        
        edge_index = graph_obj['edge_index']
        edge_feat = graph_obj['edge_feat']
        node_feat = graph_obj['node_feat']

        # Adjacency
        adj = np.zeros((self.max_num_atoms, self.max_num_atoms))
        adj[edge_index[0, :], edge_index[1, :]] = 1.0
        
        # Add the identity
        adj += np.eye(self.max_num_atoms)

        # Graph Laplacian
        laplacian = csgraph.laplacian(adj + np.eye(self.max_num_atoms), normed = True)
        
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
        laplacian = torch.Tensor(laplacian)
        
        # Learning target
        logp = torch.from_numpy(np.array([self.targets[0][idx]]))
        qed = torch.from_numpy(np.array([self.targets[1][idx]]))
        sas = torch.from_numpy(np.array([self.targets[2][idx]]))
        
        # Sample
        sample = { 
                'num_atoms': num_atoms,
                'edge_feat': edge_feat,
                'node_feat': node_feat,
                'adj': adj,
                'laplacian': laplacian,
                'logp': logp,
                'qed': qed,
                'sas': sas
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
