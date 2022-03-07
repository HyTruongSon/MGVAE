import torch
import numpy as np
import json
from torch.utils.data import Dataset
from ogb.lsc import PCQM4MDataset
from ogb.utils import smiles2graph
from scipy.sparse import csgraph
from rdkit import Chem

def get_atomic_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # AllChem.EmbedMolecule(mol)
    # conf = mol.GetConformer()
    nAtoms = mol.GetNumAtoms()
    # coords = []
    # for i in range(nAtoms):
    #    coords.append(np.array(conf.GetAtomPosition(i)))
    # coords = np.array(coords).tolist()
    features = []
    for atom in mol.GetAtoms():
        # Atomic features
        feature = []
        # Atomic number
        feature.append(atom.GetAtomicNum())
        # Ring
        if atom.IsInRing() == True:
            feature.append(1)
        else:
            feature.append(0)
        for ring_length in range(1, 10):
            if atom.IsInRingSize(ring_length):
                feature.append(1)
            else:
                feature.append(0)
        # Aromatic
        if atom.GetIsAromatic() == True:
            feature.append(1)
        else:
            feature.append(0)
        # Degree
        feature.append(atom.GetDegree())
        # Explicit valance
        feature.append(atom.GetExplicitValence())
        # Formal charge
        feature.append(atom.GetFormalCharge())
        # Implicit valance
        # feature.append(atom.GetImplicitValence())
        # Isotope
        feature.append(atom.GetIsotope())
        # Mass
        feature.append(atom.GetMass())
        # Implicit Hs
        if atom.GetNoImplicit() == True:
            feature.append(1)
        else:
            feature.append(0)
        # Number of explicit Hs
        feature.append(atom.GetNumExplicitHs())
        # Number of implicit Hs
        feature.append(atom.GetNumImplicitHs())
        # Number of radical electrons
        feature.append(atom.GetNumRadicalElectrons())
        # Total degree
        feature.append(atom.GetTotalDegree())
        # Total number of Hs
        feature.append(atom.GetTotalNumHs())
        # Total valance
        feature.append(atom.GetTotalValence())
        features.append(feature)
    atomic_features = np.array(features).tolist()
    return atomic_features

def get_in_functional_group(mol, fgroup_library):
    nAtoms = mol.GetNumAtoms()
    features = []
    for fgroup in fgroup_library:
        in_fgroup = np.zeros(nAtoms)
        matches = mol.GetSubstructMatches(fgroup)
        matched_atoms = np.array(matches).ravel()
        in_fgroup[matched_atoms.astype('int')] = 1
        features.append(in_fgroup)
    features = np.array(features).transpose()
    return features

def load_fgroup_library(library_loc = 'functional_group_list.txt'):
    with open('functional_group_list.txt') as f:
        functional_group_list = [line for line in f if not line.startswith("#")]
    return functional_group_list

def get_functional_features(smiles, fgroup_library):
    mol = Chem.MolFromSmiles(smiles)
    fgroup_features = get_in_functional_group(mol, fgroup_library)
    return fgroup_features

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

class qm9_dataset(Dataset):
    def __init__(self, directory, on_the_fly = False, max_num_atoms = 9):
        self.directory = directory
        self.smiles_fn = self.directory + '/SMILES'

        # Read the smiles
        self.smiles = read_smiles(self.smiles_fn)
        self.num_molecules = len(self.smiles)
        print('Done reading smiles')
        
        # Read the target
        self.target_names = ['alpha', 'Cv', 'G', 'gap', 'H', 'HOMO', 'LUMO', 'mu', 'omega1', 'R2', 'U', 'U0', 'ZPVE']
        self.targets = []
        for name in self.target_names:
            target = read_target(self.directory + '/targets/' + name)
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

        # Functional group
        self.functional_group_list = load_fgroup_library('functional_group_list.txt')
        self.fgroup_library = [Chem.MolFromSmarts(fgroup) for fgroup in self.functional_group_list]

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        # SMILES
        smiles = self.smiles[idx]

        if self.on_the_fly == True:
            graph_obj = smiles2graph(smiles)
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

        # Atomic features
        atom_feat = get_atomic_features(smiles)

        # Functional features
        func_feat = get_functional_features(smiles, self.fgroup_library)

        # To torch tensors
        num_atoms = torch.LongTensor(np.array([int(num_atoms)]))
        edge_feat = torch.Tensor(edge)
        node_feat = self.pad_zeros_features(torch.Tensor(node_feat))
        atom_feat = self.pad_zeros_features(torch.Tensor(atom_feat))
        func_feat = self.pad_zeros_features(torch.Tensor(func_feat))
        adj = torch.Tensor(adj)
        laplacian = torch.Tensor(laplacian)
        
        # Learning target
        alpha = torch.from_numpy(np.array([self.targets[0][idx]]))
        Cv = torch.from_numpy(np.array([self.targets[1][idx]]))
        G = torch.from_numpy(np.array([self.targets[2][idx]]))
        gap = torch.from_numpy(np.array([self.targets[3][idx]]))
        H = torch.from_numpy(np.array([self.targets[4][idx]]))
        HOMO = torch.from_numpy(np.array([self.targets[5][idx]]))
        LUMO = torch.from_numpy(np.array([self.targets[6][idx]]))
        mu = torch.from_numpy(np.array([self.targets[7][idx]]))
        omega1 = torch.from_numpy(np.array([self.targets[8][idx]]))
        R2 = torch.from_numpy(np.array([self.targets[9][idx]]))
        U = torch.from_numpy(np.array([self.targets[10][idx]]))
        U0 = torch.from_numpy(np.array([self.targets[11][idx]]))
        ZPVE = torch.from_numpy(np.array([self.targets[12][idx]]))

        # Sample
        sample = { 
                'num_atoms': num_atoms,
                'edge_feat': edge_feat,
                'node_feat': node_feat,
                'atom_feat': atom_feat,
                'func_feat': func_feat,
                'adj': adj,
                'laplacian': laplacian,
                'alpha': alpha,
                'Cv': Cv,
                'G': G,
                'gap': gap,
                'H': H,
                'HOMO': HOMO,
                'LUMO': LUMO,
                'mu': mu,
                'omega1': omega1,
                'R2': R2,
                'U': U,
                'U0': U0,
                'ZPVE': ZPVE
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
