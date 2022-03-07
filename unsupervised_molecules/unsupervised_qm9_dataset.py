import torch
import numpy as np
import json
from torch.utils.data import Dataset

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

class unsupervised_qm9_dataset(Dataset):
    def __init__(self, directory, features_fn, split = 'train', test_percent = '10'):
        self.directory = directory
        self.index_fn = self.directory + '/splits/' + str(test_percent) + '.' + split + '.index'

        # Read the index
        self.index = read_index(self.index_fn)
        self.num_molecules = self.index.shape[0]
        print('Done reading indices')
        
        # Read the unsupervised features
        self.features_fn = features_fn
        self.features = np.loadtxt(self.features_fn)
        print('Done reading the unsupervised features')

        # Read the target
        self.target_names = ['alpha', 'Cv', 'G', 'gap', 'H', 'HOMO', 'LUMO', 'mu', 'omega1', 'R2', 'U', 'U0', 'ZPVE']
        self.targets = []
        self.targets_std = []
        for name in self.target_names:
            target = read_target(self.directory + '/targets/' + name)
            self.targets.append(target)
            std = np.std(target)
            self.targets_std.append(std)
            print(name, ': std = ', std)
        print('Done reading targets')

         # Extract the split
        self.features = self.features[self.index]
        assert self.features.shape[0] == self.num_molecules
        for i in range(len(self.targets)):
            self.targets[i] = self.targets[i][self.index]
            assert self.targets[i].shape[0] == self.num_molecules
        print('Done extracting split')

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

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
                'features': torch.Tensor(self.features[idx]),
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
