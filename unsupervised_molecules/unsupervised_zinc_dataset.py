import torch
import numpy as np
import json
from torch.utils.data import Dataset

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

class unsupervised_zinc_dataset(Dataset):
    def __init__(self, directory, features_fn, split = 'new_train.index'):
        self.directory = directory
        self.index_fn = self.directory + '/' + split

        # Read the index
        index_file = open(self.index_fn)
        str = index_file.readline()
        index_file.close()
        self.index = [int(element) for element in str.split(',')]
        self.index = np.array(self.index)
        self.num_molecules = self.index.shape[0]
        print('Done reading indices:', self.num_molecules)
        
        # Read the unsupervised features
        self.features_fn = features_fn
        self.features = np.loadtxt(self.features_fn)
        print('Done reading the unsupervised features')

        # Read the target
        self.target_names = ['logp', 'qed', 'sas']
        self.targets = []
        self.targets_std = []
        for name in self.target_names:
            target = read_target(self.directory + '/' + name)
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
        logp = torch.from_numpy(np.array([self.targets[0][idx]]))
        qed = torch.from_numpy(np.array([self.targets[1][idx]]))
        sas = torch.from_numpy(np.array([self.targets[2][idx]]))

        # Sample
        sample = { 
                'features': torch.Tensor(self.features[idx]),
                'logp': logp,
                'qed': qed,
                'sas': sas
        }
        return sample
