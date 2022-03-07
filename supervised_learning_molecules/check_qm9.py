import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch import optim
from torch.utils.data import DataLoader

from qm9_dataset import qm9_dataset

def main():
    train_dataset = qm9_dataset(directory = "QM9_smiles/", split = 'train', test_percent = 10, on_the_fly = True, max_num_atoms = 9)
    test_dataset = qm9_dataset(directory = "QM9_smiles/", split = 'test', test_percent = 10, on_the_fly = True, max_num_atoms = 9)
    
    train_dataloader = DataLoader(train_dataset, 20, shuffle = True)
    test_dataloader = DataLoader(test_dataset, 20, shuffle = False)

    for batch_idx, data in enumerate(train_dataloader):
        print(data['num_atoms'])
        print(data['adj'])
        print(data['laplacian'])
        print(data['node_feat'])
        print(data['edge_feat'])
        print(data['alpha'])
        break

    for batch_idx, data in enumerate(test_dataloader):
        print(data['num_atoms'].size())
        print(data['adj'].size())
        print(data['laplacian'].size())
        print(data['node_feat'].size())
        print(data['edge_feat'].size())
        print(data['alpha'].size())
        break

if __name__ == "__main__":
    main()
