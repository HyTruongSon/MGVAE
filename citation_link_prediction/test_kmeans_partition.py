import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time

from input_data import load_data
from preprocessing import *
import args
import model

# KMeans Partition Tree
from tree import KMeans_Partitions_Tree, Multiresolution_Graph_Targets

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

dataset = 'cora'
print('Dataset:', dataset)
adj, features = load_data(dataset)

print(adj.shape)
print(features.shape)

dim = 10
n_vertices = features.shape[0]
n_clusters = 2
n_levels = 7

# PCA
mean = np.mean(features, axis = 0)
X = features - mean
cov = np.matmul(X.transpose(), X)
u, s, vh = np.linalg.svd(cov, full_matrices = True)
basis = u[:, :dim]
X = np.matmul(X, basis)

# No PCA
'''
X = features.todense()
'''

# KMeans Partitions Tree       
tree = KMeans_Partitions_Tree(features = X, n_clusters = n_clusters, n_levels = n_levels)

# Construct multiresolution targets
targets = Multiresolution_Graph_Targets(adj = adj.todense(), tree = tree)
