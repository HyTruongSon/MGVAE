import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time

from input_data import load_data
from preprocessing import *

import argparse

from snnnet.second_order_models import SnEncoder, SnDecoder, SnVAE, SnNodeChannels

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Citation Graphs')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-n', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--dataset', '-d', type = str, default = 'cora', help = 'cora / citeseer')
    parser.add_argument('--num_epoch', '-ne', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--learning_rate', '-l', type = float, default = 0.01, help = 'Initial learning rate')
    parser.add_argument('--magic_number', '-m', type = int, default = 8, help = 'To decide the number of channels')
    parser.add_argument('--num_layers', '-nl', type = int, default = 1, help = 'Number of layers')
    parser.add_argument('--kl_loss', '-k', type = int, default = 0, help = 'Use KL divergence loss or not')
    parser.add_argument('--seed', '-s', type = int, default = 12345678, help = 'Random seed')
    args = parser.parse_args()
    return args

args = _parse_args()
log_name = args.dir + "/" + args.name + ".log" 
model_name = args.dir + "/" + args.name + ".model"
LOG = open(log_name, "w")

# Fix CPU torch random seed
torch.manual_seed(args.seed)

# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)

# Fix the Numpy random seed
np.random.seed(12345678)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = 'cuda'

adj, features = load_data(args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2]))

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)) 
weight_tensor[weight_mask] = pos_weight

weight_tensor = weight_tensor.to(device = device)

input_adj = torch.FloatTensor(adj_norm.to_dense()).to(device = device)
features = features.to_dense().to(device = device)

nVertices = input_adj.size(0)
nFeatures = features.size(1)

node_out_channels = args.magic_number
node_hidden_channels = [4 * args.magic_number]
node_out_channels = args.magic_number
encoder_channels = [args.magic_number for layer in range(args.num_layers)]
latent_channels = 2 * args.magic_number
decoder_channels = [args.magic_number for layer in range(args.num_layers)]
outer_type = "individual"

input_dim = nFeatures 
hidden1_dim = 32
hidden2_dim = 16

# MGVAE core
class MGVAE_core(nn.Module):
    def __init__(self, adj, nVertices, input_dim, hidden1_dim, hidden2_dim, decoder_channels, outer_type):
        super(MGVAE_core,self).__init__()
        self.nVertices = nVertices
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.decoder_channels = decoder_channels
        self.outer_type = outer_type

        self.base_gcn = GraphConvSparse(self.input_dim, self.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(self.hidden1_dim, self.hidden2_dim, adj, activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(self.hidden1_dim, self.hidden2_dim, adj, activation=lambda x:x)
        self.decoder = SnDecoder(self.hidden2_dim, self.decoder_channels, 1, outer_type=self.outer_type)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.hidden2_dim).to(device = device)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        predict = self.decoder(torch.reshape(Z, (1, self.nVertices, self.hidden2_dim)))
        A_pred = torch.reshape(torch.sigmoid(predict), (self.nVertices, self.nVertices))
        return A_pred

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.matmul(x,self.weight)
        x = torch.matmul(self.adj, x)
        outputs = self.activation(x)
        return outputs

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

# init model and optimizer
model = MGVAE_core(input_adj, nVertices, input_dim, hidden1_dim, hidden2_dim, decoder_channels, outer_type).to(device=device)
optimizer = Adam(model.parameters(), lr=args.learning_rate)


def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# train model
best_val_roc = 0.0
for epoch in range(args.num_epoch):
    t = time.time()

    A_pred = model(features)
    optimizer.zero_grad()
    loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1).to(device = device), weight = weight_tensor)
    if args.kl_loss == 1:
        kl_divergence = 0.5 / A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)).sum(1).mean()
        loss -= kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred, adj_label.to(device = device))

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu())
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))
    LOG.write("Epoch = " + str(epoch + 1) + ". Train loss = " + str(loss.item()) + ". Train accuracy = " + str(train_acc) + ". Val ROC = " + str(val_roc)
        + ". Val AP = " + str(val_ap) + ". Time = " + str(time.time() - t) + "\n")
    
    if val_roc > best_val_roc:
        best_val_roc = val_roc
        print("Best validation updated!")
        LOG.write("Best validation updated!\n")

        torch.save(model.state_dict(), model_name)

        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")
        
        test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
        
        print("Current test ROC = " + str(test_roc) + ". Current Test AP = " + str(test_ap))
        LOG.write("Current test ROC = " + str(test_roc) + ". Current Test AP = " + str(test_ap) + "\n")

print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))
LOG.write("Test ROC = " + str(test_roc) + ". Test AP = " + str(test_ap) + "\n")
LOG.close()

