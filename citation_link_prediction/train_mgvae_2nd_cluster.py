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
    parser.add_argument('--magic_number', '-m', type = int, default = 8, help = 'To decide the number of channels')
    parser.add_argument('--num_layers', '-nl', type = int, default = 3, help = 'Number of layers')
    parser.add_argument('--num_epoch', '-ne', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--learning_rate', '-l', type = float, default = 0.01, help = 'Initial learning rate')
    parser.add_argument('--kl_loss', '-k', type = int, default = 0, help = 'Use KL divergence loss or not')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--n_clusters', '-n_clusters', type = int, default = 4, help = 'Number of clusters')
    parser.add_argument('--n_levels', '-n_levels', type = int, default = 2, help = 'Number of levels')
    parser.add_argument('--Lambda', '-Lambda', type = int, default = 0.001, help = 'Weight for the multiresolution loss')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
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
np.random.seed(args.seed)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = args.device
print(device)

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

# Clusters
clusters = []
c = 1
for l in range(args.n_levels + 1):
    clusters.append(c)
    c *= args.n_clusters

input_adj = torch.FloatTensor(adj_norm.to_dense()).to(device = device)
features = features.to_dense().to(device = device)

nVertices = input_adj.size(0)
nFeatures = features.size(1)

input_dim = nFeatures 
hidden_dim = 32
z_dim = 16

node_out_channels = args.magic_number
node_hidden_channels = [4 * args.magic_number]
node_out_channels = args.magic_number
encoder_channels = [args.magic_number for layer in range(args.num_layers)]
latent_channels = 2 * args.magic_number
decoder_channels = [args.magic_number for layer in range(args.num_layers)]
outer_type = "individual"

# Sn Multiresolution (cluster) Graph Variational Autoencoder
class MGVAE_2nd_cluster(nn.Module):
    def __init__(self, adj, clusters, input_dim, hidden_dim, z_dim, decoder_channels, outer_type, device = 'cuda'):
        super(MGVAE_2nd_cluster, self).__init__()
        self.adj = adj
        self.clusters = clusters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.decoder_channels = decoder_channels
        self.outer_type = outer_type
        self.device = device

        self.base_encoder = GraphEncoder(self.input_dim, self.hidden_dim, self.z_dim, device = device).to(device = device)
        self.base_decoder = SnDecoder(self.z_dim, self.decoder_channels, 1, outer_type = self.outer_type).to(device = device)

        self.cluster_learner = nn.ModuleList()
        self.global_encoder = nn.ModuleList()
        self.global_decoder = nn.ModuleList()
        for l in range(len(self.clusters)):
            N = self.clusters[l]
            self.cluster_learner.append(GraphCluster(self.z_dim, self.hidden_dim, N, device = device).to(device = device))
            self.global_encoder.append(GraphEncoder(self.z_dim, self.hidden_dim, self.z_dim, device = device).to(device = device))
            self.global_decoder.append(SnDecoder(self.z_dim, self.decoder_channels, 1, outer_type = self.outer_type).to(device = device))

    def forward(self, X):
        outputs = []

        # Base encoder
        base_latent, base_mean, base_logstd = self.base_encoder(self.adj, X)
        
        # Base Sn decoder
        num_vertices = self.adj.size(0)
        base_predict = self.base_decoder(torch.reshape(base_latent, (1, num_vertices, self.z_dim)))
        base_predict = torch.reshape(torch.sigmoid(base_predict), (num_vertices, num_vertices))
        
        outputs.append([base_latent, base_mean, base_logstd, base_predict, self.adj])

        l = len(self.clusters) - 1
        while l >= 0:
            if l == len(self.clusters) - 1:
                prev_adj = self.adj
                prev_latent = base_latent
            else:
                prev_adj = outputs[len(outputs) - 1][4]
                prev_latent = outputs[len(outputs) - 1][0]
                
            # Assignment score
            assign_score = self.cluster_learner[l](prev_adj, prev_latent)

            # Softmax (soft assignment)
            # assign_softmax = F.softmax(assign_score, dim = 1)

            # Gumbel softmax (hard assignment)
            assign_matrix = F.gumbel_softmax(assign_score, tau = 1, hard = True, dim = 1)

            # Print out the cluster assignment matrix
            # print(torch.sum(assign_matrix, dim = 0))

            # Shrinked latent
            shrinked_latent = torch.matmul(assign_matrix.transpose(0, 1), prev_latent)

            # Latent normalization
            shrinked_latent = F.normalize(shrinked_latent, dim = 1)

            # Shrinked adjacency
            shrinked_adj = torch.matmul(torch.matmul(assign_matrix.transpose(0, 1), prev_adj), assign_matrix)

            # Adjacency normalization
            shrinked_adj = shrinked_adj / torch.sum(shrinked_adj)

            # Global encoder
            next_latent, next_mean, next_logstd = self.global_encoder[l](shrinked_adj, shrinked_latent)

            # Global Sn decoder
            num_vertices = shrinked_adj.size(0)
            next_predict = self.global_decoder[l](torch.reshape(next_latent, (1, num_vertices, self.z_dim)))
            next_predict = torch.reshape(torch.sigmoid(next_predict), (num_vertices, num_vertices))

            outputs.append([next_latent, next_mean, next_logstd, next_predict, shrinked_adj])
            l -= 1
 
        return outputs


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, device = 'cuda', **kwargs):
        super(GraphEncoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device

        self.base_net = GraphConvSparse(self.input_dim, self.hidden_dim)
        self.mean_net = GraphConvSparse(self.hidden_dim, self.z_dim, activation=lambda x:x)
        self.logstd_net = GraphConvSparse(self.hidden_dim, self.z_dim, activation=lambda x:x)

    def forward(self, adj, X):
        hidden = self.base_net(adj, X)
        mean = self.mean_net(adj, hidden)
        logstd = self.logstd_net(adj, hidden)
        gaussian_noise = torch.randn(X.size(0), self.z_dim).to(device = self.device)
        latent = gaussian_noise * torch.exp(logstd) + mean
        return latent, mean, logstd


class GraphCluster(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, device = 'cuda', **kwargs):
        super(GraphCluster, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device

        self.base_net = GraphConvSparse(self.input_dim, self.hidden_dim)
        self.assign_net = GraphConvSparse(self.hidden_dim, self.z_dim, activation=lambda x:x)

    def forward(self, adj, X):
        hidden = self.base_net(adj, X)
        assign = self.assign_net(adj, hidden)
        return assign


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.activation = activation

    def forward(self, adj, inputs):
        x = inputs
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        outputs = self.activation(x)
        return outputs


def filter(idx, matrix):
    b = np.full(matrix.size(0), False)
    b[idx] = True
    return matrix[torch.ByteTensor(b)]


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


# init model and optimizer
model = MGVAE_2nd_cluster(input_adj, clusters, input_dim, hidden_dim, z_dim, decoder_channels, outer_type, device = device).to(device=device)
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


def multiresolution_loss(outputs, norm, adj_label, weight_tensor, device, args):
    for i in range(len(outputs)):
        latent, mean, logstd, predict, target = outputs[i]
        if i == 0:
            A_pred = predict
            loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1).to(device = device), weight = weight_tensor)
        else:
            loss += norm * args.Lambda * F.mse_loss(predict.view(-1), target.view(-1).to(device = device))
        if args.kl_loss == 1:
            kl_divergence = 0.5 / predict.size(0) * (1 + 2 * logstd - mean ** 2 - torch.exp(logstd)).sum(1).mean()
            loss -= kl_divergence
    return loss, A_pred


# train model
best_val_roc = 0.0
for epoch in range(args.num_epoch):
    t = time.time()

    outputs = model(features)
    optimizer.zero_grad()
    loss, A_pred = multiresolution_loss(outputs, norm, adj_label, weight_tensor, device, args)
    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred, adj_label.to(device = device))

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu())
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))
    LOG.write("Epoch = " + str(epoch + 1) + ". Train loss = " + str(loss.item()) + ". Train accuracy = " + str(train_acc) + ". Val ROC = " + str(val_roc) + ". Val AP = " + str(val_ap) + ". Time = " + str(time.time() - t) + "\n")
    
    if val_roc > best_val_roc:
        best_val_roc = val_roc
        print("Best validation updated!")
        LOG.write("Best validation updated!\n")

        torch.save(model.state_dict(), model_name)

        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")

        test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
        print("test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))

# Load the best validated model
'''
model.eval()
with torch.no_grad():
    model.load_state_dict(torch.load(model_name))
    outputs = model(features)
    loss, A_pred = multiresolution_loss(outputs, norm, adj_label, weight_tensor, device, args)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
'''

print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))
LOG.write("Test ROC = " + str(test_roc) + "\n")
LOG.write("Test AP = " + str(test_ap) + "\n")
LOG.close()
