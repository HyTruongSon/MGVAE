import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
import pickle
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import argparse

from qm9_dataset import qm9_dataset

from snnnet.second_order_models import SnEncoder, SnDecoder, SnVAE, SnNodeChannels

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Unsupervised feature learning by VAE')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--dataset', '-dataset', type = str, default = 'QM9_smiles', help = 'QM9_smiles/ZINC_smiles')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--kl_loss', '-kl_loss', type = int, default = 0, help = 'Use KL divergence loss or not')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--n_clusters', '-n_clusters', type = int, default = 2, help = 'Number of clusters')
    parser.add_argument('--n_levels', '-n_levels', type = int, default = 3, help = 'Number of levels of resolution')
    parser.add_argument('--n_layers', '-n_layers', type = int, default = 3, help = 'Number of layers of message passing')
    parser.add_argument('--Lambda', '-Lambda', type = float, default = 0.01, help = 'Weight for the multiresolution loss')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 32, help = 'Hidden dimension')
    parser.add_argument('--z_dim', '-z_dim', type = int, default = 32, help = 'Latent dimension')
    parser.add_argument('--pos_weight', type = float, default = 50., help = 'Positive weight')
    parser.add_argument('--magic_number', '-m', type = int, default = 16, help = 'To decide the number of channels')
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


# Multiresolution Graph Variational Autoencoder
class MGVAE_2nd(nn.Module):
    def __init__(self, clusters, num_layers, node_dim, edge_dim, hidden_dim, z_dim, decoder_channels, outer_type, device = 'cuda'):
        super(MGVAE_2nd, self).__init__()
        self.clusters = clusters
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.decoder_channels = decoder_channels
        self.outer_type = outer_type
        self.device = device

        self.base_encoder = GraphEncoder(self.num_layers, self.node_dim, self.edge_dim, self.hidden_dim, self.z_dim, device = device).to(device = device)
        self.base_decoder = SnDecoder(self.z_dim, self.decoder_channels, 1, outer_type = self.outer_type).to(device = device)

        self.cluster_learner = nn.ModuleList()
        self.global_encoder = nn.ModuleList()
        self.global_decoder = nn.ModuleList()
        for l in range(len(self.clusters)):
            N = self.clusters[l]
            self.cluster_learner.append(GraphCluster(2, self.z_dim, self.hidden_dim, N, device = device).to(device = device))
            self.global_encoder.append(GraphEncoder(2, self.z_dim, None, self.hidden_dim, self.z_dim, device = device).to(device = device))
            self.global_decoder.append(SnDecoder(self.z_dim, self.decoder_channels, 1, outer_type = self.outer_type).to(device = device))
        
        self.node_fc1 = nn.Linear(self.z_dim, 128).to(device = device)
        self.node_fc2 = nn.Linear(128, self.node_dim).to(device = device)

    def forward(self, adj, node_feat, edge_feat = None):
        outputs = []

        # Base encoder
        base_latent, base_mean, base_logstd = self.base_encoder(adj, node_feat, edge_feat)

        # Base dot-product decoder for the adjacency
        # base_predict = dot_product_decode(base_latent)
        
        # Base Sn decoder
        batch_size = adj.size(0)
        num_vertices = adj.size(1)
        base_predict = torch.reshape(torch.sigmoid(self.base_decoder(base_latent)), (batch_size, num_vertices, num_vertices))

        # Decoder for the node feature
        node_predict = torch.tanh(self.node_fc1(base_latent))
        node_predict = self.node_fc2(node_predict)

        outputs.append([base_latent, base_mean, base_logstd, base_predict, adj])

        l = len(self.clusters) - 1
        while l >= 0:
            if l == len(self.clusters) - 1:
                prev_adj = adj
                prev_latent = base_latent
            else:
                prev_adj = outputs[len(outputs) - 1][4]
                prev_latent = outputs[len(outputs) - 1][0]

            # Assignment score
            assign_score = self.cluster_learner[l](prev_adj, prev_latent)

            # Softmax (soft assignment)
            # assign_matrix = F.softmax(assign_score, dim = 2)

            # Gumbel softmax (hard assignment)
            assign_matrix = F.gumbel_softmax(assign_score, tau = 1, hard = True, dim = 2)

            # Print out the cluster assignment matrix
            # print(torch.sum(assign_matrix, dim = 0))

            # Shrinked latent
            shrinked_latent = torch.matmul(assign_matrix.transpose(1, 2), prev_latent)

            # Latent normalization
            shrinked_latent = F.normalize(shrinked_latent, dim = 1)

            # Shrinked adjacency
            shrinked_adj = torch.matmul(torch.matmul(assign_matrix.transpose(1, 2), prev_adj), assign_matrix)

            # Adjacency normalization
            shrinked_adj = shrinked_adj / torch.sum(shrinked_adj)

            # Global encoder
            next_latent, next_mean, next_logstd = self.global_encoder[l](shrinked_adj, shrinked_latent)

            # Global dot-product decoder
            # next_predict = dot_product_decode(next_latent)

            # Global Sn decoder
            next_predict = self.global_decoder[l](next_latent)

            outputs.append([next_latent, next_mean, next_logstd, next_predict, shrinked_adj])
            l -= 1

        return outputs, node_predict


class GraphEncoder(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, z_dim, use_concat_layer = True, device = 'cuda', **kwargs):
        super(GraphEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_concat_layer = use_concat_layer
        self.device = device
        
        self.node_fc1 = nn.Linear(self.node_dim, 128).to(device = device)
        self.node_fc2 = nn.Linear(128, self.hidden_dim).to(device = device)

        if self.edge_dim is not None:
            self.edge_fc1 = nn.Linear(self.edge_dim, 128).to(device = device)
            self.edge_fc2 = nn.Linear(128, self.hidden_dim).to(device = device)
        
        self.base_net = nn.ModuleList()
        self.combine_net = nn.ModuleList()
        for layer in range(self.num_layers):
            self.base_net.append(GraphConvSparse(self.hidden_dim, self.hidden_dim, device = self.device).to(device = device))
            if self.edge_dim is not None:
                self.combine_net.append(nn.Linear(2 * self.hidden_dim, self.hidden_dim).to(device = device))

        if self.use_concat_layer == False:
            self.mean_fc1 = nn.Linear(self.hidden_dim, 128).to(device = device)
            self.mean_fc2 = nn.Linear(128, self.z_dim).to(device = device)

            self.logstd_fc1 = nn.Linear(self.hidden_dim, 128).to(device = device)
            self.logstd_fc2 = nn.Linear(128, self.z_dim).to(device = device)
        else:
            self.mean_fc1 = nn.Linear((self.num_layers + 1) * self.hidden_dim, 128).to(device = device)
            self.mean_fc2 = nn.Linear(128, self.z_dim).to(device = device)
        
            self.logstd_fc1 = nn.Linear((self.num_layers + 1) * self.hidden_dim, 128).to(device = device)
            self.logstd_fc2 = nn.Linear(128, self.z_dim).to(device = device)

    def forward(self, adj, node_feat, edge_feat = None):
        node_hidden = torch.tanh(self.node_fc1(node_feat))
        node_hidden = torch.tanh(self.node_fc2(node_hidden))

        if edge_feat is not None and self.edge_dim is not None:
            edge_hidden = torch.tanh(self.edge_fc1(edge_feat))
            edge_hidden = torch.tanh(self.edge_fc2(edge_hidden))

        all_hidden = [node_hidden]
        for layer in range(len(self.base_net)):
            if layer == 0:
                hidden = self.base_net[layer](adj, node_hidden)
            else:
                hidden = self.base_net[layer](adj, hidden)

            if edge_feat is not None and self.edge_dim is not None:
                hidden = torch.cat((hidden, torch.tanh(torch.einsum('bijc,bjk->bik', edge_hidden, hidden))), dim = 2)
                hidden = torch.tanh(self.combine_net[layer](hidden))
            all_hidden.append(hidden)
        
        if self.use_concat_layer:
            hidden = torch.cat(all_hidden, dim = 2)

        mean = torch.tanh(self.mean_fc1(hidden))
        mean = self.mean_fc2(mean)

        logstd = torch.tanh(self.logstd_fc1(hidden))
        logstd = self.logstd_fc2(logstd)

        gaussian_noise = torch.randn(node_feat.size(0), node_feat.size(1), self.z_dim).to(device = self.device)
        latent = gaussian_noise * torch.exp(logstd) + mean
        return latent, mean, logstd


class GraphCluster(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, z_dim, device = 'cuda', **kwargs):
        super(GraphCluster, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device

        self.fc1 = nn.Linear(self.input_dim, 128).to(device = device)
        self.fc2 = nn.Linear(128, self.hidden_dim).to(device = device)

        self.base_net = nn.ModuleList()
        for layer in range(self.num_layers):
            self.base_net.append(GraphConvSparse(self.hidden_dim, self.hidden_dim, device = self.device).to(device = device))

        self.assign_net = GraphConvSparse(self.hidden_dim, self.z_dim, device = self.device).to(device = device)

    def forward(self, adj, X):
        hidden = torch.tanh(self.fc1(X))
        hidden = torch.tanh(self.fc2(hidden))
        for net in self.base_net:
            hidden = net(adj, hidden)
        assign = self.assign_net(adj, hidden)
        return assign


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = torch.tanh, device = 'cuda', **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation
        self.device = device

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
    predict = torch.sigmoid(torch.matmul(Z, Z.transpose(1, 2)))
    return predict


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


def multiresolution_loss(outputs, adj_label, device, pos_weight, args):
    for i in range(len(outputs)):
        latent, mean, logstd, predict, target = outputs[i]
        if i == 0:
            adj_rec = predict
            loss = F.binary_cross_entropy(adj_rec.view(-1), adj_label.view(-1), reduction = 'mean', weight = pos_weight)
            # loss = F.binary_cross_entropy(adj_rec.view(-1), adj_label.view(-1))
        else:
            loss += args.Lambda * F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
        if args.kl_loss == 1:
            kl_divergence = 0.5 / predict.size(0) * (1 + 2 * logstd - mean ** 2 - torch.exp(logstd)).sum(1).mean()
            loss -= kl_divergence
    return loss, adj_rec


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


# Dataset
dataset = qm9_dataset(directory = args.dataset, on_the_fly = True, max_num_atoms = 9)
dataloader = DataLoader(dataset, args.batch_size, shuffle = False)

U0 = []
for batch_idx, data in enumerate(dataloader):
    '''
    print(data['num_atoms'].size())
    print(data['adj'].size())
    print(data['laplacian'].size())
    print(data['node_feat'].size())
    print(data['edge_feat'].size())
    print(data['U0'].size())
    '''

    node_dim = data['node_feat'].size(2)
    atom_dim = data['atom_feat'].size(2)
    edge_dim = data['edge_feat'].size(3)
    func_dim = data['func_feat'].size(2)
    U0.append(data['U0'])
    if batch_idx >= 100:
        break
total_dim = node_dim + atom_dim + func_dim

print('Number of input vertex features:', node_dim)
print('Number of input chemical/atomic features:', atom_dim)
print('Number of functional group features:', func_dim)
print('Total number of input vertex features:', total_dim)
print('Number of input edge features:', edge_dim)


# Clusters
clusters = []
c = 1
for l in range(args.n_levels + 1):
    clusters.append(c)
    c *= args.n_clusters

# init model and optimizer
node_out_channels = args.magic_number
node_hidden_channels = [4 * args.magic_number]
node_out_channels = args.magic_number
encoder_channels = [args.magic_number for layer in range(3)]
latent_channels = 2 * args.magic_number
decoder_channels = [args.magic_number for layer in range(3)]
outer_type = "individual"

model = MGVAE_2nd(clusters = clusters, num_layers = args.n_layers, node_dim = total_dim, edge_dim = None, hidden_dim = args.hidden_dim, z_dim = args.z_dim, decoder_channels = decoder_channels, outer_type = outer_type, device = device).to(device=device)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

# Positive weight
pos_weight = torch.tensor([args.pos_weight]).to(device = device)

# train model
best_acc = 0.0
for epoch in range(args.num_epoch):
    print('--------------------------------------')
    print('Epoch', epoch)
    LOG.write('--------------------------------------\n')
    LOG.write('Epoch ' + str(epoch) + '\n')
    t = time.time()

    # feature_name = args.dir + "/" + args.name + ".epoch." + str(epoch) + ".feature"
    
    feature_name = args.dir + "/" + args.name + ".feature"

    all_latent = None

    total_loss = 0.0
    total_acc = 0.0
    nBatch = 0

    total_num_batches = len(dataloader)
    all_latent = []
    for _, data in enumerate(dataloader):
        adj = data['adj'].to(device = device)
        # laplacian = data['laplacian'].float().to(device = device)
        node_feat = data['node_feat'].float().to(device = device)
        atom_feat = data['atom_feat'].float().to(device = device)
        func_feat = data['func_feat'].float().to(device = device)
        edge_feat = data['edge_feat'].float().to(device = device)
        
        # Concatenate one-hot atomic types, chemical/atomic features and functional group features
        node_feat = torch.cat((node_feat, atom_feat, func_feat), dim = 2)

        # Just to make sure that the sample order is the same
        if nBatch < len(U0):
            assert torch.sum(torch.abs(U0[nBatch] - data['U0'])) < 1e-6

        outputs, node_predict = model(adj, node_feat, edge_feat = None)
        optimizer.zero_grad()
        loss, adj_rec = multiresolution_loss(outputs, adj, device, pos_weight, args)
        node_loss = F.mse_loss(node_predict.view(-1), node_feat.view(-1), reduction = 'mean')
        combine_loss = loss + node_loss
        
        if epoch > 0 and combine_loss.item() > 100:
            print('Bad batch')
        else:
            combine_loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        acc = get_acc(adj_rec, adj).item() 
        total_acc += acc
        if nBatch % 100 == 0:
            print('Batch', nBatch, '/', total_num_batches, ': Loss =', loss.item(), ', Node loss =', node_loss.item(), ', Accuracy =', acc)
            LOG.write('Batch ' + str(nBatch) + '/' + str(total_num_batches) + ': Loss = ' + str(loss.item()) + ', Node loss = ' + str(node_loss.item()) + ', Accuracy = ' + str(acc) + '\n')
        nBatch += 1

        # For saving the features to file
        latent = outputs[0][0].detach().cpu().numpy()
        latent = np.sum(latent, axis = 1)
        all_latent.append(latent)
    
    avg_loss = total_loss / nBatch
    avg_acc = total_acc / nBatch
    print('Average loss:', avg_loss)
    LOG.write('Average loss: ' + str(avg_loss) + '\n')
    print('Average accuracy:', avg_acc)
    LOG.write('Average accuracy: ' + str(avg_acc) + '\n')
    print("Time =", "{:.5f}".format(time.time() - t))
    LOG.write("Time = " + "{:.5f}".format(time.time() - t) + "\n")
    
    if avg_acc > best_acc:
        best_acc = avg_acc
        all_latent = np.concatenate(all_latent, axis = 0)
        np.savetxt(feature_name, all_latent)
        print('Best reconstruction accuracy updated')
        print('Saved the unsupervised features to file')

LOG.close()
