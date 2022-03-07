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

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Supervised learning')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--learning_target', '-learning_target', type = str, default = 'U0', help = 'Learning target')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--dataset', '-dataset', type = str, default = 'cora', help = 'cora / citeseer')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--n_clusters', '-n_clusters', type = int, default = 2, help = 'Number of clusters')
    parser.add_argument('--n_levels', '-n_levels', type = int, default = 3, help = 'Number of levels of resolution')
    parser.add_argument('--n_layers', '-n_layers', type = int, default = 3, help = 'Number of layers of message passing')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 32, help = 'Hidden dimension')
    parser.add_argument('--z_dim', '-z_dim', type = int, default = 32, help = 'Latent dimension')
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
class MGVAE(nn.Module):
    def __init__(self, clusters, num_layers, node_dim, edge_dim, hidden_dim, z_dim, device = 'cuda'):
        super(MGVAE, self).__init__()
        self.clusters = clusters
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device

        self.base_encoder = GraphEncoder(self.num_layers, self.node_dim, self.edge_dim, self.hidden_dim, self.z_dim, device = device).to(device = device)
        
        self.cluster_learner = nn.ModuleList()
        self.global_encoder = nn.ModuleList()
        for l in range(len(self.clusters)):
            N = self.clusters[l]
            self.cluster_learner.append(GraphCluster(self.num_layers, self.z_dim, self.hidden_dim, N, device = device).to(device = device))
            self.global_encoder.append(GraphEncoder(self.num_layers, self.z_dim, None, self.hidden_dim, self.z_dim, device = device).to(device = device))
        
        D = self.z_dim * (len(self.clusters) + 1)
        self.fc1 = nn.Linear(D, 256).to(device = device)
        self.fc2 = nn.Linear(256, 1).to(device = device)

    def forward(self, adj, node_feat, edge_feat = None):
        outputs = []

        # Base encoder
        base_latent = self.base_encoder(adj, node_feat, edge_feat)

        outputs.append([base_latent, adj])

        l = len(self.clusters) - 1
        while l >= 0:
            if l == len(self.clusters) - 1:
                prev_adj = adj
                prev_latent = base_latent
            else:
                prev_adj = outputs[len(outputs) - 1][1]
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
            next_latent = self.global_encoder[l](shrinked_adj, shrinked_latent)

            outputs.append([next_latent, shrinked_adj])
            l -= 1

        # Scalar prediction
        latent = torch.cat([torch.sum(output[0], dim = 1) for output in outputs], dim = 1)
        hidden = torch.tanh(self.fc1(latent))
        predict = self.fc2(hidden)

        return predict, latent, outputs


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
        
        self.node_fc1 = nn.Linear(self.node_dim, 256).to(device = device)
        self.node_fc2 = nn.Linear(256, self.hidden_dim).to(device = device)
        
        if self.edge_dim is not None:
            self.edge_fc1 = nn.Linear(self.edge_dim, 256).to(device = device)
            self.edge_fc2 = nn.Linear(256, self.hidden_dim).to(device = device)

        self.base_net = nn.ModuleList()
        self.combine_net = nn.ModuleList()
        for layer in range(self.num_layers):
            self.base_net.append(GraphConvSparse(self.hidden_dim, self.hidden_dim, device = self.device).to(device = device))
            if self.edge_dim is not None:
                self.combine_net.append(nn.Linear(2 * self.hidden_dim, self.hidden_dim).to(device = device))

        if self.use_concat_layer == True:
            self.latent_fc1 = nn.Linear((self.num_layers + 1) * self.hidden_dim, 256).to(device = device)
            self.latent_fc2 = nn.Linear(256, self.z_dim).to(device = device)
        else:
            self.latent_fc1 = nn.Linear(self.hidden_dim, 256).to(device = device)
            self.latent_fc2 = nn.Linear(256, self.z_dim).to(device = device)

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
        
        if self.use_concat_layer == True:
            hidden = torch.cat(all_hidden, dim = 2)

        latent = torch.tanh(self.latent_fc1(hidden))
        latent = torch.tanh(self.latent_fc2(latent))
        return latent


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

        # Option 1: Learnable clustering
        # self.base_net = nn.ModuleList()
        
        # Option 2: Fixed clustering
        self.base_net = []

        for layer in range(self.num_layers):
            self.base_net.append(GraphConvSparse(self.hidden_dim, self.hidden_dim, device = self.device).to(device = device))

        self.assign_net = GraphConvSparse(self.hidden_dim, self.z_dim, device = self.device).to(device = device)

    def forward(self, adj, X):
        hidden = torch.sigmoid(self.fc1(X))
        hidden = torch.sigmoid(self.fc2(hidden))
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
        # If we use the adjacency matrix instead of the graph Laplacian
        B = adj.size(0)
        N = adj.size(1)
        e = torch.eye(N)
        e = e.reshape((1, N, N))
        eye = e.repeat(B, 1, 1).to(device = self.device)
        D = 1.0 / torch.sum(adj + eye, dim = 2)
        adj = torch.einsum('bi,bij->bij', (D, adj))

        x = inputs
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        outputs = self.activation(x)
        return outputs


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

# Dataset
train_dataset = qm9_dataset(directory = args.dataset, split = 'train', test_percent = 10, on_the_fly = True, max_num_atoms = 9)
test_dataset = qm9_dataset(directory = args.dataset, split = 'test', test_percent = 10, on_the_fly = True, max_num_atoms = 9)

train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = False)

for batch_idx, data in enumerate(train_dataloader):
    print(data['num_atoms'].size())
    print(data['adj'].size())
    print(data['laplacian'].size())
    print(data['node_feat'].size())
    print(data['edge_feat'].size())
    print(data[args.learning_target].size())

    node_dim = data['node_feat'].size(2)
    edge_dim = data['edge_feat'].size(3)
    break
print('Number of input vertex features:', node_dim)
print('Number of input edge features:', edge_dim)

# Clusters
clusters = []
c = 1
for l in range(args.n_levels + 1):
    clusters.append(c)
    c *= args.n_clusters

# Init model and optimizer
model = MGVAE(clusters = clusters, num_layers = args.n_layers, node_dim = node_dim, edge_dim = None, hidden_dim = args.hidden_dim, z_dim = args.z_dim, device = device).to(device=device)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

# train model
best_mae = 1e9
for epoch in range(args.num_epoch):
    print('--------------------------------------')
    print('Epoch', epoch)
    LOG.write('--------------------------------------\n')
    LOG.write('Epoch ' + str(epoch) + '\n')

    # Training
    t = time.time()
    total_loss = 0.0
    nBatch = 0
    for batch_idx, data in enumerate(train_dataloader):
        adj = data['adj'].to(device = device)
        # laplacian = data['laplacian'].float().to(device = device)
        node_feat = data['node_feat'].float().to(device = device)
        edge_feat = data['edge_feat'].float().to(device = device)
        target = data[args.learning_target].float().to(device = device)

        predict, latent, outputs = model(adj, node_feat, edge_feat = None)
        optimizer.zero_grad()
        
        # Mean squared error loss
        # loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
        
        # L1 loss
        loss = F.l1_loss(predict.view(-1), target.view(-1), reduction = 'mean')
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        nBatch += 1
        if batch_idx % 100 == 0:
            print('Batch', batch_idx, '/', len(train_dataloader),': Loss =', loss.item())
            LOG.write('Batch ' + str(batch_idx) + '/' + str(len(train_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')
    
    avg_loss = total_loss / nBatch
    print('Train average loss:', avg_loss)
    LOG.write('Train average loss: ' + str(avg_loss) + '\n')
    print("Train time =", "{:.5f}".format(time.time() - t))
    LOG.write("Train time = " + "{:.5f}".format(time.time() - t) + "\n")

    # Testing
    t = time.time()
    model.eval()
    with torch.no_grad():
        sum_error = 0.0
        num_samples = 0
        for batch_idx, data in enumerate(test_dataloader):
            adj = data['adj'].to(device = device)
            # laplacian = data['laplacian'].float().to(device = device)
            node_feat = data['node_feat'].float().to(device = device)
            edge_feat = data['edge_feat'].float().to(device = device)
            target = data[args.learning_target].float().to(device = device)
            
            predict, latent, outputs = model(adj, node_feat, edge_feat)
            sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
            num_samples += adj.size(0)
        mae = sum_error / num_samples
        
        print('Test MAE:', mae)
        LOG.write('Test MAE: ' + str(mae) + '\n')
        print("Test time =", "{:.5f}".format(time.time() - t))
        LOG.write("Test time = " + "{:.5f}".format(time.time() - t) + "\n")
    
    if mae < best_mae:
        best_mae = mae
        print('Current best MAE updated:', best_mae)
        LOG.write('Current best MAE updated: ' + str(best_mae) + '\n')
        
        torch.save(model.state_dict(), model_name)

        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")

print('Best MAE:', best_mae)
LOG.write('Best MAE: ' + str(best_mae) + '\n')
LOG.close()
