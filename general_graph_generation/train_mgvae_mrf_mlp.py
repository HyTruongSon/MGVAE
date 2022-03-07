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

from graphs_dataset import graphs_dataset
from snnnet.second_order_models import SnEncoder, SnDecoder, SnVAE, SnNodeChannels
from snnnet.second_order_layers import SnConv2, SnConv2to1, LocalConv2

from mmd import *
from stats import *
import networkx as nx

from utils import *
from data import *
from create_graphs import *

def _parse_args():
    parser = argparse.ArgumentParser(description = 'General graph generation')

    # Train mu
    parser.add_argument('--train_mu', '-train_mu', type = int, default = 1, help = 'Train mu = 1, Fixed mu = 0')

    # Train sigma
    parser.add_argument('--train_sigma', '-train_sigma', type = int, default = 1, help = 'Train sigma = 1, Fixed sigma = 0')

    # L2 norm for mu
    parser.add_argument('--norm_mu', '-norm_mu', type = float, default = 0.001, help = 'L2-norm for mu')

    # L2 norm for sigma
    parser.add_argument('--norm_sigma', '-norm_sigma',  type = float, default = 0.001, help = 'L2-norm for sigma')

    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--graph_type', '-graph_type', type = str, default = 'community', help = 'community/tree/etc.')
    parser.add_argument('--num_communities', '-num_communities', type = int, default = '2', help = 'Number of communities')
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
    parser.add_argument('--pos_weight', type = float, default = 10., help = 'Positive weight')
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


# +-------------------------------+
# | Multilayer Perceptron Decoder |
# +-------------------------------+

class MLP_Decoder(nn.Module):
    def __init__(self, num_atoms, zdim, num_edge_types, num_node_features):
        super(MLP_Decoder, self).__init__()
        self.num_atoms = num_atoms
        self.zdim = zdim
        self.num_edge_types = num_edge_types
        self.num_node_features = num_node_features

        self.layers1 = []
        self.layers1.append(nn.Linear(self.num_atoms * self.zdim, 512))
        self.layers1.append(nn.Sigmoid())
        self.layers1.append(nn.Linear(512, 512))
        self.layers1.append(nn.Sigmoid())
        self.layers1.append(nn.Linear(512, self.num_atoms * self.num_atoms * self.num_edge_types))
        self.fc1 = nn.Sequential(*self.layers1)

        self.layers2 = []
        self.layers2.append(nn.Linear(self.num_atoms * self.zdim, 512))
        self.layers2.append(nn.Sigmoid())
        self.layers2.append(nn.Linear(512, self.num_atoms * self.num_node_features))
        self.fc2 = nn.Sequential(*self.layers2)

    def forward(self, inputs, mask = None):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_atoms * self.zdim))
        adj = self.fc1(inputs)
        node_features = self.fc2(inputs)
        adj = torch.reshape(adj, (batch_size, self.num_atoms, self.num_atoms, self.num_edge_types))
        node_features = torch.reshape(node_features, (batch_size, self.num_atoms, self.num_node_features))
        return adj, node_features


# +-----------------------------------+
# | Filter Nan, Inf out of the tensor |
# +-----------------------------------+

class filter_nan_inf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # Filter for Nan
        nan_filter = (input_tensor != input_tensor)

        # Filter for Inf
        inf_filter_1 = (input_tensor == float('inf'))
        inf_filter_2 = (input_tensor == float('-inf'))

        # Filtering
        output_tensor = input_tensor
        output_tensor[nan_filter] = 0
        output_tensor[inf_filter_1] = 0
        output_tensor[inf_filter_2] = 0

        # Save the tensors for backward pass
        saved_variables = [nan_filter, inf_filter_1, inf_filter_2]
        ctx.save_for_backward(*saved_variables)

        # Return the output tensor
        return output_tensor

    @staticmethod
    def backward(ctx, output_grad):
        # Saved tensors from the forward pass
        nan_filter, inf_filter_1, inf_filter_2 = ctx.saved_variables

        # Input gradient tensor creation
        input_grad = torch.zeros(output_grad.size()).to(device = output_grad.get_device())
        input_grad += output_grad

        # Filtering
        input_grad[nan_filter] = 0
        output_grad[inf_filter_1] = 0
        output_grad[inf_filter_2] = 0

        # Return the input gradient tensor
        return input_grad


# +----------------------+
# | Second-order Encoder |
# +----------------------+

class Sn2ndEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channel_list, out_channels, in_node_channels=0, nonlinearity=None, pdim1=1, skip_conn=True, l2_norm=False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channel_i = in_channels
        in_node_channel_i = in_node_channels

        total = 0
        for out_channel_i in hidden_channel_list:
            self.conv_layers.append(SnConv2(in_channel_i, out_channel_i, in_node_channels=in_node_channel_i, pdim1=pdim1)) # , l2_norm=l2_norm))
            in_channel_i = out_channel_i
            total += out_channel_i
            in_node_channel_i = 0

        if skip_conn == False:
            # First-order encoder
            self.contractive_conv_1d = SnConv2to1(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1) # , l2_norm=l2_norm)

            # Second-order encoder
            self.contractive_conv_2d = SnConv2(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1) # , l2_norm=l2_norm)
        else:
            # First-order encoder
            self.contractive_conv_1d = SnConv2to1(total, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1) # , l2_norm=l2_norm)

            # Second-order encoder
            self.contractive_conv_2d = SnConv2(total, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1) # , l2_norm=l2_norm)

        if nonlinearity is None:
            nonlinearity = nn.ReLU
        self.nonlinearity = nonlinearity()
        self.pdim1 = pdim1

        self.skip_conn = skip_conn
        self.l2_norm = l2_norm

    def forward(self, x, x_node=None, mask=None):
        if self.skip_conn == True:
            x_all = []

        for module in self.conv_layers:
            x = module(x, x_node=x_node, mask=mask)
            x_node = None
            
            # x = self.nonlinearity(x)
            x = torch.tanh(x)

            if self.skip_conn == True:
                x_all.append(x)

        if self.skip_conn == True:
            x = torch.cat(x_all, dim=3)

        x_1d = self.contractive_conv_1d(x, x_node=x_node)
        x_2d = self.contractive_conv_2d(x, x_node=x_node)
        return x_1d, x_2d


# +---------------------+
# | Markov Random Field |
# +---------------------+

class SnMRF(nn.Module):
    """
    Permutation Variational Autoencoder
    """

    def __init__(self, encoder, decoder, N, C, train_mu = True, train_sigma = True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.train_mu = train_mu
        self.train_sigma = train_sigma

        self.params_prior = []

        if self.train_mu == True:
            self.mu_prior = torch.nn.Parameter(torch.randn(N, C, device = device, dtype = torch.float, requires_grad = True))
            self.params_prior.append(self.mu_prior)
        else:
            self.mu_prior = torch.zeros(N, C, device = device, dtype = torch.float, requires_grad = False)

        if self.train_sigma == True:
            self.L_prior = torch.nn.Parameter(torch.randn(N, N, C, device = device, dtype = torch.float, requires_grad = True))
            self.params_prior.append(self.L_prior)
        else:
            self.L_prior = torch.cat([torch.eye(N, device = device, dtype = torch.float, requires_grad = False).unsqueeze(dim = 2) for c in range(C)], dim = 2)

        self.params_prior = torch.nn.ParameterList(self.params_prior)

    def forward(self, x, x_node=None, mask=None):
        # First-order & Second-order Encoder
        mu_encoder, L_encoder = self.encoder(x, x_node, mask)

        mu = mu_encoder.transpose(1, 2)
        L = L_encoder.transpose(2, 3).transpose(1, 2)

        # Sigma
        sigma = torch.matmul(L, L.transpose(2, 3))
        # sigma = torch.matmul(L.transpose(2, 3), L)

        # Reparameterization
        eps = torch.randn(mu.size()).to(device = device)
        # x_sample = mu + torch.einsum('bcij,bcj->bci', sigma, eps)
        x_sample = mu + torch.einsum('bcij,bcj->bci', L, eps)
        x_sample = x_sample.transpose(1, 2)

        # Decoder
        predict = self.decoder(x_sample, mask)

        return predict, mu_encoder, L_encoder


# +--------------------------------------------------+
# | KL-divergence between two multivariate Gaussians |
# +--------------------------------------------------+

def KL_Gaussians(mu_encoder, L_encoder, mu_prior, L_prior):
    mu_0 = mu_encoder.transpose(1, 2)
    L_0 = L_encoder.transpose(2, 3).transpose(1, 2)
    sigma_0 = torch.matmul(L_0, L_0.transpose(2, 3))
    # sigma_0 = torch.matmul(L_0.transpose(2, 3), L_0)

    mu_1 = mu_prior.transpose(1, 2)
    L_1 = L_prior.transpose(2, 3).transpose(1, 2)
    sigma_1 = torch.matmul(L_1, L_1.transpose(2, 3))
    # sigma_1 = torch.matmul(L_1.transpose(2, 3), L_1)

    # Adding noise
    batch_size = mu_encoder.size(0)
    num_nodes = mu_encoder.size(1)
    num_channels = mu_encoder.size(2)

    noise = torch.cat([torch.eye(num_nodes).unsqueeze(dim = 0) for b in range(batch_size)]) * 1e-4
    noise = torch.cat([noise.unsqueeze(dim = 1) for c in range(num_channels)], dim = 1).to(device = device)

    sigma_0 += noise
    sigma_1 += noise

    sigma_1_inverse = torch.inverse(sigma_1)

    A = torch.matmul(sigma_1_inverse, sigma_0)
    A = torch.einsum('bcii->bc', A)

    x = mu_1 - mu_0
    B = torch.einsum('bci,bcij->bcj', x, sigma_1_inverse)
    B = torch.einsum('bcj,bcj->bc', B, x)

    sign_0, logabsdet_0 = torch.slogdet(sigma_0)
    sign_1, logabsdet_1 = torch.slogdet(sigma_1)

    logabsdet_0 = torch.where(torch.isnan(logabsdet_0), torch.zeros_like(logabsdet_0), logabsdet_0)
    logabsdet_0 = torch.where(torch.isinf(logabsdet_0), torch.zeros_like(logabsdet_0), logabsdet_0)

    logabsdet_1 = torch.where(torch.isnan(logabsdet_1), torch.zeros_like(logabsdet_1), logabsdet_1)
    logabsdet_1 = torch.where(torch.isinf(logabsdet_1), torch.zeros_like(logabsdet_1), logabsdet_1)

    # logabsdet_0 = filter_nan_inf.apply(logabsdet_0)
    # logabsdet_1 = filter_nan_inf.apply(logabsdet_1)

    C = logabsdet_1 - logabsdet_0

    N = mu_0.size(2)
    KL = 0.5 * (A + B + C - N)

    return torch.mean(KL)


# +---------------+
# | Loss function |
# +---------------+

def vae_loss_function(recon_x, x, mu_encoder, L_encoder, mu_prior, L_prior, pos_weight, kld_multiplier = 1, norm_mu = 1, norm_sigma = 1):
    """
    Reconstruction + KL divergence losses summed over all elements and batch
    """
    batch_size = x.shape[0]
    recon_x_flattened = recon_x.view(batch_size, -1)
    x_flattened = x.view(batch_size, -1)

    BCE = F.binary_cross_entropy_with_logits(recon_x_flattened, x_flattened, reduction='mean', pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    num_nodes = x.shape[1]
    num_channels = mu_encoder.size(2)

    l2_mu = norm_mu * mu_prior.norm(2)
    l2_sigma = norm_sigma / L_prior.norm(2)

    mu_prior = torch.cat([mu_prior.unsqueeze(dim = 0) for b in range(batch_size)])
    L_prior = torch.cat([L_prior.unsqueeze(dim = 0) for b in range(batch_size)])

    KLD = (kld_multiplier / num_nodes) * KL_Gaussians(mu_encoder, L_encoder, mu_prior, L_prior)

    return BCE + KLD + l2_mu + l2_sigma, BCE, KLD, l2_mu, l2_sigma


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


# Dataset
graphs = create(args)
percent_train = 80
num_train = int(len(graphs) * percent_train / 100)

train_dataset = graphs_dataset(graphs[:num_train],  max_num_nodes = 20)
train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True)

test_dataset = graphs_dataset(graphs[len(graphs)-num_train:], max_num_nodes = 20)
test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = True)

for batch_idx, data in enumerate(test_dataloader):
    max_size = data['adj'].size(1)
    node_dim = data['node_feat'].size(2)
    break
print('Number of input vertex features:', node_dim)


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
encoder_channels = [args.magic_number for layer in range(6)]
latent_channels = 2 * args.magic_number
decoder_channels = [args.magic_number for layer in range(3)]
outer_type = "individual"

encoder = Sn2ndEncoder(1, encoder_channels, latent_channels, in_node_channels = node_dim)

# MLP decoder
decoder = MLP_Decoder(max_size, latent_channels, 1, node_dim)

# Sn decoder
# decoder = SnDecoder(latent_channels, decoder_channels, 1, out_node_channels = node_dim)

train_mu = True
if args.train_mu == 0:
    train_mu = False

train_sigma = True
if args.train_sigma == 0:
    train_sigma = False

model = SnMRF(encoder, decoder, max_size, latent_channels, train_mu = train_mu, train_sigma = train_sigma).to(device = device)
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

    total_loss = 0.0
    total_acc = 0.0
    nBatch = 0

    total_num_batches = len(train_dataloader)
    for _, data in enumerate(train_dataloader):
        adj = data['adj'].unsqueeze(3).to(device = device)
        node_feat = data['node_feat'].float().to(device = device)

        optimizer.zero_grad()
        (adj_rec, node_predict), mu_encoder, L_encoder = model(adj, node_feat)
        loss, BCE, KLD, l2_mu, l2_sigma = vae_loss_function(adj_rec, adj, mu_encoder, L_encoder, model.mu_prior, model.L_prior, pos_weight, norm_mu = args.norm_mu, norm_sigma = args.norm_sigma)
        node_loss = F.mse_loss(node_predict.view(-1), node_feat.view(-1), reduction = 'mean')
        combine_loss = loss + node_loss
        
        if epoch > 0 and combine_loss.item() > 100:
            print('Bad batch')
        else:
            combine_loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        acc = get_acc(torch.sigmoid(adj_rec), adj).item() 
        total_acc += acc
        if nBatch % 10 == 0:
            print('Batch', nBatch, '/', total_num_batches, ': Loss =', loss.item(), ', Node loss =', node_loss.item(), ', Accuracy =', acc)
            LOG.write('Batch ' + str(nBatch) + '/' + str(total_num_batches) + ': Loss = ' + str(loss.item()) + ', Node loss = ' + str(node_loss.item()) + ', Accuracy = ' + str(acc) + '\n')
        nBatch += 1

    avg_loss = total_loss / nBatch
    avg_acc = total_acc / nBatch
    print('Average loss:', avg_loss)
    LOG.write('Average loss: ' + str(avg_loss) + '\n')
    print('Average accuracy:', avg_acc)
    LOG.write('Average accuracy: ' + str(avg_acc) + '\n')
    print("Time =", "{:.5f}".format(time.time() - t))
    LOG.write("Time = " + "{:.5f}".format(time.time() - t) + "\n")


# Save to file
def save_graphs_to_file(graphs_set, file_name):
    file = open(file_name, 'w')
    num_graphs = len(graphs_set)
    file.write(str(num_graphs) + '\n')
    for G in graphs_set:
        A = nx.adjacency_matrix(G).todense()
        num_nodes = A.shape[0]
        file.write(str(num_nodes) + '\n')
        for u in range(num_nodes):
            for v in range(num_nodes):
                file.write(str(A[u, v]) + ' ')
            file.write('\n')
    file.close()


# Evaluation
train_set = []
train_distribution = []
for _, data in enumerate(train_dataloader):
    adj = data['adj'].to(device = device)
    node_feat = data['node_feat'].float().to(device = device)

    adj = adj.detach().cpu().numpy().tolist()
    for sample in adj:
        sample = np.array(sample)
        # Remove the diagonal
        for i in range(sample.shape[0]):
            sample[i, i] = 0
        distribution = np.sort(np.sum(sample, axis = 0))
        train_set.append(nx.from_numpy_matrix(sample))
        train_distribution.append(distribution)

save_graphs_to_file(train_set, args.dir + '/' + args.name + '.train_set')
print('Save train set to file for visualization')


test_set = []
test_distribution = []
for _, data in enumerate(test_dataloader):
    adj = data['adj'].to(device = device)
    node_feat = data['node_feat'].float().to(device = device)

    adj = adj.detach().cpu().numpy().tolist()
    for sample in adj:
        sample = np.array(sample)
        # Remove the diagonal
        for i in range(sample.shape[0]):
            sample[i, i] = 0
        distribution = np.sort(np.sum(sample, axis = 0))
        test_set.append(nx.from_numpy_matrix(sample))
        test_distribution.append(distribution)

save_graphs_to_file(test_set, args.dir + '/' + args.name + '.test_set')
print('Save test set to file for visualization')


gener_set = []
gener_distribution = []
while len(gener_set) < 1024:
    # Generation
    base_latent = torch.randn(32, max_size, args.z_dim).to(device = device)

    batch_size = 32
    mu_prior = torch.cat([model.mu_prior.unsqueeze(dim = 0) for b in range(batch_size)])
    L_prior = torch.cat([model.L_prior.unsqueeze(dim = 0) for b in range(batch_size)])

    mu = mu_prior.transpose(1, 2)
    L = L_prior.transpose(2, 3).transpose(1, 2)

    eps = torch.randn(mu.size()).to(device = device)
    x_sample = mu + torch.einsum('bcij,bcj->bci', L, eps)
    x_sample = x_sample.transpose(1, 2)

    adj_gen, _ = model.decoder(x_sample)
    adj_gen = torch.sigmoid(adj_gen)
    adj_gen = (adj_gen > 0.5).long()

    adj_gen = adj_gen.squeeze(3).float().detach().cpu().numpy().tolist()

    for sample in adj_gen:
        sample = np.array(sample)
        # Remove the diagonal
        for i in range(sample.shape[0]):
            sample[i, i] = 0
        distribution = np.sort(np.sum(sample, axis = 0))
        gener_set.append(nx.from_numpy_matrix(sample))
        gener_distribution.append(distribution)

save_graphs_to_file(gener_set, args.dir + '/' + args.name + '.gener_set')
print('Save the generation set to file for visualization')

orca_path = 'orca'

print("Degree stats (train):", degree_stats(train_set, gener_set))
LOG.write("Degree stats (train): " + str(degree_stats(train_set, gener_set)) + "\n")

print("Degree stats (test):", degree_stats(test_set, gener_set))
LOG.write("Degree stats (test): " + str(degree_stats(test_set, gener_set)) + "\n")

print("Clustering stats (train):", clustering_stats(train_set, gener_set))
LOG.write("Clustering stats (train): " + str(clustering_stats(train_set, gener_set)) + "\n")

print("Clustering stats (test):", clustering_stats(test_set, gener_set))
LOG.write("Clustering stats (test): " + str(clustering_stats(test_set, gener_set)) + "\n")

print("Orbit stats (train):", orbit_stats_all(train_set, gener_set, orca_path))
LOG.write("Orbit stats (train): " + str(orbit_stats_all(train_set, gener_set, orca_path)) + "\n")

print("Orbit stats (test):", orbit_stats_all(test_set, gener_set, orca_path))
LOG.write("Orbit stats (test): " + str(orbit_stats_all(test_set, gener_set, orca_path)) + "\n")


# Matching scheme
degree = 0
clustering = 0
orbit = 0
subsample = []
for i in range(len(test_set)):
    MIN = 1e9
    for j in range(len(gener_set)):
        diff = np.sum(np.abs(test_distribution[i] - gener_distribution[j]))
        if diff < MIN:
            MIN = diff
            match = j
    subsample.append(gener_set[match])
gener_set = subsample

print("Matching - Degree stats (test):", degree_stats(test_set, gener_set))
LOG.write("Matching - Degree stats (test): " + str(degree_stats(test_set, gener_set)) + "\n")

print("Matching - Clustering stats (test):", clustering_stats(test_set, gener_set))
LOG.write("Matching - Clustering stats (test): " + str(clustering_stats(test_set, gener_set)) + "\n")

print("Matching - Orbit stats (test):", orbit_stats_all(test_set, gener_set, orca_path))
LOG.write("Matching - Orbit stats (test): " + str(orbit_stats_all(test_set, gener_set, orca_path)) + "\n")
    
LOG.close()
