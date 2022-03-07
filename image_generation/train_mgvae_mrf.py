from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, Adagrad
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from mnist_dataset import mnist_dataset

from snnnet.second_order_models import SnEncoder, SnDecoder
from snnnet.second_order_layers import SnConv2, SnConv2to1, LocalConv2

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Image generation')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--dataset', '-dataset', type = str, default = '.', help = 'Dataset')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--cluster_height', '-cluster_height', type = int, default = 2, help = 'Cluster height')
    parser.add_argument('--cluster_width', '-cluster_width', type = int, default = 2, help = 'Cluster width')
    parser.add_argument('--n_levels', '-n_levels', type = int, default = 3, help = 'Number of levels of resolution')
    parser.add_argument('--n_layers', '-n_layers', type = int, default = 3, help = 'Number of layers of message passing')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 32, help = 'Hidden dimension')
    parser.add_argument('--z_dim', '-z_dim', type = int, default = 32, help = 'Latent dimension')
    parser.add_argument('--kl_loss', '-kl_loss', type = int, default = 1, help = 'Use KL divergence or not')
    parser.add_argument('--Lambda', '-Lambda', type = float, default = 0.01, help = 'Lambda')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    parser.add_argument('--limit_data', '-limit_data', type = int, default = 1000, help = 'Limit data')
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


# Create the adjacency matrix
def create_adj(height, width):
    height = int(height)
    width = int(width)
    N = height * width
    adj = np.zeros((N, N))
    DX = [-1, 1, 0, 0, -1, -1, 1, 1]
    DY = [0, 0, -1, 1, -1, 1, -1, 1]
    for i in range(height):
        for j in range(width):
            u = i * width + j
            for t in range(4):
                x = i + DX[t]
                y = j + DY[t]
                if x >= 0 and x < height and y >= 0 and y < width:
                    v = x * width + y
                    adj[u, v] = 1
                    adj[v, u] = 1
    adj = torch.Tensor(adj)
    return adj


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
    def __init__(self, in_channels, hidden_channel_list, out_channels, in_node_channels=0, nonlinearity=None, pdim1=1, skip_conn=False, l2_norm=False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channel_i = in_channels
        in_node_channel_i = in_node_channels
        for out_channel_i in hidden_channel_list:
            self.conv_layers.append(SnConv2(in_channel_i, out_channel_i, in_node_channels=in_node_channel_i, pdim1=pdim1)) # , l2_norm=l2_norm))
            in_channel_i = out_channel_i
            in_node_channel_i = 0

        # First-order encoder
        self.contractive_conv_1d = SnConv2to1(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1) # , l2_norm=l2_norm)

        # Second-order encoder
        self.contractive_conv_2d = SnConv2(in_channel_i, out_channels, in_node_channels=in_node_channel_i, pdim1=pdim1) # , l2_norm=l2_norm)

        if nonlinearity is None:
            nonlinearity = nn.Tanh
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
            x = self.nonlinearity(x)

            if self.skip_conn == True:
                x_all.append(x)

        if self.skip_conn == True:
            x = torch.cat(x_all, dim=3)

        x_1d = self.nonlinearity(self.contractive_conv_1d(x, x_node=x_node))
        x_2d = self.nonlinearity(self.contractive_conv_2d(x, x_node=x_node))
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
            # self.mu_prior = torch.nn.Parameter(torch.zeros(N, C, device = device, dtype = torch.float, requires_grad = True))
            self.params_prior.append(self.mu_prior)
        else:
            self.mu_prior = torch.zeros(N, C, device = device, dtype = torch.float, requires_grad = False)

        if self.train_sigma == True:
            self.L_prior = torch.nn.Parameter(torch.randn(N, N, C, device = device, dtype = torch.float, requires_grad = True))
            self.params_prior.append(self.L_prior)
        else:
            self.L_prior = torch.cat([torch.eye(N, device = device, dtype = torch.float, requires_grad = False).unsqueeze(dim = 2) for c in range(C)], dim = 2)

        self.params_prior = torch.nn.ParameterList(self.params_prior)

    def forward(self, adj, node_features):
        # First-order & Second-order Encoder
        mu_encoder, L_encoder = self.encoder(adj, node_features)

        mu = mu_encoder.transpose(1, 2)
        L = L_encoder.transpose(2, 3).transpose(1, 2)

        # Sigma
        sigma = torch.matmul(L, L.transpose(2, 3))

        # Reparameterization
        eps = torch.randn(mu.size()).to(device = device)
        latent = mu + torch.einsum('bcij,bcj->bci', L, eps)
        latent = latent.transpose(1, 2)

        # Decoder
        predict = self.decoder(latent)

        return latent, mu_encoder, L_encoder, predict


# +-------------------------------+
# | Multilayer Perceptron Decoder |
# +-------------------------------+

class MLP_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 512):
        super(MLP_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.layers = []
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.fc(inputs)


# +--------------------------------------------------+
# | KL-divergence between two multivariate Gaussians |
# +--------------------------------------------------+

def KL_Gaussians(mu_encoder, L_encoder, mu_prior, L_prior):
    # Dimensions
    batch_size = mu_encoder.size(0)
    num_nodes = mu_encoder.size(1)
    num_channels = mu_encoder.size(2)

    mu_0 = mu_encoder.transpose(1, 2)
    L_0 = L_encoder.transpose(2, 3).transpose(1, 2)
    sigma_0 = torch.matmul(L_0, L_0.transpose(2, 3))
    # sigma_0 = torch.matmul(L_0.transpose(2, 3), L_0)

    mu_prior = torch.cat([mu_prior.unsqueeze(dim = 0).clone() for b in range(batch_size)])
    L_prior = torch.cat([L_prior.unsqueeze(dim = 0).clone() for b in range(batch_size)])

    mu_1 = mu_prior.transpose(1, 2)
    L_1 = L_prior.transpose(2, 3).transpose(1, 2)
    sigma_1 = torch.matmul(L_1, L_1.transpose(2, 3))
    # sigma_1 = torch.matmul(L_1.transpose(2, 3), L_1)

    # Adding noise
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


# +-----------------------------------------------+
# | Multiresolution Graph Variational Autoencoder |
# +-----------------------------------------------+

def create_batch_adj(adj, batch_size):
    adj = adj.unsqueeze(dim = 0).unsqueeze(dim = 3)
    adj = torch.cat([adj.clone() for b in range(batch_size)], dim = 0)
    return adj

class MGVAE(nn.Module):
    def __init__(self, image_height, image_width, cluster_height, cluster_width, n_layers, n_levels, input_dim, hidden_dim, z_dim, device = 'cuda', batch_size = 20):
        super(MGVAE, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.cluster_height = cluster_height
        self.cluster_width = cluster_width
        self.n_layers = n_layers
        self.n_levels = n_levels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size

        self.encoder_channels = [self.hidden_dim for idx in range(self.n_layers)]

        # Base adjacency
        self.base_adj = create_adj(image_height, image_width).to(device = device)
        self.base_adj = create_batch_adj(self.base_adj, self.batch_size)

        # Base encoder
        self.base_encoder = Sn2ndEncoder(1, self.encoder_channels, self.z_dim, self.input_dim).to(device = device)

        # Base decoder (MLP)
        self.base_decoder = MLP_Decoder(self.z_dim, self.input_dim).to(device = device)
        
        # Base MRF
        self.base_mrf = SnMRF(self.base_encoder, self.base_decoder, self.image_height * self.image_width, self.z_dim).to(device = device)

        self.local_adj = []
        self.global_adj = []
        self.assign = []
        for l in range(self.n_levels):
            self.local_adj.append([])
            self.global_adj.append([])
            self.assign.append([])

        l = self.n_levels - 1
        h = self.image_height
        w = self.image_width
        while l >= 0:
            self.local_adj[l] = create_adj(self.cluster_height, self.cluster_width).to(device = self.device)
            self.local_adj[l] = create_batch_adj(self.local_adj[l], self.batch_size)

            h = int(h / self.cluster_height)
            w = int(w / self.cluster_width)

            self.global_adj[l] = create_adj(h, w).to(device = self.device)
            self.global_adj[l] = create_batch_adj(self.global_adj[l], self.batch_size)

            self.assign[l] = np.zeros((h * self.cluster_height * w * self.cluster_width, h * w))
            for i in range(h):
                for j in range(w):
                    to_index = i * w + j
                    x0 = i * self.cluster_height
                    y0 = j * self.cluster_width
                    for dx in range(self.cluster_height):
                        for dy in range(self.cluster_width):
                            x = x0 + dx
                            y = y0 + dy
                            from_index = x * w * self.cluster_width + y
                            self.assign[l][from_index, to_index] = 1.0
            self.assign[l] = torch.Tensor(self.assign[l]).to(device = self.device)
            l -= 1

        # Hierarchy
        self.global_encoder = nn.ModuleList()
        self.global_decoder = nn.ModuleList()
        self.global_mrf = nn.ModuleList()

        for l in range(self.n_levels):
            height = self.image_height // pow(self.cluster_height, self.n_levels - l)
            width = self.image_width // pow(self.cluster_width, self.n_levels - l)
            
            encoder = Sn2ndEncoder(1, self.encoder_channels, self.z_dim, self.z_dim).to(device = device)
            decoder = MLP_Decoder(self.z_dim, self.input_dim).to(device = device)
            mrf = SnMRF(encoder, decoder, height * width, self.z_dim).to(device = device)

            self.global_encoder.append(encoder)
            self.global_decoder.append(decoder)
            self.global_mrf.append(mrf)


    def create_targets(self, image, height, width, n_levels):
        image = image.detach().cpu().numpy()
        batch_size = image.shape[0]
        n_channels = image.shape[1]

        targets = []
        for l in range(n_levels):
            targets.append([])
        
        l = n_levels - 1
        while l >= 0:
            new_height = image.shape[2] // height
            new_width = image.shape[3] // width
            target = np.zeros((batch_size, n_channels, new_height, new_width))
            
            for i in range(new_height):
                for j in range(new_width):
                    x1 = i * height
                    y1 = j * width
                    x2 = (i + 1) * height
                    y2 = (j + 1) * width
                    batch = image[:, :, x1:x2, y1:y2]
                    average = np.mean(batch, axis = (2, 3))
                    target[:, :, i, j] = average[:, :]

            image = target
            targets[l] = torch.Tensor(target).transpose(1, 2).transpose(2, 3).to(device = device)

            l -= 1
        
        return targets

    def forward(self, image):
        targets = self.create_targets(image, self.cluster_height, self.cluster_width, self.n_levels)

        outputs = []

        image = image.transpose(1, 2).transpose(2, 3)
        N = image.size(1) * image.size(2)
        node_features = torch.reshape(image, (image.size(0), N, image.size(3)))

        # Base MRF
        base_latent, base_mu_encoder, base_L_encoder, base_predict = self.base_mrf(self.base_adj, node_features)

        outputs.append([base_latent, base_mu_encoder, base_L_encoder, base_predict, image, self.base_mrf.mu_prior, self.base_mrf.L_prior])

        l = self.n_levels - 1
        h = self.image_height
        w = self.image_width

        while l >= 0:
            h = int(h / self.cluster_height)
            w = int(w / self.cluster_width)

            target = torch.reshape(targets[l], (image.size(0), h, w, image.size(3)))
            
            if l == self.n_levels - 1:
                prev_latent = base_latent
                       
            node_features = torch.einsum('ki,bic->bkc', self.assign[l].transpose(0, 1), prev_latent)

            # Global MRF
            next_latent, next_mu_encoder, next_L_encoder, next_predict = self.global_mrf[l](self.global_adj[l], node_features)

            outputs.append([next_latent, next_mu_encoder, next_L_encoder, next_predict, target, self.global_mrf[l].mu_prior, self.global_mrf[l].L_prior])

            prev_latent = next_latent

            l -= 1

        return outputs


# +----------------------+
# | Multiresolution Loss |
# +----------------------+

def multiresolution_loss(outputs, device, args):
    all_rec = []
    all_target = []
    for i in range(len(outputs)):
        latent, mu_encoder, L_encoder, predict, target, mu_prior, L_prior = outputs[i]
        predict = predict.contiguous()
        target = target.contiguous()
        all_target.append(target)
        rec = torch.reshape(predict, target.size())
        all_rec.append(rec)
        if i == 0:
            loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
        else:
            loss += args.Lambda * F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
        if args.kl_loss == 1:
            num_nodes = mu_encoder.size(1)
            if i == 0:
                kl_divergence = KL_Gaussians(mu_encoder, L_encoder, mu_prior, L_prior) / num_nodes
            else:
                kl_divergence += KL_Gaussians(mu_encoder, L_encoder, mu_prior, L_prior) / num_nodes
    if args.kl_loss == 1:
        return loss + kl_divergence, loss, kl_divergence, all_rec, all_target
    return loss, loss, None, all_rec, all_target


# Dataset
if args.dataset == 'cifar10':
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    if args.dataset == 'mnist':
        trainset = mnist_dataset(directory = './mnist', split = 'train', limit_data = args.limit_data)
        testset = mnist_dataset(directory = './mnist', split = 'test', limit_data = args.limit_data)

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers = 2)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size, shuffle = False, num_workers = 2)


for _, data in enumerate(train_dataloader):
    image, label = data
    color_channels = image.size(1)
    image_height = image.size(2)
    image_width = image.size(3)
    break
print('Image height:', image_height)
print('Image width:', image_width)
print('Color channels:', color_channels)

# Model creation
model = MGVAE(image_height = image_height, image_width = image_width, cluster_height = args.cluster_height, cluster_width = args.cluster_width, n_layers = args.n_layers, n_levels = args.n_levels, input_dim = color_channels, hidden_dim = args.hidden_dim, z_dim = args.z_dim, device = device, batch_size = args.batch_size).to(device=device)
optimizer = Adagrad(model.parameters(), lr = args.learning_rate)


# Train model
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
    for batch_idx, data in enumerate(train_dataloader):
        image, label = data
        image = image.to(device = device)

        optimizer.zero_grad()
        outputs = model(image)
        loss, recon_loss, kl_divergence, _, _ = multiresolution_loss(outputs, device, args)
        
        if epoch > 0 and loss.item() > 100:
            print('Bad batch')
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if nBatch % 100 == 0:
            if args.kl_loss == 1:
                print('Batch', nBatch, '/', total_num_batches, ': Loss =', loss.item(), ', Recon loss =', recon_loss.item(), ', KL divergence =', kl_divergence.item())
            else:
                print('Batch', nBatch, '/', total_num_batches, ': Loss =', loss.item())

            LOG.write('Batch ' + str(nBatch) + '/' + str(total_num_batches) + ': Loss = ' + str(loss.item()) + '\n')

        nBatch += 1

    avg_loss = total_loss / nBatch
    print('Average loss:', avg_loss)
    LOG.write('Average loss: ' + str(avg_loss) + '\n')
    print("Time =", "{:.5f}".format(time.time() - t))
    LOG.write("Time = " + "{:.5f}".format(time.time() - t) + "\n")

    # Save the model
    torch.save(model.state_dict(), args.dir + "/" + args.name + "_epoch_" + str(epoch) + ".model")
    print("Model is saved to " + args.dir + "/" + args.name + "_epoch_" + str(epoch) + ".model")

    # Visualization
    model.eval()
    with torch.no_grad():
        idx = 0
        for _, data in enumerate(test_dataloader):
            image, label = data
            image = image.to(device = device)

            outputs = model(image)
            loss, recon_loss, kl_divergence, all_rec, all_target = multiresolution_loss(outputs, device, args)

            for b in range(image.size(0)):
                img = image[b, :, :, :]
                img = img.transpose(0, 1).transpose(1, 2)
                img = img.squeeze()

                '''
                plt.clf()
                plt.imshow(img.detach().cpu())
                plt.savefig(args.dir + "/visualization/Epoch_" + str(epoch) + "_idx_" + str(idx) + "_input.png")
                '''

                for l in range(len(all_rec)):
                    target = all_target[l]
                    target = target[b, :, :, :]

                    plt.clf()
                    if args.dataset == 'mnist':
                        plt.imshow(np.squeeze(target.detach().cpu(), axis = 2), cmap = 'gray')
                    else:
                        plt.imshow(target.detach().cpu())
                    plt.savefig(args.dir + "/visualization/Epoch_" + str(epoch) + "_idx_" + str(idx) + "_target_resolution_" + str(args.n_levels - l) + ".png")

                    rec = all_rec[l]
                    rec = rec[b, :, :, :]

                    plt.clf()
                    if args.dataset == 'mnist':
                        plt.imshow(np.squeeze(rec.detach().cpu(), axis = 2), cmap = 'gray')
                    else:
                        plt.imshow(rec.detach().cpu())
                    plt.savefig(args.dir + "/visualization/Epoch_" + str(epoch) + "_idx_" + str(idx) + "_rec_resolution_" + str(args.n_levels - l) + ".png")
                idx += 1

            if idx >= 40:
                break
    print('Done visualization of reconstruction for the test set')

    # Generation
    model.eval()
    with torch.no_grad():
        all_base_predict = []
        all_predict = []
        l = args.n_levels - 1
        while l >= 0:
            all_predict.append([])
            l -= 1

        for _, data in enumerate(test_dataloader):
            image, label = data
            batch_size = image.size(0)
            image_height = image.size(2)
            image_width = image.size(3)
            color_channels = image.size(1)

            base_latent = torch.randn(batch_size, args.z_dim).to(device = device)
            
            # Base decoder
            mu_prior = torch.cat([model.base_mrf.mu_prior.unsqueeze(dim = 0) for b in range(batch_size)])
            L_prior = torch.cat([model.base_mrf.L_prior.unsqueeze(dim = 0) for b in range(batch_size)])

            mu = mu_prior.transpose(1, 2)
            L = L_prior.transpose(2, 3).transpose(1, 2)

            eps = torch.randn(mu.size()).to(device = device)
            latent = mu + torch.einsum('bcij,bcj->bci', L, eps)
            latent = latent.transpose(1, 2)

            base_predict = model.base_decoder(latent)
            base_predict = torch.reshape(base_predict, (batch_size, image_height, image_width, color_channels)).detach().cpu().numpy()
            all_base_predict.append(base_predict)

            l = args.n_levels - 1
            h = image_height
            w = image_width

            while l >= 0:
                h = int(h / args.cluster_height)
                w = int(w / args.cluster_width)
                
                # Decoder
                mu_prior = torch.cat([model.global_mrf[l].mu_prior.unsqueeze(dim = 0) for b in range(batch_size)])
                L_prior = torch.cat([model.global_mrf[l].L_prior.unsqueeze(dim = 0) for b in range(batch_size)])

                mu = mu_prior.transpose(1, 2)
                L = L_prior.transpose(2, 3).transpose(1, 2)

                eps = torch.randn(mu.size()).to(device = device)
                latent = mu + torch.einsum('bcij,bcj->bci', L, eps)
                latent = latent.transpose(1, 2)
                
                predict = model.global_decoder[l](latent)
                predict = torch.reshape(predict, (batch_size, h, w, color_channels)).detach().cpu().numpy()
                all_predict[l].append(predict)

                l -= 1

            '''
            if len(all_base_predict) >= 100:
                break
            '''

        name = args.dir + "/" + args.name + ".epoch_" + str(epoch)

        all_base_predict = np.concatenate(all_base_predict, axis = 0)
        np.save(name + ".highest_resolution", all_base_predict)

        l = args.n_levels - 1
        while l >= 0:
            all_predict[l] = np.concatenate(all_predict[l], axis = 0)
            np.save(name + ".resolution_level_" + str(l), all_predict[l])
            l -= 1
        print("Done generation")

LOG.close()

'''
for batch_idx, data in enumerate(trainloader):
    image, label = data
    print(image.size())
    image = image[0, :, :, :]
    image = image.transpose(0, 1).transpose(1, 2)
    print(image)
    print(image.size())
    print(label.size())
    plt.clf()
    plt.imshow(image.squeeze())
    plt.savefig("noname_" + str(batch_idx) + ".png")
    if batch_idx == 10:
        break
'''
