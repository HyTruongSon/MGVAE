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

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Supervised learning')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
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


# Multilayer Perceptron Decoder
class MLP_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = []
        self.layers.append(nn.Linear(self.input_dim, 512))
        self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(512, 512))
        self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(512, self.output_dim))
        self.layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.fc(inputs)


# Multiresolution Graph Variational Autoencoder
class MGVAE(nn.Module):
    def __init__(self, image_height, image_width, cluster_height, cluster_width, n_layers, n_levels, input_dim, hidden_dim, z_dim, device = 'cuda'):
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

        # Base
        self.base_encoder = GraphEncoder(self.n_layers, self.input_dim, None, self.hidden_dim, self.z_dim, device = device).to(device = device)
        self.base_decoder = MLP_Decoder(self.z_dim, self.input_dim).to(device = device)
        self.base_adj = create_adj(image_height, image_width).to(device = device)

        # Hierarchy
        self.local_encoder = nn.ModuleList()
        self.local_decoder = nn.ModuleList()
        self.local_pooler = nn.ModuleList()
        self.global_decoder = nn.ModuleList()
        
        self.local_adj = []
        self.global_adj = []

        for l in range(self.n_levels):
            # self.local_encoder.append(GraphEncoder(self.n_layers, self.z_dim, None, self.hidden_dim, self.z_dim, device = device).to(device = device))
            # self.local_decoder.append(MLP_Decoder(self.z_dim, self.input_dim).to(device = device))
            self.local_pooler.append(GraphPooler(self.n_layers, self.z_dim, self.hidden_dim, self.z_dim, device = device).to(device = device))
            self.global_decoder.append(MLP_Decoder(self.z_dim, self.input_dim).to(device = device))

            self.local_adj.append([])
            self.global_adj.append([])

        l = self.n_levels - 1
        h = self.image_height
        w = self.image_width

        self.assign = []
        for l in range(self.n_levels):
            self.assign.append([])
        
        while l >= 0:
            self.local_adj[l] = create_adj(self.cluster_height, self.cluster_width).to(device = self.device)
            h = int(h / self.cluster_height)
            w = int(w / self.cluster_width)
            self.global_adj[l] = create_adj(h, w).to(device = self.device)
            
            self.assign[l] = np.zeros((h, w, self.cluster_height * self.cluster_width))
            for i in range(h):
                for j in range(w):
                    x0 = i * self.cluster_height
                    y0 = j * self.cluster_width
                    for dx in range(self.cluster_height):
                        for dy in range(self.cluster_width):
                            idx = dx * self.cluster_width + dy
                            x = x0 + dx
                            y = y0 + dy
                            v = x * w * self.cluster_width + y
                            self.assign[l][i, j, idx] = v
                    # print(self.assign[l][i, j, :])
            l -= 1

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

        # Base encoder
        base_latent, base_mean, base_logvar = self.base_encoder(self.base_adj, node_features)
        
        # Base decoder
        base_predict = self.base_decoder(base_latent)

        outputs.append([base_latent, base_mean, base_logvar, base_predict, image])

        l = self.n_levels - 1
        h = self.image_height
        w = self.image_width

        while l >= 0:
            h = int(h / self.cluster_height)
            w = int(w / self.cluster_width)

            target = torch.reshape(targets[l], (image.size(0), h, w, image.size(3)))
            
            all_latent = []
            all_mean = []
            all_logvar = []

            batch_size = image.size(0)
            local_adj = torch.cat([self.local_adj[l].unsqueeze(dim = 0) for b in range(batch_size)], dim = 0)
            global_adj = torch.cat([self.global_adj[l].unsqueeze(dim = 0) for b in range(batch_size)], dim = 0)

            for i in range(h):
                for j in range(w):
                    idx = self.assign[l][i, j, :]
                    assign = np.zeros((self.cluster_height * self.cluster_width, h * self.cluster_height * w * self.cluster_width))
                    for v in range(self.cluster_height * self.cluster_width):
                        assign[v, int(idx[v])] = 1.0
                    assign = torch.Tensor(assign).to(device = self.device)
                    
                    if l == self.n_levels - 1:
                        node_features = torch.einsum('mn,bnk->bmk', assign, base_latent)
                    else:
                        node_features = torch.einsum('mn,bnk->bmk', assign, prev_latent)
                    
                    # Local encoder
                    # local_latent, local_mean, local_logvar = self.local_encoder[l](local_adj, node_features)

                    # Local decoder
                    # local_predict = self.local_decoder[l](local_latent)

                    # Local pooler
                    local_latent, local_mean, local_logvar = self.local_pooler[l](local_adj, node_features)

                    all_latent.append(local_latent)
                    all_mean.append(local_mean)
                    all_logvar.append(local_logvar)

            all_latent = torch.cat(all_latent, dim = 1)
            all_mean = torch.cat(all_mean, dim = 1)
            all_logvar = torch.cat(all_logvar, dim = 1)

            # Global decoder
            global_predict = self.global_decoder[l](all_latent)
           
            outputs.append([all_latent, all_mean, all_logvar, global_predict, target])

            prev_latent = all_latent

            l -= 1

        return outputs

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
                self.combine_net.append(nn.Linear(2 * self.hidden_dim, self.hidden_dim, device = self.device).to(device = device))

        if self.use_concat_layer == False:
            self.mean_fc1 = nn.Linear(self.hidden_dim, 128).to(device = device)
            self.mean_fc2 = nn.Linear(128, self.z_dim).to(device = device)

            self.logvar_fc1 = nn.Linear(self.hidden_dim, 128).to(device = device)
            self.logvar_fc2 = nn.Linear(128, self.z_dim).to(device = device)
        else:
            self.mean_fc1 = nn.Linear((self.num_layers + 1) * self.hidden_dim, 128).to(device = device)
            self.mean_fc2 = nn.Linear(128, self.z_dim).to(device = device)

            self.logvar_fc1 = nn.Linear((self.num_layers + 1) * self.hidden_dim, 128).to(device = device)
            self.logvar_fc2 = nn.Linear(128, self.z_dim).to(device = device)

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

        logvar = torch.tanh(self.logvar_fc1(hidden))
        logvar = self.logvar_fc2(logvar)

        gaussian_noise = torch.randn(node_feat.size(0), node_feat.size(1), self.z_dim).to(device = self.device)
        latent = gaussian_noise * torch.exp(0.5 * logvar) + mean
        return latent, mean, logvar


class GraphPooler(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, z_dim, device = 'cuda', **kwargs):
        super(GraphPooler, self).__init__(**kwargs)
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
        
        self.mean_net = GraphConvSparse(self.hidden_dim, self.z_dim, activation = F.tanh)
        self.logvar_net = GraphConvSparse(self.hidden_dim, self.z_dim, activation = F.tanh)
    
    def forward(self, adj, X):
        hidden = torch.tanh(self.fc1(X))
        hidden = torch.tanh(self.fc2(hidden))
        for net in self.base_net:
            hidden = net(adj, hidden)
        mean = self.mean_net(adj, hidden)
        logvar = self.logvar_net(adj, hidden)
        pooled_mean = torch.mean(mean, dim = 1).unsqueeze(dim = 1)
        pooled_logvar = torch.mean(logvar, dim = 1).unsqueeze(dim = 1)
        batch_size = mean.size(0)
        gaussian_noise = torch.randn(batch_size, 1, self.z_dim).to(device = self.device)
        latent = gaussian_noise * torch.exp(0.5 * pooled_logvar) + pooled_mean
        return latent, pooled_mean, pooled_logvar


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


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


def multiresolution_loss(outputs, device, args):
    all_rec = []
    all_target = []
    for i in range(len(outputs)):
        latent, mean, logvar, predict, target = outputs[i]
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
            kl_divergence = 0.5 * (1 + logvar - mean ** 2 - torch.exp(logvar)).sum(1).mean()
            loss -= kl_divergence
    return loss, all_rec, all_target


# Dataset
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
model = MGVAE(image_height = image_height, image_width = image_width, cluster_height = args.cluster_height, cluster_width = args.cluster_width, n_layers = args.n_layers, n_levels = args.n_levels, input_dim = color_channels, hidden_dim = args.hidden_dim, z_dim = args.z_dim, device = device).to(device=device)
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
        loss, _, _ = multiresolution_loss(outputs, device, args)
        
        if epoch > 0 and loss.item() > 100:
            print('Bad batch')
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if nBatch % 10 == 0:
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
            loss, all_rec, all_target = multiresolution_loss(outputs, device, args)

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
                    plt.imshow(target.detach().cpu())
                    plt.savefig(args.dir + "/visualization/Epoch_" + str(epoch) + "_idx_" + str(idx) + "_target_resolution_" + str(args.n_levels - l) + ".png")

                    rec = all_rec[l]
                    rec = rec[b, :, :, :]

                    plt.clf()
                    plt.imshow(rec.detach().cpu())
                    plt.savefig(args.dir + "/visualization/Epoch_" + str(epoch) + "_idx_" + str(idx) + "_rec_resolution_" + str(args.n_levels - l) + ".png")
                idx += 1

            if idx >= 20:
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

            base_latent = torch.randn(batch_size, image_height * image_width, args.z_dim).to(device = device)
            base_predict = model.base_decoder(base_latent)
            base_predict = torch.reshape(base_predict, (batch_size, image_height, image_width, color_channels)).detach().cpu().numpy()
            all_base_predict.append(base_predict)

            l = args.n_levels - 1
            h = image_height
            w = image_width

            while l >= 0:
                h = int(h / args.cluster_height)
                w = int(w / args.cluster_width)
                
                latent = torch.randn(batch_size, h * w, args.z_dim).to(device = device)
                predict = model.global_decoder[l](latent)
                predict = torch.reshape(predict, (batch_size, h, w, color_channels)).detach().cpu().numpy()
                all_predict[l].append(predict)

                l -= 1

            '''
            if len(all_base_predict) >= 100:
                break
            '''

        name = args.dir + "/" + args.name + "_epoch_" + str(epoch)

        all_base_predict = np.concatenate(all_base_predict, axis = 0)
        np.save(name + ".all_base_predict", all_base_predict)

        l = args.n_levels - 1
        while l >= 0:
            all_predict[l] = np.concatenate(all_predict[l], axis = 0)
            np.save(name + ".predict_level_" + str(l), all_predict[l])
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
