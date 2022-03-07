import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from PIL import Image
import PIL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, Adagrad
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from mnist_dataset import mnist_dataset


def draw_(data, output):
    num_samples = data.shape[0]
    color_channels = data.shape[3]
    for sample in range(num_samples):
        image = data[sample, :, :, :] * 255
        figure = image.astype('uint8')
        if color_channels == 1:
            PIL_image = Image.fromarray(figure.squeeze(2))
        else:
            PIL_image = Image.fromarray(figure).convert('RGB')
        PIL_image.save(output + "/" + str(sample) + ".png")
        if sample % 100 == 0:
            print('Done', sample)


def draw(directory, filename, output):
    data = np.load(directory + "/" + filename)
    draw_(data, output)

def reduce(input_data, cluster_height = 2, cluster_width = 2):
    num_samples = input_data.shape[0]
    input_height = input_data.shape[1]
    input_width = input_data.shape[2]
    color_channels = input_data.shape[3]
    output_height = input_height // cluster_height
    output_width = input_width // cluster_width
    output_data = np.zeros((num_samples, output_height, output_width, color_channels))
    for sample in range(num_samples):
        for c in range(color_channels):
            for i in range(output_height):
                for j in range(output_width):
                    x = i * cluster_height
                    y = j * cluster_width
                    output_data[sample, i, j, c] = np.mean(input_data[sample, x:x+cluster_height, y:y+cluster_width, c])
    return output_data

# Test data
testset = mnist_dataset(directory = './mnist', split = 'test')
test_dataloader = torch.utils.data.DataLoader(testset, batch_size = 20, shuffle = False, num_workers = 2)

images = []
for _, data in enumerate(test_dataloader):
    image, label = data
    images.append(image.transpose(1, 2).transpose(2, 3).detach().cpu().numpy())
images = np.concatenate(images, axis = 0)
print(images.shape)

draw_(images, 'test_32x32')
draw_(reduce(images), 'test_16x16')
print('Done zoom to 16x16')
draw_(reduce(images, 4, 4), 'test_8x8')
print('Done zoom to 8x8')

# Generated examples
directory = 'train_mgvae_conv-mnist'
filename = 'train_mgvae_conv.mnist.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.cluster_height.2.cluster_width.2.n_levels.2.n_layers.4.Lambda.1.hidden_dim.256.z_dim.256.resolution_level_2.npy'
output = 'generation_32x32'

draw(directory, filename, output)

filename = 'train_mgvae_conv.mnist.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.cluster_height.2.cluster_width.2.n_levels.2.n_layers.4.Lambda.1.hidden_dim.256.z_dim.256.resolution_level_1.npy'
output = 'generation_16x16'

draw(directory, filename, output)

filename = 'train_mgvae_conv.mnist.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.cluster_height.2.cluster_width.2.n_levels.2.n_layers.4.Lambda.1.hidden_dim.256.z_dim.256.resolution_level_0.npy'
output = 'generation_8x8'

draw(directory, filename, output)

