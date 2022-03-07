import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader
import os
import time

from unsupervised_zinc_dataset import unsupervised_zinc_dataset

from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Unsupervised feature learning by VAE')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--dataset', '-dataset', type = str, default = 'ZINC_12k', help = 'Directory to the dataset')
    parser.add_argument('--features_fn', '-features_fn', type = str, default = '', help = 'Unsupervised features file')
    parser.add_argument('--learning_target', '-learning_target', type = str, default = 'logp', help = 'Learning target')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
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


# Dataset
train_dataset = unsupervised_zinc_dataset(directory = args.dataset, features_fn = args.features_fn, split = 'new_train.index')
val_dataset = unsupervised_zinc_dataset(directory = args.dataset, features_fn = args.features_fn, split = 'new_val.index')
test_dataset = unsupervised_zinc_dataset(directory = args.dataset, features_fn = args.features_fn, split = 'new_test.index')

train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True)

for batch_idx, data in enumerate(train_dataloader):
    print(data['features'].size())
    print(data[args.learning_target].size())
    features_dim = data['features'].size(1)
    break
print('Number of features:', features_dim)


# Multilayer Perceptron
class MLP(nn.Module):
    def __init__(self, features_dim, hidden_dim = 512, device = 'cuda', **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.bottom_layer = nn.Linear(self.features_dim, self.hidden_dim).to(device = device)
        # self.hidden_layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim).to(device = device)
        # self.hidden_layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(device = device)
        self.top_layer = nn.Linear(self.hidden_dim, 1).to(device = device)

    def forward(self, features):
        hidden = torch.tanh(self.bottom_layer(features))
        # hidden = torch.tanh(self.hidden_layer_1(hidden))
        # hidden = torch.tanh(self.hidden_layer_2(hidden))
        return self.top_layer(hidden)


# Init model and optimizer
model = MLP(features_dim = features_dim, device = device).to(device=device)
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
        features = data['features'].to(device = device)
        target = data[args.learning_target].float().to(device = device)

        predict = model(features)
        optimizer.zero_grad()
        loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        nBatch += 1
        if batch_idx % 1000 == 0:
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
        for batch_idx, data in enumerate(val_dataloader):
            features = data['features'].to(device = device)
            target = data[args.learning_target].float().to(device = device)

            predict = model(features)
            sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
            num_samples += features.size(0)
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

