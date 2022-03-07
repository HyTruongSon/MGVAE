import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import equivariant_linear_pytorch as eq

# +---------+
# | Encoder |
# +---------+

class Encoder(nn.Module):
    def __init__(self, input_order, output_order, encoder_channels, input_dim, output_dim, use_deterministic_encoder = False, device = 'cuda'):
        super().__init__()
        self.input_order = input_order
        self.output_order = output_order
        self.encoder_channels = encoder_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.use_deterministic_encoder = use_deterministic_encoder

        # build network architecture
        self.fc0 = nn.Linear(self.input_dim, self.input_dim).to(device = self.device)
        self.equi_layers = nn.ModuleList()
        if self.input_order == 1:
            self.equi_layers.append(eq.layer_1_to_2(self.input_dim, self.encoder_channels[0], device = self.device).to(device = self.device))
        else:
            self.equi_layers.append(eq.layer_2_to_2(self.input_dim, self.encoder_channels[0], device = self.device).to(device = self.device))
        L = len(self.encoder_channels)
        for layer in range(1, L):
            self.equi_layers.append(eq.layer_2_to_2(self.encoder_channels[layer - 1], self.encoder_channels[layer], device = self.device).to(device = self.device))
        if self.output_order == 1:
            self.equi_layers.append(eq.layer_2_to_1(self.encoder_channels[L - 1], self.output_dim, device = self.device).to(device = self.device))
        else:
            self.equi_layers.append(eq.layer_2_to_2(self.encoder_channels[L - 1], self.output_dim, device = self.device).to(device = self.device))
        
        if self.use_deterministic_encoder == True:
            self.fc1 = nn.Linear(self.output_dim, self.output_dim).to(device = self.device)
        else:
            self.fc1 = nn.Linear(self.output_dim, self.output_dim).to(device = self.device)
            self.fc2 = nn.Linear(self.output_dim, self.output_dim).to(device = self.device)

        # parameters
        self.params = []
        for layer in self.equi_layers:
            self.params = self.params + list(layer.parameters())
        self.params = self.params + list(self.fc1.parameters())
        if self.use_deterministic_encoder == False:
            self.params = self.params + list(self.fc2.parameters())
        self.params = torch.nn.ParameterList(self.params)

    def forward(self, inputs):
        inputs = self.fc0(inputs)
        inputs = torch.tanh(inputs)
        
        if self.input_order == 1:
            outputs = inputs.transpose(1, 2)
        else:
            outputs = inputs.transpose(2, 3).transpose(1, 2)
        
        L = len(self.equi_layers)
        for layer in range(L):
            # Leaky ReLU
            # outputs = F.leaky_relu(self.equi_layers[layer](outputs))
            # Sigmoid
            outputs = torch.tanh(self.equi_layers[layer](outputs))
        
        if self.output_order == 1:
            outputs = outputs.transpose(1, 2)
        else:
            outputs = outputs.transpose(1, 2).transpose(2, 3)
        
        if self.use_deterministic_encoder == True:
            return self.fc1(outputs)
        
        mean = self.fc1(outputs)
        logstd = self.fc2(outputs)
        gaussian_noise = torch.randn(mean.size()).to(device = self.device)
        latent = gaussian_noise * torch.exp(logstd) + mean
        return latent, mean, logstd

# +---------+
# | Decoder |
# +---------+

class Decoder(nn.Module):
    def __init__(self, encoder_channels, input_dim, output_dim, device = 'cuda'):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # build network architecture
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.encoder_channels[0]).to(device = self.device))
        L = len(self.encoder_channels)
        for layer in range(1, L):
            self.layers.append(nn.Linear(self.encoder_channels[layer - 1], self.encoder_channels[layer]).to(device = self.device))
        self.layers.append(nn.Linear(self.encoder_channels[L - 1], self.output_dim).to(device = self.device))

        # parameters
        self.params = []
        for layer in self.layers:
            self.params = self.params + list(layer.parameters())

    def forward(self, inputs):
        for layer in range(len(self.layers)):
            if layer == 0:
                outputs = self.layers[layer](inputs)
            else:
                outputs = self.layers[layer](outputs)
        return outputs

