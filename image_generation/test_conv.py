import torch
import torch.nn as nn

class Generator_32(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Generator_32, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.z_dim, self.hidden_dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            # state size. hidden_dim x 4 x 4
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            # state size. hidden_dim x 8 x 8
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            # state size. hidden_dim x 16 x 16
            nn.ConvTranspose2d(self.hidden_dim, self.output_dim, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (output_dim) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

class Generator_16(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Generator_16, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.z_dim, self.hidden_dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            # state size. hidden_dim x 4 x 4
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            # state size. hidden_dim x 8 x 8
            nn.ConvTranspose2d(self.hidden_dim, self.output_dim, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (output_dim) x 16 x 16
        )

    def forward(self, input):
        return self.main(input)

class Generator_8(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Generator_8, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.z_dim, self.hidden_dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            # state size. hidden_dim x 4 x 4
            nn.ConvTranspose2d(self.hidden_dim, self.output_dim, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (output_dim) x 8 x 8
        )

    def forward(self, input):
        return self.main(input)


batch_size = 20
z_dim = 256
hidden_dim = 256
output_dim = 3

Z = torch.randn(batch_size, z_dim, 1, 1)

G = Generator_32(z_dim, hidden_dim, output_dim)
output = G(Z)
print(output.size())

G = Generator_16(z_dim, hidden_dim, output_dim)
output = G(Z)
print(output.size())

G = Generator_8(z_dim, hidden_dim, output_dim)
output = G(Z)
print(output.size())

