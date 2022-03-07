import torch
import numpy as np
import json
from torch.utils.data import Dataset

class mnist_rotation_dataset(Dataset):
    def __init__(self, directory, split = 'train', limit_data = -1):
        self.directory = directory
        self.split = split
        self.file_fn = self.directory + '/rotated_' + self.split + '.npz'

        data = np.load(self.file_fn)
        self.x = data['x']
        self.labels = data['y']
        self.num_samples = self.x.shape[0]

        self.original_height = 28
        self.original_width = 28
        self.height = 32
        self.width = 32

        # Limit data
        if limit_data != -1:
            assert limit_data > 0
            assert limit_data <= self.num_samples
            self.num_samples = limit_data
            self.x = self.x[:self.num_samples]
            self.labels = self.labels[:self.num_samples]
    
        # Images
        self.images = []
        for sample in range(self.num_samples):
            image = np.zeros((self.height, self.width))
            image[2:30,2:30] = np.reshape(self.x[sample], (28, 28))[:, :]
            self.images.append(image)

        assert len(self.images) == len(self.labels)
        assert len(self.images) == self.num_samples
        print('Number of samples:', self.num_samples)

        for i in range(100):
            for x in range(self.height):
                line = ''
                for y in range(self.width):
                    if self.images[i][x, y] > 0:
                        line += '1'
                    else:
                        line += '0'
                print(line)
            print(self.labels[i])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Image
        image = torch.Tensor(self.images[idx])
        image = image.unsqueeze(dim = 0)

        # Label
        label = torch.Tensor(np.array([self.labels[idx]]))

        # Sample
        return image, label

