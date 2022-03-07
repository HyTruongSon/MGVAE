import torch
import numpy as np
import json
from torch.utils.data import Dataset

class mnist_dataset(Dataset):
    def __init__(self, directory, split = 'train', limit_data = -1):
        self.directory = directory
        self.split = split

        if self.split == 'train':
            self.image_fn = self.directory + '/train-images.idx3-ubyte'
            self.label_fn = self.directory + '/train-labels.idx1-ubyte'
            self.num_samples = 60000
        else:
            self.image_fn = self.directory + '/t10k-images.idx3-ubyte'
            self.label_fn = self.directory + '/t10k-labels.idx1-ubyte'
            self.num_samples = 10000

        self.original_height = 28
        self.original_width = 28
        self.height = 32
        self.width = 32

        # Limit data
        if limit_data != -1:
            assert limit_data > 0
            assert limit_data <= self.num_samples
            self.num_samples = limit_data
    
        # Images
        self.images = []
        file = open(self.image_fn, 'rb')
        for idx in range(16):
            byte = int.from_bytes(file.read(1), 'big')
        for sample in range(self.num_samples):
            image = np.zeros((self.height, self.width))
            for i in range(self.original_height):
                for j in range(self.original_width):
                    byte = int.from_bytes(file.read(1), 'big')
                    image[i + 2, j + 2] = byte / 255.0
            self.images.append(image)
        file.close()

        # Labels
        self.labels = []
        file = open(self.label_fn, 'rb')
        for idx in range(8):
            file.read(1)
        for sample in range(self.num_samples):
            label = int.from_bytes(file.read(1), 'big')
            self.labels.append(label)
        file.close()

        assert len(self.images) == len(self.labels)
        assert len(self.images) == self.num_samples
        print('Number of samples:', self.num_samples)

        '''
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
        '''

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

