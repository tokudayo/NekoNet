import torch, os
import numpy as np
from model import Net
from torchvision.io import read_image
from torchvision import transforms, utils


class DataPipeline():
    def __init__(self, path, bs, tsnf=None, device='cpu'):
        self.path = path
        self.bs = bs
        self.tsnf = tsnf
        self.class_data = []
        self.num_class = len(os.listdir(path))
        for c in range(self.num_class):
            class_path = path + '/' + str(c)
            self.class_data.append(os.listdir(class_path))
            
    def example_from_class(self, c):
        return np.random.choice(self.class_data[c])

    def class_size(self, c):
        return len(self.class_data[c])

    def get(self):
        priority = np.random.permutation(self.num_class)
        batchX = []
        batchY = []
        for c in priority:
            class_path = self.path + '/' + str(c)
            for imname in self.class_data[c]:
                if len(batchX) == self.bs: break
                img = read_image(class_path + '/' + imname)
                img = self.tsnf(img)
                batchX.append(img)
                batchY.append(c)
            if len(batchX) == self.bs: break
        return torch.Tensor(batchX), torch.Tensor(batchY)
