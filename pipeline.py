import os, torch
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Resize

class DataPipeline():
    def __init__(self, path, batch_size, tsnf=None, device='cpu'):
        self.path = path
        self.batch_size = batch_size
        self.tsnf = tsnf
        self.class_data = []
        self.num_class = len(os.listdir(path))
        for c in range(self.num_class):
            class_path = path + '/' + str(c)
            self.class_data.append(os.listdir(class_path))

    
    def generator(self):
        self.priority = np.random.permutation(self.num_class)
        self.ptr = 0
        self.full_loop = False
        while self.full_loop == False:
            batchX = []
            batchY = []
            while len(batchX) < self.batch_size:
                c = self.priority[self.ptr]
                class_path = self.path + '/' + str(c)
                for imname in self.class_data[c]:
                    if len(batchX) == self.batch_size: break
                    img = read_image(class_path + '/' + imname)
                    if self.tsnf: img = self.tsnf(img)
                    batchX.append(img)
                    batchY.append(c)
                self.ptr += 1
                if self.ptr == self.num_class:
                    self.ptr = 0
                    self.full_loop = True
            yield torch.stack(batchX), torch.Tensor(batchY)

    def example_from_class(self, c):
        return np.random.choice(self.class_data[c])

    def class_size(self, c):
        return len(self.class_data[c])