import os, torch
import numpy as np
from torchvision.io import read_image

def params_info(net):
    total = 0
    trainable = 0
    for param in net.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    
    print(f'Total parameters: {total}')
    print(f'Trainable: {trainable}')
    print(f'Non-trainable: {total - trainable}')


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

    def __iter__(self):
        self.priority = np.random.permutation(self.num_class)
        self.ptr = 0
        self.full_loop = False
        return self
    
    def generator(self):
        self.priority = np.random.permutation(self.num_class)
        self.ptr = 0
        self.full_loop = False
        while self.full_loop == False:
            batchX = []
            batchY = []
            while len(batchX) < self.bs:
                c = self.priority[self.ptr]
                class_path = self.path + '/' + str(c)
                for imname in self.class_data[c]:
                    if len(batchX) == self.bs: break
                    img = read_image(class_path + '/' + imname)
                    img = self.tsnf(img)
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