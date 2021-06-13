import os, torch
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Resize
from matplotlib import pyplot as plt


class DataPipeline():
    def __init__(self, path, bs, tsnf=None, device='cpu', image_size=None):
        self.path = path
        self.bs = bs
        self.tsnf = tsnf
        self.image_size = image_size
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
            while len(batchX) < self.bs:
                c = self.priority[self.ptr]
                class_path = self.path + '/' + str(c)
                for imname in self.class_data[c]:
                    if len(batchX) == self.bs: break
                    img = read_image(class_path + '/' + imname)
                    if self.image_size:
                        img = Resize(self.image_size)(img)
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


def batch_to_images(X, y):
    X = X.to('cpu').numpy()
    batch_size = X.shape[0]
    r = c = int(np.ceil(np.sqrt(batch_size)))
    fig = plt.figure(figsize=(r, c))
    for index, img in enumerate(X):
        # Adds a subplot at the 1st position
        fig.add_subplot(r, c, index + 1)
        # showing image
        plt.imshow(img.transpose(1, 2, 0))
        plt.axis('off')
        plt.title(y[index].item())
    plt.show()