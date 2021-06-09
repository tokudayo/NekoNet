import torch, os
import numpy as np
from model import Net
from torchvision.io import read_image
from torchvision import transforms, utils


class DataPipeline():
    def __init__(self, path, bs):
        self.path = path
        self.bs = bs
        self.class_data = []
        self.num_class = len(os.listdir(path))
        for c in range(self.num_class):
            class_path = path + '/' + str(c)
            self.class_data.append(os.listdir(class_path))
            

    def example_from_class(self, c):
        return np.random.choice(self.class_data[c])

