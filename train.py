import torch, os
import numpy as np
from model import Net
from torchvision.io import read_image
from torchvision import transforms, utils


class DataPipeline():
    def __init__(self, path, bs, tsnf):
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
        class_pointer = 0
        batch = []
        for c in priority:
            if len(batch) + self.class_size(c) > self.bs:
                # random selection
                pass
            batch.append()