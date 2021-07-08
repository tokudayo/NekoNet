import os, torch
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Resize

class DataLoader():
    def __init__(self, path, tsnf=None, cache_path=None):
        self.path = path
        self.tsnf = tsnf
        self.class_data = []
        if cache_path:
            try:
                self.cache = torch.load(cache_path)
            except:
                self.cache = None
        else:
            self.cache = None
        self.num_class = len(os.listdir(path))
        for c in range(self.num_class):
            class_path = path + '/' + str(c)
            self.class_data.append(os.listdir(class_path))
        
    def generator(self, batch_size=32):
        self.priority = np.random.permutation(self.num_class)
        self.ptr = 0
        self.full_loop = False
        while self.full_loop == False:
            batchX = []
            batchY = []
            while len(batchX) < batch_size:
                c = self.priority[self.ptr]
                for imname in self.class_data[c]:
                    if len(batchX) == batch_size: break
                    if self.cache is None:
                        img = self.read(self.path_to(c, imname))
                    else:
                        img = self.cache[c][imname]
                    batchX.append(img)
                    batchY.append(c)
                self.ptr += 1   
                if self.ptr == self.num_class:
                    self.ptr = 0
                    self.full_loop = True
            yield torch.stack(batchX), torch.Tensor(batchY)

    def read(self, path):
        img = read_image(path)
        if self.tsnf: img = self.tsnf(img)
        return img

    def example_from_class(self, c):
        return np.random.choice(self.class_data[c])

    def class_size(self, c):
        return len(self.class_data[c])

    def path_to(self, c, e):
        return self.path + '/' + str(c) + '/' + e

    def random_pair(self, same_class=False, transform=True):
        if same_class:
            c = np.random.randint(0, self.num_class)
            p1 = p2 = ''
            while p1 == p2:
                p1, p2 = np.random.choice(self.class_data[c], 2)
            p1 = self.path_to(c, p1)
            p2 = self.path_to(c, p2)
        else:
            c1 = np.random.randint(0, self.num_class)
            p1 = np.random.choice(self.class_data[c1])
            p1 = self.path_to(c1, p1)
            c2 = np.random.randint(0, self.num_class)
            p2 = np.random.choice(self.class_data[c2])
            p2 = self.path_to(c2, p2)
        return p1, p2

    def count_all(self):
        total = 0
        for c in range(self.num_class):
            total += len(self.class_data[c])
        return total

    def create_cache(self, outpath):
        cachelist = []
        print("Hello")
        for c in range(self.num_class):
            d = dict()
            for e in self.class_data[c]:
                img = self.read(self.path_to(c, e))
                d[e] = img
            cachelist.append(d)
        print(cachelist)
        torch.save(cachelist, outpath)