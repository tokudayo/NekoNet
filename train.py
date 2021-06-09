import torch, os
import numpy as np
from model import Net
from torchvision.io import read_image
import torchvision.transforms as T


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
        return torch.stack(batchX), torch.Tensor(batchY)


dl = DataPipeline('./data', 15, tsnf = T.Compose([T.Resize((224, 224)),
                                                  lambda x : x/255.0,
                                                  T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))

print('ok')
a, b = dl.get()
print(a.shape)
print(b.shape)