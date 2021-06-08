import torch
from torch import nn
import numpy as np
from torchinfo import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x

net = Net()
summary(net, (1, 3, 224, 224))