import torch
from torch import nn
from torch import functional as F
from torchinfo import summary

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p = 2, dim = -1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        self.backbone.classifier = Identity()
        self.fc = nn.Linear(1280, 64)
        self.l2_norm = L2Norm()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.l2_norm(x)
        return x


if __name__ == "__main__":
    net = Net()
    print(net)
