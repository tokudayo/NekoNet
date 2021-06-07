import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x

net = Net()
print(net.parameters())

for param in net.parameters():
    print(param.shape)

    

