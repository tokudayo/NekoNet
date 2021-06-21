import timm
import torchvision.transforms as T
from torch import nn
from torch.nn import functional as F
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


class EffNetV2S128(nn.Module):
    def __init__(self):
        super(EffNetV2S128, self).__init__()
        # self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        self.backbone.classifier = Identity()
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1280, 128)
        self.l2_norm = L2Norm()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.l2_norm(x)
        return x

class MobileNetV3L64(nn.Module):
    def __init__(self):
        super(MobileNetV3L64, self).__init__()
        # self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True)
        self.backbone.classifier = Identity()
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1280, 64)
        self.l2_norm = L2Norm()

        # Required transformation of [0; 255] (3, H, W) tensor input
        self.transform = T.Compose([T.Resize((224, 224)),
                         T.transforms.ColorJitter(brightness = .5, contrast = 0.3),
                         lambda x : x/255.0,
                         T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.l2_norm(x)
        return x

if __name__ == "__main__":
    pass
