from utils.general import attempt_download, load_weight
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
        return F.normalize(x, p = 2., dim = -1)

class EfficientNetV2B0_128(nn.Module):
    def __init__(self, pretrained=False):
        super(EfficientNetV2B0_128, self).__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_b0', pretrained=not pretrained, exportable=True)
        self.backbone.classifier = Identity()
        self.fc = nn.Linear(1280, 128)
        self.l2_norm = L2Norm()
        if pretrained:
            attempt_download('./models/efficientnetv2_b0_state.pt', 'https://github.com/20toduc01/NekoNet/releases/download/v1.1/effnetv2_b0_128_state.pt')
            load_weight(self, './models/efficientnetv2_b0_state.pt')

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.l2_norm(x)
        return x

class EfficientNetV2S_128(nn.Module):
    def __init__(self):
        super(EfficientNetV2S_128, self).__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True, exportable=True)
        self.backbone.classifier = Identity()
        self.fc = nn.Linear(1280, 128)
        self.l2_norm = L2Norm()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.l2_norm(x)
        return x


class MobileNetV3L_64(nn.Module):
    def __init__(self):
        super(MobileNetV3L_64, self).__init__()
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, exportable=True)
        self.backbone.classifier = Identity()
        self.fc = nn.Linear(1280, 64)
        self.l2_norm = L2Norm()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.l2_norm(x)
        return x

