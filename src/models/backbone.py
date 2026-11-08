from typing import Tuple, List
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFPNBackbone(nn.Module):
    """ResNet-50 backbone with Feature Pyramid Network (FPN).
    
    Generates feature maps P3, P4, P5 with strides 8, 16, 32 and uniform channels C = 256.
    """

    def __init__(self, pretrained: bool = False, out_channels: int = 256):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Stage 0 & 1 (stride 4)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        # C3 (stride 8, channels 512)
        self.layer2 = resnet.layer2
        # C4 (stride 16, channels 1024)
        self.layer3 = resnet.layer3
        # C5 (stride 32, channels 2048)
        self.layer4 = resnet.layer4

        # Lateral 1x1 convolutions
        self.lat_c5 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(512, out_channels, kernel_size=1)

        # Smooth 3x3 convolutions
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Bottom-up path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)

        c3 = self.layer2(c1)  # Stride 8
        c4 = self.layer3(c3)  # Stride 16
        c5 = self.layer4(c4)  # Stride 32

        # Top-down FPN path
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat_c3(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        f3 = self.smooth_p3(p3)  # P3: stride 8, C=256 (F1 in paper notation)
        f4 = self.smooth_p4(p4)  # P4: stride 16, C=256 (F2 in paper notation)
        f5 = self.smooth_p5(p5)  # P5: stride 32, C=256 (F3 in paper notation)

        return f3, f4, f5

