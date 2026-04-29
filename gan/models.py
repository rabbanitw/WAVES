"""Pix2Pix-style components for the watermark-removal GAN.

* Generator: residual U-Net at 256x256 (G(x) = clamp(x + alpha * tanh(net(x)))).
  ~7M params. Pix2Pix's U-Net 256 layout, slimmed.
* Discriminator: 70x70 PatchGAN. ~2.8M params.
* Classifier (C_eval): ImageNet-pretrained ResNet-18 with a 2-class head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -------- Generator (U-Net 256, residual) --------

class _UNetDown(nn.Module):
    def __init__(self, in_c, out_c, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.net(x)
        return torch.cat([x, skip], dim=1)


class GeneratorUNet(nn.Module):
    """Residual U-Net for 256x256 inputs.

    Output is `clamp(x + alpha * tanh(delta), -1, 1)` where `delta` comes from
    the U-Net decoder. This biases G toward small perturbations of the input.
    """

    def __init__(self, in_c: int = 3, out_c: int = 3, ngf: int = 64,
                 alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

        self.d1 = _UNetDown(in_c, ngf, normalize=False)        # 128
        self.d2 = _UNetDown(ngf, ngf * 2)                       # 64
        self.d3 = _UNetDown(ngf * 2, ngf * 4)                   # 32
        self.d4 = _UNetDown(ngf * 4, ngf * 8, dropout=0.5)      # 16
        self.d5 = _UNetDown(ngf * 8, ngf * 8, dropout=0.5)      # 8
        self.d6 = _UNetDown(ngf * 8, ngf * 8, dropout=0.5)      # 4
        self.d7 = _UNetDown(ngf * 8, ngf * 8, dropout=0.5)      # 2
        self.d8 = _UNetDown(ngf * 8, ngf * 8, normalize=False)  # 1

        self.u1 = _UNetUp(ngf * 8, ngf * 8, dropout=0.5)        # 2
        self.u2 = _UNetUp(ngf * 16, ngf * 8, dropout=0.5)       # 4
        self.u3 = _UNetUp(ngf * 16, ngf * 8, dropout=0.5)       # 8
        self.u4 = _UNetUp(ngf * 16, ngf * 8)                    # 16
        self.u5 = _UNetUp(ngf * 16, ngf * 4)                    # 32
        self.u6 = _UNetUp(ngf * 8, ngf * 2)                     # 64
        self.u7 = _UNetUp(ngf * 4, ngf)                         # 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_c, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        delta = self.final(u7)
        return torch.clamp(x + self.alpha * delta, -1.0, 1.0)


# -------- Discriminator (70x70 PatchGAN) --------

class PatchDiscriminator(nn.Module):
    def __init__(self, in_c: int = 3, ndf: int = 64):
        super().__init__()

        def block(in_c, out_c, normalize=True, stride=2):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_c, ndf, normalize=False),    # 128
            *block(ndf, ndf * 2),                   # 64
            *block(ndf * 2, ndf * 4),               # 32
            *block(ndf * 4, ndf * 8, stride=1),     # 31
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),  # 30
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 1, 30, 30]


# -------- Classifier (C_eval) --------

class BinaryClassifier(nn.Module):
    """ResNet-18 (ImageNet-pretrained) with a 2-class head for orig vs edit."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        # ImageNet normalization buffers (input is [-1, 1])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_neg1_1: torch.Tensor) -> torch.Tensor:
        x01 = (x_neg1_1 + 1.0) / 2.0
        x = (x01 - self.mean) / self.std
        return self.backbone(x)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    G = GeneratorUNet()
    D = PatchDiscriminator()
    C = BinaryClassifier(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    print(f"G  params={count_params(G)/1e6:5.2f}M  out={tuple(G(x).shape)}")
    print(f"D  params={count_params(D)/1e6:5.2f}M  out={tuple(D(x).shape)}")
    print(f"C  params={count_params(C)/1e6:5.2f}M  out={tuple(C(x).shape)}")
