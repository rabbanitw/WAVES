"""Binary classifier for distinguishing real photographs from Nano-Banana
edits / generations. Used as the SynthID-shaped surrogate detector
referenced by the experiments in this kit."""

import torch
import torch.nn as nn
from torchvision import models


class BinaryClassifier(nn.Module):
    """ResNet-{18,50} ImageNet-pretrained, with a 2-class head:
       label 0 = real photograph (e.g. Open Images), label 1 = Nano-Banana
       generated or edited.

    The model accepts inputs in [-1, 1] (matching the rest of the kit's
    data pipeline) and does the ImageNet mean/std normalisation
    internally."""

    def __init__(self, pretrained: bool = True, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet101(weights=weights)
        else:
            raise ValueError(f"backbone={backbone!r}, expected resnet18/50/101")
        net.fc = nn.Linear(net.fc.in_features, 2)
        self.backbone = net
        # ImageNet stats applied inside forward() so callers can hand us
        # [-1, 1] tensors directly.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_neg1_1: torch.Tensor) -> torch.Tensor:
        x01 = (x_neg1_1 + 1.0) / 2.0
        x = (x01 - self.mean) / self.std
        return self.backbone(x)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for bb in ("resnet18", "resnet50"):
        C = BinaryClassifier(pretrained=False, backbone=bb)
        x = torch.randn(2, 3, 256, 256)
        print(f"{bb:10s}  params={count_params(C)/1e6:5.2f}M  out={tuple(C(x).shape)}")
