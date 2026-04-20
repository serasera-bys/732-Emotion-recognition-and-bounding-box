from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class BaselineEmotionCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet18EmotionClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_convnextv2_pico(num_classes: int, pretrained: bool) -> nn.Module:
    import timm

    candidate_names = (
        "convnextv2_pico.fcmae_ft_in22k_in1k",
        "convnextv2_pico.fcmae_ft_in1k",
        "convnextv2_pico",
        "timm/convnextv2_pico.fcmae_ft_in1k",
    )
    last_error: Exception | None = None
    for name in candidate_names:
        try:
            model = timm.create_model(name, pretrained=pretrained)
            model.reset_classifier(num_classes=num_classes)
            return model
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Failed to create ConvNeXtV2 Pico model: {last_error}")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "cnn":
        return BaselineEmotionCNN(num_classes=num_classes)
    if arch == "resnet18":
        return ResNet18EmotionClassifier(num_classes=num_classes)
    if arch == "convnextv2_pico":
        return build_convnextv2_pico(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported architecture: {arch}")
