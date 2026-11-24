"""ResNet9 architecture used to classify 38 plant leaf conditions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_block(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
    """Convolution + batch normalization + ReLU, optionally followed by max pooling."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    """Training and validation helpers shared by image classifiers."""

    def training_step(self, batch):
        images, labels = batch
        return F.cross_entropy(self(images), labels)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        return {"val_loss": F.cross_entropy(out, labels).detach(), "val_accuracy": accuracy(out, labels)}

    @staticmethod
    def validation_epoch_end(outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        return {"val_loss": loss, "val_accuracy": acc}


class ResNet9(ImageClassificationBase):
    """Compact residual network for 256x256 RGB leaf images."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)  # 128 x 64 x 64
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)  # 256 x 16 x 16
        self.conv4 = conv_block(256, 512, pool=True)  # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)
