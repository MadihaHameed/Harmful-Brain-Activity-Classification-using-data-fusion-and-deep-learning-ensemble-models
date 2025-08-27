# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 22:16:06 2025

@author: madih
"""

import torch.nn as nn
from torchvision import models
from timm import create_model

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.model(x)

class Xception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = create_model('xception', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.model(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class ReLViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    def forward(self, x):
        return self.model(x)
