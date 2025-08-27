# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 22:15:18 2025

@author: madih
"""

from config import num_classes, device
from models import ResNet50, Xception, VGG16, DenseNet, MobileNetV2, ReLViT
from train import train_and_evaluate
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import plot_and_save_metrics

models_dict = {
    'ResNet50': ResNet50(num_classes),
    'Xception': Xception(num_classes),
    'VGG16': VGG16(num_classes),
    'DenseNet121': DenseNet(num_classes),
    'MobileNetV2': MobileNetV2(num_classes),
    'ReLViT': ReLViT(num_classes)
}

histories, model_names = [], []
for name, model in models_dict.items():
    print(f"\nTraining {name}...\n")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
    history = train_and_evaluate(model, criterion, optimizer, scheduler, name, num_epochs=25)
    histories.append(history)
    model_names.append(name)

plot_and_save_metrics(histories, model_names)
print("✅ Training and evaluation completed for all models.")
