# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 22:16:35 2025

@author: madih
"""

import os
import torch
from torchvision import datasets

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
train_dir = 'D:/Codes/all-model-run-seperatly-for-pr-curve/Split/train'
val_dir = 'D:/Codes/all-model-run-seperatly-for-pr-curve/Split/val'
test_dir = 'D:/Codes/all-model-run-seperatly-for-pr-curve/Split/test'

# Save directory
save_dir = "D:/Codes/all-model-run-seperatly-for-pr-curve/results"
os.makedirs(save_dir, exist_ok=True)

# Parameters
img_size = (299, 299)
batch_size = 16
num_classes = len(datasets.ImageFolder(train_dir).classes)
