# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 22:15:34 2025

@author: madih
"""

import matplotlib.pyplot as plt
import os
from config import save_dir

def plot_and_save_metrics(histories, model_names):
    epochs = len(histories[0]["train_acc"])
    for i, history in enumerate(histories):
        plt.figure(figsize=(8, 6), dpi=300)
        plt.subplot(2, 1, 1)
        plt.plot(range(1, epochs+1), history["train_acc"], label='Train Acc', color='blue')
        plt.plot(range(1, epochs+1), history["val_acc"], label='Val Acc', color='green', linestyle='--')
        plt.plot(range(1, epochs+1), history["test_acc"], label='Test Acc', color='red', linestyle=':')
        plt.legend(); plt.title(f'{model_names[i]} Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(range(1, epochs+1), history["train_loss"], label='Train Loss', color='blue')
        plt.plot(range(1, epochs+1), history["val_loss"], label='Val Loss', color='green', linestyle='--')
        plt.plot(range(1, epochs+1), history["test_loss"], label='Test Loss', color='red', linestyle=':')
        plt.legend(); plt.title(f'{model_names[i]} Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_names[i]}_metrics.png"))
        plt.close()
