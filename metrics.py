# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 22:15:51 2025

@author: madih
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import os
from config import save_dir, num_classes
from data_loader import class_names

def compute_classification_iou(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    ious = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        IoU = TP / (TP + FP + FN + 1e-7)
        ious.append(IoU)
    return np.mean(ious), ious

def compute_mota_motp_classification(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    missed = np.sum(np.sum(cm, axis=1) - np.diag(cm))
    false_pos = np.sum(np.sum(cm, axis=0) - np.diag(cm))
    mota = 1 - (missed + false_pos) / len(y_true)
    motp = np.mean([cm[i, i] / max(cm[i, :].sum(), 1) for i in range(num_classes)])
    return mota, motp

def calculate_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specificities = []
    for i in range(num_classes):
        TN = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        FP = np.sum(cm[:, i]) - cm[i, i]
        specificities.append(TN / (TN + FP + 1e-7))
    return np.mean(specificities), specificities

def plot_class_heatmap(probabilities, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(probabilities, xticklabels=class_names, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title("Classification Confidence Heatmap")
    plt.savefig(save_path)
    plt.close()

def plot_roc_pr_curves(y_true, y_probs, model_name):
    y_bin = label_binarize(y_true, classes=range(num_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, pr_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # ROC
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {class_names[i]} (AUC={roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(); plt.title(f'{model_name} ROC')
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"))
    plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {class_names[i]} (AUC={pr_auc[i]:.2f})')
    plt.legend(); plt.title(f'{model_name} PR Curve')
    plt.savefig(os.path.join(save_dir, f"{model_name}_pr_curve.png"))
    plt.close()
