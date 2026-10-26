# ============================================================
# utils/evaluation.py — Metrics & Confusion Matrix
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CLASS_NAMES, OUTPUT_DIR


def compute_metrics(y_true, y_pred):
    """
    Returns a dict containing:
        accuracy, precision, recall, f1  (macro-averaged)
    """
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall"   : recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1"       : f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_classification_report(y_true, y_pred):
    """Prints per-class precision / recall / F1 for classes present in data."""
    present_labels = sorted(set(y_true) | set(y_pred))
    short_names    = [CLASS_NAMES[i].split("___")[-1][:30] for i in present_labels]
    print(classification_report(y_true, y_pred,
                                labels=present_labels,
                                target_names=short_names,
                                zero_division=0))


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """
    Saves / shows the confusion matrix as a heatmap.
    Mirrors Fig. 4 of the paper.
    """
    present_labels = sorted(set(y_true) | set(y_pred))
    short          = [CLASS_NAMES[i].split("___")[-1][:15] for i in present_labels]
    cm             = confusion_matrix(y_true, y_pred, labels=present_labels)

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Vision Transformer Plant Disease Recognition", fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    path = save_path or os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"[Evaluation] Confusion matrix saved → {path}")
    plt.show()


def plot_accuracy_curve(train_accs, val_accs, save_path: str = None):
    """
    Plots training vs validation accuracy curve.
    Mirrors Fig. 3 of the paper.
    """
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_accs, "b-o", label="Training Accuracy",   markersize=4)
    plt.plot(epochs, val_accs,   "r-o", label="Validation Accuracy", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve — Vision Transformer (ViT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = save_path or os.path.join(OUTPUT_DIR, "accuracy_curve.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"[Evaluation] Accuracy curve saved → {path}")
    plt.show()


def plot_loss_curve(train_losses, val_losses, save_path: str = None):
    """Plots training vs validation loss."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, "b-o", label="Training Loss",   markersize=4)
    plt.plot(epochs, val_losses,   "r-o", label="Validation Loss", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curve — Vision Transformer (ViT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = save_path or os.path.join(OUTPUT_DIR, "loss_curve.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"[Evaluation] Loss curve saved → {path}")
    plt.show()