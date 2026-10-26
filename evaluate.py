#!/usr/bin/env python3
# ============================================================
# evaluate.py — Evaluate a trained ViT checkpoint
#
# Usage:
#   python evaluate.py --checkpoint checkpoints/best_model.pth
# ============================================================

import os
import argparse
import torch
import torch.nn as nn

from config          import DATA_DIR, CHECKPOINT_DIR
from models.vit_model import VisionTransformer, load_checkpoint
from utils.dataset    import get_data_loaders
from utils.evaluation import (compute_metrics, print_classification_report,
                               plot_confusion_matrix)


@torch.no_grad()
def run_evaluation(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, all_labels, all_preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")

    # Load data
    _, _, test_loader = get_data_loaders(args.data_dir, batch_size=32)

    # Load model
    model     = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    load_checkpoint(model, args.checkpoint, device=device)

    # Run evaluation
    loss, acc, y_true, y_pred = run_evaluation(
        model, test_loader, criterion, device)

    metrics = compute_metrics(y_true, y_pred)
    print("\n" + "="*55)
    print(f"  Test Loss      : {loss:.4f}")
    print(f"  Test Accuracy  : {acc:.2f}%")
    print(f"  Test Precision : {metrics['precision']*100:.2f}%")
    print(f"  Test Recall    : {metrics['recall']*100:.2f}%")
    print(f"  Test F1-Score  : {metrics['f1']*100:.2f}%")
    print("="*55)

    print("\nPer-class report:")
    print_classification_report(y_true, y_pred)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred)


# ── Windows multiprocessing guard ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--data_dir",   type=str, default=DATA_DIR)
    args = parser.parse_args()
    main(args)
