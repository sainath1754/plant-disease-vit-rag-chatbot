#!/usr/bin/env python3
# ============================================================
# train.py — Train / Fine-tune ViT on PlantVillage
#
# Usage (PyCharm terminal):
#   python train.py
#   python train.py --epochs 10 --batch_size 16
#   python train.py --resume
#
# ⚠️  Windows MUST call this inside  if __name__ == "__main__":
#     (already done at the bottom of this file)
# ============================================================

import os
import sys
import argparse
import time
import platform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config import (
    DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    WEIGHT_DECAY, SCHEDULER_STEP, SCHEDULER_GAMMA,
)
from models.vit_model  import VisionTransformer, save_checkpoint, load_checkpoint
from utils.dataset     import get_data_loaders
from utils.evaluation  import (compute_metrics, plot_accuracy_curve,
                                plot_loss_curve, plot_confusion_matrix,
                                print_classification_report)


# ── Device ────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon
    return torch.device("cpu")


# ── Training epoch ────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [Train]",
                unit="batch", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        preds       = model(images).argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        n          += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/n:.1f}%")

    return total_loss / n, 100.0 * correct / n


# ── Validation / Test ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, split="Val"):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels  = [], []

    for images, labels in tqdm(loader, desc=f"  [{split}]", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits  = model(images)
        loss    = criterion(logits, labels)
        preds   = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        n          += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / n, 100.0 * correct / n, all_labels, all_preds


# ── Model builder ─────────────────────────────────────────────

def build_model(use_pretrained, device):
    """
    Pretrained mode (default):
        Loads timm's vit_base_patch16_224 ImageNet weights.
        Freezes most layers, fine-tunes head + last 3 blocks only.
        → 1 epoch on CPU takes ~5-15 min depending on CPU.

    Scratch mode (--scratch flag):
        Trains full ViT from random init.
        → needs a GPU; on CPU can take hours per epoch.
    """
    if use_pretrained:
        try:
            import timm
            from config import NUM_CLASSES
            print("[Model] Loading pretrained ViT-Base from timm …")
            model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=NUM_CLASSES,
            )
            # Freeze everything
            for p in model.parameters():
                p.requires_grad = False
            # Unfreeze head
            for p in model.head.parameters():
                p.requires_grad = True
            # Unfreeze last 3 transformer blocks
            for block in model.blocks[-3:]:
                for p in block.parameters():
                    p.requires_grad = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"[Model] Pretrained loaded. "
                  f"Trainable: {trainable:,} / {total:,} params "
                  f"({100*trainable/total:.1f}%)")
            return model.to(device)

        except ImportError:
            print("[Model] timm not found. Install it for much faster training:")
            print("        pip install timm")
            print("[Model] Falling back to training from scratch …")

    print("[Model] Training Vision Transformer from scratch.")
    model = VisionTransformer().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] Scratch ViT. Total params: {total:,}")
    return model


# ── Main (inside guard — required for Windows) ───────────────

def main(args):
    device = get_device()

    print("\n" + "="*60)
    print("  Plant Disease ViT — Training")
    print(f"  Platform  : {platform.system()} {platform.machine()}")
    print(f"  Device    : {device}")
    print(f"  Mode      : {'Scratch' if args.scratch else 'Fine-tune pretrained'}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print("="*60 + "\n")

    # Data loaders (num_workers=0 auto-set on Windows)
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir, args.batch_size)

    # Model
    model     = build_model(use_pretrained=not args.scratch, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

    # Resume
    start_epoch, best_val_acc = 0, 0.0
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if args.resume and os.path.exists(best_ckpt):
        start_epoch, best_val_acc = load_checkpoint(model, best_ckpt, optimizer, device)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs)

        vl_loss, vl_acc, _, _ = evaluate(
            model, val_loader, criterion, device, split="Val")

        scheduler.step()

        train_losses.append(tr_loss);  val_losses.append(vl_loss)
        train_accs.append(tr_acc);     val_accs.append(vl_acc)

        flag = " ✅ best" if vl_acc > best_val_acc else ""
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.1f}%  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.1f}%{flag}  "
              f"({time.time()-t0:.0f}s)")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_checkpoint(model, optimizer, epoch, vl_acc, best_ckpt)

        epoch_ckpt = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")
        save_checkpoint(model, optimizer, epoch, vl_acc, epoch_ckpt)

    # ── Final test ────────────────────────────────────────────
    print("\n── Final Test Evaluation ──")
    load_checkpoint(model, best_ckpt, device=device)
    _, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, split="Test")

    m = compute_metrics(y_true, y_pred)
    print(f"  Test Accuracy  : {test_acc:.2f}%")
    print(f"  Precision      : {m['precision']*100:.2f}%")
    print(f"  Recall         : {m['recall']*100:.2f}%")
    print(f"  F1-Score       : {m['f1']*100:.2f}%")
    print_classification_report(y_true, y_pred)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_accuracy_curve(train_accs, val_accs)
    plot_loss_curve(train_losses, val_losses)
    plot_confusion_matrix(y_true, y_pred)

    print(f"\n✅ Done. Best val accuracy: {best_val_acc:.2f}%")


# ── Windows multiprocessing guard — DO NOT REMOVE ─────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,   default=DATA_DIR)
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from checkpoints/best_model.pth")
    parser.add_argument("--scratch", action="store_true",
                        help="Train from random init (slow on CPU)")
    args = parser.parse_args()
    main(args)
