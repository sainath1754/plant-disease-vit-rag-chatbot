#!/usr/bin/env python3
# ============================================================
# inference.py — Single Image Inference + RAG Advisory
#
# Usage:
#   python inference.py --image path\to\leaf.jpg
#   python inference.py --image path\to\leaf.jpg --top_k 5
# ============================================================

import os
import argparse
import torch
import torch.nn.functional as F

from config              import CHECKPOINT_DIR, CLASS_NAMES, NUM_CLASSES
from models.rag_module   import RAGAdvisoryModule
from utils.preprocessing import preprocess_image


# ── Detect which architecture the checkpoint uses ────────────

def _is_timm_checkpoint(state_dict: dict) -> bool:
    """
    Returns True if checkpoint was saved from a timm ViT model.
    timm uses:  patch_embed.proj.*  and  mlp.fc1 / mlp.fc2
    Custom ViT: patch_embed.projection.*  and  mlp.net.*
    """
    all_keys = " ".join(state_dict.keys())
    return "mlp.fc1" in all_keys or "patch_embed.proj.weight" in all_keys


# ── Load model from checkpoint ────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    ckpt       = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"]

    if _is_timm_checkpoint(state_dict):
        print("[Inference] Detected timm ViT checkpoint — loading timm model …")
        import timm
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=NUM_CLASSES,
        )
    else:
        print("[Inference] Detected custom ViT checkpoint — loading custom model …")
        from models.vit_model import VisionTransformer
        model = VisionTransformer()

    model.load_state_dict(state_dict)
    model.to(device).eval()

    print(f"[Inference] Loaded  epoch={ckpt.get('epoch','?')}  "
          f"val_acc={ckpt.get('val_acc', 0.0):.2f}%")
    return model


# ── Predict ───────────────────────────────────────────────────

def predict(model, image_path, device, top_k=5):
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    top_probs, top_idx = probs.topk(min(top_k, NUM_CLASSES))
    preds = [
        {"class": CLASS_NAMES[i.item()], "confidence": p.item() * 100}
        for p, i in zip(top_probs, top_idx)
    ]
    return preds[0]["class"], preds[0]["confidence"], preds


# ── Main ─────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt):
        print(f"\nERROR: No checkpoint found at {ckpt}")
        print("Please run:  python train.py --epochs 10 --batch_size 16")
        return

    model = load_model(ckpt, device)

    print(f"\n[Inference] Analysing: {args.image}")
    pred_class, confidence, top_preds = predict(
        model, args.image, device, top_k=args.top_k)

    print("\n" + "="*60)
    print(f"  Predicted Class : {pred_class}")
    print(f"  Confidence      : {confidence:.2f}%")
    print("\n  Top Predictions:")
    for i, p in enumerate(top_preds, 1):
        bar = "█" * int(p["confidence"] / 5)
        print(f"  {i}. {p['class']:<44} {p['confidence']:6.2f}%  {bar}")
    print("="*60)

    print("\n[RAG] Fetching treatment recommendation …")
    rag    = RAGAdvisoryModule()
    result = rag.generate_recommendation(pred_class)

    print("\n" + "="*60)
    print(f"  Crop   : {result['crop']}")
    print(f"  Disease: {result['disease']}")
    print("\n── Recommendation ──────────────────────────────────────")
    print(result["recommendation"])
    print("="*60)


# ── Windows guard ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    main(args)