# ============================================================
# utils/preprocessing.py — Image Preprocessing Utilities
# ============================================================

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMAGE_SIZE


# ── Single-image inference pipeline ──────────────────────────

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def preprocess_image(image_input) -> torch.Tensor:
    """
    Accepts a file path (str), PIL Image, or numpy array.
    Returns a normalised tensor of shape (1, 3, 224, 224).
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image type: provide path, PIL Image, or numpy array.")

    tensor = _INFERENCE_TRANSFORM(img)
    return tensor.unsqueeze(0)   # add batch dim → (1, C, H, W)


def validate_image(image_path: str) -> bool:
    """Returns True if the file is a readable RGB image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img.verify()
        return True
    except Exception:
        return False


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation for visualisation.
    Input : (C, H, W) tensor
    Output: uint8 numpy array (H, W, 3)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
