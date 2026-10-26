# ============================================================
# models/vit_model.py — Vision Transformer (ViT) for Plant Disease Classification
#
# Architecture (per paper):
#   • Patch size  : 16×16
#   • Embed dim   : 768
#   • Heads       : 12
#   • Layers      : 12
#   • Dropout     : 0.1
#   • Classes     : 38  (PlantVillage)
# ============================================================

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS,
                    NUM_LAYERS, MLP_RATIO, DROPOUT, NUM_CLASSES)


# ── 1. Patch Embedding ────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits image (B, C, H, W) into non-overlapping 16×16 patches,
    projects each flattened patch to embed_dim.
    """

    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        assert img_size % patch_size == 0, \
            "Image size must be divisible by patch size."

        self.num_patches  = (img_size // patch_size) ** 2   # 196
        self.patch_size   = patch_size
        self.embed_dim    = embed_dim

        # Conv2d with kernel/stride = patch_size acts as the linear projection
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.projection(x)          # (B, embed_dim, 14, 14)
        x = x.flatten(2)               # (B, embed_dim, 196)
        x = x.transpose(1, 2)          # (B, 196, embed_dim)
        return x


# ── 2. Multi-Head Self-Attention ──────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 dropout=DROPOUT):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.qkv     = nn.Linear(embed_dim, embed_dim * 3)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ── 3. MLP Block ──────────────────────────────────────────────

class MLPBlock(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, mlp_ratio=MLP_RATIO,
                 dropout=DROPOUT):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── 4. Transformer Encoder Block ──────────────────────────────

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 mlp_ratio=MLP_RATIO, dropout=DROPOUT):
        super().__init__()
        self.norm1  = nn.LayerNorm(embed_dim)
        self.attn   = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.mlp    = MLPBlock(embed_dim, mlp_ratio, dropout)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))   # residual + attention
        x = x + self.drop(self.mlp(self.norm2(x)))    # residual + FFN
        return x


# ── 5. Vision Transformer (complete model) ────────────────────

class VisionTransformer(nn.Module):
    """
    ViT-Base implementation as described in the paper.

    Forward pass:
        image (B, 3, 224, 224)
          → patch embedding   (B, 196, 768)
          → prepend [CLS] token  → (B, 197, 768)
          → add positional encoding
          → 12 × TransformerEncoderBlock
          → take [CLS]  → (B, 768)
          → classification head  → (B, 38)
    """

    def __init__(self,
                 img_size   = IMAGE_SIZE,
                 patch_size = PATCH_SIZE,
                 in_channels= 3,
                 num_classes= NUM_CLASSES,
                 embed_dim  = EMBED_DIM,
                 num_heads  = NUM_HEADS,
                 num_layers = NUM_LAYERS,
                 mlp_ratio  = MLP_RATIO,
                 dropout    = DROPOUT):
        super().__init__()

        self.patch_embed  = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches       = self.patch_embed.num_patches  # 196

        # Learnable [CLS] token & positional encoding
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed    = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop     = nn.Dropout(dropout)

        # Stack of 12 transformer encoder blocks
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: [CLS] → num_classes
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)

        # 1. Patch embedding
        x = self.patch_embed(x)                              # (B, 196, 768)

        # 2. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, 768)
        x   = torch.cat([cls, x], dim=1)                    # (B, 197, 768)

        # 3. Add positional encoding
        x   = self.pos_drop(x + self.pos_embed)

        # 4. Transformer encoder
        x   = self.blocks(x)                                 # (B, 197, 768)
        x   = self.norm(x)

        # 5. Use [CLS] representation for classification
        cls_out = x[:, 0]                                    # (B, 768)
        logits  = self.head(cls_out)                         # (B, 38)
        return logits


# ── Checkpoint helpers ────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"    : epoch,
        "val_acc"  : val_acc,
        "model"    : model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"[ViT] Checkpoint saved → {path}  (val_acc={val_acc:.2f}%)")


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model"]

    # ── Key remapping: handle checkpoints saved under old layer names ──
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        # patch_embed.proj  →  patch_embed.projection
        new_k = new_k.replace("patch_embed.proj.", "patch_embed.projection.")
        # mlp.fc1  →  mlp.net.0   (first Linear in Sequential)
        new_k = new_k.replace(".mlp.fc1.", ".mlp.net.0.")
        # mlp.fc2  →  mlp.net.3   (second Linear in Sequential)
        new_k = new_k.replace(".mlp.fc2.", ".mlp.net.3.")
        remapped[new_k] = v

    model.load_state_dict(remapped)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ViT] Loaded checkpoint from {path}  "
          f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?'):.2f}%)")
    return ckpt.get("epoch", 0), ckpt.get("val_acc", 0.0)