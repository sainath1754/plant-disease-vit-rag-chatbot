# ============================================================
# config.py — Central Configuration
# Plant Pathology Recognition (ViT + RAG)
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "PlantVillage")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")

# ── Dataset ──────────────────────────────────────────────────
IMAGE_SIZE   = 224          # ViT input (224×224)
PATCH_SIZE   = 16           # Each patch is 16×16 px  →  (224/16)² = 196 patches
NUM_CHANNELS = 3
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42

# ── ViT Hyperparameters ──────────────────────────────────────
EMBED_DIM       = 768       # Embedding dimension (ViT-Base)
NUM_HEADS       = 12        # Multi-head self-attention heads
NUM_LAYERS      = 12        # Transformer encoder layers
MLP_RATIO       = 4         # Feed-forward hidden dim multiplier
DROPOUT         = 0.1       # Dropout for regularisation

# ── Training ─────────────────────────────────────────────────
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4      # Adam LR as per paper
NUM_EPOCHS      = 30
WEIGHT_DECAY    = 1e-4
SCHEDULER_STEP  = 10        # LR decay every N epochs
SCHEDULER_GAMMA = 0.5

# ── Disease classes (PlantVillage 38 classes) ─────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)   # 38

# ── RAG / Knowledge retrieval ─────────────────────────────────
SBERT_MODEL   = "all-MiniLM-L6-v2"   # Sentence-BERT model
FAISS_INDEX   = os.path.join(BASE_DIR, "knowledge_base", "faiss_index.bin")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base", "docs")
TOP_K_DOCS    = 3                     # Retrieved documents per query

# ── Web app ──────────────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
DEBUG    = True
