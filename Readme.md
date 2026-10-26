# 🌿 Plant Pathology Recognition Using Vision Transformer in a Knowledge Retrieval-Assisted Conversational Agronomic Advisory Framework

> A deep learning system that detects crop diseases from leaf images using Vision Transformers (ViT) and delivers treatment recommendations through a RAG-based conversational advisory module.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Why This Project Stands Out](#why-this-project-stands-out)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [How the Project Works](#how-the-project-works)
- [Installation & Setup](#installation--setup)
- [Execution Steps](#execution-steps)
- [Commands for Project Review Panel](#commands-for-project-review-panel)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Overview

Agricultural crop losses due to plant diseases account for **20–40% of global production annually**. This project addresses that challenge by combining:

1. **Vision Transformer (ViT)** — trained on the PlantVillage dataset (87,000+ labeled leaf images, 38 disease classes, 14 crop species) to classify plant diseases with **97.8% accuracy**.
2. **Retrieval-Augmented Generation (RAG)** — using Sentence-BERT embeddings and FAISS vector search to retrieve agronomic knowledge and generate actionable treatment advice via a large language model.
3. **Web Interface** — a user-facing platform where farmers can upload leaf images and receive disease diagnosis plus crop management recommendations in real time.

---

## Why This Project Stands Out

| Feature | Benefit |
|---|---|
| Vision Transformer over CNN | Captures **global spatial relationships** across the entire leaf surface, not just local patches |
| Patch-based self-attention | Detects **fine-grained pathological features** missed by traditional convolution |
| RAG-based advisory | Goes beyond classification — gives **practical treatment plans** grounded in agricultural knowledge |
| PlantVillage dataset | Industry-standard benchmark with **87,000+ images** ensuring reliability |
| 97.8% validation accuracy | Outperforms CNN (92.4%), MobileNet (94.1%), and ResNet50 (96.3%) |
| Web deployment | Accessible to farmers without technical expertise |

Traditional CNN-based models are limited to local receptive fields. This project's ViT architecture uses **multi-head self-attention** to model long-range dependencies across leaf regions, making it significantly better at distinguishing visually similar disease symptoms.

---

## Project Structure

```
plant_disease_vit/                  # Root project directory
│
├── checkpoints/                    # Saved model weights from training
│   ├── best_model.pth              # Best performing checkpoint (used for inference)
│   ├── epoch_001.pth               # Checkpoint saved after epoch 1
│   ├── epoch_002.pth               # Checkpoint saved after epoch 2
│   ├── epoch_003.pth               # Checkpoint saved after epoch 3
│   ├── epoch_004.pth               # Checkpoint saved after epoch 4
│   └── epoch_005.pth               # Checkpoint saved after epoch 5
│
├── data/                           # PlantVillage dataset (raw and processed)
│
├── knowledge_base/                 # Agronomic documents for RAG retrieval
│
├── models/                         # Core model architecture and RAG logic
│   ├── __init__.py
│   ├── vit_model.py                # Vision Transformer architecture definition
│   └── rag_module.py               # RAG pipeline (Sentence-BERT + FAISS + LLM)
│
├── outputs/                        # Evaluation results, plots, and reports
│
├── TEST-IMAGES/                    # Sample leaf images for inference testing
│
├── utils/                          # Helper utilities (metrics, visualizations)
│
├── venv/                           # Python virtual environment (not committed to Git)
│
├── web/                            # Web application module
│   ├── __init__.py
│   ├── app.py                      # Flask web server entry point
│   ├── static/
│   │   └── style.css               # Frontend stylesheet
│   └── templates/
│       └── index.html              # Leaf image upload and results UI
│
├── config.py                       # Hyperparameters and path configurations
├── evaluate.py                     # Model evaluation script
├── inference.py                    # Single-image inference script
├── requirements.txt                # All Python dependencies
├── commands for execution.txt      # Quick-reference command list
└── README.md                       # This file
```

---

## File Descriptions

### `models/vit_model.py`
Defines the **Vision Transformer (ViT) architecture**. Handles image patch extraction (16×16 pixels), linear patch embedding, positional encoding, and stacks 12 transformer encoder layers with multi-head self-attention and feed-forward blocks. The classification token output is passed to a fully connected head for 38-class disease prediction.

### `models/rag_module.py`
Implements the complete **Retrieval-Augmented Generation pipeline**. Encodes disease labels using Sentence-BERT, performs FAISS vector similarity search against the knowledge base, retrieves the most relevant agronomic documents, and passes them as context to an LLM to generate structured treatment recommendations.

### `evaluate.py`
Loads `checkpoints/best_model.pth` and runs the trained model on the test set. Computes and reports **accuracy, precision, recall, F1-score**, and generates the confusion matrix. Output results are saved to the `outputs/` folder.

### `inference.py`
Accepts a single leaf image path as input, loads `checkpoints/best_model.pth`, runs the ViT classification pipeline, and triggers the RAG advisory module to return the predicted disease class along with treatment recommendations. Used for quick testing on images in `TEST-IMAGES/`.

### `config.py`
Central configuration file. Stores all model hyperparameters (patch size=16, embedding dim=768, attention heads=12, batch size=32, learning rate=0.0001), dataset paths, checkpoint paths, and RAG settings.

### `web/app.py`
The Flask web server entry point. Serves the frontend interface (`templates/index.html`), handles leaf image uploads, routes images through the ViT classification model, triggers the RAG advisory module, and returns disease predictions with treatment recommendations to the user.

### `web/templates/index.html`
The frontend UI page. Provides the image upload form, displays the predicted disease class, and renders the RAG-generated treatment recommendations returned by the Flask server.

### `web/static/style.css`
Stylesheet for the web interface. Controls the layout and visual presentation of the upload form and results page.

### `checkpoints/best_model.pth`
The saved weights of the best-performing Vision Transformer model from training. This is the primary model file loaded during evaluation, inference, and web serving.

### `checkpoints/epoch_00X.pth`
Per-epoch weight snapshots saved during training. These allow rollback to any specific training epoch if needed for analysis or comparison.

### `knowledge_base/`
Contains the agronomic reference documents used by the RAG module. These documents cover disease-specific treatment protocols, pesticide recommendations, and crop management guidance that the LLM uses to generate advisory responses.

### `TEST-IMAGES/`
A collection of sample leaf images for quick inference testing and panel demonstration without needing to source external images.

### `outputs/`
Stores all evaluation artefacts including accuracy/loss graphs, confusion matrix plots, and classification reports generated by `evaluate.py`.

### `requirements.txt`
Lists all Python package dependencies required to run the project. Install with `pip install -r requirements.txt` inside the activated virtual environment.

---

## How the Project Works

```
User Uploads Leaf Image
         │
         ▼
  Image Preprocessing
  (resize, normalize)              ← config.py
         │
         ▼
  Vision Transformer (ViT)         ← models/vit_model.py
  - Patch embedding (16×16)
  - Positional encoding
  - 12× Multi-Head Self-Attention
  - Feed-Forward layers
         │
         ▼
  Disease Class Predicted
  (e.g., "Tomato Early Blight")
         │
         ▼
  RAG Advisory Module              ← models/rag_module.py
  - Sentence-BERT query encoding
  - FAISS vector similarity search ← knowledge_base/
  - LLM treatment generation
         │
         ▼
  Web Interface Response           ← web/app.py + templates/index.html
  (Disease label + Treatment Recommendations)
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for training; CPU works for inference)
- pip package manager

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/plant_disease_vit.git
cd plant_disease_vit
```

### Step 2 — Create and Activate a Virtual Environment
```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```
> ⚠️ Always ensure the virtual environment is activated before running any project commands. Your terminal prompt should show `(venv)` as a prefix.

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure Paths
Review `config.py` and confirm dataset and checkpoint paths match your local setup.

---

## Execution Steps

> ✅ **Note:** The model has already been trained, evaluated, and inference-tested prior to the panel review. `checkpoints/best_model.pth` contains the final trained weights. Steps 3 and 4 below are included for reproducibility documentation only. **For the panel demo, start directly from Step 5.**

### 1. Activate the Virtual Environment *(required every session)*
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies *(first time only)*
```bash
pip install -r requirements.txt
```

### 3. Train the Vision Transformer Model *(already completed — weights saved)*
```bash
python models/vit_model.py
```
Trains ViT on the PlantVillage dataset. Saves per-epoch and best checkpoints to `checkpoints/`.

### 4. Evaluate Model Performance *(already completed — results in outputs/)*
```bash
python evaluate.py
```
Loads `checkpoints/best_model.pth` and generates accuracy, F1-score, and confusion matrix.

### 5. Run Inference on a Test Image ✅ *(ready for demo)*
```bash
python inference.py --image TEST-IMAGES/<your_leaf_image>.jpg
```
Returns the predicted disease class and RAG-generated treatment recommendation.

### 6. Launch the Web Application ✅ *(ready for demo)*
```bash
python web/app.py
```
Access the application at `http://localhost:5000`

---

## Commands for Project Review Panel

> The model is **pre-trained and ready** at `checkpoints/best_model.pth`. Use the commands below in order during your panel demonstration.

```bash
# Step 1 — Activate the virtual environment (always do this first)
venv\Scripts\activate

# Step 2 — Confirm all dependencies are installed
pip list

# Step 3 — Confirm GPU / CPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# Step 4 — Confirm saved model weights exist
python -c "import os; print('Model found:', os.path.exists('checkpoints/best_model.pth'))"

# Step 5 — Show model parameter count (proves ViT architecture)
python -c "from models.vit_model import VisionTransformer; m = VisionTransformer(); print(sum(p.numel() for p in m.parameters()), 'total parameters')"

# Step 6 — Re-run evaluation to show live metrics
python evaluate.py

# Step 7 — Run inference on a sample test image
python inference.py --image TEST-IMAGES/<your_leaf_image>.jpg

# Step 8 — Launch the web application for live demo
python web/app.py
```

---

## Model Performance

| Method | Architecture | Accuracy (%) | Remarks |
|---|---|---|---|
| CNN-Based Model | Convolutional Neural Network | 92.4 | Local feature extractor; weak global context |
| MobileNet | Lightweight CNN | 94.1 | Fast but reduced accuracy for visually similar diseases |
| ResNet50 | Deep Residual Network | 96.3 | Deeper layers for better feature learning |
| **Proposed ViT Model** | **Vision Transformer** | **97.8** | Global pattern recognition via patch embedding and self-attention |

**Training Configuration:**

| Parameter | Value |
|---|---|
| Patch Size | 16 × 16 px |
| Embedding Dimension | 768 |
| Attention Heads | 12 |
| Batch Size | 32 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Dropout | 0.1 |
| Loss Function | Cross-Entropy |
| Dataset Split | 70% Train / 15% Val / 15% Test |

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Vision Transformer | HuggingFace Transformers |
| Image Processing | OpenCV, Torchvision |
| Semantic Embeddings | Sentence-BERT |
| Vector Search | FAISS |
| Language Generation | Large Language Model (LLM) |
| Web Framework | Flask / FastAPI |
| Dataset | PlantVillage (87,000+ images, 38 classes) |

---

## References

1. Balafas et al. (2023). Machine Learning and Deep Learning for Plant Disease Classification. *IEEE Access, 11*, 114352–114377.
2. Maurya et al. (2024). A Lightweight Meta-Ensemble Approach for Plant Disease Detection. *IEEE Access, 12*, 28096–28108.
3. Shafik et al. (2023). A Systematic Literature Review on Plant Disease Detection. *IEEE Access, 11*, 59174–59203.
4. PlantVillage Dataset — Hughes & Salathé (2015).

> Full reference list available in the accompanying research paper.

---

## License

This project is developed for academic research purposes. See `LICENSE` for details.

---

*Built as part of a research paper on Vision Transformer-based plant pathology recognition and knowledge-assisted agronomic advisory systems.*