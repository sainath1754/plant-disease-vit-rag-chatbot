#!/usr/bin/env python3
# ============================================================
# web/app.py — Flask Web Application
# Usage:  python web\app.py   then open http://localhost:5000
# ============================================================

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from PIL   import Image

from config              import CHECKPOINT_DIR, CLASS_NAMES, NUM_CLASSES, \
                                WEB_HOST, WEB_PORT, DEBUG
from models.rag_module   import RAGAdvisoryModule
from utils.preprocessing import preprocess_image

app    = Flask(__name__, template_folder="templates", static_folder="static")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_rag   = None


# ── Same all-keys detection used in inference.py ─────────────

def _load_model():
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("[Web] WARNING: No checkpoint found. Run train.py first.")
        from models.vit_model import VisionTransformer
        return VisionTransformer().to(DEVICE)

    saved      = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = saved["model"]
    all_keys   = " ".join(state_dict.keys())

    if "mlp.fc1" in all_keys or "patch_embed.proj.weight" in all_keys:
        print("[Web] Detected timm ViT checkpoint — loading timm model …")
        import timm
        model = timm.create_model("vit_base_patch16_224",
                                  pretrained=False, num_classes=NUM_CLASSES)
    else:
        print("[Web] Detected custom ViT checkpoint — loading custom model …")
        from models.vit_model import VisionTransformer
        model = VisionTransformer()

    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"[Web] Model ready  epoch={saved.get('epoch','?')}  "
          f"val_acc={saved.get('val_acc', 0.0):.2f}%")
    return model


def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def get_rag():
    global _rag
    if _rag is None:
        _rag = RAGAdvisoryModule()
    return _rag


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    try:
        pil_img = Image.open(file.stream).convert("RGB")
        tensor  = preprocess_image(pil_img).to(DEVICE)

        model = get_model()
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]

        top_probs, top_idx = probs.topk(5)
        top_preds = [
            {"class": CLASS_NAMES[i.item()],
             "confidence": round(p.item() * 100, 2)}
            for p, i in zip(top_probs, top_idx)
        ]

        pred_class = top_preds[0]["class"]
        result     = get_rag().generate_recommendation(pred_class)

        return jsonify({
            "success"        : True,
            "predicted_class": pred_class,
            "confidence"     : top_preds[0]["confidence"],
            "crop"           : result["crop"],
            "disease"        : result["disease"],
            "recommendation" : result["recommendation"],
            "top_predictions": top_preds,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    context = data.get("disease_context", "")
    if not message:
        return jsonify({"reply": "Please enter a question."}), 400
    try:
        docs  = get_rag().retrieve(f"{context} {message}")
        parts = [f"Based on agronomic knowledge for '{context or 'your query'}':\n"]
        for d in docs[:2]:
            parts.append(f"• {d['title']}: {d['text'][:300].strip()}…")
        parts.append("\nConsult your local extension office for field-specific advice.")
        return jsonify({"reply": "\n".join(parts)})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 500


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    get_model()
    get_rag()
    print(f"\n[Web] Running at http://localhost:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, debug=DEBUG, use_reloader=False)