# ============================================================
# models/rag_module.py — RAG Advisory Module
#
# Pipeline (per paper):
#   disease label
#     → Sentence-BERT embedding
#     → FAISS similarity search  (top-k docs)
#     → retrieved context + query
#     → LLM (or template engine) generates treatment advice
# ============================================================

import os
import json
import pickle
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SBERT_MODEL, FAISS_INDEX, KNOWLEDGE_DIR, TOP_K_DOCS


# ── Lazy imports (heavy libraries) ───────────────────────────

def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL)


def _load_faiss():
    import faiss
    return faiss


# ── RAG Module ────────────────────────────────────────────────

class RAGAdvisoryModule:
    """
    Retrieval-Augmented Generation module:
      1. Encodes disease label with Sentence-BERT
      2. Searches FAISS index for most relevant agronomic docs
      3. Generates treatment recommendation
    """

    def __init__(self, knowledge_dir: str = KNOWLEDGE_DIR,
                 index_path: str = FAISS_INDEX, top_k: int = TOP_K_DOCS):
        self.knowledge_dir = knowledge_dir
        self.index_path    = index_path
        self.top_k         = top_k

        self._sbert    = None   # loaded on first use
        self._index    = None
        self._docs     = []     # list of {"title": ..., "text": ...}
        self._doc_meta = []     # parallel metadata list

        self._build_or_load_index()

    # ── Index management ──────────────────────────────────────

    def _build_or_load_index(self):
        """Build FAISS index from JSON knowledge docs, or load cached version."""
        faiss = _load_faiss()

        meta_path = self.index_path + ".meta.pkl"
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        if os.path.exists(self.index_path) and os.path.exists(meta_path):
            print("[RAG] Loading cached FAISS index …")
            self._index = faiss.read_index(self.index_path)
            with open(meta_path, "rb") as f:
                self._docs, self._doc_meta = pickle.load(f)
            print(f"[RAG] Index loaded: {self._index.ntotal} vectors.")
            return

        print("[RAG] Building FAISS index from knowledge docs …")
        self._sbert = _load_sentence_transformer()
        self._load_documents()

        if not self._docs:
            print("[RAG] Warning: No documents found. Using built-in fallback knowledge.")
            self._docs     = _FALLBACK_DOCS
            self._doc_meta = [{"source": "built-in"}] * len(_FALLBACK_DOCS)

        texts      = [d["title"] + " " + d["text"] for d in self._docs]
        embeddings = self._sbert.encode(texts, convert_to_numpy=True,
                                        show_progress_bar=True).astype("float32")

        dim          = embeddings.shape[1]
        self._index  = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)

        faiss.write_index(self._index, self.index_path)
        with open(meta_path, "wb") as f:
            pickle.dump((self._docs, self._doc_meta), f)

        print(f"[RAG] Index built and saved: {self._index.ntotal} vectors.")

    def _load_documents(self):
        """Load all .json files from knowledge_dir."""
        if not os.path.isdir(self.knowledge_dir):
            return
        for fname in os.listdir(self.knowledge_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.knowledge_dir, fname)
                with open(fpath) as f:
                    data = json.load(f)
                    for entry in data:
                        self._docs.append(entry)
                        self._doc_meta.append({"source": fname})

    # ── Public API ────────────────────────────────────────────

    def retrieve(self, query: str) -> list[dict]:
        """
        Returns top-k most relevant documents for the query.
        Each doc: {"title": str, "text": str, "score": float}
        """
        if self._sbert is None:
            self._sbert = _load_sentence_transformer()

        q_vec = self._sbert.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self._index.search(q_vec, self.top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._docs):
                doc = dict(self._docs[idx])
                doc["score"] = float(dist)
                results.append(doc)
        return results

    def generate_recommendation(self, disease_label: str) -> dict:
        """
        Main entry point.
        Returns:
            {
              "disease"       : str,
              "crop"          : str,
              "retrieved_docs": [...],
              "recommendation": str
            }
        """
        crop, disease = _parse_label(disease_label)
        query         = f"{crop} {disease} treatment pesticide management"

        retrieved = self.retrieve(query)
        context   = "\n\n".join(
            f"[{i+1}] {d['title']}\n{d['text']}"
            for i, d in enumerate(retrieved)
        )

        recommendation = _compose_recommendation(crop, disease, retrieved)

        return {
            "disease"       : disease,
            "crop"          : crop,
            "retrieved_docs": retrieved,
            "recommendation": recommendation,
        }


# ── Helpers ───────────────────────────────────────────────────

def _parse_label(label: str) -> tuple[str, str]:
    """
    'Tomato___Early_blight'  →  ('Tomato', 'Early blight')
    'Apple___healthy'        →  ('Apple', 'Healthy (No disease detected)')
    """
    parts   = label.split("___")
    crop    = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    if disease.lower() == "healthy":
        disease = "Healthy (No disease detected)"
    return crop, disease


def _compose_recommendation(crop: str, disease: str,
                             docs: list[dict]) -> str:
    """
    Compose a structured treatment recommendation from retrieved docs.
    In production replace this with an LLM call (e.g. GPT / Llama).
    """
    if "healthy" in disease.lower():
        return (
            f"✅ Your {crop} plant appears HEALTHY.\n\n"
            "Preventive measures:\n"
            "• Continue regular crop monitoring.\n"
            "• Maintain proper irrigation and nutrient balance.\n"
            "• Rotate crops annually to prevent soil-borne pathogens.\n"
            "• Remove and destroy any infected plant debris promptly."
        )

    sections = [
        f"🌿 Crop   : {crop}",
        f"🦠 Disease: {disease}",
        "",
        "── Treatment Recommendations ──────────────────────────",
    ]

    if docs:
        for i, doc in enumerate(docs, 1):
            sections.append(f"\n📄 Source {i}: {doc['title']}")
            # Use first 600 characters of the retrieved knowledge
            excerpt = doc["text"][:600].strip()
            sections.append(excerpt)
    else:
        sections.append("No specific knowledge documents retrieved.")

    sections += [
        "",
        "── General Agronomic Advice ───────────────────────────",
        "• Apply recommended fungicide / bactericide as per label instructions.",
        "• Remove and destroy infected leaves to limit disease spread.",
        "• Avoid overhead irrigation; use drip irrigation to keep foliage dry.",
        "• Maintain proper plant spacing for good air circulation.",
        "• Monitor weather conditions — humid/warm conditions accelerate spread.",
        "• Consult your local agricultural extension office for region-specific advice.",
    ]

    return "\n".join(sections)


# ── Fallback built-in knowledge (used when no JSON docs found) ─

_FALLBACK_DOCS = [
    {
        "title": "Tomato Early Blight Management",
        "text" : (
            "Early blight (Alternaria solani) causes dark, concentric ring lesions on lower leaves. "
            "Treatment: Apply chlorothalonil or mancozeb fungicide every 7-10 days. "
            "Remove infected leaves. Mulch to reduce soil splash. "
            "Use disease-resistant tomato varieties. Ensure adequate calcium supply."
        )
    },
    {
        "title": "Tomato Late Blight Control",
        "text" : (
            "Late blight (Phytophthora infestans) produces water-soaked lesions turning brown. "
            "Treatment: Apply copper-based or systemic fungicide (metalaxyl). "
            "Destroy infected plants completely. Avoid planting in previously infected areas. "
            "Use certified disease-free seeds."
        )
    },
    {
        "title": "Potato Late Blight Management",
        "text" : (
            "Potato late blight is caused by Phytophthora infestans. "
            "Apply preventive fungicide sprays of mancozeb or cymoxanil. "
            "Hill up potato rows to protect tubers. Harvest early if blight is severe. "
            "Store only disease-free tubers."
        )
    },
    {
        "title": "Apple Scab Control",
        "text" : (
            "Apple scab (Venturia inaequalis) presents as olive-green spots. "
            "Spray with captan, myclobutanil, or lime sulfur during pre-bloom. "
            "Rake and destroy fallen leaves. Prune for good canopy airflow. "
            "Apply dormant oil spray in early spring."
        )
    },
    {
        "title": "Corn Common Rust Treatment",
        "text" : (
            "Corn common rust (Puccinia sorghi) shows as small, brick-red pustules. "
            "Apply triazole fungicides (propiconazole) at early infection stage. "
            "Plant resistant hybrids. Scout fields regularly from V6 stage onward. "
            "Avoid excessive nitrogen fertilization."
        )
    },
    {
        "title": "Grape Black Rot Management",
        "text" : (
            "Grape black rot (Guignardia bidwellii) causes circular leaf lesions. "
            "Apply mancozeb, myclobutanil, or captan from bud break. "
            "Remove mummified berries. Prune to improve air circulation. "
            "Maintain weed-free rows to reduce humidity."
        )
    },
    {
        "title": "Pepper Bacterial Spot Control",
        "text" : (
            "Bacterial spot (Xanthomonas campestris) causes water-soaked leaf spots. "
            "Apply copper-based bactericides. Use certified disease-free transplants. "
            "Avoid working in wet fields to prevent spread. "
            "Remove and destroy severely infected plants."
        )
    },
    {
        "title": "General Plant Disease Preventive Measures",
        "text" : (
            "1. Crop rotation: rotate susceptible crops every 2-3 seasons. "
            "2. Sanitation: remove plant debris after harvest. "
            "3. Resistant varieties: use certified disease-resistant seeds. "
            "4. Proper fertilization: balanced NPK reduces disease susceptibility. "
            "5. Water management: avoid leaf wetness; use drip irrigation. "
            "6. Regular scouting: early detection is the most cost-effective control."
        )
    },
]
