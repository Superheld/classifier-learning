"""
Track B (P1): Text → Bedeutungsvektor. Encoder laden, encodieren, cachen.

Track-B-lokal, weil es sentence-transformers braucht (im Gegensatz zu den
geteilten Root-Helfern data_utils/eval_utils). Embeddings sind teuer zu rechnen,
darum cachen wir sie pro (Encoder, Split) als .npy — beim zweiten Lauf kommt es
aus dem Cache. Der Cache-Ordner ist gitignored (*.npy, regenerierbar).
"""

import os

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def encode(model_name, texts, cache_key, split):
    """Text-Liste → Embedding-Matrix (N × dim), gecacht.

    model_name : HF-Name des Encoders, z.B. "sentence-transformers/all-MiniLM-L6-v2"
    cache_key  : kurzer Name für die Cache-Datei, z.B. "minilm"
    split      : "train" / "test" (getrennte Cache-Dateien)

    normalize_embeddings=True (Einheitslänge) — gut für LogReg und Cosinus-kNN.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{cache_key}_{split}.npy")
    if os.path.exists(path):
        print(f"[cache] {cache_key}/{split}: geladen")
        return np.load(path)

    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[compute] {model_name} / {split} auf {device} …")
    model = SentenceTransformer(model_name, device=device)
    vecs = model.encode(
        texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    np.save(path, vecs)
    print(f"[gespeichert] {cache_key}/{split}: shape={vecs.shape}")
    return vecs


def context_window(model_name):
    """Kontextfenster (max_seq_length) des Encoders — der P1-Check aus der CURRICULUM."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name).max_seq_length
