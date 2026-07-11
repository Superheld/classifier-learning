# %% [markdown]
# # Track B — P1 „Vorbereiten" (allgemein für alle Encoder)
#
# ## Der Generationssprung (Gen 1 → Gen 2)
#
# Track A rechnete mit **Wort-Zählungen** (TF-IDF): „card" ist „card", und „card"
# und „karte" sind fremde Symbole. Track B rechnet mit **Bedeutung**: ein
# vortrainierter **Encoder** macht aus jedem Text einen **dichten Vektor**, in dem
# semantisch Ähnliches nah beieinander liegt. Der Klassifikator-**Kopf** (LogReg,
# kNN …) lernt dann auf diesen Vektoren.
#
# **feature extraction** = der Encoder bleibt *eingefroren* (kein Training), wir
# nutzen ihn nur als Merkmals-Fabrik. Das Menü hat zwei Achsen: **Encoder**
# (klein → stark) × **Kopf**.
#
# Dieses Notebook ist die **gemeinsame Vorarbeit** für *alle* Encoder-Notebooks:
# Passen unsere Texte überhaupt in den Encoder? Und wie funktioniert das
# Encodieren + Cachen? Die einzelnen Encoder (`klein/minilm.py`, `stark/…`) bauen
# darauf auf und wiederholen das nicht.

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except (NameError, AttributeError):
    pass

root = Path.cwd()
while not (root / "data_utils.py").exists() and root != root.parent:
    root = root.parent
for d in (root, root / "03_track_b_embeddings"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import matplotlib.pyplot as plt

from data_utils import load_banking77

train_texts, _ = load_banking77("train")
print(f"{len(train_texts)} Trainingstexte")

# %% [markdown]
# ## Kontextfenster — passt der Text in den Encoder?
#
# Jeder Encoder hat ein **Kontextfenster** (`max_seq_length`, in *Tokens*): alles
# darüber wird abgeschnitten. Bei GNAD war genau das das Problem (97 % der Artikel
# länger als 128 Tokens → halbe Information verloren). Deshalb der Pflicht-Check
# *vor* dem Encodieren.
#
# **Tokens ≠ Wörter:** ein Tokenizer zerlegt Wörter in Sub-Wort-Stücke
# („waiting" → „wait", „##ing"). Wir messen mit einem repräsentativen Tokenizer
# (MiniLM/BERT-Stil) — andere Encoder liegen ähnlich.

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
token_counts = [len(tok.encode(t, add_special_tokens=True)) for t in train_texts]

import numpy as np

tc = np.array(token_counts)
print(f"Tokens je Anfrage: median {int(np.median(tc))}, 95%-Perzentil {int(np.percentile(tc,95))}, max {tc.max()}")
for window in (256, 512):
    über = (tc > window).mean() * 100
    print(f"  Kontextfenster {window}: {über:.1f} % der Texte würden abgeschnitten")

plt.figure(figsize=(8, 4))
plt.hist(tc, bins=40, color="#5B8FF9")
plt.axvline(256, color="#E8684A", linestyle="--", label="Fenster 256 (MiniLM)")
plt.xlabel("Tokens je Anfrage")
plt.ylabel("Anzahl")
plt.title("banking77 — Token-Längen (Track B P1)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Deuten
# - **Alles kurz** (median ~13 Tokens, max ~110) → passt in *jedes* gängige
#   Encoder-Fenster (256/512). **Kein Chunking nötig.**
# - Das ist das **invertierte GNAD-Problem**: dort mussten lange Dokumente
#   gestückelt + gemittelt werden; hier ist das Kontextfenster kein Thema.
# - Genau darum sind kurze Texte das **Heimspiel** von Embeddings (CURRICULUM P4).

# %% [markdown]
# ## Encodieren & Cachen (das Werkzeug)
#
# Das Rechnen der Vektoren ist teuer (Modell-Forward-Pass je Text), aber **einmalig**
# pro Encoder — darum cachen wir. `embeddings.encode(model_name, texts, key, split)`
# lädt den Encoder (beim ersten Mal Download), rechnet auf der Apple-GPU (MPS) und
# legt das Ergebnis als `.npy` ab. Beim zweiten Lauf kommt es sofort aus dem Cache.
#
# Jedes Encoder-Notebook ruft nur noch `encode(...)` auf — die Vorarbeit hier
# gilt für alle.
