# %% [markdown]
# # Track B — P1 „Vorbereiten" (allgemein für alle Encoder)
#
# ## Der Generationssprung (Gen 1 → Gen 2)
#
# Track A rechnete mit **Wort-Zählungen** (TF-IDF): „card" ist „card", und „card"
# und „karte" sind fremde Symbole ohne Beziehung. Track B rechnet mit **Bedeutung**:
# ein vortrainierter **Encoder** macht aus jedem Text einen **dichten Vektor**, in
# dem semantisch Ähnliches nah beieinander liegt — auch ohne gemeinsame Wörter. Der
# Klassifikator-**Kopf** (LogReg, kNN …) lernt dann auf diesen Vektoren.
#
# Der Fachbegriff ist **feature extraction**: der Encoder bleibt **eingefroren**
# (kein Training!), wir nutzen ihn nur als Merkmals-Fabrik. Das ist die Grenze von
# Track B — den Encoder *aufzutauen* und mitzutrainieren ist erst Track C (Gen 3a).
#
# ## Die Landkarte des Tracks
#
# Das Menü hat **zwei Achsen**, und die Notebooks sind danach sortiert:
#
# | Achse | Werte | wo |
# |---|---|---|
# | **Encoder** (das Lineal) | klein → stark | `klein/minilm.py`, `stark/mpnet.py`, `stark/e5.py` |
# | **Kopf** (der Klassifikator) | LogReg · kNN · LinearSVC … | im Kopf-Zyklus (P3, in `mpnet.py`) |
#
# Dieses Notebook ist die **gemeinsame Vorarbeit** für *alle* Encoder-Notebooks und
# beantwortet zwei Fragen, bevor irgendein Modell gebaut wird: (1) Passen unsere
# Texte überhaupt in die Encoder? (2) Wie funktioniert das Encodieren + Cachen?

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
# Jeder Encoder hat ein **Kontextfenster** (`max_seq_length`, gemessen in *Tokens*):
# alles darüber wird **abgeschnitten** — der Rest des Textes ist für das Modell
# unsichtbar. Deshalb ein Pflicht-Check *vor* dem Encodieren.
#
# **Tokens ≠ Wörter:** ein Tokenizer zerlegt Wörter in Sub-Wort-Stücke
# („waiting" → „wait", „##ing"). Wir zählen mit einem repräsentativen Tokenizer
# (MiniLM/BERT-Stil); andere gängige Encoder liegen ähnlich.
#
# **Warum das hier ein Nicht-Problem ist — und wann es eins wäre:** Im
# Vorgänger-Datensatz **GNAD** (lange deutsche News-Artikel) waren 97 % der Texte
# länger als 128 Tokens — halbe Information ging beim Abschneiden verloren, man musste
# stückeln und mitteln. banking77 sind kurze Kundenanfragen; wir erwarten das
# **umgekehrte** Bild.

# %%
from transformers import AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
token_counts = [len(tok.encode(t, add_special_tokens=True)) for t in train_texts]
tc = np.array(token_counts)

print(f"Tokens je Anfrage: median {int(np.median(tc))}, "
      f"95%-Perzentil {int(np.percentile(tc, 95))}, max {tc.max()}")
for window in (256, 512):
    über = (tc > window).mean() * 100
    print(f"  Kontextfenster {window}: {über:.1f} % der Texte würden abgeschnitten")

# %% [markdown]
# Und dasselbe als Bild — die ganze Verteilung auf einen Blick:

# %%
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
#   Encoder-Fenster (256/512). **Kein Chunking nötig** — 0 % wird abgeschnitten.
# - Das ist das **invertierte GNAD-Problem**: dort mussten lange Dokumente gestückelt
#   werden; hier ist das Kontextfenster schlicht kein Thema.
# - Genau darum sind kurze Texte das **Heimspiel** von Embeddings (CURRICULUM P4):
#   der ganze Sinn steckt in einem Fenster, nichts geht verloren.

# %% [markdown]
# ## Encodieren & Cachen (das Werkzeug)
#
# Das Rechnen der Vektoren ist teuer (ein Modell-Forward-Pass je Text), aber
# **einmalig** pro Encoder — darum cachen wir.
# `embeddings.encode(model_name, texts, key, split)` lädt den Encoder (beim ersten
# Mal Download), rechnet auf der Apple-GPU (**MPS**) und legt das Ergebnis als `.npy`
# ab. Beim zweiten Lauf kommt es sofort aus dem Cache (der Ordner ist gitignored —
# regenerierbar).
#
# Jedes Encoder-Notebook ruft dann nur noch `encode(...)` auf. Ein Encoder wie **E5**
# braucht zusätzlich einen `prefix=` — auch das kapselt der Helfer (siehe `stark/e5.py`).
