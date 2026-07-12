# %% [markdown]
# # Track B · klein — MiniLM + LogReg (banking77)
#
# ## P2 „Bauen" — der erste Encoder
#
# **MiniLM** (`all-MiniLM-L6-v2`) ist der Start-Encoder der *kleinen* Familie:
# schnell, **384** Dimensionen, eine solide Baseline für Gen 2. Die allgemeine
# Vorarbeit — der Generationssprung Gen 1 → Gen 2, der Kontextfenster-Check und die
# Encode/Cache-Mechanik — steht in **`../vorbereiten.py`** und wird hier nicht
# wiederholt. Dieses Notebook ist der erste konkrete Bau: **encodieren → Kopf →
# messen.**
#
# Die Latte: Track A getunt = **90,25 %** (TF-IDF + LogReg). Kann ein *kleiner*
# semantischer Encoder das schon schlagen?

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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_banking77
from eval_utils import evaluate_and_save
from embeddings import encode

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Intents: {len(set(train_labels))}")

# %% [markdown]
# ## Schritt 1 — Encodieren: Text → Bedeutungsvektor
#
# Jede Anfrage wird zu einem 384-dim Vektor, in dem semantisch Ähnliches nah
# beieinander liegt. Teuer zu rechnen, aber einmalig pro Encoder — darum gecacht
# (`.npy`); der zweite Lauf ist sofort da. Mechanik: `../vorbereiten.py`.

# %%
X_train = encode(MODEL, train_texts, "minilm", "train")
X_test = encode(MODEL, test_texts, "minilm", "test")
print(f"X_train: {X_train.shape}  (Anfragen × Embedding-Dimension)")

# %% [markdown]
# ## Kurzer Blick: was *ist* so ein Embedding?
#
# Keine Wort-Zählung mehr (wie TF-IDF), sondern 384 dichte Kommazahlen pro Anfrage.
# Der Encoder liefert sie **L2-normalisiert** (Länge ≈ 1) — praktisch, weil dann
# Cosinus-Ähnlichkeit und lineare Köpfe gut damit rechnen.

# %%
import numpy as np

print(f"Erste Anfrage: {train_texts[0]!r}")
print(f"Vektor (erste 5 von 384): {np.round(X_train[0][:5], 3)}")
print(f"Vektorlänge (L2-Norm):     {np.linalg.norm(X_train[0]):.3f}  (≈ 1, normalisiert)")

# %% [markdown]
# ## Schritt 2 — Kopf trainieren
#
# Auf die eingefrorenen Embeddings kommt **derselbe Kopf wie der Gen-1-Sieger**
# (LogReg) — das macht den Vergleich fair: gleicher Klassifikator, nur die Merkmale
# sind jetzt Bedeutung statt Wort-Zählung. Weil die Embeddings dicht sind (nicht
# spärlich wie TF-IDF), ist das Training schnell.

# %%
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)
p = clf.predict(X_test)

# %% [markdown]
# ## Schritt 3 — Messen
#
# Gleiches Besteck wie in Track A (Accuracy + Macro-F1), damit die Zahlen direkt
# vergleichbar sind.

# %%
acc = accuracy_score(test_labels, p)
macro_f1 = f1_score(test_labels, p, average="macro")
print(f"MiniLM + LogReg   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"Track A (TF-IDF+LogReg getunt): 90.25 %   ← die Latte")

evaluate_and_save("B_minilm_logreg", test_labels, p,
                  model="MiniLM + LogReg", note="P2 plain, frozen embeddings",
                  scores=clf.predict_proba(X_test), classes=clf.classes_, score_type="proba")

# %% [markdown]
# ## Wo irrt das Modell?
#
# Zwei Blicke wie in Track A — die Confusion zeigt, *welche* Intent-Paare der
# Encoder verwechselt.

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

plot_confusion_matrix(test_labels, p, title="MiniLM + LogReg — Confusion Matrix")
plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)

# %% [markdown]
# ## Deuten
#
# - Schlägt der **kleine** Encoder schon die getunte TF-IDF-Latte, oder braucht es
#   die starken Encoder (mpnet/E5) aus P3? (Und das *plain*, ohne einen getunten Kopf.)
# - Verwechselt MiniLM **dieselben** Intent-Paare wie TF-IDF (→ liegt an den Daten,
#   die Paare sind inhärent nah) oder **andere** (→ die Semantik trägt wirklich etwas
#   bei, wo Wort-Überlappung versagte)? Der Abgleich mit Track As Confusion ist der Test.
#
# **✓ Checkpoint (CURRICULUM):** In welchem Szenario schlagen Embeddings
# Bag-of-Words *deutlich* — und warum? (Stichwort: wenig Labels, Synonyme, kein
# gemeinsames Vokabular zwischen Anfrage und Trainingstext.)
