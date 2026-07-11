# %% [markdown]
# # Track B · klein — MiniLM + LogReg (banking77)
#
# ## P2 „Bauen" — erster Encoder
#
# **MiniLM** (`all-MiniLM-L6-v2`) ist der Start-Encoder der kleinen Familie:
# schnell, 384 Dimensionen, solide Baseline. Die allgemeine Vorarbeit (Konzept,
# Kontextfenster-Check, Encode/Cache-Mechanik) steht in `../vorbereiten.py` — hier
# bauen wir nur noch: **encodieren → Kopf → messen.**
#
# Die Latte: Track A getunt = **90,25 %** (TF-IDF + LogReg).

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

from data_utils import load_banking77, save_result
from embeddings import encode

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Intents: {len(set(train_labels))}")

# %% [markdown]
# ## Encodieren (aus dem Cache, sonst rechnen)
# Text → 384-dim Bedeutungsvektor. Details in `../vorbereiten.py`.

# %%
X_train = encode(MODEL, train_texts, "minilm", "train")
X_test = encode(MODEL, test_texts, "minilm", "test")
print(f"X_train: {X_train.shape}  (Anfragen × Embedding-Dimension)")

# %% [markdown]
# ## Kopf trainieren und messen
#
# LogReg auf den eingefrorenen Embeddings — derselbe Kopf wie der Gen-1-Sieger,
# fairer Vergleich. Die Embeddings sind dicht (nicht spärlich wie TF-IDF), das
# Training ist schnell.

# %%
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)
p = clf.predict(X_test)

acc = accuracy_score(test_labels, p)
macro_f1 = f1_score(test_labels, p, average="macro")
print(f"MiniLM + LogReg   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"Track A (TF-IDF+LogReg getunt): 90.25 %   ← die Latte")

save_result("B_minilm_logreg", acc, macro_f1=round(macro_f1, 4),
            model="MiniLM + LogReg", note="P2 plain, frozen embeddings")

# %% [markdown]
# ## Wo irrt das Modell?

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

plot_confusion_matrix(test_labels, p, title="MiniLM + LogReg — Confusion Matrix")
plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)

# %% [markdown]
# ## Deuten
#
# - Schlägt der **kleine** Encoder schon die getunte TF-IDF-Latte, oder braucht es
#   die starken Encoder (E5/BGE) aus P3?
# - Verwechselt MiniLM **dieselben** Intent-Paare wie TF-IDF (→ liegt an den Daten)
#   oder andere (→ die Semantik trägt wirklich etwas bei)?
#
# **✓ Checkpoint (CURRICULUM):** In welchem Szenario schlagen Embeddings
# Bag-of-Words deutlich — und warum? (Stichwort: wenig Labels, Synonyme.)
