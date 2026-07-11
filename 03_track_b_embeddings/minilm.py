# %% [markdown]
# # Track B — Gen 2: Embeddings — MiniLM + LogReg (banking77)
#
# ## Der Generationssprung
#
# Track A rechnete mit **Wort-Zählungen** (TF-IDF): „card" ist „card", „card"
# und „karte" sind fremd. Track B rechnet mit **Bedeutung**: ein vortrainierter
# Encoder macht aus jedem Text einen **dichten Vektor**, in dem semantisch Ähnliches
# nah beieinander liegt. Der Klassifikator (hier LogReg) lernt dann auf diesen
# Vektoren.
#
# **feature extraction** = der Encoder bleibt *eingefroren* (kein Training), wir
# nutzen ihn nur als Merkmals-Fabrik. Start-Encoder laut CURRICULUM: die
# **MiniLM-Familie** (klein, schnell).
#
# Die Latte: Track A getunt = **90,25 %** (LogReg). Schlägt Bedeutung die Wörter?

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
# Root (data_utils, eval_utils) + Track-B-Ordner (embeddings.py liegt track-lokal)
for d in (root, root / "03_track_b_embeddings"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_banking77, save_result
from embeddings import context_window, encode

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Intents: {len(set(train_labels))}")

# %% [markdown]
# ## P1 — Kontextfenster prüfen (der CURRICULUM-Check)
#
# Bevor wir encodieren: passt unser Text ins Fenster des Encoders? Bei GNAD war
# genau das das Problem (97 % länger als 128 Tokens). Hier sollten wir das
# Gegenteil sehen — kurze Kundenanfragen.

# %%
max_len = context_window(MODEL)
word_counts = [len(t.split()) for t in train_texts]
print(f"Encoder-Kontextfenster (max_seq_length): {max_len} Tokens")
print(f"Unsere Texte: median {sorted(word_counts)[len(word_counts)//2]} Wörter, "
      f"max {max(word_counts)} Wörter")
print("→ passt locker; kein Chunking nötig (anders als bei langen Dokumenten).")

# %% [markdown]
# ## P1 — Encodieren (Text → Bedeutungsvektor)
#
# Der Encoder rechnet jeden Text in einen Vektor. Beim ersten Mal wird das Modell
# geladen (Download ~90 MB) und auf der Apple-GPU (MPS) gerechnet; danach kommt es
# aus dem Cache.

# %%
X_train = encode(MODEL, train_texts, "minilm", "train")
X_test = encode(MODEL, test_texts, "minilm", "test")
print(f"X_train: {X_train.shape}  (Anfragen × Embedding-Dimension)")

# %% [markdown]
# ## P2 — Kopf trainieren und messen
#
# LogReg auf den eingefrorenen Embeddings — derselbe Kopf wie der Gen-1-Sieger,
# fairer Vergleich. `max_iter` großzügig; die Embeddings sind dicht (nicht spärlich
# wie TF-IDF), das Training ist schnell.

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
# Zwei Fragen an das Ergebnis:
# - Schlägt der **kleine** Encoder schon die getunte TF-IDF-Latte, oder braucht es
#   die starken Encoder (E5/BGE) aus P3?
# - Verwechselt MiniLM **dieselben** Intent-Paare wie TF-IDF (dann liegt's an den
#   Daten), oder andere (dann trägt die Semantik wirklich etwas bei)?
#
# **✓ Checkpoint (CURRICULUM):** In welchem Szenario schlagen Embeddings
# Bag-of-Words deutlich — und warum? (Stichwort: wenig Labels, Synonyme.)
