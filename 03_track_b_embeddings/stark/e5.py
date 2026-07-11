# %% [markdown]
# # Track B · stark — E5 + LogReg (banking77)
#
# ## P3 — retrieval-optimierter Encoder
#
# **intfloat/e5-base-v2**: aus der E5-Familie, die die MTEB-Embedding-Leaderboards
# anführt. 768-dim. **Wichtige Konvention:** E5 wurde mit Präfixen trainiert
# (`"query: "` / `"passage: "`) — ohne den Präfix liefert es *schlechtere* Vektoren.
# Für Klassifikation nutzt man `"query: "` für alle Texte (der `encode`-Helfer
# hängt ihn an). Ein typischer Encoder-Stolperstein — genau darum geht P1/P3.
#
# Latte: MiniLM 90,83 % · mpnet (siehe Nachbardatei) · Track A 90,25 %.

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

MODEL = "intfloat/e5-base-v2"
PREFIX = "query: "  # E5-Konvention — ohne diesen Praefix wird E5 schlechter

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}")

# %% [markdown]
# ## Encodieren (mit E5-Präfix) + Kopf + messen

# %%
X_train = encode(MODEL, train_texts, "e5base", "train", prefix=PREFIX)
X_test = encode(MODEL, test_texts, "e5base", "test", prefix=PREFIX)
print(f"X_train: {X_train.shape}")

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)
p = clf.predict(X_test)

acc = accuracy_score(test_labels, p)
macro_f1 = f1_score(test_labels, p, average="macro")
print(f"E5 + LogReg   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"MiniLM 90.83 %   |   Track A 90.25 %")

save_result("B_e5_logreg", acc, macro_f1=round(macro_f1, 4),
            model="E5-base + LogReg", note="P3 retrieval-Encoder (query:-Praefix), frozen")

# %% [markdown]
# ## Wo irrt das Modell?

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

plot_confusion_matrix(test_labels, p, title="E5 + LogReg — Confusion Matrix")
plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)

# %% [markdown]
# ## Deuten
# E5/BGE sind auf *Retrieval* optimiert (Ähnlichkeit finden) — überträgt sich das
# auf Intent-Klassifikation? Und: hat der `"query: "`-Präfix wirklich geholfen?
# (Zum Prüfen: einmal ohne Präfix encodieren und vergleichen — die Lektion, dass
# Encoder-Konventionen zählen.)
