# %% [markdown]
# # Track B · stark — mpnet + LogReg (banking77)
#
# ## P3 — stärkerer Encoder
#
# **all-mpnet-base-v2**: dasselbe SBERT-Prinzip wie MiniLM, aber größer (768 statt
# 384 Dimensionen, mächtigeres Basismodell). Gilt als eines der stärksten
# „general purpose"-SBERT-Modelle. Kein Prefix nötig. Vorarbeit: `../vorbereiten.py`.
#
# Latte: MiniLM (klein) = **90,83 %** · Track A getunt = 90,25 %.

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

MODEL = "sentence-transformers/all-mpnet-base-v2"

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}")

# %% [markdown]
# ## Encodieren + Kopf + messen

# %%
X_train = encode(MODEL, train_texts, "mpnet", "train")
X_test = encode(MODEL, test_texts, "mpnet", "test")
print(f"X_train: {X_train.shape}  (768-dim statt MiniLMs 384)")

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)
p = clf.predict(X_test)

acc = accuracy_score(test_labels, p)
macro_f1 = f1_score(test_labels, p, average="macro")
print(f"mpnet + LogReg   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"MiniLM war 90.83 %   |   Track A 90.25 %")

save_result("B_mpnet_logreg", acc, macro_f1=round(macro_f1, 4),
            model="mpnet + LogReg", note="P3 staerkerer Encoder, frozen")

# %% [markdown]
# ## Wo irrt das Modell?

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

plot_confusion_matrix(test_labels, p, title="mpnet + LogReg — Confusion Matrix")
plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)

# %% [markdown]
# ## Deuten
# Bringt die doppelte Dimension + das größere Modell echten Vorsprung, oder ist
# MiniLM auf kurzen, klaren Kundenanfragen schon nah am Plafond? Mehr Dimension ≠
# automatisch besser — die Messung entscheidet.
