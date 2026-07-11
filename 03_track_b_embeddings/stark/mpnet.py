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

# %% [markdown]
# # P3 — Kopf-Zyklus (auf den mpnet-Embeddings)
#
# Jetzt optimieren wir den **Kopf** auf dem besten Encoder. Derselbe ehrliche
# Zyklus wie in Track A — nur läuft er auf *fertigen Embeddings* statt TF-IDF.
# Genau dafür haben wir den generischen Kern `optimization.greedy_search`
# rausgezogen: der Loop ist derselbe, nur die `evaluate`-Funktion ist anders.

# %% [markdown]
# ## Val-Split auf den Embeddings
# Gleiche Aufteilung wie F2 (15 %, stratifiziert, Seed 42) — nur direkt auf der
# Embedding-Matrix, mit denselben Parametern → dieselbe Partition.

# %%
from sklearn.model_selection import train_test_split

from optimization import greedy_search

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, train_labels, test_size=0.15, stratify=train_labels, random_state=42
)
print(f"Trainingsteil: {len(y_tr)}   Validierung: {len(y_val)}")

# %% [markdown]
# ## LogReg-C auf val greedy tunen
# Auf dichten Embeddings liegt das C-Optimum oft anders als bei TF-IDF (dort C=10).

# %%
def evaluate(cfg):
    clf = LogisticRegression(max_iter=1000, **cfg)
    clf.fit(X_tr, y_tr)
    pr = clf.predict(X_val)
    return accuracy_score(y_val, pr), f1_score(y_val, pr, average="macro")


experiments = [
    ("C=0.3", {"C": 0.3}),
    ("C=3", {"C": 3}),
    ("C=10", {"C": 10}),
    ("C=30", {"C": 30}),
    ("class_weight=balanced", {"class_weight": "balanced"}),
]
best_clf, proto_df = greedy_search(evaluate, experiments)

from eval_utils import plot_rounds

print(proto_df.to_string(index=False))
print(f"Beste LogReg-Config: {best_clf or 'Default'}")
plot_rounds(proto_df, "mpnet-Kopf (LogReg) — Optimierungsrunden")

# %% [markdown]
# ## Kopf-*Wechsel* — LogReg vs. kNN vs. LinearSVC (auf val)
# Anderes Konzept: **kNN (Cosinus)** klassifiziert per Nachbarschaft — laut
# CURRICULUM stark bei vielen Klassen × wenig Beispielen. Fairer Vergleich auf val.

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

heads = {
    "LogReg (getunt)": LogisticRegression(max_iter=1000, **best_clf),
    "kNN cosine k=15": KNeighborsClassifier(n_neighbors=15, metric="cosine"),
    "LinearSVC": LinearSVC(),
}
head_scores = {}
for name, h in heads.items():
    h.fit(X_tr, y_tr)
    head_scores[name] = f1_score(y_val, h.predict(X_val), average="macro")
    print(f"{name:<18} val Macro-F1 {head_scores[name]*100:.2f} %")

best_head_name = max(head_scores, key=head_scores.get)
print(f"→ bester Kopf auf val: {best_head_name}")

# %% [markdown]
# ## Finale Messung — Testset, genau EINMAL

# %%
final = heads[best_head_name]
final.fit(X_train, train_labels)
pt = final.predict(X_test)
test_acc = accuracy_score(test_labels, pt)
test_f1 = f1_score(test_labels, pt, average="macro")
print(f"mpnet + {best_head_name} (getunt)  test-Acc: {test_acc*100:.2f} %   Macro-F1: {test_f1*100:.2f} %")
print(f"mpnet + LogReg plain war: 92.23 %")

save_result("B_mpnet_tuned", test_acc, macro_f1=round(test_f1, 4),
            model=f"mpnet + {best_head_name}", note="P3 Kopf-Zyklus (val-getunt), test 1x")
