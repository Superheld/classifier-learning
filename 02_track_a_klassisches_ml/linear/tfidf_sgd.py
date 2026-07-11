# %% [markdown]
# # Track A · linear — TF-IDF + SGDClassifier (banking77)
#
# Dritte lineare Familie. **SGDClassifier** trainiert *dasselbe* lineare Modell
# wie LogReg/LinearSVC, aber per **stochastischem Gradientenabstieg** (kleine
# Schritte bergab statt geschlossener Lösung). Das skaliert auf riesige Daten und
# gibt Zugriff auf verschiedene Verlustfunktionen: `loss="hinge"` (= linear SVM,
# Default), `loss="log_loss"` (= LogReg). Gleiche vier Schritte wie gehabt.
#
# Referenz: LogReg getunt 90,25 % · LinearSVC 89,50 %.

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
for d in (root, root / "02_track_a_klassisches_ml"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_banking77, load_banking77_split, save_result

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Intents: {len(set(train_labels))}")

# %% [markdown]
# ## P2 — pure
# TF-IDF mit Defaults, SGD mit festem `random_state` (stochastisch → sonst nicht
# reproduzierbar).

# %%
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

clf = SGDClassifier(random_state=42)
clf.fit(X_train, train_labels)
preds = clf.predict(X_test)
acc = accuracy_score(test_labels, preds)
print(f"SGD pure  Accuracy: {acc*100:.2f} %   (LogReg pure 87.78 %, LinearSVC pure 89.47 %)")

save_result("A_plain_tfidf_sgd", acc,
            macro_f1=round(f1_score(test_labels, preds, average="macro"), 4),
            model="TF-IDF + SGD", note="P2 plain")

# %% [markdown]
# ## P3 — optimieren (auf val)
# Neben den Feature-Knöpfen die SGD-eigenen: `loss` (hinge vs. log_loss),
# `alpha` (Regularisierung, Default 1e-4), `class_weight`.

# %%
from experiment import tune

tr_texts, tr_labels, val_texts, val_labels = load_banking77_split()

experiments = [
    ("Bigramme (1,2)", {"ngram_range": (1, 2)}, None),
    ("min_df=2", {"min_df": 2}, None),
    ("sublinear_tf", {"sublinear_tf": True}, None),
    ("loss=log_loss", None, {"loss": "log_loss"}),
    ("alpha=1e-5", None, {"alpha": 1e-5}),
    ("class_weight=balanced", None, {"class_weight": "balanced"}),
]
best_vec, best_clf, proto_df = tune(
    lambda kw: SGDClassifier(random_state=42, **kw),
    experiments, tr_texts, tr_labels, val_texts, val_labels,
)

# %%
from eval_utils import plot_per_class_f1, plot_rounds, plot_top_confusions

print(proto_df.to_string(index=False))
print(f"Beste Config:  TF-IDF={best_vec or 'Default'}   SGD={best_clf or 'Default'}")
plot_rounds(proto_df, "SGD — Optimierungsrunden")

# %% [markdown]
# ## Finale Messung — Testset, genau EINMAL

# %%
full_texts, full_labels = load_banking77("train")
vec = TfidfVectorizer(**best_vec)
Xtr = vec.fit_transform(full_texts)
Xte = vec.transform(test_texts)
clf = SGDClassifier(random_state=42, **best_clf)
clf.fit(Xtr, full_labels)
p = clf.predict(Xte)
test_acc = accuracy_score(test_labels, p)
test_f1 = f1_score(test_labels, p, average="macro")
print(f"SGD getunt  test-Acc: {test_acc*100:.2f} %   Macro-F1: {test_f1*100:.2f} %")

save_result("A_tuned_tfidf_sgd", test_acc, macro_f1=round(test_f1, 4),
            model="TF-IDF + SGD", config=f"vec={best_vec}, clf={best_clf}",
            note="P3 getunt, test 1x")

plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)
