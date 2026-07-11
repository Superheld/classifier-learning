# %% [markdown]
# # Track A · naive_bayes — TF-IDF + ComplementNB (banking77)
#
# Andere Sippe: **Naive Bayes** rechnet nicht mit gelernten Gewichten, sondern
# mit **Wort-Wahrscheinlichkeiten je Klasse** (wie typisch ist „atm" für
# atm_support?) — extrem schnell, ein klassischer Text-Baseline.
# **ComplementNB** ist die Variante *für unbalancierte* Daten: sie schätzt jede
# Klasse gegen das Komplement aller anderen. Unser train ist unbalanciert (35–187
# je Intent), also der passende NB-Kandidat.
#
# NB hat kein `C`/`class_weight` — die Stellschraube ist `alpha` (Glättung).
# Referenz: LogReg getunt 90,25 %.

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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB

from data_utils import load_banking77, load_banking77_split, save_result

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(
    f"train: {len(train_texts)}   test: {len(test_texts)}   Intents: {len(set(train_labels))}"
)

# %% [markdown]
# ## P2 — pure

# %%
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

clf = ComplementNB()
clf.fit(X_train, train_labels)
preds = clf.predict(X_test)
acc = accuracy_score(test_labels, preds)
print(f"ComplementNB pure  Accuracy: {acc * 100:.2f} %   (LogReg pure 87.78 %)")

save_result(
    "A_plain_tfidf_complementnb",
    acc,
    macro_f1=round(f1_score(test_labels, preds, average="macro"), 4),
    model="TF-IDF + ComplementNB",
    note="P2 plain",
)

# %% [markdown]
# ## P3 — optimieren (auf val)
# Feature-Knöpfe + `alpha` (Glättung, Default 1.0; kleiner = weniger geglättet)
# und `norm` (ComplementNBs Gewichts-Normalisierung).

# %%
from tuning import tune

tr_texts, tr_labels, val_texts, val_labels = load_banking77_split()

experiments = [
    ("Bigramme (1,2)", {"ngram_range": (1, 2)}, None),
    ("min_df=2", {"min_df": 2}, None),
    ("sublinear_tf", {"sublinear_tf": True}, None),
    ("alpha=0.3", None, {"alpha": 0.3}),
    ("alpha=0.1", None, {"alpha": 0.1}),
    ("norm=True", None, {"norm": True}),
]
best_vec, best_clf, proto_df = tune(
    lambda kw: ComplementNB(**kw),
    experiments,
    tr_texts,
    tr_labels,
    val_texts,
    val_labels,
)

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_rounds, plot_top_confusions

print(proto_df.to_string(index=False))
print(
    f"Beste Config:  TF-IDF={best_vec or 'Default'}   ComplementNB={best_clf or 'Default'}"
)
plot_rounds(proto_df, "ComplementNB — Optimierungsrunden")

# %% [markdown]
# ## Finale Messung — Testset, genau EINMAL

# %%
full_texts, full_labels = load_banking77("train")
vec = TfidfVectorizer(**best_vec)
Xtr = vec.fit_transform(full_texts)
Xte = vec.transform(test_texts)
clf = ComplementNB(**best_clf)
clf.fit(Xtr, full_labels)
p = clf.predict(Xte)
test_acc = accuracy_score(test_labels, p)
test_f1 = f1_score(test_labels, p, average="macro")
print(
    f"ComplementNB getunt  test-Acc: {test_acc * 100:.2f} %   Macro-F1: {test_f1 * 100:.2f} %"
)

save_result(
    "A_tuned_tfidf_complementnb",
    test_acc,
    macro_f1=round(test_f1, 4),
    model="TF-IDF + ComplementNB",
    config=f"vec={best_vec}, clf={best_clf}",
    note="P3 getunt, test 1x",
)

plot_confusion_matrix(test_labels, p, title="ComplementNB getunt — Confusion Matrix")
plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)
