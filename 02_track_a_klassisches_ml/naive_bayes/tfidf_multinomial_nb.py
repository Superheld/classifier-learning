# %% [markdown]
# # Track A · naive_bayes — TF-IDF + MultinomialNB (banking77)
#
# Der **Lehrbuch-Textklassifikator**: MultinomialNB modelliert je Klasse, wie
# wahrscheinlich jedes Wort ist, und multipliziert (im Log) durch. Blitzschnell,
# die klassische Baseline in jedem NLP-Kurs. Auf unbalancierten Daten meist etwas
# schwächer als ComplementNB — genau das prüfen wir.
#
# Stellschraube wie bei jedem NB: `alpha` (Glättung). Referenz: LogReg 90,25 %.

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
from sklearn.naive_bayes import MultinomialNB

from data_utils import load_banking77, load_banking77_split, save_result

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Intents: {len(set(train_labels))}")

# %% [markdown]
# ## P2 — pure

# %%
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

clf = MultinomialNB()
clf.fit(X_train, train_labels)
preds = clf.predict(X_test)
acc = accuracy_score(test_labels, preds)
print(f"MultinomialNB pure  Accuracy: {acc*100:.2f} %   (ComplementNB zum Vergleich!)")

save_result("A_plain_tfidf_multinomialnb", acc,
            macro_f1=round(f1_score(test_labels, preds, average="macro"), 4),
            model="TF-IDF + MultinomialNB", note="P2 plain")

# %% [markdown]
# ## P3 — optimieren (auf val)
# Feature-Knöpfe + `alpha` (Glättung) und `fit_prior` (Klassen-Vorwissen nutzen
# oder gleichverteilt annehmen — relevant, weil train unbalanciert ist).

# %%
from experiment import tune

tr_texts, tr_labels, val_texts, val_labels = load_banking77_split()

experiments = [
    ("Bigramme (1,2)", {"ngram_range": (1, 2)}, None),
    ("min_df=2", {"min_df": 2}, None),
    ("sublinear_tf", {"sublinear_tf": True}, None),
    ("alpha=0.3", None, {"alpha": 0.3}),
    ("alpha=0.1", None, {"alpha": 0.1}),
    ("fit_prior=False", None, {"fit_prior": False}),
]
best_vec, best_clf, proto_df = tune(
    lambda kw: MultinomialNB(**kw),
    experiments, tr_texts, tr_labels, val_texts, val_labels,
)

# %%
from eval_utils import plot_per_class_f1, plot_rounds, plot_top_confusions

print(proto_df.to_string(index=False))
print(f"Beste Config:  TF-IDF={best_vec or 'Default'}   MultinomialNB={best_clf or 'Default'}")
plot_rounds(proto_df, "MultinomialNB — Optimierungsrunden")

# %% [markdown]
# ## Finale Messung — Testset, genau EINMAL

# %%
full_texts, full_labels = load_banking77("train")
vec = TfidfVectorizer(**best_vec)
Xtr = vec.fit_transform(full_texts)
Xte = vec.transform(test_texts)
clf = MultinomialNB(**best_clf)
clf.fit(Xtr, full_labels)
p = clf.predict(Xte)
test_acc = accuracy_score(test_labels, p)
test_f1 = f1_score(test_labels, p, average="macro")
print(f"MultinomialNB getunt  test-Acc: {test_acc*100:.2f} %   Macro-F1: {test_f1*100:.2f} %")

save_result("A_tuned_tfidf_multinomialnb", test_acc, macro_f1=round(test_f1, 4),
            model="TF-IDF + MultinomialNB", config=f"vec={best_vec}, clf={best_clf}",
            note="P3 getunt, test 1x")

plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)
