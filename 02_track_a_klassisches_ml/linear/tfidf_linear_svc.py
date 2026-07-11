# %% [markdown]
# # Track A — Gen 1: TF-IDF + LinearSVC (banking77)
#
# ## P2 „Bauen" — pure, zum Vergleich mit LogReg
#
# Gleiche vier Schritte wie beim LogReg-Notebook (`Text → TF-IDF → Modell →
# messen`), **nur der Klassifikator wechselt**. So sehen wir apples-to-apples,
# wie sich zwei Gen-1-Modelle auf denselben Daten schlagen.
#
# **LinearSVC** (lineare Support Vector Machine) sucht die Trenn-Ebene mit dem
# größten **Abstand** (Margin) zwischen den Klassen — ein anderes Prinzip als
# LogReg (das Wahrscheinlichkeiten modelliert). Bei hochdimensionalem,
# spärlichem TF-IDF-Text ist LinearSVC oft einen Tick besser.
#
# Referenz: plain **LogReg = 87,78 %**.

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

# Autoreload: geänderte Module (z.B. data_utils.py) neu laden, ohne Kernel-
# Neustart. Läuft nur im Jupyter-Kernel; als reines Skript wird es übersprungen.
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except (NameError, AttributeError):
    pass

root = Path.cwd()
while not (root / "data_utils.py").exists() and root != root.parent:
    root = root.parent
# Root (geteilte Helfer: data_utils, eval_utils) + Track-A-Ordner (experiment.py
# liegt track-lokal) auf den Importpfad — egal von wo Zed den Kernel startet.
for d in (root, root / "02_track_a_klassisches_ml"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from data_utils import load_banking77, load_banking77_split, save_result

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Intents: {len(set(train_labels))}")

# %% [markdown]
# ## Schritt 1 — TF-IDF (identisch zum LogReg-Lauf)
#
# Bewusst *dieselben* Default-Einstellungen wie beim plain LogReg — nur so ist
# der Vergleich fair (gleiche Features, anderer Klassifikator).

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
print(f"X_train: {X_train.shape}   Vokabular: {len(vectorizer.vocabulary_)}")

# %% [markdown]
# ## Schritt 2 — LinearSVC trainieren
#
# `dual="auto"` lässt sklearn die effizientere Rechenform wählen (bei uns mehr
# Beispiele als Features). `max_iter` großzügig, damit die Optimierung sicher
# durchkonvergiert.

# %%
from sklearn.svm import LinearSVC

clf = LinearSVC(dual="auto", max_iter=5000)
clf.fit(X_train, train_labels)
print("Modell trainiert.")

# %% [markdown]
# ## Schritt 3 — messen

# %%
from sklearn.metrics import accuracy_score, f1_score

preds = clf.predict(X_test)
acc = accuracy_score(test_labels, preds)
macro_f1 = f1_score(test_labels, preds, average="macro")

print(f"LinearSVC  Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"LogReg     Accuracy: 87.78 %   (plain-Referenz)")
print(f"Differenz: {(acc-0.8778)*100:+.2f} Prozentpunkte")

save_result("A_plain_tfidf_linsvc", acc, macro_f1=round(macro_f1, 4),
            model="TF-IDF + LinearSVC", note="P2 plain, keine Optimierung")

# %% [markdown]
# ## Deuten & nächster Schritt
#
# Zwei Gen-1-Modelle, gleiche Features, ehrlicher Vergleich. Welches vorn liegt,
# hängt oft an Kleinigkeiten — beide sind in derselben Liga.
#
# Auch dieses Modell ist **ungetunt**. Der nächste Schritt wäre derselbe
# P3-Zyklus wie bei LogReg (Val-Split, eine Änderung pro Runde) — hier drehen
# sich die Schrauben an `C` und den TF-IDF-Parametern.
#
# **✓ Checkpoint:** LinearSVC liefert keine Wahrscheinlichkeiten (`predict_proba`),
# LogReg schon. Warum könnte das bei der geplanten S2-Kaskade (Modell gibt ab,
# wenn unsicher) ein Argument *für* LogReg sein?

# %% [markdown]
# # P3 — „Optimieren"
#
# Derselbe ehrliche Zyklus wie beim LogReg: auf dem **Val-Split** vergleichen,
# eine Änderung pro Runde, Testset erst am Ende ein einziges Mal. Nur der
# Klassifikator ist LinearSVC — mal sehen, ob dieselben Knöpfe gleich wirken.

# %%
from experiment import tune

tr_texts, tr_labels, val_texts, val_labels = load_banking77_split()
print(f"Trainingsteil: {len(tr_texts)}   Validierung: {len(val_texts)}")

# %% [markdown]
# ## Die Experimente
# Gleiche Feature-Knöpfe wie beim LogReg (Bigramme, `min_df`, `sublinear_tf`),
# dann `C` und Klassengewichte. Bei LinearSVC liegt der `C`-Sweet-Spot oft
# *anders* als bei LogReg — genau deshalb messen wir statt zu raten.

# %%
experiments = [
    ("Bigramme (1,2)", {"ngram_range": (1, 2)}, None),
    ("min_df=2", {"min_df": 2}, None),
    ("sublinear_tf", {"sublinear_tf": True}, None),
    ("C=0.3", None, {"C": 0.3}),
    ("C=3", None, {"C": 3}),
    ("C=10", None, {"C": 10}),
    ("class_weight=balanced", None, {"class_weight": "balanced"}),
]
best_vec, best_clf, proto_df = tune(
    lambda kw: LinearSVC(dual="auto", max_iter=5000, **kw),
    experiments, tr_texts, tr_labels, val_texts, val_labels,
)

# %% [markdown]
# ## Rundenprotokoll & Best-Config

# %%
from eval_utils import plot_rounds

print(proto_df.to_string(index=False))
print(f"\nBeste Config:  TF-IDF={best_vec or 'Default'}   LinearSVC={best_clf or 'Default'}")
print(f"Beste val Macro-F1: {proto_df['val_macroF1'].max():.2f} %")

plot_rounds(proto_df, "LinearSVC — Optimierungsrunden")

# %% [markdown]
# ## Finale Messung — Testset, genau EINMAL
# Best-Config auf vollem train fitten, ein einziges Mal auf test messen.

# %%
full_texts, full_labels = load_banking77("train")

vec = TfidfVectorizer(**best_vec)
Xtr = vec.fit_transform(full_texts)
Xte = vec.transform(test_texts)
clf = LinearSVC(dual="auto", max_iter=5000, **best_clf)
clf.fit(Xtr, full_labels)
p = clf.predict(Xte)

test_acc = accuracy_score(test_labels, p)
test_f1 = f1_score(test_labels, p, average="macro")

print(f"LinearSVC getunt   test-Acc: {test_acc*100:.2f} %   Macro-F1: {test_f1*100:.2f} %")
print(f"LinearSVC plain    test-Acc: 89.47 %")
print(f"LogReg getunt      test-Acc: 90.25 %   (der zu schlagende Wert)")

save_result("A_tuned_tfidf_linsvc", test_acc, macro_f1=round(test_f1, 4),
            model="TF-IDF + LinearSVC", config=f"vec={best_vec}, clf={best_clf}",
            note="P3 getunt, Config auf val gewählt, test 1x gemessen")

# %% [markdown]
# ## Wo irrt das Modell?

# %%
from eval_utils import plot_per_class_f1, plot_top_confusions

plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)

# %% [markdown]
# ## Deuten
# Vergleich der beiden getunten Gen-1-Modelle — plus die Frage, ob sie *dieselben*
# Intents verwechseln (dann liegt es an den Daten/Features, nicht am Modell) oder
# unterschiedliche (dann macht die Modellwahl einen echten Unterschied).
