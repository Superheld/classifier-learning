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
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from data_utils import load_banking77, save_result

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
