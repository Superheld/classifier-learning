# %% [markdown]
# # Track A — Gen 1: Klassisches ML (banking77)
#
# ## P2 „Bauen" — die plainste Version
#
# Ziel hier ist **nicht** ein gutes Modell, sondern das **Vorgehen einmal
# komplett zu sehen**. Der kanonische erste Textklassifikator:
#
# ```
# Text  ──TF-IDF──▶  Zahlen  ──LogReg──▶  Vorhersage  ──▶  messen
# ```
#
# Nichts getunt, kein Val-Split, keine Tricks (das kommt in P3 „Optimieren").
# Als Referenz: die Majority-Baseline aus F1 lag bei **1,30 %** — das ist die
# Latte, die dieses Modell überspringen muss.

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

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
# ## Schritt 1 — Text zu Zahlen: TF-IDF
#
# Ein Modell rechnet mit Zahlen, nicht mit Wörtern. **TF-IDF** macht aus jedem
# Text einen Zahlenvektor: pro Wort ein Gewicht, das hoch ist, wenn das Wort in
# *diesem* Text oft vorkommt (TF), aber selten über *alle* Texte (IDF). Häufige
# Füllwörter bekommen so wenig Gewicht, unterscheidende Wörter viel.
#
# `fit_transform` auf train: lernt das Vokabular **und** rechnet die Vektoren.
# `transform` auf test: nutzt *dasselbe* Vokabular (test darf nichts „lehren").

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()  # alles Default — bewusst nichts getunt
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

print(f"X_train: {X_train.shape}  (Anfragen x Vokabular)")
print(f"X_test : {X_test.shape}")
print(f"Vokabulargröße: {len(vectorizer.vocabulary_)}")

# %% [markdown]
# ## Schritt 2 — Modell trainieren: Logistische Regression
#
# **Logistische Regression** lernt pro Intent ein Gewicht je Wort: welche Wörter
# sprechen für, welche gegen diesen Intent. Aus den Gewichten × TF-IDF-Werten
# wird für jeden Intent ein Score, der höchste gewinnt. `max_iter=1000`, damit
# die Optimierung sicher durchkonvergiert (Default 100 reicht bei 77 Klassen
# nicht immer).

# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)  # hier passiert das eigentliche „Lernen"
print("Modell trainiert.")

# %% [markdown]
# ## Schritt 3 — Vorhersagen und messen
#
# Das Modell tippt für jede test-Anfrage einen Intent. **Accuracy** = Anteil
# richtig getippter Anfragen. (Weil test balanciert ist, ist Macro-F1 hier
# ähnlich — das ehrliche Messbesteck kommt in F2.)

# %%
from sklearn.metrics import accuracy_score, f1_score

preds = clf.predict(X_test)
acc = accuracy_score(test_labels, preds)
macro_f1 = f1_score(test_labels, preds, average="macro")

print(f"Accuracy : {acc*100:.2f} %")
print(f"Macro-F1 : {macro_f1*100:.2f} %")
print(f"Baseline : 1.30 %  ->  Sprung: +{(acc-0.0130)*100:.1f} Prozentpunkte")

save_result("A_plain_tfidf_logreg", acc, macro_f1=round(macro_f1, 4),
            model="TF-IDF + LogReg", note="P2 plain, keine Optimierung")

# %% [markdown]
# ## Deuten & nächster Schritt
#
# Du hast das ganze Vorgehen einmal gesehen: vektorisieren → trainieren →
# vorhersagen → messen. Vier Zeilen echtes ML.
#
# Die Zahl ist ein **ehrlicher Startpunkt**, kein Endstand — nichts ist getunt
# (keine n-Gramme, kein `min_df`, keine Stoppwörter entfernt, Regularisierung
# auf Default). Genau das ist der Stoff für **P3 „Optimieren"**: eine Hypothese,
# eine Änderung, neu messen — aber *dann* mit einem Val-Split (F2), damit wir
# nicht am Testset schummeln.
#
# **✓ Checkpoint:** Warum ruft man auf dem test-Set `transform` und nicht
# `fit_transform`? Was würde schiefgehen, wenn man das Vokabular auf train+test
# gemeinsam lernt?
