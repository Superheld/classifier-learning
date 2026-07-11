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

import pandas as pd

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

from data_utils import load_banking77, load_banking77_split, save_result

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

# %% [markdown]
# # P3 — „Optimieren"
#
# Jetzt tunen wir — aber **ehrlich**. Die Regel: Wer Knöpfe gegen das *Testset*
# dreht und den besten behält, hat am Testset mitoptimiert; die Zahl wird Fiktion.
# Deshalb der **Val-Split** (F2): ein Stück aus train, auf dem wir vergleichen.
# Das Testset bleibt zu — es wird erst ganz am Ende **ein einziges Mal** angefasst.
#
# Der Zyklus je Runde:
# 1. Messlineal fix: **Macro-F1 auf val**.
# 2. **Genau eine** Änderung gegenüber dem bisher Besten.
# 3. Neu messen. Besser → behalten, schlechter → verwerfen.
# 4. Ins Rundenprotokoll. Wiederholen bis Plateau.

# %% [markdown]
# ## Val-Setup & Helfer
#
# `run(...)` fittet TF-IDF+LogReg auf dem **Trainingsteil** und misst auf **val**.
# `consider(...)` nimmt den bisher besten Stand, ändert **einen** Knopf, misst,
# und behält die Änderung nur, wenn val besser wird. So ist jedes Experiment
# eine Zeile — und die Entscheidung sichtbar.

# %%
tr_texts, tr_labels, val_texts, val_labels = load_banking77_split()
print(f"Trainingsteil: {len(tr_texts)}   Validierung: {len(val_texts)}")


def run(vec_kwargs, clf_kwargs):
    """Fittet auf tr_, misst auf val_. Rückgabe: (val_accuracy, val_macro_f1)."""
    vec = TfidfVectorizer(**vec_kwargs)
    Xtr = vec.fit_transform(tr_texts)
    Xval = vec.transform(val_texts)
    clf = LogisticRegression(max_iter=1000, **clf_kwargs)
    clf.fit(Xtr, tr_labels)
    p = clf.predict(Xval)
    return accuracy_score(val_labels, p), f1_score(val_labels, p, average="macro")


protocol = []          # das Rundenprotokoll
best_vec, best_clf = {}, {}   # der bisher beste Stand (leer = alle Defaults)

acc0, f1_0 = run(best_vec, best_clf)
best_f1 = f1_0
protocol.append({"schritt": "Start (Default)", "val_macroF1": round(f1_0 * 100, 2),
                 "val_acc": round(acc0 * 100, 2), "behalten": "—"})
print(f"Ist-Stand auf val:  Macro-F1 {f1_0*100:.2f} %   Acc {acc0*100:.2f} %")


def consider(label, vec_change=None, clf_change=None):
    """Testet EINE Änderung auf dem bisher Besten und behält sie nur, wenn besser."""
    global best_vec, best_clf, best_f1
    cand_vec = {**best_vec, **(vec_change or {})}
    cand_clf = {**best_clf, **(clf_change or {})}
    acc, f1 = run(cand_vec, cand_clf)
    better = f1 > best_f1
    print(f"{label:<24} val Macro-F1 {f1*100:5.2f} %  "
          f"(bisher best {best_f1*100:.2f} %)  -> {'BEHALTEN' if better else 'verworfen'}")
    protocol.append({"schritt": label, "val_macroF1": round(f1 * 100, 2),
                     "val_acc": round(acc * 100, 2),
                     "behalten": "ja" if better else "nein"})
    if better:
        best_vec, best_clf, best_f1 = cand_vec, cand_clf, f1

# %% [markdown]
# ## Runde 1 — Bigramme
# *Hypothese:* In der F1-EDA teilten sich die Verwechsel-Paare Einzelwörter
# (`card`, `not`, `working`). Erst Wortpaare wie „not working" oder „hasn't
# arrived" trennen sie. `ngram_range=(1,2)` nimmt Uni- **und** Bigramme.

# %%
consider("Bigramme (1,2)", vec_change={"ngram_range": (1, 2)})

# %% [markdown]
# ## Runde 2 — seltene Wörter wegschneiden
# *Hypothese:* 47 % des Vokabulars waren Hapax (1× Wörter: Namen, Tippfehler).
# `min_df=2` wirft alles raus, was in <2 Anfragen vorkommt → weniger Rauschen.

# %%
consider("min_df=2", vec_change={"min_df": 2})

# %% [markdown]
# ## Runde 3 — TF logarithmisch
# *Hypothese:* Ob ein Wort 1× oder 5× vorkommt, ist bei kurzen Texten weniger
# wichtig als *dass* es vorkommt. `sublinear_tf=True` dämpft die Häufigkeit
# (1 + log tf) — ein Standard-Textkniff.

# %%
consider("sublinear_tf", vec_change={"sublinear_tf": True})

# %% [markdown]
# ## Runde 4 — Regularisierung C
# `C` steuert, wie stark LogReg große Gewichte bestraft (klein = mehr Bestrafung
# = einfacheres Modell). Default ist 1. Wir probieren mehrere Werte auf dem
# bisher besten Feature-Stand — der greedy-Vergleich behält den besten.

# %%
for c in [0.1, 3, 10]:   # 1 ist der aktuelle Default-Stand
    consider(f"C={c}", clf_change={"C": c})

# %% [markdown]
# ## Runde 5 — Klassengewichte
# *Hypothese:* train ist unbalanciert (35–187 Beispiele/Intent, F1).
# `class_weight="balanced"` gibt seltenen Intents mehr Gewicht — könnte Macro-F1
# heben (das über alle Klassen gleich mittelt).

# %%
consider("class_weight=balanced", clf_change={"class_weight": "balanced"})

# %% [markdown]
# ## Rundenprotokoll & Best-Config

# %%
print(pd.DataFrame(protocol).to_string(index=False))
print(f"\nBeste Config gefunden:  TF-IDF={best_vec or 'Default'}   LogReg={best_clf or 'Default'}")
print(f"Beste val Macro-F1:     {best_f1*100:.2f} %")

# %% [markdown]
# ## Finale Messung — Testset, genau EINMAL
#
# Jetzt und nur jetzt fassen wir test an. Wir fitten die Best-Config auf dem
# **vollen** train (Trainingsteil + val zusammen — jetzt darf das val wieder
# mittrainieren) und messen ein einziges Mal auf test. Das ist die ehrliche Zahl.

# %%
full_texts, full_labels = load_banking77("train")

vec = TfidfVectorizer(**best_vec)
Xtr = vec.fit_transform(full_texts)
Xte = vec.transform(test_texts)
clf = LogisticRegression(max_iter=1000, **best_clf)
clf.fit(Xtr, full_labels)
p = clf.predict(Xte)

test_acc = accuracy_score(test_labels, p)
test_f1 = f1_score(test_labels, p, average="macro")

print(f"getunt   test-Accuracy: {test_acc*100:.2f} %   Macro-F1: {test_f1*100:.2f} %")
print(f"plain    test-Accuracy: 87.78 %")
print(f"Gewinn durch Tuning:    +{(test_acc-0.8778)*100:.2f} Prozentpunkte")

save_result("A_tuned_tfidf_logreg", test_acc, macro_f1=round(test_f1, 4),
            model="TF-IDF + LogReg", config=f"vec={best_vec}, clf={best_clf}",
            note="P3 getunt, Config auf val gewählt, test 1x gemessen")

# %% [markdown]
# ## Deuten
#
# - Der **Sprung von plain zu getunt** kommt fast immer aus den *Features*
#   (Bigramme, min_df), selten aus dem Klassifikator-Feintuning — merk dir das
#   für die anderen Tracks.
# - Was auf **val nicht half, ist verworfen** — auch wenn es „hätte helfen
#   sollen". Genau das ist die Disziplin: nicht die Theorie entscheidet, die
#   Messung entscheidet.
# - Wir haben test **zweimal** angefasst (plain P2 + getunt P3) — nie dazwischen.
#   Diese Zahl ist ehrlich vergleichbar mit dem, was Track B/C/D später liefern.
