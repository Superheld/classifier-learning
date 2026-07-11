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
# ## Val-Setup
#
# Der Optimierungszyklus selbst (auf val messen, eine Änderung, besser →
# behalten) steckt jetzt zentral in **`experiment.tune()`** — F3, einmal gebaut,
# von jedem Modell genutzt. Hier bleibt nur das Modellspezifische: der Val-Split
# und gleich die Liste der Experimente.

# %%
from experiment import tune

tr_texts, tr_labels, val_texts, val_labels = load_banking77_split()
print(f"Trainingsteil: {len(tr_texts)}   Validierung: {len(val_texts)}")

# %% [markdown]
# ## Die Experimente
# Jede Zeile: `(Label, TF-IDF-Änderung, LogReg-Änderung)` — genau **eine**
# Änderung pro Runde, `tune()` probiert sie greedy auf dem bisher Besten durch.
# Die Hypothesen (aus der F1-EDA):
# - **Bigramme**: die Verwechsel-Paare teilten sich Einzelwörter (`card`, `not`,
#   `working`); erst „not working" trennt sie.
# - **min_df=2**: die 47 % Hapax-Wörter (Namen, Tippfehler) wegschneiden.
# - **sublinear_tf**: Häufigkeit dämpfen (1 + log tf) — Standard-Textkniff.
# - **C**: LogReg-Regularisierung (klein = einfacheres Modell), Default 1.
# - **class_weight**: train ist unbalanciert → seltene Intents höher gewichten.

# %%
experiments = [
    ("Bigramme (1,2)", {"ngram_range": (1, 2)}, None),
    ("min_df=2", {"min_df": 2}, None),
    ("sublinear_tf", {"sublinear_tf": True}, None),
    ("C=0.1", None, {"C": 0.1}),
    ("C=3", None, {"C": 3}),
    ("C=10", None, {"C": 10}),
    ("class_weight=balanced", None, {"class_weight": "balanced"}),
]
best_vec, best_clf, proto_df = tune(
    lambda kw: LogisticRegression(max_iter=1000, **kw),
    experiments, tr_texts, tr_labels, val_texts, val_labels,
)

# %% [markdown]
# ## Rundenprotokoll & Best-Config

# %%
from eval_utils import plot_rounds

print(proto_df.to_string(index=False))
print(f"\nBeste Config:  TF-IDF={best_vec or 'Default'}   LogReg={best_clf or 'Default'}")
print(f"Beste val Macro-F1: {proto_df['val_macroF1'].max():.2f} %")

plot_rounds(proto_df, "LogReg — Optimierungsrunden")

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
# ## Wo irrt das Modell? — Fehlerbilder
#
# Zwei Blicke auf die getunten Test-Vorhersagen. Die **Verwechslungen** sind der
# Reality-Check zur F1-EDA: dort hatten wir per Wort-Überlappung Paare wie
# `card_payment_not_recognised ↔ direct_debit_payment_not_recognised` als
# Verwechsel-Kandidaten vorhergesagt — treffen sie ein?

# %%
from eval_utils import plot_per_class_f1, plot_top_confusions

plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)

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
