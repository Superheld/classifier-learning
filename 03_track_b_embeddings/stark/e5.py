# %% [markdown]
# # Track B · stark — E5 + LogReg (banking77)
#
# ## P3 — der retrieval-optimierte Encoder (und eine Lehre)
#
# **intfloat/e5-base-v2** kommt aus der **E5-Familie**, die die
# [MTEB](https://huggingface.co/spaces/mteb/leaderboard)-Embedding-Leaderboards
# anführt — auf dem Papier also *stärker* als MiniLM oder mpnet. Die spannende
# Frage dieses Notebooks ist deshalb nicht „schlägt es die Latte?", sondern:
# **überträgt sich ein Retrieval-Spitzenplatz auf unsere Klassifikation?**
# (Spoiler: Das Ergebnis ist der Grund, warum diese Datei existiert.)
#
# Latte bisher in Track B: MiniLM **90,83 %** · mpnet plain **92,23 %** (Nachbardatei)
# · Track A getunt 90,25 %.

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

MODEL = "intfloat/e5-base-v2"
PREFIX = "query: "  # E5-Konvention — siehe nächste Zelle

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}")

# %% [markdown]
# ## Der Encoder-Stolperstein: E5 will einen Präfix
#
# E5 ist ein **asymmetrischer** Encoder: es wurde darauf trainiert, *Suchanfragen*
# und *Dokumente* getrennt zu behandeln, jeweils mit einem Präfix vor dem Text —
# `"query: "` für die Anfrage, `"passage: "` für das Dokument. Ohne Präfix bekommt
# E5 einen Input, wie er ihn im Training nie gesehen hat, und liefert **schlechtere**
# Vektoren.
#
# Für Klassifikation gibt es keine „Dokumente", nur Texte — die E5-Autoren empfehlen,
# alles als `"query: "` zu behandeln. Genau solche modellspezifischen Konventionen
# sind das, was P1/P3 lehren soll: *ein stärkeres Modell falsch bedient ist
# schwächer als ein schwächeres richtig bedient.* Unser `encode`-Helfer hängt den
# Präfix an jeden Text, bevor er rechnet.

# %% [markdown]
# ## Schritt 1 — Encodieren (mit E5-Präfix)
#
# Text → 768-dim Bedeutungsvektor, gecacht als `.npy` (Details in `../vorbereiten.py`).
# Der `prefix=`-Parameter ist der einzige Unterschied zum MiniLM/mpnet-Aufruf.

# %%
X_train = encode(MODEL, train_texts, "e5base", "train", prefix=PREFIX)
X_test = encode(MODEL, test_texts, "e5base", "test", prefix=PREFIX)
print(f"X_train: {X_train.shape}   (Anfragen × 768)")

# %% [markdown]
# ## Schritt 2 — Kopf trainieren
#
# Exakt derselbe Kopf wie bei MiniLM/mpnet: LogReg auf den eingefrorenen Embeddings.
# Nur so ist der Vergleich fair — die *einzige* geänderte Variable ist der Encoder.

# %%
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)
p = clf.predict(X_test)

# %% [markdown]
# ## Schritt 3 — Messen (und die Überraschung)

# %%
acc = accuracy_score(test_labels, p)
macro_f1 = f1_score(test_labels, p, average="macro")
print(f"E5 + LogReg   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"Zum Vergleich:  MiniLM 90.83 %  ·  mpnet 92.23 %  ·  Track A 90.25 %")

save_result("B_e5_logreg", acc, macro_f1=round(macro_f1, 4),
            model="E5-base + LogReg", note="P3 retrieval-Encoder (query:-Praefix), frozen")

# %% [markdown]
# ## Wo irrt das Modell?

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

plot_confusion_matrix(test_labels, p, title="E5 + LogReg — Confusion Matrix")
plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)

# %% [markdown]
# ## Deuten — „stärker ≠ besser"
#
# E5 landet bei **87,06 %** — und damit **unter** dem kleinen MiniLM (90,83 %), unter
# mpnet (92,23 %) und sogar unter der getunten TF-IDF aus Track A (90,25 %). Der
# MTEB-Leader ist auf *unserer* Aufgabe der schwächste Encoder im Feld. Warum?
#
# - **MTEB-Rang ≠ Klassifikations-Güte.** E5 ist auf **Retrieval** optimiert — die
#   *richtigen Dokumente zu einer Anfrage finden*. Das belohnt eine andere Geometrie
#   des Vektorraums als „trenne 77 Intents linear". Ein Bestenlisten-Platz in Disziplin
#   A sagt wenig über Disziplin B.
# - **mpnet ist ein Klassifikations-/Semantik-Allrounder**, genau für „ähnliche Sätze
#   nah beieinander" gebaut — das passt hier besser als E5s Retrieval-Spezialisierung.
# - Es ist dasselbe Muster wie später in Track C: **die Basis + ihr Trainingsziel
#   entscheiden, nicht das Etikett „stärker".** Die Messung schlägt die Erwartung.
#
# **Kontroll-Experiment (die eigentliche P1-Lehre):** War der `"query: "`-Präfix
# überhaupt nötig? Einmal ohne Präfix encodieren (`prefix=""`, neuer `cache_key`)
# und vergleichen — die Zahl sollte *fallen*. So wird aus „man sagt, E5 braucht den
# Präfix" ein selbst gemessenes Faktum.
#
# **✓ Checkpoint:** Ein Kollege schlägt vor, „einfach das Modell mit dem höchsten
# MTEB-Score" zu nehmen. Was entgegnest du — mit dieser Zahl als Beleg?
