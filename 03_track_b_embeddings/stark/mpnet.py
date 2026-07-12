# %% [markdown]
# # Track B · stark — mpnet + Kopf (banking77)
#
# ## P3 — stärkerer Encoder + Kopf-Zyklus
#
# **all-mpnet-base-v2** ist dasselbe SBERT-Prinzip wie MiniLM, nur größer: **768**
# statt 384 Dimensionen, mächtigeres Basismodell. Es gilt als einer der stärksten
# „general purpose"-Satz-Encoder — gebaut dafür, dass semantisch ähnliche Sätze nah
# beieinander liegen. Kein Präfix nötig (anders als E5). Vorarbeit: `../vorbereiten.py`.
#
# Dieses Notebook macht **zwei Dinge**:
# 1. **Encoder-Wechsel** messen: bringt das größere mpnet ggü. MiniLM echten Vorsprung?
# 2. **Kopf-Zyklus**: den Klassifikator *auf* den mpnet-Embeddings optimieren (F3).
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
# ## Schritt 1 — Encodieren
#
# Text → 768-dim Vektor (doppelt so breit wie MiniLMs 384), gecacht als `.npy`.
# Beim ersten Lauf lädt + rechnet mpnet auf der Apple-GPU (MPS), danach kommt es
# aus dem Cache.

# %%
X_train = encode(MODEL, train_texts, "mpnet", "train")
X_test = encode(MODEL, test_texts, "mpnet", "test")
print(f"X_train: {X_train.shape}  (768-dim statt MiniLMs 384)")

# %% [markdown]
# ## Schritt 2 — Plain-Kopf: bringt das größere Modell was?
#
# Erst der ehrliche Baseline-Griff für diesen Encoder: **derselbe LogReg wie bei
# MiniLM**, nichts getunt. So misst der Vergleich nur den *Encoder-Wechsel* MiniLM →
# mpnet — eine geänderte Variable.

# %%
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
# Der Encoder ist gesetzt (mpnet, **eingefroren**). Jetzt optimieren wir das Einzige,
# das noch beweglich ist: den **Kopf**. Derselbe ehrliche Zyklus wie in Track A —
# nur läuft er auf *fertigen Embeddings* statt auf einer TF-IDF-Matrix. Genau dafür
# haben wir den generischen Kern `optimization.greedy_search` rausgezogen: der Loop
# ist identisch, nur die `evaluate`-Funktion ist anders.
#
# Wir drehen an **zwei** Dingen: erst den *Hyperparametern* des LogReg-Kopfs, dann
# probieren wir *andere Kopf-Typen* durch.

# %% [markdown]
# ## Val-Split auf den Embeddings
#
# Gleiche Disziplin wie F2: optimiert wird auf einem **Validierungs**-Teil, nie am
# Testset. Gleiche Aufteilung wie überall (15 %, stratifiziert, Seed 42) — nur
# direkt auf der Embedding-Matrix. Gleiche Parameter → dieselbe Partition wie in A.

# %%
from sklearn.model_selection import train_test_split

from optimization import greedy_search

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, train_labels, test_size=0.15, stratify=train_labels, random_state=42
)
print(f"Trainingsteil: {len(y_tr)}   Validierung: {len(y_val)}")

# %% [markdown]
# ## Runde 1 — LogReg-`C` auf val greedy tunen
#
# `C` steuert die Regularisierung (klein = einfacheres, stärker gezügeltes Modell).
# Interessant: bei TF-IDF (Track A, spärliche Vektoren) lag das Optimum bei `C=10`.
# Auf **dichten** Embeddings liegt es oft *anders* — die Merkmale sind hier wenige
# hundert dichte Zahlen statt zehntausender spärlicher, also ein anderes Regime. Wir
# raten nicht, wir messen den Dreier 0,3 / 3 / 10 / 30 plus Klassengewichte.

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
# ## Runde 2 — Kopf-*Wechsel*: LogReg vs. kNN vs. LinearSVC
#
# Bisher haben wir *einen* Kopf-Typ justiert. Jetzt stellen wir drei
# grundverschiedene Köpfe gegeneinander — fair auf demselben val:
# - **LogReg (getunt)**: die beste Config aus Runde 1, unsere Referenz.
# - **kNN (Cosinus, k=15)**: klassifiziert nach **Nachbarschaft** — welche 15
#   Trainingsvektoren liegen (im Cosinus-Sinn) am nächsten? Das Curriculum nennt kNN
#   stark bei *vielen Klassen × wenig Beispielen* — genau unser Fall (77 × ~130).
# - **LinearSVC**: linearer Klassifikator mit **maximalem Rand** zwischen den
#   Klassen; auf dichten Embeddings kantet er LogReg oft knapp aus.

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
#
# Der Sieger-Kopf (auf val gewählt) wird auf dem **vollen** Trainingsset neu gefittet
# und **einmal** auf test gemessen. Das ist die ehrliche Zahl, vergleichbar mit A/C.

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

# %% [markdown]
# ## Deuten
#
# - Der Kopf-Zyklus hebt mpnet auf **94,12 %** — das ist **am BERT-Dach** (~93–94 %
#   auf banking77), und das mit einem **eingefrorenen** Encoder plus getuntem Kopf.
#   Gen 2 schlägt hier Gen 1 klar, und setzt die Latte, an der Track C sich messen muss.
# - Beachte, *woher* der Gewinn kam: der **Kopf-Wechsel** (LinearSVC) brachte mehr als
#   das C-Feintuning. Wie in Track A gilt: die größten Sprünge kommen selten aus dem
#   letzten Hyperparameter, sondern aus einer strukturell besseren Wahl.
# - Alles hier lag **über** dem eingefrorenen Encoder — der Encoder selbst blieb fix.
#   Das ist die Grenze von Track B; ihn *aufzutauen* ist Track C.
#
# **✓ Checkpoint:** Warum fitten wir den Sieger-Kopf für die finale Messung noch
# einmal auf `X_train` (dem *vollen* Trainingsset), obwohl wir ihn beim Tunen nur auf
# `X_tr` (dem Trainingsteil) trainiert hatten?
