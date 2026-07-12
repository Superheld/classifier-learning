# %% [markdown]
# # Track C · hybrid — finegetunter RoBERTa + getauschter Kopf (banking77)
#
# ## Die Kombination C × B
#
# Bisher zwei getrennte Welten:
# - **Track B** nahm einen *eingefrorenen* Encoder und tunte den **Kopf** darauf
#   (mpnet + LinearSVC = 94,12 %).
# - **Track C** taute den Encoder auf und ließ ihn *seinen eigenen* Softmax-Kopf
#   mitlernen (RoBERTa = 93,89 %).
#
# Dieses Notebook mischt beides: Wir nehmen den **finegetunten RoBERTa** (aus
# `modelle/roberta_ft`, gespeichert in `voll/roberta.py`), benutzen ihn als
# **eingefrorenen Feature-Extraktor** — und fahren darauf den **Track-B-Kopf-Zyklus**
# (LogReg tunen, dann Kopf-Wechsel LinearSVC/kNN).
#
# **Die Frage:** Der finegetunte Encoder hat seine Repräsentation schon für *seinen*
# linearen Kopf geformt. Bringt ein *anderer* Kopf (LinearSVC max-margin, kNN
# Nachbarschaft) darauf noch etwas — oder ist der Kopf, sobald der Encoder angepasst
# ist, zweitrangig? (Erwartung aus dem Gespräch: eher marginal — genau das wollen wir
# *messen*.) Streng genommen ist das schon halb Synthese; als Experiment lebt es hier.

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
for d in (root, root / "04_track_c_finetuning"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import numpy as np
import torch

from data_utils import load_banking77
from eval_utils import evaluate_and_save

MODELL_DIR = root / "modelle" / "roberta_ft"
assert MODELL_DIR.exists(), (
    f"{MODELL_DIR} fehlt — erst voll/roberta.py laufen lassen und die "
    "Persistenz-Zelle ausführen (speichert den finegetunten Encoder)."
)

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}")

# %% [markdown]
# ## Schritt 1 — Den finegetunten Encoder laden
#
# `AutoModel` lädt aus `roberta_ft` nur den **RoBERTa-Körper** (der Klassifikations-
# Kopf wird verworfen — den wollen wir ja gerade *nicht*). Das sind die für banking77
# *aufgetauten* Gewichte — anders als Track Bs generischer Frozen-Encoder.

# %%
from transformers import AutoModel, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
encoder = AutoModel.from_pretrained(MODELL_DIR).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(MODELL_DIR)

laengen = np.array([len(tokenizer(t)["input_ids"]) for t in train_texts])
MAX_LEN = int(np.ceil(np.percentile(laengen, 99) / 8) * 8)
print(f"Encoder auf {DEVICE}.   MAX_LEN = {MAX_LEN}")

# %% [markdown]
# ## Schritt 2 — Embeddings ziehen (Feature Extraction)
#
# Jeden Text durch den eingefrorenen Encoder schicken und den **`<s>`-Vektor**
# (Position 0 des letzten Hidden-States) nehmen — das ist die Sentence-Repräsentation,
# die RoBERTas Kopf beim Finetunen gelesen hat, also die, die die Anpassung geformt
# hat. **L2-normalisiert** (wie in Track B — gut für LogReg & Cosinus-kNN). Teuer,
# darum als `.npy` gecacht.

# %%
CACHE = root / "04_track_c_finetuning" / "hybrid" / "cache"


@torch.no_grad()
def extract(texts, cache_key, batch_size=64):
    path = CACHE / f"{cache_key}.npy"
    if path.exists():
        print(f"[cache] {cache_key}: geladen")
        return np.load(path)
    vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size], padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(DEVICE)
        cls = encoder(**enc).last_hidden_state[:, 0]  # <s>-Token
        cls = torch.nn.functional.normalize(cls, dim=1)  # L2
        vecs.append(cls.cpu().numpy())
    v = np.vstack(vecs)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, v)
    print(f"[gespeichert] {cache_key}: {v.shape}")
    return v


X_train = extract(train_texts, "roberta_ft_train")
X_test = extract(test_texts, "roberta_ft_test")
print(f"X_train: {X_train.shape}  (finegetunte RoBERTa-Embeddings)")

# %% [markdown]
# # Kopf-Zyklus (identisch zu Track B · mpnet)
#
# Ab hier ist es **wörtlich der Track-B-Ansatz** auf diesen Embeddings: Val-Split,
# LogReg-`C` greedy auf val tunen (geteilter F3-Kern `optimization.greedy_search`),
# dann Kopf-*Wechsel* LogReg vs. kNN vs. LinearSVC. Die sklearn-Köpfe arbeiten direkt
# mit den Label-Namen (wie in Track B).

# %% [markdown]
# ## Val-Split + LogReg-`C` tunen

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from optimization import greedy_search

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, train_labels, test_size=0.15, stratify=train_labels, random_state=42
)


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
plot_rounds(proto_df, "RoBERTa-ft-Kopf (LogReg) — Optimierungsrunden")

# %% [markdown]
# ## Kopf-Wechsel — LogReg vs. kNN vs. LinearSVC (auf val)

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

# %%
final = heads[best_head_name]
final.fit(X_train, train_labels)
p = final.predict(X_test)
test_acc = accuracy_score(test_labels, p)
test_f1 = f1_score(test_labels, p, average="macro")
print(f"RoBERTa-ft Embeddings + {best_head_name}   test-Acc: {test_acc*100:.2f} %   Macro-F1: {test_f1*100:.2f} %")
print(f"Vergleich:  RoBERTa eigener Softmax-Kopf 93,89 %  ·  B mpnet frozen 94,12 %")

# evaluate_and_save: speichert zusätzlich alle Test-Vorhersagen nach
# predictions/C_hybrid_roberta.json (Fehler-Analyse). p/test_labels sind Label-NAMEN.
if hasattr(final, "predict_proba"):
    _scores, _stype = final.predict_proba(X_test), "proba"
else:
    _scores, _stype = final.decision_function(X_test), "decision_function"
evaluate_and_save(
    "C_hybrid_roberta",
    test_labels, p,
    model=f"RoBERTa-ft Embeddings + {best_head_name}",
    note="C×B Hybrid: finegetunter Encoder (frozen) + Track-B-Kopf, test 1x",
    scores=_scores, classes=final.classes_, score_type=_stype,
)

# %% [markdown]
# ## Wo irrt das Modell? — Fehlerbilder

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

plot_confusion_matrix(test_labels, p, title=f"RoBERTa-ft + {best_head_name} — Confusion Matrix")
plot_top_confusions(test_labels, p, top=15)
plot_per_class_f1(test_labels, p, worst=20)

# %% [markdown]
# ## Deuten
#
# Drei Zahlen entscheiden die Frage:
# - **RoBERTa eigener Softmax-Kopf: 93,89 %** (voll/roberta.py)
# - **RoBERTa-ft Embeddings + getauschter Kopf: hier**
# - mpnet frozen + LinearSVC: 94,12 %
#
# - **Kaum ein Unterschied zum Softmax-Kopf?** Dann ist bestätigt: sobald der Encoder
#   *aufgetaut* ist, sind seine Features schon linear separierbar — der Kopf-Wechsel
#   holt fast nichts mehr. Genau umgekehrt zu Track B, wo der Kopf-Wechsel viel
#   brachte (dort war der Encoder *nicht* angepasst). Die Lehre: der Kopf zählt, *bis*
#   der Encoder die Arbeit übernimmt.
# - **Ein spürbarer Sprung (in die eine oder andere Richtung)?** Dann trägt die Wahl
#   der Kopf-Geometrie auch auf angepassten Features noch — interessanter Sonderfall.
#
# So oder so schließt das die Frage sauber ab, statt sie im Kopf zu lassen — und es
# ist der Baustein für die Synthese: **Komponenten über Generationen hinweg
# kombinieren**, ehrlich gemessen.
#
# **✓ Checkpoint:** In Track B brachte der Kopf-Wechsel (LogReg→LinearSVC) einen
# klaren Sprung, hier vermutlich kaum. Was sagt dieser Unterschied darüber aus, *wo*
# bei Gen 2 vs. Gen 3 die eigentliche Arbeit passiert?
