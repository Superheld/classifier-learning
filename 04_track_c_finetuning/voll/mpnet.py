# %% [markdown]
# # Track C · voll — mpnet finetunen (banking77)
#
# ## Der Champion, jetzt aufgetaut
#
# mpnet (`all-mpnet-base-v2`) war in **Track B unser Bester** — aber *eingefroren*:
# 94,12 % mit getuntem Kopf, „am BERT-Dach ohne Finetuning". Jetzt drehen wir genau
# an der Schraube, die dort fix blieb: wir **tauen mpnet auf** und trainieren seine
# Gewichte mit. Die Frage dieses Notebooks:
#
# > Der schon für Satz-Semantik vortrainierte Encoder war eingefroren kaum zu
# > schlagen — bringt ihn *finetunen* über die 94,12 %? Oder war er eingefroren
# > bereits so nah am Optimum, dass Auftauen nichts mehr holt (oder gar overfittet)?
#
# Das ist die interessanteste Basis im ganzen Track: nicht ein generisches Modell
# (DistilBERT/RoBERTa), sondern der *aufgabennahe Spezialist*. **Mechanik = wie
# `roberta.py`**; ich kommentiere hier nur das mpnet-Spezifische.

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

import math

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_utils import load_banking77, save_result

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Device: {DEVICE}")

# %% [markdown]
# ## Labels, Tokenizer, max_length, Kopf
#
# Wie gehabt (Details in `roberta.py`). Ein Wort zum Tokenizer: mpnet bringt seinen
# **eigenen** mit — `<s>`/`</s>` als Rahmen (wie RoBERTa), aber lowercase-WordPiece
# ohne `Ġ`. Und die Basis, die wir laden, ist der *sentence-transformers*-Checkpoint:
# `AutoModelForSequenceClassification` nimmt dessen MPNet-Encoder (die satz-
# angepassten Gewichte — genau der Punkt!), verwirft den SBERT-Pooling-Kopf und
# setzt einen frischen 77-Klassen-Kopf drauf.

# %%
labels_sorted = sorted(set(train_labels))
label2id = {name: i for i, name in enumerate(labels_sorted)}
id2label = {i: name for name, i in label2id.items()}
NUM_LABELS = len(labels_sorted)
y_train = [label2id[l] for l in train_labels]
y_test = [label2id[l] for l in test_labels]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
laengen = np.array([len(tokenizer(t)["input_ids"]) for t in train_texts])
MAX_LEN = int(np.ceil(np.percentile(laengen, 99) / 8) * 8)
print(f"{NUM_LABELS} Intents.   MAX_LEN = {MAX_LEN} "
      f"(schneidet {(laengen > MAX_LEN).mean() * 100:.2f} % an)")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
)
print(f"Modell: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} Mio Parameter.")

# %% [markdown]
# ## Gemeinsame Helfer (wie in roberta.py)

# %%
def to_dataset(texts, labels):
    ds = Dataset.from_dict({"text": texts, "labels": labels})
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text"],
    )


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


train_ds = to_dataset(train_texts, y_train)
test_ds = to_dataset(test_texts, y_test)
collator = DataCollatorWithPadding(tokenizer)

# %% [markdown]
# # P2 — „Bauen" (plain, gleiche Einstellungen wie roberta/distilbert)
#
# LR 5e-5, 3 Epochen, kein Tuning, test genau einmal — damit der Vergleich der
# *Basen* fair bleibt (einzige geänderte Variable: das Modell).

# %%
args = TrainingArguments(
    output_dir=str(root / "04_track_c_finetuning/runs/mpnet/plain"),
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    dataloader_pin_memory=False,
)
trainer = Trainer(
    model=model, args=args, train_dataset=train_ds,
    data_collator=collator, processing_class=tokenizer,
)
trainer.train()

# %% [markdown]
# ## Finale Messung — Testset, genau EINMAL

# %%
pred = trainer.predict(test_ds)
preds = np.argmax(pred.predictions, axis=-1)
acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average="macro")
print(f"mpnet finetuned (plain)   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"Latte:  mpnet FROZEN 94,12 %  ·  RoBERTa finetuned 93,89 %")

save_result(
    "C_plain_mpnet_ft",
    acc,
    macro_f1=round(macro_f1, 4),
    model="mpnet (finetuned)",
    note="P2 plain, Standardwerte, test 1x",
)

# %% [markdown]
# ## Wo irrt das Modell? — Fehlerbilder

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

pred_names = [id2label[int(p)] for p in preds]
plot_confusion_matrix(test_labels, pred_names, title="mpnet finetuned plain — Confusion Matrix")
plot_top_confusions(test_labels, pred_names, top=15)
plot_per_class_f1(test_labels, pred_names, worst=20)

# %% [markdown]
# # P3 — „Optimieren" (LR-Dreier + Early Stopping)
#
# Gleicher Zyklus wie in `roberta.py`: Val-Split, drei Learning Rates auf val, das
# beste Modell per Early Stopping, dann Refit auf vollem Train und test einmal.

# %%
from sklearn.model_selection import train_test_split

tr_texts, va_texts, tr_y, va_y = train_test_split(
    train_texts, y_train, test_size=0.15, stratify=y_train, random_state=42
)
tr_ds = to_dataset(tr_texts, tr_y)
va_ds = to_dataset(va_texts, va_y)


def search_lr(lr):
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
    )
    a = TrainingArguments(
        output_dir=str(root / f"04_track_c_finetuning/runs/mpnet/lr_{lr:.0e}"),
        num_train_epochs=6,
        per_device_train_batch_size=32,
        learning_rate=lr,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=False,
    )
    t = Trainer(
        model=m, args=a, train_dataset=tr_ds, eval_dataset=va_ds,
        data_collator=collator, processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    t.train()
    evals = [(h["epoch"], h["eval_macro_f1"]) for h in t.state.log_history if "eval_macro_f1" in h]
    best_epoch, best_f1 = max(evals, key=lambda e: e[1])
    return best_f1, best_epoch


LRS = [1e-5, 2e-5, 3e-5]
proto = {}
for lr in LRS:
    f1, ep = search_lr(lr)
    proto[lr] = (f1, ep)
    print(f"LR {lr:.0e}:  beste val Macro-F1 {f1*100:5.2f} %  @ Epoche {ep:.0f}")

best_lr, (best_val_f1, best_epoch) = max(proto.items(), key=lambda kv: kv[1][0])
print(f"\n→ beste LR: {best_lr:.0e}  (val Macro-F1 {best_val_f1*100:.2f} %, Epoche {best_epoch:.0f})")

# %% [markdown]
# ## Finale Messung — Refit auf vollem Train, Testset genau EINMAL

# %%
final_epochs = max(1, math.ceil(best_epoch))
m_final = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
)
a_final = TrainingArguments(
    output_dir=str(root / "04_track_c_finetuning/runs/mpnet/final"),
    num_train_epochs=final_epochs,
    per_device_train_batch_size=32,
    learning_rate=best_lr,
    warmup_steps=100,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    dataloader_pin_memory=False,
)
t_final = Trainer(
    model=m_final, args=a_final, train_dataset=train_ds,
    data_collator=collator, processing_class=tokenizer,
)
t_final.train()

pred = t_final.predict(test_ds)
preds = np.argmax(pred.predictions, axis=-1)
acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average="macro")
print(f"\nmpnet finetuned (LR-getunt)   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"Latte:  mpnet FROZEN 94,12 %  ·  RoBERTa finetuned 93,89 %")

save_result(
    "C_tuned_mpnet_ft",
    acc,
    macro_f1=round(macro_f1, 4),
    model="mpnet (finetuned, LR-getunt)",
    config=f"lr={best_lr:.0e}, epochs={final_epochs}, warmup=100steps",
    note="P3 LR auf val gewählt, refit full train, test 1x",
)

# %% [markdown]
# ## Wo irrt das getunte Modell? — Fehlerbilder

# %%
pred_names = [id2label[int(p)] for p in preds]
plot_confusion_matrix(test_labels, pred_names, title="mpnet finetuned getunt — Confusion Matrix")
plot_top_confusions(test_labels, pred_names, top=15)
plot_per_class_f1(test_labels, pred_names, worst=20)

# %% [markdown]
# ## Deuten
#
# Drei mpnet-Zahlen liegen jetzt nebeneinander — und der Vergleich ist der Kern:
# - **frozen + getunter Kopf: 94,12 %** (Track B)
# - **finetuned plain / LR-getunt: hier**
#
# Zwei mögliche Geschichten, beide lehrreich:
# - **Finetuning zieht vorbei:** dann trägt selbst beim starken Satz-Encoder das
#   Auftauen noch etwas bei — die Aufgabe braucht mehr als eine feste Repräsentation.
# - **Es zieht nicht (oder fällt gar):** dann war mpnet eingefroren schon so nah am
#   Plafond, dass Auftauen auf ~10k Beispielen nur Overfitting-Risiko bringt, keinen
#   Gewinn. Das wäre die stärkste Bestätigung des Track-B-Befunds — ein
#   aufgabennaher *eingefrorener* Encoder ist schwer zu schlagen, und der ganze
#   Finetuning-Aufwand zahlt sich nicht immer aus (→ Kosten-Argument der Synthese).
#
# **✓ Checkpoint:** Angenommen finetuned mpnet landet ~gleichauf mit frozen mpnet.
# Welchen der beiden würdest du in Produktion nehmen — und mit welchem Argument
# jenseits der Accuracy?
