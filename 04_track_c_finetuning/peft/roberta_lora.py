# %% [markdown]
# # Track C · peft — RoBERTa + LoRA (banking77)
#
# ## Parameter-effizientes Finetuning
#
# Es gibt zwei Wege, einen Encoder *aufzutauen* (das Spektrum aus Track C):
# - **Full Fine-tuning** (`voll/roberta.py`): alle 125 Mio Gewichte trainieren → 93,89 %.
# - **PEFT / LoRA** (dieses Notebook): Basis einfrieren, nur kleine Adapter trainieren.
#
# Dies ist ein **eigenes Konzept**, darum eine eigene Datei — vergleichbar gegen das
# volle Finetuning übers Scoreboard. Mechanik (Tokenisieren, Trainer, LR/Early
# Stopping) wie in `voll/roberta.py`; hier steht die LoRA-Sache im Zentrum.
#
# ## Was ist LoRA?
#
# **LoRA (Low-Rank Adaptation)**:
# - Die vortrainierten Gewichte werden **eingefroren** — kein einziges angefasst.
# - In ausgewählte Schichten (hier die **Attention**-Projektionen `query` & `value`)
#   werden **zwei kleine Matrizen** `A` (d×r) und `B` (r×d) mit winzigem **Rang** `r`
#   (z.B. 8) eingeschoben. Die Schicht rechnet `W + B·A` — das eingefrorene `W` plus
#   einen niedrig-rangigen Zusatz. Nur `A` und `B` werden trainiert.
# - **Intuition:** die *Anpassung*, die eine Aufgabe braucht, ist viel
#   niederdimensionaler als die volle Gewichtsmatrix — eine dünne Korrektur, kein
#   Neuschreiben. Rang 8 fängt den Großteil ein.
#
# **Der Gewinn:** nur ~1 % der Parameter trainieren → weniger Speicher, schneller,
# Adapter ein paar MB statt 500. **Die Frage:** wie nah kommt der Sparmodus an die
# 93,89 % des vollen Finetunings?

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
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_utils import load_banking77
from eval_utils import evaluate_and_save

MODEL_NAME = "FacebookAI/roberta-base"

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")

# %% [markdown]
# ## Vorbereitung (wie voll/roberta.py — Labels, Tokenizer, Daten)

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
print(f"{NUM_LABELS} Intents.   MAX_LEN = {MAX_LEN}")


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

tr_texts, va_texts, tr_y, va_y = train_test_split(
    train_texts, y_train, test_size=0.15, stratify=y_train, random_state=42
)
tr_ds = to_dataset(tr_texts, tr_y)
va_ds = to_dataset(va_texts, va_y)

# %% [markdown]
# ## Schritt 1 — LoRA-Adapter auf die Basis setzen
#
# `LoraConfig` beschreibt das Rezept, `get_peft_model` friert die Basis ein und hängt
# die Adapter dran. `print_trainable_parameters` zeigt, wie klein der trainierte
# Anteil ist. `lora_alpha` skaliert den Zusatz (Faktor `alpha/r`). Der
# Klassifikations-Kopf bleibt voll trainierbar (`modules_to_save`) — er ist frisch.

# %%
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # RoBERTas Attention-Projektionen
    lora_dropout=0.1,
    modules_to_save=["classifier"],
)


def build_lora_model():
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
    )
    return get_peft_model(base, lora_config)


lora_model = build_lora_model()
lora_model.print_trainable_parameters()

# %% [markdown]
# ## Schritt 2 — LoRA trainieren (auf val Epochen finden)
#
# Gleiche Trainer-Mechanik wie der LR-Dreier in `voll/roberta.py`. Zwei Unterschiede:
# LoRA mag eine **höhere Learning Rate** (nur kleine Matrizen zu bewegen — hier 3e-4
# statt 3e-5) und braucht oft **mehr Epochen** (der Umweg über die Adapter lernt
# langsamer). Early Stopping auf val findet den Punkt.

# %%
LORA_LR = 3e-4


def train_lora(model, train_dataset, epochs, eval_dataset, out):
    args = TrainingArguments(
        output_dir=str(root / f"04_track_c_finetuning/runs/lora/{out}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        learning_rate=LORA_LR,
        warmup_steps=100,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=1,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=False,
    )
    cb = [EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset is not None else None
    t = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator, processing_class=tokenizer,
        compute_metrics=compute_metrics, callbacks=cb,
    )
    t.train()
    return t


t_search = train_lora(lora_model, tr_ds, epochs=8, eval_dataset=va_ds, out="search")
evals = [(h["epoch"], h["eval_macro_f1"]) for h in t_search.state.log_history if "eval_macro_f1" in h]
lora_best_epoch, lora_best_f1 = max(evals, key=lambda e: e[1])
print(f"\nLoRA beste val Macro-F1 {lora_best_f1*100:.2f} %  @ Epoche {lora_best_epoch:.0f}")

# %% [markdown]
# ## Schritt 3 — Refit auf vollem Train, Testset genau EINMAL

# %%
lora_epochs = max(1, math.ceil(lora_best_epoch))
print(f"LoRA-Refit: {lora_epochs} Epochen, volles Train.")

lora_final = build_lora_model()
t_lora_final = train_lora(lora_final, train_ds, epochs=lora_epochs, eval_dataset=None, out="final")

pred = t_lora_final.predict(test_ds)
preds = np.argmax(pred.predictions, axis=-1)
acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average="macro")
print(f"\nRoBERTa + LoRA   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"Vergleich:  volles Finetuning 93,89 %  ·  B mpnet 94,12 %")

# Label-NAMEN via id2label; Scores = softmax der Logits → predictions/C_lora_roberta*.
y_true_names = [id2label[int(i)] for i in y_test]
pred_names = [id2label[int(p)] for p in preds]
_proba = np.exp(pred.predictions - pred.predictions.max(axis=1, keepdims=True))
_proba /= _proba.sum(axis=1, keepdims=True)
evaluate_and_save(
    "C_lora_roberta",
    y_true_names, pred_names,
    model="RoBERTa-base (LoRA)",
    config=f"r=8, alpha=16, lr={LORA_LR:.0e}, epochs={lora_epochs}",
    note="C peft/LoRA-Sparmodus, test 1x",
    scores=_proba, classes=[id2label[i] for i in range(NUM_LABELS)], score_type="proba",
)

# %% [markdown]
# ## Wo irrt das Modell? — Fehlerbilder

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

pred_names = [id2label[int(p)] for p in preds]
plot_confusion_matrix(test_labels, pred_names, title="RoBERTa + LoRA — Confusion Matrix")
plot_top_confusions(test_labels, pred_names, top=15)
plot_per_class_f1(test_labels, pred_names, worst=20)

# %% [markdown]
# ## Deuten — was Sparmodus kostet (oder eben nicht)
#
# Stell drei Zahlen nebeneinander: **volles Finetuning (93,89 %)** vs. **LoRA** vs.
# der **trainierte Parameter-Anteil** aus Schritt 1. Die Lehre steckt im Verhältnis:
# - Kommt LoRA mit ~1 % der trainierten Gewichte auf ~99 % der Accuracy, ist das der
#   ganze Punkt — **fast gratis** in Speicher/Zeit/Artefaktgröße, praktisch kein
#   Qualitätsverlust. Genau darum ist LoRA bei knapper GPU der Standard.
# - Ein spürbarer Abstand hieße: für *diese* Aufgabe braucht es mehr Kapazität — dann
#   mit höherem Rang `r` oder mehr Ziel-Schichten (auch `key`, Feed-Forward) angehen.
#
# **✓ Checkpoint:** Du betreibst 10 verschiedene Intent-Klassifikatoren. Warum ist
# LoRA hier dem vollen Finetuning betrieblich haushoch überlegen — selbst wenn die
# Accuracy identisch wäre?
