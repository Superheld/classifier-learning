# %% [markdown]
# # Track C — Gen 3a: Finetuning mit stärkerer Basis (banking77)
#
# ## Warum dieses zweite Modell?
#
# `distilbert.py` hat den *aufgetauten* DistilBERT plain auf **90,08 %** gebracht —
# im Rauschen gleichauf mit dem getunten TF-IDF (A) und **4 Punkte unter** dem
# *eingefrorenen* mpnet aus Track B (94,12 %). Lehre: „aufgetaut" schlägt „eingefroren"
# **nicht automatisch** — die *Basis* entscheidet. DistilBERT ist ein kleines,
# generisches Modell; mpnet war ein für Satz-Semantik vortrainierter Spezialist.
#
# Deshalb hier die ehrliche Gegenprobe: **dieselbe Mechanik, stärkere Basis.**
# `FacebookAI/roberta-base` (~125 Mio Parameter, besser & länger vortrainiert als
# BERT-base). Einzige geänderte Variable ggü. `distilbert.py` ist das Modell — so
# misst der Vergleich sauber, was die Basis bringt.
#
# **Mechanik = identisch zu `distilbert.py`** (Tokenisieren → Kopf → Trainer →
# einmal test messen). Ich kommentiere hier vor allem, was sich durch RoBERTa ändert;
# für die ausführlichen Erklärungen siehe `distilbert.py`.

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

import torch

from data_utils import load_banking77

MODEL_NAME = "FacebookAI/roberta-base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Device: {DEVICE}")

# %% [markdown]
# ## Schritt 1 — Labels: Namen → Zahlen (wie gehabt)
#
# Identisch zu `distilbert.py`: eine eigene, für beide Splits gemeinsame Abbildung
# `label_text` → ID, plus `id2label` als lesbarer Rückweg.

# %%
labels_sorted = sorted(set(train_labels))
label2id = {name: i for i, name in enumerate(labels_sorted)}
id2label = {i: name for name, i in label2id.items()}
NUM_LABELS = len(labels_sorted)

y_train = [label2id[l] for l in train_labels]
y_test = [label2id[l] for l in test_labels]
print(f"{NUM_LABELS} Intents.")

# %% [markdown]
# ## Schritt 2 — RoBERTas Tokenizer ist ein *anderer*
#
# Wichtig: jeder Encoder bringt seinen **eigenen** Tokenizer mit, und RoBERTas ist
# ein anderer Typ als DistilBERTs. DistilBERT nutzte WordPiece (`##`-Fortsetzungen,
# `[CLS]`/`[SEP]`). RoBERTa nutzt **byte-level BPE**:
# - Das **`Ġ`** vor einem Token steht für ein vorangestelltes **Leerzeichen** —
#   markiert also einen Wortanfang. Tokens *ohne* `Ġ` sind Wort-Fortsetzungen.
# - **`<s>`** und **`</s>`** umrahmen den Text (RoBERTas Pendant zu `[CLS]`/`[SEP]`).
#
# Genau deshalb muss der Tokenizer exakt zum Modell passen — dieselben IDs bedeuten
# bei RoBERTa und DistilBERT völlig verschiedene Dinge.

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

beispiel = "What can I do if my card still hasn't arrived after 2 weeks?"
enc = tokenizer(beispiel)
print("Text  :", beispiel)
print("Token :", tokenizer.convert_ids_to_tokens(enc["input_ids"]))

# %% [markdown]
# ## Schritt 3 — `max_length` aus den Daten (neu berechnet)
#
# BPE zerlegt anders als WordPiece, also können die Token-Längen leicht abweichen —
# wir rechnen sie für RoBERTas Tokenizer frisch aus, statt DistilBERTs Wert zu
# übernehmen. banking77 bleibt kurz.

# %%
import numpy as np

laengen = np.array([len(tokenizer(t)["input_ids"]) for t in train_texts])
for p in (50, 95, 99, 100):
    print(f"  {p:>3}. Perzentil: {int(np.percentile(laengen, p)):>3} Token")

MAX_LEN = int(np.ceil(np.percentile(laengen, 99) / 8) * 8)
abgeschnitten = (laengen > MAX_LEN).mean() * 100
print(f"\n→ MAX_LEN = {MAX_LEN}   (schneidet nur {abgeschnitten:.2f} % an)")

# %% [markdown]
# ## Schritt 4 — Klassifikations-Kopf auf RoBERTa
#
# Wie bei DistilBERT: vortrainierter Körper + **frischer, zufälliger** Kopf
# (`num_labels=77`). Auch hier meldet transformers die neuen `classifier`-Gewichte
# als untrainiert — erwartet.

# %%
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModell geladen: {n_params / 1e6:.1f} Mio Parameter (vs. DistilBERT 67,0).")

# %% [markdown]
# # P2 — „Bauen" (plain, exakt dieselben Einstellungen wie DistilBERT)
#
# Damit der Vergleich sauber ist, ändern wir **nur die Basis** — alle Trainings-
# Einstellungen bleiben wie in `distilbert.py`: LR 5e-5, 3 Epochen, Batch 32, kein
# Tuning, test genau einmal.
#
# **Ein ehrlicher Vorbehalt:** RoBERTa gilt als **LR-empfindlicher** als DistilBERT —
# bei 5e-5 *kann* das Training gelegentlich kippen (Loss bleibt flach/steigt, Acc
# landet bei ~random). Falls das passiert, ist es kein Bug, sondern genau die
# Instabilität, die P3 mit einer kleineren Learning Rate + Warmup einfängt. Beim
# `train()` also auf den Loss schauen: fällt er, ist alles gut.

# %%
from datasets import Dataset
from transformers import DataCollatorWithPadding


def to_dataset(texts, labels):
    ds = Dataset.from_dict({"text": texts, "labels": labels})
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text"],
    )


train_ds = to_dataset(train_texts, y_train)
test_ds = to_dataset(test_texts, y_test)
collator = DataCollatorWithPadding(tokenizer)
print(train_ds)

# %% [markdown]
# ## Trainer & Training
#
# RoBERTa-base ist rund doppelt so groß wie DistilBERT — auf MPS dauert der Lauf
# entsprechend länger. Bleibt Batch 32 im Speicher eng, ist `per_device_train_batch_size=16`
# die einzige nötige Änderung.

# %%
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=str(root / "04_track_c_finetuning/runs"),
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    dataloader_pin_memory=False,  # pin_memory wird auf MPS nicht unterstützt
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator,
    processing_class=tokenizer,
)
trainer.train()

# %% [markdown]
# ## Finale Messung — Testset, genau EINMAL

# %%
from sklearn.metrics import accuracy_score, f1_score

from data_utils import save_result

pred = trainer.predict(test_ds)
preds = np.argmax(pred.predictions, axis=-1)

acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average="macro")
print(f"RoBERTa finetuned (plain)   Accuracy: {acc * 100:.2f} %   Macro-F1: {macro_f1 * 100:.2f} %")
print(f"Latte:  B mpnet 94,12 %  ·  C DistilBERT 90,08 %  ·  A LogReg 90,25 %")

save_result(
    "C_plain_roberta",
    acc,
    macro_f1=round(macro_f1, 4),
    model="RoBERTa-base (finetuned)",
    note="P2 plain, Standardwerte, test 1x",
)

# %% [markdown]
# ## Deuten & nächster Schritt
#
# Die eine Frage, sauber isoliert: **holt die stärkere Basis den Rückstand auf?**
# - Landet RoBERTa deutlich über DistilBERTs 90,08 %, ist bewiesen: bei Finetuning
#   trägt die *Basis* den Löwenanteil, nicht das Auftauen an sich.
# - Kommt es schon plain nah an oder über B's 94,12 %, hat Gen 3 Gen 2 hier zum
#   ersten Mal echt geschlagen.
# - Bleibt es flach/instabil: LR-Empfindlichkeit — dann zeigt P3, wie viel eine
#   kleinere Learning Rate rettet.
#
# **P3** kommt dann auf diesem Sieger-Modell: Val-Split, LR-Dreier + Warmup, Early
# Stopping, Klassengewichte, LoRA als Sparmodus — test bleibt bis zur finalen
# P3-Messung zu.

# %% [markdown]
# # P3 — „Optimieren"
#
# RoBERTa plain lag bei **93,17 %** — nur ~1 Punkt unter B (94,12 %) und noch ganz
# ungetunt. Jetzt drehen wir die Schrauben aus dem Curriculum, allen voran die
# **Learning Rate** — laut Curriculum „die empfindlichste Schraube". Ziel: B überholen.
#
# Die Disziplin bleibt: optimiert wird **auf val**, nie am Testset. Neu gegenüber
# A/B ist, dass val jetzt zusätzlich das **Early Stopping** trägt — es sagt dem
# Training, wann es aufhören soll, bevor es auswendig lernt (overfittet).
#
# **Ehrlicher Zeit-Hinweis:** P3 finetunt RoBERTa **mehrfach** (ein Lauf je LR + ein
# finaler). Auf MPS ist das der teure Teil — rechne mit **~1 Stunde** insgesamt (Early
# Stopping kürzt oft ab). Wer es eilig hat, testet statt drei nur zwei LRs.

# %% [markdown]
# ## Val-Split (F2-Disziplin)
#
# 15 % aus dem Trainingsset abzweigen, stratifiziert, Seed 42 — dieselbe Aufteilung
# wie überall. `to_dataset` (aus P2) macht daraus tokenisierte `datasets`.

# %%
from sklearn.model_selection import train_test_split

tr_texts, va_texts, tr_y, va_y = train_test_split(
    train_texts, y_train, test_size=0.15, stratify=y_train, random_state=42
)
tr_ds = to_dataset(tr_texts, tr_y)
va_ds = to_dataset(va_texts, va_y)
print(f"Trainingsteil: {len(tr_y)}   Validierung: {len(va_y)}")

# %% [markdown]
# ## Messfunktion für den Trainer
#
# Damit der `Trainer` nach jeder Epoche auf val misst, braucht er `compute_metrics`.
# Wir geben ihm unser gewohntes Besteck (Accuracy + Macro-F1) zurück; nach
# **Macro-F1** wird das beste Modell gewählt (fair bei unbalancierten Klassen).

# %%
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

# %% [markdown]
# ## Der LR-Dreier — ein Finetuning je Learning Rate
#
# `search_lr` finetunt RoBERTa mit einer gegebenen LR auf dem Trainingsteil,
# misst nach **jeder** Epoche auf val und stoppt, sobald sich val 2 Epochen lang
# nicht mehr bessert (`EarlyStoppingCallback`). `load_best_model_at_end` holt am
# Ende automatisch das val-beste Modell zurück. Zurück kommen die beste val-Macro-F1
# und die Epoche, in der sie fiel — Letztere brauchen wir gleich fürs finale Refit.
#
# Das ist derselbe F3-Gedanke wie `greedy_search` (auf val vergleichen, Bestes
# behalten) — nur ist hier *jede* Auswertung ein voller Trainingslauf, darum halten
# wir die Suche klein (drei LRs) statt die volle Greedy-Maschine zu fahren.
#
# Gesucht wird **niedriger** als das plain-5e-5: RoBERTa mag kleinere Learning Rates.

# %%
from transformers import EarlyStoppingCallback


def search_lr(lr):
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
    )
    a = TrainingArguments(
        output_dir=str(root / f"04_track_c_finetuning/runs/lr_{lr:.0e}"),
        num_train_epochs=6,
        per_device_train_batch_size=32,
        learning_rate=lr,
        warmup_steps=100,  # erste ~100 Schritte die LR hochrampen — stabilisiert RoBERTa
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=False,  # pin_memory wird auf MPS nicht unterstützt
    )
    t = Trainer(
        model=m,
        args=a,
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        data_collator=collator,
        processing_class=tokenizer,
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
#
# Wie in A/B: die auf val gewählte Config (beste LR, Epochen-Budget aus dem
# Sieger-Lauf) wird auf dem **vollen** Trainingsset neu trainiert — jetzt darf val
# wieder mittrainieren — und **einmal** auf test gemessen. Kein Early Stopping mehr
# (kein val übrig), stattdessen die feste Epochenzahl, bei der val vorhin gipfelte.

# %%
import math

from eval_utils import evaluate_and_save

final_epochs = max(1, math.ceil(best_epoch))
print(f"Refit: LR {best_lr:.0e}, {final_epochs} Epochen, volles Train ({len(train_texts)}).")

m_final = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
)
a_final = TrainingArguments(
    output_dir=str(root / "04_track_c_finetuning/runs/final"),
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
    model=m_final,
    args=a_final,
    train_dataset=train_ds,  # volles Train aus P2
    data_collator=collator,
    processing_class=tokenizer,
)
t_final.train()

pred = t_final.predict(test_ds)
preds = np.argmax(pred.predictions, axis=-1)
acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average="macro")
print(f"\nRoBERTa finetuned (LR-getunt)   Accuracy: {acc*100:.2f} %   Macro-F1: {macro_f1*100:.2f} %")
print(f"Latte:  B mpnet 94,12 %  ·  RoBERTa plain 93,17 %")

# evaluate_and_save braucht Label-NAMEN; der Trainer liefert Integer-IDs (y_test, preds).
# id2label übersetzt zurück — dann landen die Vorhersagen in predictions/C_tuned_roberta.json.
y_true_names = [id2label[int(i)] for i in y_test]
y_pred_names = [id2label[int(i)] for i in preds]
evaluate_and_save(
    "C_tuned_roberta",
    y_true_names, y_pred_names,
    model="RoBERTa-base (finetuned, LR-getunt)",
    config=f"lr={best_lr:.0e}, epochs={final_epochs}, warmup=100steps",
    note="P3 LR auf val gewählt, refit full train, test 1x",
)

# %% [markdown]
# ## Wo irrt das getunte Modell? — Fehlerbilder
#
# Fehler-Analyse auf dem besten Modell (LR-getunt), gleiches Besteck wie A/B. Die
# vorhergesagten IDs übersetzen wir per `id2label` in lesbare Intent-Namen. Spannend:
# verwechselt der finegetunte RoBERTa dieselben Paare wie die eingefrorenen Encoder
# in Track B — oder hat das Auftauen andere Grenzfälle verschoben?

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

pred_names = [id2label[int(p)] for p in preds]

plot_confusion_matrix(test_labels, pred_names, title="RoBERTa getunt — Confusion Matrix")
plot_top_confusions(test_labels, pred_names, top=15)
plot_per_class_f1(test_labels, pred_names, worst=20)

# %% [markdown]
# ## Modell sichern (Persistenz)
#
# Der finegetunte RoBERTa (93,89 %) lebt bis hier nur im Kernel-RAM — ein Neustart
# und er ist weg. Wir schreiben ihn ins gitignorete `modelle/`, damit andere
# Notebooks (der Kopf-Zyklus in `hybrid/`, später das Dashboard) ihn laden können,
# **ohne neu zu finetunen**. `save_pretrained` legt Gewichte + Config + Tokenizer im
# HF-Standardformat ab; zurückgeladen wird mit `from_pretrained(MODELL_DIR)`.

# %%
MODELL_DIR = root / "modelle" / "roberta_ft"
t_final.model.save_pretrained(MODELL_DIR)
tokenizer.save_pretrained(MODELL_DIR)
print(f"gespeichert nach {MODELL_DIR}  (laden: from_pretrained(MODELL_DIR))")

# %% [markdown]
# ## Deuten & nächster Schritt
#
# - **Über B (94,12 %)?** Dann hat Gen 3 hier zum ersten Mal Gen 2 geschlagen — der
#   aufgetaute, LR-getunte Encoder überholt den besten eingefrorenen. Wenn knapp
#   darunter: die beiden liegen im Messrauschen, und die *Kosten*-Seite (Track C
#   braucht GPU-Training, B nur einen Kopf) wird zum eigentlichen Unterscheider (P4).
# - **Was die LR gebracht hat:** der Abstand zwischen bester und schlechtester LR im
#   Dreier zeigt schwarz auf weiß, warum das Curriculum sie „die empfindlichste
#   Schraube" nennt — dieselbe Architektur, nur eine andere Zahl.
#
# **Nächster Schritt — LoRA:** derselbe RoBERTa, aber *parameter-effizient*
# finetunen (nur ~1 % der Gewichte). Eigenes Konzept → eigene Datei:
# **`peft/roberta_lora.py`**. Der Vergleich full vs. LoRA läuft übers
# Scoreboard/Dashboard, nicht in diesem Notebook.
