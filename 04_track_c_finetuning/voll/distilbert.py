# %% [markdown]
# # Track C — Gen 3a: Finetuning (banking77)
#
# ## P1 „Vorbereiten"
#
# In Track B war der Encoder ein **fixes Lineal**: mpnet & Co. haben Vektoren
# geliefert, wir haben nur den kleinen Kopf trainiert — die Encoder-Gewichte
# blieben eingefroren (94,12 % „am BERT-Dach *ohne* Finetuning"). Jetzt **tauen
# wir den Encoder auf**: seine Gewichte lernen mit, speziell für unsere 77 Intents.
# Das ist der einzige Hebel, der über die 94 % hinaus kann — und die eigene
# Modell-Generation.
#
# ```
# Track B:  Text ──[Encoder EINGEFROREN]──▶ Vektor ──[Kopf trainiert]──▶ Klasse
# Track C:  Text ──[Encoder TRAINIERT + Kopf]──────────────────────────▶ Klasse
# ```
#
# **P1 ist reine Vorbereitung** — noch kein Training. Wir bringen den Text in die
# Form, die ein Transformer frisst (Subword-Token), wählen `max_length` bewusst
# und setzen dem vortrainierten Encoder einen frischen Klassifikations-Kopf auf.
# Das eigentliche Training (das Auftauen) ist P2.
#
# Werkzeug ab hier: **Hugging Face transformers** auf **MPS** (Apple-GPU) — nicht
# mehr scikit-learn. Modell: **distilbert-base-uncased** (klein, schnell, englisch;
# ~66 Mio Parameter, ~40 % schneller als BERT-base bei ~97 % Leistung).

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

# Autoreload: geänderte Module ohne Kernel-Neustart neu laden. Nur im Kernel.
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except (NameError, AttributeError):
    pass

root = Path.cwd()
while not (root / "data_utils.py").exists() and root != root.parent:
    root = root.parent
# Root (data_utils, eval_utils) + Track-C-Ordner auf den Importpfad.
for d in (root, root / "04_track_c_finetuning"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import torch

from data_utils import load_banking77

MODEL_NAME = "distilbert-base-uncased"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")
print(f"train: {len(train_texts)}   test: {len(test_texts)}   Device: {DEVICE}")

# %% [markdown]
# ## Schritt 1 — Labels: Namen → Zahlen (und zurück)
#
# Ein Transformer-Kopf gibt für jede Klasse eine Zahl aus, also brauchen die
# Labels ganzzahlige IDs `0..76`. Wir bauen die Abbildung aus den sortierten
# `label_text`-Namen selbst — **train und test mit *derselben* Abbildung**, sonst
# zeigen die IDs auf verschiedene Intents. `id2label` ist der Rückweg: damit werden
# Vorhersagen später wieder lesbar (`card_arrival` statt `11`).

# %%
labels_sorted = sorted(set(train_labels))
label2id = {name: i for i, name in enumerate(labels_sorted)}
id2label = {i: name for name, i in label2id.items()}
NUM_LABELS = len(labels_sorted)

y_train = [label2id[l] for l in train_labels]
y_test = [label2id[l] for l in test_labels]

print(f"{NUM_LABELS} Intents.  Beispiel-Mapping:")
for name in labels_sorted[:3]:
    print(f"  {label2id[name]:>2} ↔ {name}")
print(f"  … bis {NUM_LABELS - 1} ↔ {labels_sorted[-1]}")

# %% [markdown]
# ## Schritt 2 — Subword-Tokenisierung
#
# TF-IDF (Track A) zerlegte Text in ganze Wörter. Ein Transformer nutzt **Subword-
# Token**: häufige Wörter bleiben ein Token, seltene zerfallen in Wortstücke. So
# kommt das Modell mit *jedem* Wort klar (auch Tippfehlern, Namen) ohne unendliches
# Vokabular. Wichtig: **jedes Modell bringt seinen eigenen Tokenizer mit** — er muss
# exakt zum vortrainierten Encoder passen, sonst zeigen die IDs ins Leere.
#
# Wir schauen uns die Zerlegung an einem echten Beispiel an — `##`-Präfixe markieren
# Wortfortsetzungen, `[CLS]`/`[SEP]` sind Steuertoken, die BERT-Modelle umrahmen.

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

beispiel = "What can I do if my card still hasn't arrived after 2 weeks?"
enc = tokenizer(beispiel)
print("Text  :", beispiel)
print("Token :", tokenizer.convert_ids_to_tokens(enc["input_ids"]))
print("IDs   :", enc["input_ids"])

# %% [markdown]
# ## Schritt 3 — `max_length` bewusst wählen
#
# Jeder Text wird auf eine feste Länge gebracht: kürzere werden mit `[PAD]`
# aufgefüllt, längere **abgeschnitten** (`truncation`). Zu kurz → wir werfen Inhalt
# weg; zu lang → verschenkter Speicher/Rechenzeit (Aufwand wächst mit der Länge).
# Also nicht raten — **die Verteilung der Token-Längen anschauen** und danach wählen
# (derselbe Reflex wie die F1-EDA: die Messung entscheidet). banking77 sind kurze
# Kundenanfragen, wir erwarten wenig.

# %%
import numpy as np

# Länge in Token je Trainingstext (ohne Padding/Truncation, nur zählen).
laengen = np.array([len(tokenizer(t)["input_ids"]) for t in train_texts])
for p in (50, 95, 99, 100):
    print(f"  {p:>3}. Perzentil: {int(np.percentile(laengen, p)):>3} Token")

# p99 großzügig aufgerundet → deckt ~alle Texte ab, ohne Speicher zu verschwenden.
MAX_LEN = int(np.ceil(np.percentile(laengen, 99) / 8) * 8)
abgeschnitten = (laengen > MAX_LEN).mean() * 100
print(f"\n→ MAX_LEN = {MAX_LEN}   (schneidet nur {abgeschnitten:.2f} % der Texte an)")

# %% [markdown]
# ## Schritt 4 — Der Klassifikations-Kopf kommt drauf
#
# `AutoModelForSequenceClassification` lädt den **vortrainierten** DistilBERT-Körper
# (sein Sprachwissen aus dem Pretraining) und setzt obendrauf einen **frischen,
# zufällig initialisierten** Klassifikations-Kopf mit 77 Ausgängen. `num_labels`
# sagt, wie viele Klassen; `id2label`/`label2id` reisen im Modell mit, damit
# Vorhersagen lesbar bleiben.
#
# **Erwartete Warnung:** transformers meldet, dass die `classifier`-Gewichte neu und
# untrainiert sind — genau so soll es sein. In P2 lernt der Kopf von Grund auf, und
# der Körper wird **mit** angepasst (das „Auftauen"). Ein Vorhersage-Versuch *jetzt*
# wäre reines Raten.

# %%
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModell geladen: {n_params / 1e6:.1f} Mio Parameter, {NUM_LABELS} Ausgänge.")

# %% [markdown]
# ## Deuten & nächster Schritt
#
# Das Rohmaterial für P2 steht: Texte lassen sich in Subword-Token übersetzen
# (`tokenizer`), `MAX_LEN` ist begründet gewählt, Labels sind IDs mit lesbarem
# Rückweg, und der vortrainierte Encoder trägt einen frischen 77-Klassen-Kopf.
#
# **P2 „Bauen"** verdrahtet das zum Training: die Splits als `datasets`-Objekte
# tokenisieren, dynamisches Padding (`DataCollatorWithPadding`), und der `Trainer`
# fährt die eigentliche Optimierung auf MPS — Encoder **und** Kopf zusammen. Danach
# messen wir **test genau einmal**. Rechne mit Minuten statt Sekunden: das ist der
# Preis des Auftauens (und Stoff für die P4-Einordnung „war der Aufwand es wert?").
#
# **✓ Checkpoint:** Warum darf der Test-Split die `label2id`-Abbildung *nutzen*,
# aber nicht *mitbestimmen*? Und warum braucht Track C eine GPU/MPS, Track A und B
# (fast) nicht?

# %% [markdown]
# # P2 — „Bauen"
#
# Jetzt verdrahten wir das Vorbereitete zum **ersten Training**. Regel dieser
# Phase: **Standardwerte, kein Tuning** — ein ehrlicher erster Lauf, kein Feilen.
# (Das Feilen ist P3.) Trotzdem oft schon ein Bestwert-Kandidat, weil ein
# aufgetauter Transformer viel Kapazität mitbringt — nur eben „um Größenordnungen
# teurer" als A/B: Sekunden werden Minuten.
#
# Drei neue Teile gegenüber A/B:
# 1. **`datasets`-Objekte** statt Listen — der `Trainer` erwartet dieses Format.
# 2. **Dynamisches Padding** (`DataCollatorWithPadding`): jeder Batch wird nur auf
#    die Länge seines *längsten* Textes gepolstert, nicht global auf `MAX_LEN` —
#    spart Rechenzeit.
# 3. Der **`Trainer`**: fährt die Trainingsschleife (Vorwärts → Loss → Gradient →
#    Gewichte anpassen), automatisch auf MPS. Encoder **und** Kopf lernen zusammen.

# %% [markdown]
# ## Schritt 1 — Splits als tokenisierte `datasets`
#
# Aus `(texts, labels)` wird je Split ein `Dataset`, dann tokenisiert (mit
# `truncation` auf `MAX_LEN`). Die Label-Spalte muss **`labels`** heißen — unter
# diesem Namen reicht der `Trainer` sie ans Modell weiter. Die Rohtext-Spalte
# entfernen wir; das Modell rechnet nur mit Token-IDs.

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
collator = DataCollatorWithPadding(tokenizer)  # polstert je Batch dynamisch
print(train_ds)

# %% [markdown]
# ## Schritt 2 — Trainings-Einstellungen (alle auf Standard)
#
# `TrainingArguments` ist das Kontrollpult. Bewusst **nichts getunt**:
# - `learning_rate=5e-5` ist der HF-Standard — die *empfindlichste* Schraube, aber
#   erst P3-Stoff.
# - `num_train_epochs=3`: dreimal durch die Trainingsdaten.
# - `eval_strategy="no"`: **wir messen während des Trainings nicht** — das Testset
#   bleibt zu, bis wir es am Ende genau einmal anfassen. Dieselbe Disziplin wie A/B.
# - `save_strategy="no"`: keine Zwischen-Checkpoints auf Platte (Lernlauf).
# - `report_to="none"`: kein externes Logging (wandb & Co.).
#
# MPS wird automatisch genutzt (kein CPU-Flag gesetzt).

# %%
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=str(root / "04_track_c_finetuning/runs"),
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-5,  # HF-Default — bewusst nicht getunt
    eval_strategy="no",  # test bleibt während des Trainings zu
    save_strategy="no",
    logging_steps=50,  # alle 50 Schritte den Loss zeigen
    report_to="none",
    dataloader_pin_memory=False,  # pin_memory wird auf MPS nicht unterstützt
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator,
    processing_class=tokenizer,  # 5.x: früher hieß das `tokenizer=`
)

# %% [markdown]
# ## Schritt 3 — Training: das Auftauen
#
# `trainer.train()` fährt die Schleife. Anders als in A/B werden **alle 67 Mio
# Gewichte** angepasst, nicht nur ein kleiner Kopf. Der `loss` (Trainingsfehler)
# sollte fallen — sinkt er nicht, stimmt meist die Learning Rate (kommt in P3).
# Auf MPS dauert das **einige Minuten** — der Moment, in dem Finetuning „teuer" wird.

# %%
trainer.train()

# %% [markdown]
# ## Schritt 4 — Finale Messung: Testset, genau EINMAL
#
# Erst jetzt fassen wir test an. `trainer.predict` liefert für jede Anfrage 77
# Roh-Scores (Logits); der höchste ist die Vorhersage. Gemessen mit **demselben
# Besteck wie A/B** (Accuracy + Macro-F1), damit die Zahlen vergleichbar bleiben.

# %%
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_utils import save_result

pred = trainer.predict(test_ds)
preds = np.argmax(pred.predictions, axis=-1)

acc = accuracy_score(y_test, preds)
macro_f1 = f1_score(y_test, preds, average="macro")
print(f"DistilBERT finetuned (plain)   Accuracy: {acc * 100:.2f} %   Macro-F1: {macro_f1 * 100:.2f} %")
print(f"Latte:  B mpnet getunt 94,12 %  ·  A LogReg getunt 90,25 %")

save_result(
    "C_plain_distilbert",
    acc,
    macro_f1=round(macro_f1, 4),
    model="DistilBERT (finetuned)",
    note="P2 plain, Standardwerte, test 1x",
)

# %% [markdown]
# ## Wo irrt das Modell? — Fehlerbilder
#
# Dasselbe Fehler-Besteck wie in Track A/B, damit die Bilder vergleichbar sind. Das
# Modell sagt Integer-IDs vorher — wir übersetzen sie über `id2label` zurück in
# lesbare Intent-Namen, sonst stünden in der Confusion-Matrix nur Zahlen.

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

pred_names = [id2label[int(p)] for p in preds]

plot_confusion_matrix(test_labels, pred_names, title="DistilBERT plain — Confusion Matrix")
plot_top_confusions(test_labels, pred_names, top=15)
plot_per_class_f1(test_labels, pred_names, worst=20)

# %% [markdown]
# ## Deuten & nächster Schritt
#
# Zum ersten Mal hat sich der Encoder **selbst** bewegt — kein fixes Lineal mehr.
# Schlägt der plain-Lauf schon B (94,12 %)? Wenn ja: das ist die Kraft des
# Auftauens, *ohne* eine einzige getunte Schraube. Wenn nein: Finetuning ist
# empfindlich, und genau da setzt P3 an.
#
# **P3 „Optimieren"** dreht dann die Stellschrauben aus dem Curriculum — allen voran
# die **Learning Rate + Warmup** (die empfindlichste), dazu **Early Stopping am
# Val-Set** und **LoRA** als Sparmodus. Dafür brauchen wir wieder einen Val-Split
# (wie F2), auf dem wir vergleichen — das Testset bleibt bis zur finalen P3-Messung
# unberührt.
#
# **✓ Checkpoint:** Wir haben test hier genau *einmal* gemessen und *nicht* während
# des Trainings evaluiert. Was wäre der ehrliche Fehler gewesen, hätten wir
# `eval_strategy="epoch"` aufs Testset laufen lassen und die beste Epoche behalten?

