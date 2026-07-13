# %% [markdown]
# # Track D · lokal — Llama-3.2-1B, few-shot (banking77)
#
# ## P3 „Optimieren" — der erste echte Kontext-Schritt
#
# Der Boden aus `llama1b_zero.py`: nacktes zero-shot holt auf diesem 1B-Modell nur
# ~9 %, und ein Viertel der Antworten ist nicht mal ein parsbares Label. Das Modell
# kennt die 77 Kategorie-*Namen*, aber nicht ihre *Bedeutung*.
#
# **Few-shot** ändert das: wir legen dem Prompt **gelöste Beispiele** bei — je
# Kategorie zwei echte Anfragen aus den Trainingsdaten mit ihrem richtigen Label.
# Das Modell liest sie im Kontext und leitet daraus ab, *was* die Kategorien
# bedeuten und *in welchem Format* es antworten soll. Das nennt man
# **In-Context-Learning**: das Modell „lernt" für die Dauer *dieses einen Aufrufs*
# aus dem Prompt — **kein Gewicht ändert sich** (das kommt erst beim Finetuning).
#
# Gemessen wird — wie es die Disziplin verlangt — auf **val**: Prompt-Tuning ist
# Optimieren, und der Testsatz bleibt für die *finale* Zahl reserviert. Wir
# vergleichen direkt gegen zero-shot auf demselben Val-Subsample.

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
for d in (root, root / "05_track_d_llm"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_banking77_split
from llm_client import get_client, classify, stratified_subsample

client = get_client()
MODEL = "Llama-3.2-1B-Instruct-4bit"

# %% [markdown]
# ## Schritt 1 — Daten, Labels, Splits
#
# `load_banking77_split()` trennt den Trainingsteil stratifiziert in *train* und
# *val*. Der **Testsatz bleibt unangefasst**. Aus *train* holen wir gleich die
# Demo-Beispiele, auf *val* messen wir.

# %%
tr_t, tr_l, val_t, val_l = load_banking77_split()
labels = sorted(set(tr_l))            # die 77 Intent-Namen
print(f"{len(labels)} Intents · train: {len(tr_t)} · val: {len(val_t)}")

# %% [markdown]
# ## Schritt 2 — die Demos wählen: 2 Beispiele je Kategorie aus *train*
#
# `stratified_subsample` zieht mit festem Seed **2 Beispiele je Klasse** → 154
# gelöste (Anfrage, Label)-Paare. Wichtig:
#
# - **aus `train`**, nicht aus val/test → kein Leakage (wir zeigen dem Modell keine
#   Beispiele, auf denen wir es später messen).
# - **2 je Klasse**, damit *jeder* der 77 Intents im Prompt vertreten ist. Das ist
#   lang (154 Beispiele), aber es ist ein **stabiler Prefix**, den der Server cached
#   — jede Test-Anfrage teilt sich denselben Demo-Block.

# %%
demo_t, demo_l = stratified_subsample(tr_t, tr_l, per_class=2, seed=42)
print(f"Demos: {len(demo_t)} (= 2 × {len(labels)})")
print("Beispiel-Demo:", repr(demo_t[0]), "→", demo_l[0])

# %% [markdown]
# ## Schritt 3 — den few-shot-Prompt bauen
#
# Die klassische few-shot-Form im Chat: die Demos werden als **abwechselnde
# Gesprächsturns** vorgelegt (`user` = Anfrage, `assistant` = das richtige Label),
# *bevor* die echte Anfrage kommt. Das Modell sieht also 154 gelöste Runden und
# dann die neue Frage — und antwortet im gelernten Muster.
#
# Zum Vergleich bauen wir auch den **zero-shot**-Prompt von `llama1b_zero.py` nach
# (nur System-Anweisung + Label-Liste, keine Beispiele), damit wir den Effekt der
# Demos *isoliert* sehen.

# %%
label_block = "\n".join(labels)
SYSTEM = (
    "You are an intent classifier for a bank's customer support.\n"
    "Classify the customer's message into exactly ONE of the intent labels below.\n"
    "Respond with ONLY the exact label, nothing else.\n\n"
    "Valid labels:\n" + label_block
)

# zero-shot: System + Anfrage
def make_messages_zero(text):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}]

# few-shot: System + 154 Demo-Turns + Anfrage
demo_turns = []
for dt, dl in zip(demo_t, demo_l):
    demo_turns.append({"role": "user", "content": dt})
    demo_turns.append({"role": "assistant", "content": dl})

def make_messages_few(text):
    return [{"role": "system", "content": SYSTEM}] + demo_turns + [{"role": "user", "content": text}]

# Blick auf den Anfang des few-shot-Prompts (erste zwei Demo-Runden):
for m in make_messages_few("PLATZHALTER")[1:5]:
    print(f"  [{m['role']}] {m['content'][:60]}")
print(f"  … (insgesamt {len(demo_turns)} Demo-Turns) …")
print(f"  [user] PLATZHALTER  ← hier steht die echte Anfrage")

# %% [markdown]
# ## Schritt 4 — messen: zero-shot vs. few-shot auf val
#
# Wir klassifizieren dasselbe **Val-Subsample** (2 je Klasse = 154 Anfragen) einmal
# zero-shot, einmal few-shot, und stellen die Kennzahlen nebeneinander. `classify`
# cached die Antworten — der erste Lauf rechnet (~1 Min.), jeder weitere ist sofort da.
#
# Kennzahlen: **Accuracy** (Anteil exakt richtig), **Macro-F1** (F1 je Klasse,
# ungewichtet gemittelt) und **unknown** (Antworten ohne verwertbares Label).

# %%
ev_t, ev_l = stratified_subsample(val_t, val_l, per_class=2, seed=42)   # 154 val, kein Overlap mit demos (train)

def bewerten(name, make_messages, cache_tag):
    preds, raws = classify(client, MODEL, ev_t, make_messages, labels,
                           cache_tag=cache_tag, max_tokens=30, progress=False)
    acc = accuracy_score(ev_l, preds)
    mf1 = f1_score(ev_l, preds, average="macro", zero_division=0)
    unk = preds.count("unknown")
    print(f"{name:14s}  Accuracy={acc:.3f}  Macro-F1={mf1:.3f}  unknown={unk:>3}/{len(preds)}")
    return preds, raws

print(f"Messe auf {len(ev_t)} Val-Anfragen …\n")
preds_zero, _ = bewerten("zero-shot", make_messages_zero, "llama1b_zero_val")
preds_few, raws_few = bewerten("few-shot 2/Kl", make_messages_few, "llama1b_fewshot_val")

# %% [markdown]
# ## Schritt 5 — die Fehler *sehen*: Confusion-Matrix (few-shot)
#
# Dieselben Werkzeuge wie in Track A/B/C — jetzt auf die few-shot-Vorhersagen:
# Confusion-Matrix (Diagonale = richtig), häufigste Verwechslungen, und F1 je Intent
# (schwächste zuerst). So siehst du, *welche* Kategorien das kleine Modell auch mit
# Beispielen noch durcheinanderwirft.

# %%
from eval_utils import plot_confusion_matrix, plot_per_class_f1, plot_top_confusions

plot_confusion_matrix(ev_l, preds_few, title="Llama-1B few-shot (2/Kl) — Confusion Matrix (val)")
plot_top_confusions(ev_l, preds_few, top=15)
plot_per_class_f1(ev_l, preds_few, worst=20)

# %% [markdown]
# ## Schritt 6 — deuten und die Brücke zum Finetuning
#
# Was die Zahlen (deine, aus dem Lauf oben) erzählen sollten:
#
# - **Few-shot hebt die Accuracy deutlich** gegenüber zero-shot — auf demselben
#   winzigen 1B-Modell, nur durch Beispiele im Kontext.
# - **`unknown` bricht ein**: die Demos zeigen das exakte Format, also hört das
#   Modell auf, drumherum zu reden. Ein großer Teil des zero-shot-Elends war gar
#   nicht „Verstehen", sondern „Format".
# - **Und trotzdem: weit unter Track A/B/C (~94 %).** Ein 1B nutzt 154 Beispiele im
#   Kontext nur begrenzt — In-Context-Learning ersetzt kein Training.
#
# Genau hier setzt dein nächster Schritt an: **statt die Beispiele jedes Mal in den
# Prompt zu legen, trainieren wir sie dem kleinen Modell ein** (LoRA-Finetuning).
# Das ist die Pointe, die du selbst formuliert hast — „es muss ja nichts können
# außer Bedeutung erkennen". Derselbe Gedanke steckt hinter Track C (dort ein kleiner
# *Encoder*, RoBERTa, auf 94 % finegetunt). Der Bogen wird rund:
#
# **dasselbe Llama-1B: zero-shot (~9 %) → few-shot (?) → finegetunt (?).**
#
# ✓ Checkpoint P3-few-shot: In-Context-Learning ist der erste echte Kontext-Hebel;
# er heilt vor allem das Format und hebt die Bedeutung an. Nächste Datei:
# LoRA-Finetuning des kleinen Modells.
