# %% [markdown]
# # Track D · lokal — Llama-3.2-1B, zero-shot (banking77)
#
# ## P2 „Bauen" — der erste konkrete LLM-Lauf
#
# **Llama-3.2-1B** ist das kleinste Modell auf dem Server: 1 Milliarde Parameter,
# 4-bit quantisiert, ~0,2 s pro Anfrage. Ideal, um die **Pipeline** billig
# aufzubauen und die Mechanik zu lernen — die inhaltliche Stärke kommt später mit
# größeren Modellen.
#
# Die gemeinsame Vorarbeit — der Generationssprung Gen 3b, die Server-Mechanik, die
# Zeit-statt-Geld-Kostenachse und die Mess-Disziplin — steht in
# **`../vorbereiten.py`** und wird hier nicht wiederholt. Dieses Notebook ist der
# erste Bau: **Prompt formulieren → klassifizieren → ehrlich messen.**
#
# Die Latte: Track A getunt = ~90 %, Track B/C = ~94 %. Zero-shot heißt: das Modell
# hat banking77 **nie gesehen** — wir beschreiben ihm nur die Aufgabe. Wie weit
# trägt das?

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

from data_utils import load_banking77_split, save_result
from llm_client import get_client, classify, stratified_subsample

client = get_client()
MODEL = "Llama-3.2-1B-Instruct-4bit"

# %% [markdown]
# ## Schritt 1 — die Kategorien als Prompt-Material
#
# In Track A–C waren die 77 Labels bloße Zielwerte, die das Modell aus Daten lernte.
# Hier sind sie **Input**: das Modell erfährt die erlaubten Kategorien nur, weil wir
# sie ihm in den Prompt schreiben. „Data Prep ist Prompt-Design" — die Vorbereitung
# ist dieser Text.

# %%
tr_t, tr_l, val_t, val_l = load_banking77_split()
labels = sorted(set(tr_l))          # die 77 Intent-Namen
print(f"{len(labels)} Intents. Ausschnitt:")
print("  " + "\n  ".join(labels[:6]) + "\n  …")

# Iterier-Subsample: 2 Beispiele je Klasse aus val (fester Seed, reproduzierbar).
# Schnell genug für eine Lernschleife; die finale Zahl kommt später am vollen Test.
sub_t, sub_l = stratified_subsample(val_t, val_l, per_class=2)
print(f"\nIterier-Subsample: {len(sub_t)} Anfragen (2×{len(labels)}) aus val")

# %% [markdown]
# ## Schritt 2 — der zero-shot-Prompt
#
# Der naive erste Versuch: eine `system`-Anweisung mit der Aufgabe und der Liste der
# 77 erlaubten Labels, dann die Anfrage als `user`-Nachricht. Kein gelöstes
# Beispiel (deshalb *zero*-shot). Wir bitten explizit um **nur das Label** — ob das
# kleine Modell sich daran hält, ist die erste offene Frage.
#
# Der Prompt lebt bewusst *hier* im Notebook (nicht im Helfer): er ist die
# Stellschraube, die wir in P3 tunen — er soll sichtbar sein und sich verändern.

# %%
label_block = "\n".join(labels)
SYSTEM = (
    "You are an intent classifier for a bank's customer support.\n"
    "Classify the customer's message into exactly ONE of the intent labels below.\n"
    "Respond with ONLY the exact label, nothing else — no explanation, no punctuation.\n\n"
    "Valid labels:\n" + label_block
)

def make_messages(text):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]

# Ein Blick auf den fertigen Prompt für die erste Anfrage:
print(make_messages(sub_t[0])[0]["content"][:300], "…\n")
print("user:", make_messages(sub_t[0])[1]["content"])

# %% [markdown]
# ## Schritt 3 — klassifizieren (mit Cache)
#
# `classify` ruft für jede Anfrage den Server, cached die Antwort auf Platte und
# parst sie zu einem gültigen Label (oder `"unknown"`, wenn nichts passt — wir
# schummeln kein Label herbei). Der erste Lauf rechnet, jeder weitere ist sofort da.

# %%
preds, raws = classify(client, MODEL, sub_t, make_messages, labels,
                       cache_tag="llama1b_p2_zeroshot", max_tokens=30)

# %% [markdown]
# ## Schritt 4 — ehrlich messen
#
# **Accuracy** = Anteil exakt richtiger Labels. **Macro-F1** = Mittel der
# F1-Werte je Klasse (Precision und Recall pro Intent, dann ungewichtet gemittelt) —
# behandelt alle 77 Intents gleich, egal wie häufig. Dazu zählen wir, wie oft das
# Modell gar kein verwertbares Label lieferte (`unknown`).

# %%
acc = accuracy_score(sub_l, preds)
mf1 = f1_score(sub_l, preds, average="macro", zero_division=0)
n_unknown = preds.count("unknown")

print(f"Accuracy:  {acc:.3f}")
print(f"Macro-F1:  {mf1:.3f}")
print(f"unknown:   {n_unknown}/{len(preds)}  ({n_unknown / len(preds):.0%} ohne verwertbares Label)")
print(f"Baseline (Majority): ~0,013   |   Track A: ~0,90   Track B/C: ~0,94")

# %% [markdown]
# ## Schritt 5 — hinsehen, *warum* es scheitert
#
# Die Zahl ist ernüchternd — weit unter Track A/B/C. Das ist kein Bug, sondern die
# ehrliche Lektion von Track D: „beschreib die Aufgabe einem LLM" ist keine Magie,
# und ein **kleines** lokales Modell ist im Nachteil. Sehen wir uns die Rohantworten
# an — oft liegt das Problem nicht im *Verstehen*, sondern im *Format*:

# %%
import itertools

print("Fehlklassifikationen (Anfrage → roh → geparst | gold):\n")
misses = [(t, r, p, g) for t, r, p, g in zip(sub_t, raws, preds, sub_l) if p != g]
for t, r, p, g in itertools.islice(misses, 8):
    print(f"  {t[:50]!r}")
    print(f"     roh   = {r!r}")
    print(f"     pred  = {p}   gold = {g}\n")

# %% [markdown]
# ## ✓ Checkpoint P2
#
# Was wir gelernt haben:
#
# - **Die Pipeline steht:** lokaler Server → `openai`-SDK → Prompt → Label → Messung,
#   inklusive Platten-Cache. Alles Weitere baut darauf auf.
# - **Zero-shot mit einem 1B-Modell reicht nicht.** Die Accuracy liegt weit unter
#   den trainierten Tracks. Zwei Ursachen mischen sich: das kleine Modell *versteht*
#   manche Anfragen schlicht falsch, und es hält sich oft nicht ans *Format*
#   (Fettung, Zusatztext, Wortsalat) → viele `unknown`.
# - Damit ist klar, **was P3 angehen muss**: das Format erzwingen (z.B. Ausgabe
#   strenger constrainen / guided decoding), dem Modell **few-shot-Beispiele** geben,
#   und stärkere Modelle als Vergleich heranziehen — jede Änderung wieder auf val
#   gemessen, damit wir sehen, ob sie wirklich hilft.
#
# Wir speichern diese naive P2-Zahl als Track-D-Startpunkt. (Gemessen auf dem
# Val-Subsample — die *finale* test-Zahl kommt in P3 nach dem Tunen.)

# %%
save_result(
    "D_llama1b_zeroshot_p2",
    acc,
    macro_f1=round(mf1, 4),
    model="Llama-3.2-1B-Instruct-4bit (lokal, oMLX)",
    note=f"P2 naiv zero-shot, val-Subsample 2x77, unknown={n_unknown}",
    score_type="none",
)

# %% [markdown]
# ## P3 „Optimieren" — folgt
#
# Nächste Schritte in *dieser* Datei (Konvention: eine Datei wächst von *pure* zu
# *getunt*):
#
# 1. **Format erzwingen** — die Ausgabe auf genau ein gültiges Label zwingen
#    (Prompt-Härtung, ggf. oMLX guided decoding). Ziel: `unknown` → 0.
# 2. **Few-shot** — ein paar gelöste Beispiele in den Prompt (In-Context-Learning).
# 3. **Vergleich** — dasselbe mit einem stärkeren Modell (Nemotron 30B /
#    Qwen-Opus-Distilled), um die Kosten-/Nutzenkurve *Zeit ↔ Qualität* zu sehen.
# 4. **Finale Messung** — beste Prompt-Variante einmal auf dem **vollen** Testsatz.
