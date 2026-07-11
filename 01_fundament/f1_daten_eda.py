# %% [markdown]
# # F1 — Daten & EDA (banking77)
#
# **Frage:** Was steckt in den Daten — *bevor* irgendein Modell sie sieht?
#
# EDA (Explorative Datenanalyse) heißt: Zeit mit den **Daten** verbringen, nicht
# mit Modellen. Wir arbeiten das ROADMAP-Programm ab und schauen banking77 von
# allen Seiten an: Verteilung, Längen, **welche Begriffe vorkommen**, Duplikate &
# Leakage, Sonderzeichen, Überlappung der Klassen, echte Beispiele lesen. Ganz am
# Ende legen wir die **Latte** (Majority-Baseline) — die Zahl, die jedes echte
# Modell überspringen muss.
#
# Datensatz: **banking77** — echte englische Bank-Kundenanfragen, 77 Intents,
# kurze Sätze. Das ist **Intent-Klassifikation** (Customer Service), ein anderes
# Regime als lange Nachrichtenartikel.
#
# Bedienung in Zed: Cursor in eine Zelle, **Cmd+Enter** führt sie aus. Führe die
# Setup-Zelle zuerst aus — alle anderen Zellen bauen auf ihren Variablen auf.

# %% [markdown]
# ## Setup
#
# `data_utils.py` liegt im Projekt-Root, dieses Notebook in `01_fundament/`.
# Die Zelle findet den Root (dort liegt `data_utils.py`) — egal von wo Zed den
# Kernel startet — und macht ihn importierbar.

# %%
import sys
from pathlib import Path

root = Path.cwd()
while not (root / "data_utils.py").exists() and root != root.parent:
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from data_utils import load_banking77

train_texts, train_labels = load_banking77("train")
test_texts, test_labels = load_banking77("test")

print(f"Projekt-Root: {root}")
print(f"train: {len(train_texts)} Anfragen   test: {len(test_texts)} Anfragen")
print(f"Intents: {len(set(train_labels))}\n")

# Ein paar echte Zeilen ansehen
peek = pd.DataFrame({"text": train_texts[:6], "intent": train_labels[:6]})
print(peek.to_string(index=False))

# %% [markdown]
# ## 1. Klassenverteilung — train vs. test
#
# Erste Frage jeder EDA: Wie viele Klassen, wie groß, **wie ungleich** — und ist
# der Split *stratifiziert* (dieselbe Verteilung in train und test)?

# %%
dist = pd.DataFrame({
    "train": pd.Series(train_labels).value_counts(),
    "test":  pd.Series(test_labels).value_counts(),
}).fillna(0).astype(int)
dist["train_%"] = (dist["train"] / dist["train"].sum() * 100).round(2)

print(f"Intents: {dist.shape[0]}")
print("\n--- 5 größte Intents (train) ---")
print(dist.sort_values("train", ascending=False).head(5))
print("\n--- 5 kleinste Intents (train) ---")
print(dist.sort_values("train", ascending=True).head(5))

print(f"\ntrain je Intent:  min {dist['train'].min()}  "
      f"median {int(dist['train'].median())}  max {dist['train'].max()}")
print(f"test  je Intent:  min {dist['test'].min()}  "
      f"median {int(dist['test'].median())}  max {dist['test'].max()}")

# %%
# Verteilung als Balken (nach train-Größe sortiert)
ax = dist.sort_values("train")[["train", "test"]].plot.barh(
    figsize=(8, 14), width=0.85)
ax.set_title("banking77 — Beispiele je Intent (train vs. test)")
ax.set_xlabel("Anzahl Beispiele")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Deuten
# - **train ist unbalanciert** (min ~35, max ~187), **test ist balanciert**
#   (~40 pro Intent). Der Split ist also *nicht* stratifiziert im üblichen Sinn —
#   test wurde bewusst gleichverteilt gebaut.
# - Folge für die **Majority-Baseline**: Sie tippt immer den häufigsten
#   train-Intent. Weil test gleichverteilt ist, trifft das nur ~1/77 ≈ **1,3 %**.
#   Die Latte liegt also sehr niedrig — 77 Klassen sind schwer.
# - Weil test balanciert ist, sind **Accuracy** und **Macro-F1** hier näher
#   beieinander als bei GNAD. Trotzdem messen wir später beides (F2).

# %% [markdown]
# ## 2. Textlängen
#
# Wie lang sind die Anfragen — in Wörtern und Zeichen? Das entscheidet mit,
# welche Modelle passen: Bei GNAD waren 97 % der Artikel länger als 128 Tokens
# (Track-B-Problem). Hier erwarten wir das Gegenteil — kurze Sätze.

# %%
lengths = pd.DataFrame({
    "intent": train_labels,
    "n_words": [len(t.split()) for t in train_texts],
    "n_chars": [len(t) for t in train_texts],
})
print("Wörter je Anfrage:", lengths["n_words"].describe()[["min", "50%", "max"]].to_dict())
print("Zeichen je Anfrage:", lengths["n_chars"].describe()[["min", "50%", "max"]].to_dict())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
lengths["n_words"].plot.hist(bins=40, ax=axes[0], title="Länge in Wörtern")
axes[0].set_xlabel("Wörter")
lengths["n_chars"].plot.hist(bins=40, ax=axes[1], title="Länge in Zeichen")
axes[1].set_xlabel("Zeichen")
plt.tight_layout()
plt.show()

# %%
# Gibt es systematisch längere/kürzere Intents?
by_intent = lengths.groupby("intent")["n_words"].median().sort_values()
print("--- kürzeste Intents (Median Wörter) ---")
print(by_intent.head(5))
print("\n--- längste Intents (Median Wörter) ---")
print(by_intent.tail(5))

# %% [markdown]
# ### Deuten
# - **Kurz-Text-Regime**: wenige Wörter pro Anfrage. Das dreht die Karten
#   gegenüber GNAD — Embeddings (Track B) haben kein Kontextfenster-Problem, und
#   TF-IDF (Track A) hat pro Beispiel *weniger* Signal.
# - Ausreißer nach oben (max Zeichen) lohnt sich anzuschauen — Copy-Paste,
#   mehrere Fragen in einer? (In Abschnitt 6 prüfen wir Artefakte.)

# %% [markdown]
# ## 3. Vokabular — welche Begriffe kommen vor?
#
# Jetzt der Blick, den du wolltest: *was steht da eigentlich drin?* Wir
# tokenisieren simpel (Kleinschreibung, an Nicht-Buchstaben trennen) und zählen.
# Einmal roh (dann dominieren Füllwörter) und einmal ohne **Stoppwörter**
# (die inhaltsleeren „the/a/to/my"), damit die *inhaltlichen* Begriffe auftauchen.

# %%
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def tokenize(text):
    """Kleinschreibung, Wörter aus Buchstaben (Zahlen/Satzzeichen fallen weg)."""
    return re.findall(r"[a-z]+", text.lower())

all_tokens = [tok for t in train_texts for tok in tokenize(t)]
vocab = Counter(all_tokens)
content = Counter(tok for tok in all_tokens if tok not in ENGLISH_STOP_WORDS)

print(f"Tokens gesamt: {len(all_tokens):,}")
print(f"Vokabulargröße (verschiedene Wörter): {len(vocab):,}")
hapax = sum(1 for w, c in vocab.items() if c == 1)
print(f"Einmal-Wörter (Hapax): {hapax:,}  ({hapax/len(vocab)*100:.0f} % des Vokabulars)")

print("\n--- Top 15 ROH (mit Füllwörtern) ---")
for w, c in vocab.most_common(15):
    print(f"  {c:>5}  {w}")
print("\n--- Top 15 INHALTLICH (ohne Stoppwörter) ---")
for w, c in content.most_common(15):
    print(f"  {c:>5}  {w}")

# %% [markdown]
# ### Deuten
# - Die inhaltlichen Top-Wörter (card, money, transfer, payment …) verraten die
#   **Domäne**: Karten, Überweisungen, Konto. Genau das Signal, das TF-IDF und
#   Embeddings später aufgreifen.
# - Viele **Hapax-Wörter** (nur 1×) sind normal — Namen, Tippfehler, Seltenes.
#   Sie werden später oft weggeschnitten (`min_df` in Track A).

# %% [markdown]
# ## 4. Charakteristische Begriffe pro Intent
#
# Interessanter als „häufig insgesamt" ist: *welche Wörter kennzeichnen einen
# Intent?* Wir nehmen die häufigsten inhaltlichen Wörter je Intent — eine erste,
# ehrliche Vorschau auf das, was Track A (TF-IDF) formalisiert.

# %%
texts_by_intent = {}
for t, l in zip(train_texts, train_labels):
    texts_by_intent.setdefault(l, []).append(t)

def top_words(texts, k=8):
    c = Counter(tok for t in texts for tok in tokenize(t)
                if tok not in ENGLISH_STOP_WORDS)
    return [w for w, _ in c.most_common(k)]

for intent in ["card_arrival", "atm_support", "lost_or_stolen_card",
               "exchange_rate", "top_up_by_bank_transfer_charge"]:
    print(f"{intent:<32} {', '.join(top_words(texts_by_intent[intent]))}")

# %% [markdown]
# ### Deuten
# Manche Intents haben ein glasklares Leitwort (`atm` → atm_support). Andere
# teilen sich Wörter (`card` taucht überall auf) — das ist der Keim späterer
# **Verwechslungen**. Genau das quantifizieren wir in Abschnitt 7.

# %% [markdown]
# ## 5. Duplikate & Leakage
#
# **Der wichtigste Hygiene-Check.** Zwei Fragen:
# 1. Gibt es exakte Duplikate *innerhalb* von train? (Redundanz)
# 2. Tauchen train-Anfragen **wortgleich im test** auf? Dann hat das Modell die
#    Antwort schon „gesehen" → **Leakage**, die Testzahl wäre geschönt.

# %%
train_norm = [t.strip().lower() for t in train_texts]
test_norm = [t.strip().lower() for t in test_texts]

dups_train = sum(c - 1 for c in Counter(train_norm).values() if c > 1)
overlap = set(train_norm) & set(test_norm)

print(f"Exakte Duplikate innerhalb train: {dups_train} "
      f"(von {len(train_texts)} Anfragen)")
print(f"train-Anfragen, die wortgleich im test stehen (Leakage): {len(overlap)}")
if overlap:
    print("\nBeispiele für Overlap:")
    for ex in list(overlap)[:5]:
        print(f"  - {ex[:80]}")

# %% [markdown]
# ### Deuten
# - Duplikate innerhalb train sind meist harmlos (dieselbe Frage mehrfach gelabelt).
# - **Overlap train↔test ist das kritische Signal.** Ist er groß, muss man ihn
#   entfernen, sonst misst man teils Auswendiglernen. Ist er ~0, ist der Split
#   sauber. Die Zahl oben sagt, woran wir sind — merk sie dir.

# %% [markdown]
# ## 6. Sonderzeichen & Encoding-Artefakte
#
# Steckt Müll drin? Platzhalter (`{{...}}`), HTML, kaputtes Encoding, leere
# Anfragen, viele Zahlen? Solche Artefakte verzerren später Features.

# %%
import unicodedata

non_ascii = [t for t in train_texts if any(ord(c) > 127 for c in t)]
empty = [t for t in train_texts if not t.strip()]
with_digits = [t for t in train_texts if any(c.isdigit() for c in t)]
with_braces = [t for t in train_texts if "{" in t or "}" in t]

print(f"Anfragen mit Nicht-ASCII-Zeichen: {len(non_ascii)}")
print(f"Leere Anfragen:                   {len(empty)}")
print(f"Anfragen mit Ziffern:             {len(with_digits)}")
print(f"Anfragen mit {{ oder }}:            {len(with_braces)}")

if non_ascii:
    print("\nBeispiele mit Nicht-ASCII (welche Zeichen?):")
    for t in non_ascii[:5]:
        specials = {c: unicodedata.name(c, "?") for c in set(t) if ord(c) > 127}
        print(f"  {t[:70]}   -> {specials}")

# %% [markdown]
# ### Deuten
# Nicht-ASCII kann legitim sein (£, €, Akzente) oder Müll (kaputtes Encoding).
# Die Beispiele zeigen, was es ist. Ziffern sind bei Bank-Anfragen normal
# (Beträge, Kartennummern) — man entscheidet später, ob man sie behält.

# %% [markdown]
# ## 7. Vokabular-Überlappung zwischen Intents
#
# Welche Intents *reden über dasselbe*? Wir vergleichen die Top-Wörter jedes
# Intent-Paars (Jaccard: geteilte / gesamte Wörter). Hohe Überlappung = wo das
# Modell später am ehesten **verwechselt**. Datengetriebene Vorahnung.

# %%
top_sets = {intent: set(top_words(texts, k=15))
            for intent, texts in texts_by_intent.items()}
intents = list(top_sets)

pairs = []
for i in range(len(intents)):
    for j in range(i + 1, len(intents)):
        a, b = intents[i], intents[j]
        inter = top_sets[a] & top_sets[b]
        union = top_sets[a] | top_sets[b]
        if union:
            pairs.append((len(inter) / len(union), a, b, inter))

pairs.sort(reverse=True)
print("--- Top 10 überlappende Intent-Paare (Jaccard der Top-15-Wörter) ---")
for score, a, b, shared in pairs[:10]:
    print(f"  {score:.2f}  {a}  <->  {b}")
    print(f"        geteilt: {', '.join(sorted(shared))}")

# %% [markdown]
# ### Deuten
# Das sind die Kandidaten für die spätere Confusion-Matrix (F2). Wenn zwei
# Intents fast dieselben Wörter benutzen, braucht das Modell feinere Signale als
# nur Wort-Häufigkeit — ein Argument, das später *für* Embeddings/BERT spricht.

# %% [markdown]
# ## 8. Ein Dutzend Beispiele lesen
#
# Zum Schluss der EDA das Wichtigste, was keine Statistik ersetzt: **lesen.**
# Was würdest *du* als Mensch als Signal nutzen? Wo ist es mehrdeutig?

# %%
for intent in ["card_arrival", "lost_or_stolen_card", "exchange_rate"]:
    print(f"\n=== {intent} ===")
    for t in texts_by_intent[intent][:6]:
        print(f"  - {t}")

# %% [markdown]
# ### ✓ Checkpoint (aus der ROADMAP)
# Nenne drei Dinge, die dir nur die EDA sagen konnte — und welche
# Modell-Entscheidung jede davon beeinflusst. Und: Warum kann ein Spam-Filter
# mit 95 % Accuracy nutzlos sein?

# %% [markdown]
# ## 9. Die Latte — Majority-Baseline
#
# Der Abschluss von F1: die simpelste denkbare „Modell"-Zeile. `DummyClassifier`
# tippt immer die häufigste train-Klasse. Das ist keine echte Modellierung,
# sondern die **Referenz**, die jedes Modell schlagen muss.

# %%
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(train_texts, train_labels)          # „lernt" nur die häufigste Klasse
acc = dummy.score(test_texts, test_labels)
print(f"Majority-Baseline auf test: {acc*100:.2f} %")
print(f"(Zufall bei 77 gleichverteilten Klassen: {100/77:.2f} %)")

# %% [markdown]
# ## Fazit F1
#
# Was uns die EDA *vor* jedem Modell gesagt hat:
# - **77 Intents, kurze Texte** → Intent-Klassifikation, anderes Regime als GNAD.
# - **train unbalanciert, test balanciert** → Baseline extrem niedrig (~1,3 %),
#   Accuracy und Macro-F1 nah beieinander.
# - **Vokabular** ist domänen-typisch (card/transfer/payment); manche Intents
#   teilen sich Wörter → Verwechslungs-Kandidaten schon jetzt sichtbar.
# - **Duplikate/Leakage**: der Overlap-Check aus Abschnitt 5 sagt, ob der Split
#   sauber ist.
#
# Damit ist der Boden bereitet. Nächster Fundament-Schritt: **F2 Messwesen** —
# der Val-Split und das ehrliche Messbesteck, bevor wir in Track A das erste
# echte Modell bauen.
