# %% [markdown]
# # F1 — Daten & EDA
#
# **Frage:** Was steckt in den Daten — *bevor* irgendein Modell sie sieht?
#
# EDA (Explorative Datenanalyse) ist die erste eigene Station: Zeit mit den
# *Daten* verbringen, nicht mit Modellen. Wir arbeiten das ROADMAP-Programm
# Schritt für Schritt ab. Heute: Setup + **Klassenverteilung**.
#
# Bedienung in Zed: Cursor in eine Zelle, **Cmd+Enter** führt sie aus.

# %% [markdown]
# ## Setup
#
# Das Notebook liegt in `01_fundament/`, `gnad_utils.py` aber im Projekt-Root.
# Diese Zelle findet den Root (dort liegt `gnad_utils.py`) — egal, von wo Zed
# den Kernel startet — und macht ihn importierbar.

# %%
import sys
from pathlib import Path

root = Path.cwd()
while not (root / "gnad_utils.py").exists() and root != root.parent:
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
from gnad_utils import load_gnad

train_texts, train_labels = load_gnad("train")
test_texts, test_labels = load_gnad("test")

print(f"Projekt-Root: {root}")
print(f"train: {len(train_texts)} Artikel   test: {len(test_texts)} Artikel")
print(f"Beispiel-Label: {train_labels[0]!r}")
print(f"Beispiel-Text:  {train_texts[0][:120]}...")

# %% [markdown]
# ## Klassenverteilung — train vs. test
#
# Erste Frage jeder EDA: Wie viele Klassen, wie groß, und **wie ungleich**?
# Und: Ist der Split *stratifiziert* — also dieselbe Verteilung in train und
# test? Wenn nicht, misst man später Äpfel gegen Birnen.
#
# Wir stellen absolute Zahlen und Anteile (%) nebeneinander.

# %%
dist = pd.DataFrame({
    "train": pd.Series(train_labels).value_counts(),
    "test":  pd.Series(test_labels).value_counts(),
})
dist["train_%"] = (dist["train"] / dist["train"].sum() * 100).round(1)
dist["test_%"]  = (dist["test"]  / dist["test"].sum()  * 100).round(1)
dist = dist.sort_values("train", ascending=False)
print(dist)

print(f"\nKlassen: {dist.shape[0]}")
print(f"Größte : {dist.index[0]}  ({dist['train_%'].iloc[0]} %)")
print(f"Kleinste: {dist.index[-1]}  ({dist['train_%'].iloc[-1]} %)")
print(f"Verhältnis größte/kleinste: {dist['train'].iloc[0] / dist['train'].iloc[-1]:.1f}x")

# %% [markdown]
# ### Deuten (zusammen)
#
# Schau dir die `train_%`- und `test_%`-Spalten an:
# - Stimmen train- und test-Anteile je Klasse überein → Split ist stratifiziert.
# - Wie stark ist die **Unwucht** (Verhältnis größte/kleinste Klasse)?
#
# Das ist der Grund, warum die **Majority-Baseline** = größte Klasse (%) ist —
# und warum *Accuracy allein* bei Unwucht in die Irre führt (kommt in F2).
#
# **Nächste EDA-Schritte** (bauen wir als weitere Zellen): Textlängen-Verteilung
# pro Klasse (Histogramme), Duplikate & Leakage über den Split, häufigste Wörter
# pro Klasse, Vokabular-Überlappung Inland↔International.
