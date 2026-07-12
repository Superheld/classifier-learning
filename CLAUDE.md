# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Was das ist

Ein **Lernprojekt**, kein Produkt. Ziel: einen Textklassifikator bauen **und dabei
das Feld verstehen** — von klassischem ML über Embeddings und Finetuning bis zum LLM.
Bruce lernt, du bist Tutor/Mentor. Alles ist auf Deutsch (Code-Kommentare,
Doku, Notebook-Prosa), lernenden-orientiert und an echten Daten ehrlich gemessen.

**Aufgabe/Datensatz:** [banking77](https://huggingface.co/datasets/PolyAI/banking77) —
englische Bank-Kundenanfragen, **77 Intents**, kurze Sätze. Liegt als zwei Parquet
(`data/banking77/train|test.parquet`), geladen über `data_utils.load_banking77()`.
(GNAD-Daten in `data/10kGNAD/` sind Alt-Reste, ungenutzt. `data/support/` sind
Nebendatensätze, gitignored, via `data/support/download_*.py` holbar.)

## Notebooks im Zellen-Format (das Ausführungsmodell)

Es gibt **keine** `.ipynb` und **kein** Test-Suite/CLI. Die `.py`-Modell-Dateien sind
**Notebooks im `# %%`-Zellenformat**, gedacht für **Zeds REPL** (Cmd+Enter pro Zelle).
Kein `.ipynb` → bessere Git-Diffs. Du führst sie normalerweise nicht als Skript aus;
Bruce fährt die Zellen interaktiv. Beim Bearbeiten die Zellstruktur und den
Erklär-Ton (Markdown-Zellen `# %% [markdown]`) beibehalten.

- **Notebooks sind Lehrmaterial — immer gut dokumentieren.** Das ist keine Zier,
  sondern der Zweck: Bruce lernt daran. In **kleinen Schritten** vorgehen, jeden mit
  einer Markdown-Zelle (`# %% [markdown]`) einleiten, die erklärt **was** wir tun,
  **wie** es funktioniert und **warum** — vor der Code-Zelle, nicht nachträglich.
  Fachbegriffe beim ersten Auftreten kurz erden (TF-IDF, Macro-F1, `fit`/`transform`).
  Lieber eine Zelle mehr und ein Gedanke pro Zelle als ein dichter Block. Am Ende
  eines Abschnitts kurz **deuten** (was sagt die Zahl?) und oft ein `# ✓ Checkpoint:`
  als Verständnisfrage. Ton: erklärend, konkret, ohne Jargon-Nebel — wie ein Tutor,
  der mitdenkt. Bestehende Dateien (`tfidf_logreg.py`) sind die Referenz für Dichte
  und Ton.
- **Konvention: eine Datei pro Modell**, benannt nach dem Modell (`tfidf_logreg.py`,
  `mpnet.py`). Jede Datei **wächst von *pure* (P2) zu *getunt* (P3)** — nicht neue
  Dateien pro Phase, sondern dieselbe Datei verlängern.
- **Setup-Zelle immer zuerst.** Sie enthält (a) einen `autoreload`-Guard
  (`try: get_ipython()… except (NameError, AttributeError): pass`) und (b) ein
  Root-Bootstrap: von `Path.cwd()` nach oben bis `data_utils.py` gefunden ist, dann
  Root **und** Track-Ordner auf `sys.path`. Dadurch laufen Notebooks egal von wo Zed
  den Kernel startet. Dieses Muster in jeder neuen Modell-Datei übernehmen.
- **Kernel-Cache-Falle:** Geänderte importierte Module werden dank `autoreload`
  meist live neu geladen; bei Zweifel Kernel neu starten.

## Kapitel-Struktur

Ordner = Kapitel; jeder Track (A–D) = eine „Generation" von Modellen; jeder Track
durchläuft **P1 Vorbereiten → P2 Bauen → P3 Optimieren → P4 Einordnen**.

| Ordner | Inhalt |
|---|---|
| `01_fundament/` | F1–F4: Daten & EDA, Messwesen, Optimierungszyklus, Theorie |
| `02_track_a_klassisches_ml/` | Gen 1: TF-IDF + LogReg / LinearSVC / SGD / Naive Bayes (Unterordner `linear/`, `naive_bayes/`) |
| `03_track_b_embeddings/` | Gen 2: eingefrorene SBERT-Embeddings + linearer Kopf (`klein/`, `stark/`) |
| `04_track_c_finetuning/` | Gen 3a: BERT & Co. finetunen (offen) |
| `05_track_d_llm/` | Gen 3b: LLM per Prompt (offen; `anthropic` in requirements) |
| `06_synthese/` | Gesamtvergleich + Mini-App |

## Geteilter Kern vs. track-lokale Module

Root-Helfer werden von **allen** Tracks importiert; track-spezifische Module liegen
im jeweiligen Track-Ordner (auf `sys.path` durch die Setup-Zelle).

- `data_utils.py` — `load_banking77(split)`, `load_banking77_split(val_size, seed=42)`
  (stratifiziert, für Val), `save_result(name, accuracy, **extra)` schreibt nach
  `results.json` (als Rangliste sortiert).
- `eval_utils.py` — Fehler-Visualisierung: `plot_per_class_f1`, `plot_top_confusions`,
  `plot_confusion_matrix`, `plot_rounds`.
- `optimization.py` — **`greedy_search(evaluate, experiments)`**: der track-unabhängige
  F3-Optimierungskern. Fährt den Zyklus (Default auf val messen → je Experiment **genau
  eine** Änderung → besser an val-Macro-F1 → behalten). Der Loop lebt **einmal im Root**;
  *wie* eine Config zu einem Score wird, bringt jeder Track als eigene `evaluate`-Funktion
  mit (Track A: `02_track_a_klassisches_ml/tuning.py::tune`, das TF-IDF-Matrix+Kopf baut).
- `03_track_b_embeddings/embeddings.py` — `encode(model_name, texts, cache_key, split,
  prefix="")`: Encoder laden, encodieren, als `.npy` cachen (Cache gitignored,
  regenerierbar). Manche Encoder brauchen einen Prefix (E5: `"query: "`), MiniLM/mpnet nicht.

## Das Kernprinzip: ehrlich messen

Diese Disziplin ist der *Punkt* des Projekts — nicht verwässern:

- **Baseline zuerst** (Majority ≈ 1,30 %) als Latte.
- **Nur am Val-Split tunen** (`load_banking77_split()`), **nie am Testset.**
- Das **Testset wird pro Track genau zweimal** angefasst: naive P2-Messung und finale
  P3-Messung. Nie dazwischen. In P3 die Best-Config auf dem **vollen** train (Trainingsteil
  + val) refitten, dann test 1× messen.
- **Die Messung entscheidet, nicht die Theorie.** Wenn eine Hypothese auf val durchfällt
  (z.B. Bigramme), wird sie verworfen und das offen so benannt — auch im Notebook-Text.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Danach Notebook in Zed öffnen, Setup-Zelle ausführen. Einstieg: `01_fundament/f1_daten_eda.py`.
Python 3.14, macOS; Embedding-Encoder nutzen `mps` falls verfügbar.

## Weitere Doku

- `README.md` — Projekt-Orientierung + Scoreboard (test-Accuracy je Ansatz).
- `CURRICULUM.md` — das Curriculum (datensatz-unabhängig, CRISP-DM + Generationen).
- `KONZEPTE.md` — Handwerks-Handbuch (**enthält noch GNAD-Reste**, werden nachgezogen).
- `results.json` — generierte Ergebnis-Rangliste (via `data_utils.save_result`).

## Arbeitsweise in diesem Repo

- **Erklären an etablierten Frameworks erden** (CRISP-DM, Modell-Generationen), Quellen
  konkret nennen, im Zweifel per Docs/WebFetch verifizieren — **keine je neu erfundenen
  Taxonomien**. Bruce will festen Boden, prüft als Entwickler/Tester nach.
- Neue Modell-Datei = Konvention kopieren: Setup-Zelle, `save_result`-Aufruf mit
  aussagekräftigem Namen, P2-pure-dann-P3-getunt-Aufbau, `tune()`/`greedy_search` für P3.
