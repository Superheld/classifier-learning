# Klassifizierung — Lernprojekt Textklassifikation

Einen Textklassifikator bauen **und dabei das Feld verstehen** — von klassischem
ML über Embeddings und Finetuning bis zum LLM. Kein Produkt, ein Lernpfad:
Bruce lernt, die AI ist Tutor. Gearbeitet wird an echten Daten, ehrlich gemessen.

**Aufgabe:** [banking77](https://huggingface.co/datasets/PolyAI/banking77) —
echte englische Bank-Kundenanfragen, **77 Intents**, kurze Sätze.
Intent-Klassifikation (Customer Service). Liegt in `data/banking77/`, geladen
über `data_utils.load_banking77()`.

## Struktur

Ordner = Kapitel. Das Curriculum steht in **`CURRICULUM.md`** (CRISP-DM als Prozess,
vier Strategie-Tracks über drei „Generationen" der Modelle).

| Ordner | Inhalt |
|---|---|
| `01_fundament/` | Daten & EDA, Messwesen, Experiment-Setup, Theorie (F1–F4) |
| `02_track_a_klassisches_ml/` | Gen 1: TF-IDF + LogReg / LinearSVC / Naive Bayes |
| `03_track_b_embeddings/` | Gen 2: semantische Embeddings |
| `04_track_c_finetuning/` | Gen 3a: BERT & Co. finetunen |
| `05_track_d_llm/` | Gen 3b: LLM per Prompt |
| `06_synthese/` | Gesamtvergleich + Mini-App |

Jeder Track durchläuft dieselben vier Phasen: **P1 Vorbereiten → P2 Bauen →
P3 Optimieren → P4 Einordnen**.

## Notebooks

Die `.py`-Dateien sind **Notebooks im Zellen-Format** (`# %%`), gedacht für Zeds
REPL (Cmd+Enter je Zelle). Kein `.ipynb` — bessere Git-Diffs.

- **Konvention:** eine Datei pro Modell, benannt nach dem Modell
  (`tfidf_logreg.py`, `tfidf_linear_svc.py` …). Jede Datei wächst von *pure*
  (P2) zu *getunt* (P3).
- **Setup-Zelle immer zuerst** ausführen — alle anderen bauen auf ihr auf.
- **`autoreload`** ist eingebaut: geänderte Module (z.B. `data_utils.py`) werden
  ohne Kernel-Neustart neu geladen.

## Ehrlich messen (das Kernprinzip)

- **Baseline zuerst** (Majority): die Latte, die jedes Modell überspringen muss.
- **Val-Split** (`data_utils.load_banking77_split()`) zum Optimieren — **nie am
  Testset tunen**, sonst wird die Zahl Fiktion.
- Das **Testset** wird pro Track genau **zweimal** angefasst: naive P2-Messung
  und finale P3-Messung. Nie dazwischen.

## Stand & Ergebnisse (test, Accuracy)

| Ansatz | pure | getunt |
|---|---|---|
| Majority-Baseline | 1,30 % | — |
| **A** · linear · LogReg | 87,78 % | **90,25 %** |
| **A** · linear · LinearSVC | 89,47 % | 89,50 % |
| **A** · linear · SGD | 88,23 % | 88,95 % |
| **A** · naive_bayes · MultinomialNB | 78,77 % | 87,03 % |
| **A** · naive_bayes · ComplementNB | 70,06 % | 79,88 % |
| **B/C/D** | offen | offen |

Gen-1-Befund: lineare Sippe (~88–90 %) schlägt Naive Bayes klar; getunter
LogReg führt. Fundament: **F1 (EDA)** ✅ · **F2 (Val-Split)** ✅ ·
**F3 (Optimierungszyklus** `experiment.tune()`**)** ✅ · F4 offen.
Referenz-Obergrenze auf banking77 (BERT): ~93–94 %.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dann ein Notebook in Zed öffnen, Setup-Zelle ausführen, losgehen.
Empfohlener Einstieg: `01_fundament/f1_daten_eda.py`.

## Weitere Doku

- **`CURRICULUM.md`** — das Curriculum (datensatz-unabhängig, wiederverwendbar).
- `MODELL-LANDKARTE.md`, `KONZEPTE.md` — Nachschlagewerke (teils noch aus der
  Vorgänger-Phase, werden nachgezogen).
- `results.json` — Ergebnis-Sammlung (generiert, via `data_utils.save_result`).
