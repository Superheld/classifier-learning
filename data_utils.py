"""
Gemeinsame Hilfsfunktionen für alle Tracks.

Aktueller Datensatz: banking77 (PolyAI) — echte englische Bank-Kundenanfragen,
77 Intents, kurze Sätze. Intent-Klassifikation (Customer Service).

Format: zwei Parquet-Dateien (train/test) mit den Spalten
  - text        : die Kundenanfrage
  - label       : Intent als Integer 0..76
  - label_text  : Intent als lesbarer Name, z.B. "card_arrival"
Wir nutzen label_text als Label — lesbar in Reports und Confusion-Matrizen.
"""

import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "banking77")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")


def load_banking77(split="train"):
    """Lädt train.parquet oder test.parquet.

    Rückgabe: (texts, labels) — zwei gleich lange Listen.
    texts[i] ist die Kundenanfrage, labels[i] der Intent-Name (label_text).
    """
    path = os.path.join(DATA_DIR, f"{split}.parquet")
    df = pd.read_parquet(path)
    return df["text"].tolist(), df["label_text"].tolist()


def load_banking77_split(val_size=0.15, seed=42):
    """Teilt train.parquet stratifiziert in Trainings- und Validierungsteil.

    Rückgabe: (tr_texts, tr_labels, val_texts, val_labels).
    Das Testset bleibt unberührt (dafür `load_banking77("test")`).

    Warum das nötig ist (F2): Beim Optimieren vergleicht man Varianten und behält
    die beste. Täte man das am Testset, hätte man daran „mitoptimiert" und die
    finale Zahl wäre geschönt. Der Val-Split ist der Vergleichsplatz; der feste
    Seed macht ihn reproduzierbar.
    """
    texts, labels = load_banking77("train")
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=val_size, stratify=labels, random_state=seed
    )
    return tr_texts, tr_labels, val_texts, val_labels


def save_result(name, accuracy, **extra):
    """Speichert das Ergebnis eines Ansatzes in results.json.

    So kann die Synthese am Ende alle Ansätze gegenüberstellen.
    """
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results = json.load(f)
    results[name] = {"accuracy": round(accuracy, 4), **extra}
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[gespeichert] {name}: accuracy={accuracy:.4f} -> results.json")
