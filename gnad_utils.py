"""
Gemeinsame Hilfsfunktionen für alle Lernstufen.

Der 10kGNAD-Datensatz ist eine CSV-Datei mit Semikolon als Trennzeichen
und einfachen Anführungszeichen (') als Quotechar — beides untypisch,
deshalb kapseln wir das Einlesen hier einmal zentral.
"""

import csv
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "10kGNAD")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")

# Ohne dieses Limit bricht das csv-Modul bei sehr langen Artikeln ab.
csv.field_size_limit(sys.maxsize)


def load_gnad(split="train"):
    """Lädt train.csv oder test.csv.

    Rückgabe: (texts, labels) — zwei gleich lange Listen.
    texts[i] ist der Artikeltext, labels[i] die zugehörige Kategorie.
    """
    path = os.path.join(DATA_DIR, f"{split}.csv")
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";", quotechar="'")
        for row in reader:
            labels.append(row[0])
            texts.append(row[1])
    return texts, labels


def save_result(name, accuracy, **extra):
    """Speichert das Ergebnis einer Stufe in results.json.

    So kann 05_vergleich.py am Ende alle Ansätze gegenüberstellen.
    """
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results = json.load(f)
    results[name] = {"accuracy": round(accuracy, 4), **extra}
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[gespeichert] {name}: accuracy={accuracy:.4f} -> results.json")
