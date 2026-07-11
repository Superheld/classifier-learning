"""
Mess- und Visualisierungswerkzeug (F2 „Messwesen") — von allen Tracks genutzt.

Zwei Blicke auf die Fehler eines Klassifikators:
  - plot_per_class_f1   : welche Intents sitzen, welche wackeln?
  - plot_top_confusions : welche Intent-Paare wirft das Modell durcheinander?
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def plot_per_class_f1(y_true, y_pred, worst=None, title="F1 je Intent"):
    """Horizontaler Balken: F1 je Klasse, schwächste unten.

    worst=N zeigt nur die N schwächsten Intents (da sitzt die Musik).
    """
    labels = sorted(set(y_true) | set(y_pred))
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    order = f1.argsort()  # schlechteste zuerst
    labs = [labels[i] for i in order]
    vals = [f1[i] for i in order]
    if worst:
        labs, vals = labs[:worst], vals[:worst]
        title = f"{title} — schwächste {worst}"
    fig, ax = plt.subplots(figsize=(8, max(3, len(labs) * 0.22)))
    ax.barh(labs, vals, color="#5B8FF9")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_top_confusions(y_true, y_pred, top=15, title="Häufigste Verwechslungen"):
    """Horizontaler Balken: die häufigsten Fehl-Paare (wahr → getippt)."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True)
    pairs = pairs[:top]
    lab = [f"{t}  →  {p}" for _, t, p in pairs][::-1]
    val = [c for c, _, _ in pairs][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(pairs) * 0.35)))
    ax.barh(lab, val, color="#E8684A")
    ax.set_xlabel("Anzahl (wahr → getippt)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
