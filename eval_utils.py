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


def plot_rounds(proto_df, title="Optimierungsrunden"):
    """Balken der val-Macro-F1 je Runde. Grün = behalten, rot = verworfen, grau = Start."""
    colors = {"ja": "#3D9970", "nein": "#E8684A", "—": "#AAAAAA"}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(proto_df["schritt"], proto_df["val_macroF1"],
           color=[colors[b] for b in proto_df["behalten"]])
    ax.set_ylabel("val Macro-F1 (%)")
    ax.set_ylim(proto_df["val_macroF1"].min() - 3, proto_df["val_macroF1"].max() + 1)
    ax.set_title(f"{title} (grün behalten · rot verworfen)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, normalize=True, title="Confusion Matrix"):
    """Vollständige Confusion-Matrix als Heatmap: Zeile = wahr, Spalte = getippt.

    normalize=True: je Zeile auf 1 normiert (Recall-Verteilung je wahrer Klasse),
    damit unterschiedlich große Klassen vergleichbar sind. Bei 77 Intents ist die
    Matrix dicht — die dunkle Diagonale ist gut (richtig), off-diagonale Flecken
    sind die Verwechslungen. Für die konkreten Paare: plot_top_confusions.
    """
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(14, 13))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("getippt (predicted)")
    ax.set_ylabel("wahr (true)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
