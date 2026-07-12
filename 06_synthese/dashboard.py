"""
06_synthese — Dashboard (Gradio).

Interaktive Ansicht unserer Ergebnisse. KEINE `# %%`-Notebook-Datei, sondern eine
richtige App — starten mit:

    python 06_synthese/dashboard.py

Das öffnet einen lokalen Webserver auf http://127.0.0.1:7860 (offline, kein Account,
kein Netz). Mit Strg+C beenden.

--- Gradio in drei Begriffen (dein Einstieg) ---
- **Blocks**   : der Container für die ganze Oberfläche. Alles, was man `with gr.Blocks()`
                 hineinschreibt, wird zu Web-Elementen.
- **Component** : ein einzelnes UI-Element — `gr.Dataframe` (Tabelle), `gr.BarPlot`
                 (Balken), `gr.Markdown` (Text). Man gibt ihm Daten, es rendert sie.
- **launch()**  : startet den Webserver und serviert die Blocks im Browser.

Diese erste Scheibe ist rein **anzeigend** (liest `results.json`) — noch kein Modell,
keine Eingabe. Die kommen in späteren Tabs (Embedding-Explorer, Live-Klassifikator).
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Diese Datei liegt in 06_synthese/, das Projekt-Root ist eine Ebene höher.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # damit data_utils gefunden wird (Test-Labels)

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # headless: Figuren für Gradio erzeugen, kein Fenster nötig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors

from data_utils import load_banking77

RESULTS = ROOT / "results.json"
PRED_DIR = ROOT / "predictions"  # echte Test-Vorhersagen je Modell (via eval_utils.evaluate_and_save)

# Der Name jedes Ergebnisses beginnt mit dem Track-Buchstaben (A_…, B_…, C_…).
# Daraus leiten wir die „Generation" ab — die Landkarte, die alle Tracks verbindet.
GENERATION = {
    "A": "Gen 1 · Klassisch (TF-IDF)",
    "B": "Gen 2 · Embeddings (frozen)",
    "C": "Gen 3a · Finetuning",
    "D": "Gen 3b · LLM",
}

# --- Raum-Explorer: Embeddings im (projizierten) Raum ---
# Test-TEXTE und -Labels sind index-gleich zu den gecachten Embedding-Matrizen (alle
# in der Reihenfolge von load_banking77("test") gerechnet). Die Texte brauchen wir, um
# einen „falsch sitzenden" Punkt aufmachen und LESEN zu können.
TEST_TEXTS, TEST_LABELS = load_banking77("test")
TEST_TEXTS = np.array(TEST_TEXTS, dtype=object)
TEST_LABELS = np.array(TEST_LABELS)

# Eine Handvoll Intents statt aller 77 — sonst 77 Farben = unlesbarer Regenbogen.
# Zwei bekannte Verwechsel-Paare + zwei klar verschiedene, damit Überlappung UND
# saubere Trennung im selben Bild sichtbar werden.
SELECTED_INTENTS = [
    "card_arrival", "card_delivery_estimate",             # oft verwechselt (Karte unterwegs)
    "exchange_rate", "card_payment_wrong_exchange_rate",  # oft verwechselt (Wechselkurs)
    "card_linking", "pending_cash_withdrawal",            # klar verschieden
]

ENCODERS = {
    "mpnet (frozen · Track B)": ROOT / "03_track_b_embeddings/cache/mpnet_test.npy",
    "RoBERTa-ft (finetuned · Track C)": ROOT / "04_track_c_finetuning/hybrid/cache/roberta_ft_test.npy",
}


def load_scoreboard():
    """Liest results.json → aufbereitete Tabelle (DataFrame), nach Accuracy sortiert.

    Warum DataFrame: sowohl `gr.Dataframe` als auch `gr.BarPlot` nehmen genau das
    entgegen — eine Aufbereitung, zwei Ansichten.
    """
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = []
    for key, v in results.items():
        rows.append(
            {
                "Modell": v.get("model", key),
                "Generation": GENERATION.get(key[0].upper(), key[0]),
                "Accuracy %": round(v["accuracy"] * 100, 2),
                "Macro-F1 %": round(v.get("macro_f1", 0) * 100, 2),
                "Key": key,
            }
        )
    df = pd.DataFrame(rows).sort_values("Accuracy %", ascending=False, ignore_index=True)
    return df


# Explizite plain→getunt-Paare (P2 → P3). Bewusst per Hand, nicht aus den Keys
# geparst: die Namensschemata sind nicht einheitlich (B nutzt "..._logreg" statt
# "..._plain"), und eine explizite Liste ist ehrlicher und robuster als Rate-Parsing.
# (Ansätze ohne getunte Gegenstücke — DistilBERT, MiniLM, E5, LoRA, Hybrid — fehlen
# hier bewusst; sie haben kein plain↔getunt-Paar.)
TUNING_PAIRS = [
    ("A · TF-IDF + LogReg",    "A_plain_tfidf_logreg",        "A_tuned_tfidf_logreg"),
    ("A · TF-IDF + LinearSVC", "A_plain_tfidf_linsvc",        "A_tuned_tfidf_linsvc"),
    ("A · TF-IDF + SGD",       "A_plain_tfidf_sgd",           "A_tuned_tfidf_sgd"),
    ("A · MultinomialNB",      "A_plain_tfidf_multinomialnb", "A_tuned_tfidf_multinomialnb"),
    ("A · ComplementNB",       "A_plain_tfidf_complementnb",  "A_tuned_tfidf_complementnb"),
    ("B · mpnet + Kopf",       "B_mpnet_logreg",              "B_mpnet_tuned"),
    ("C · RoBERTa (full FT)",  "C_plain_roberta",             "C_tuned_roberta"),
    ("C · mpnet (full FT)",    "C_plain_mpnet_ft",            "C_tuned_mpnet_ft"),
]


def load_tuning_deltas():
    """Je Strategie: Accuracy plain (P2), getunt (P3) und die Differenz Δ in Punkten.

    Δ ist der eigentliche Blick: *wie viel Luft* hatte das Optimieren? Sehr ungleich —
    genau das ist die Lehre (Naive Bayes hatte viel, die lineare Sippe kaum).
    """
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = []
    for label, plain_key, tuned_key in TUNING_PAIRS:
        if plain_key not in results or tuned_key not in results:
            continue  # fehlt ein Ergebnis, Paar überspringen (nicht crashen)
        plain = results[plain_key]["accuracy"] * 100
        tuned = results[tuned_key]["accuracy"] * 100
        rows.append(
            {
                "Strategie": label,
                "Generation": GENERATION.get(label[0].upper(), label[0]),
                "plain %": round(plain, 2),
                "getunt %": round(tuned, 2),
                "Δ Punkte": round(tuned - plain, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("Δ Punkte", ascending=False, ignore_index=True)


def _subset(cache_path):
    """Lädt eine gecachte Embedding-Matrix und filtert auf die SELECTED_INTENTS.

    Gibt drei index-gleiche Arrays zurück: Embeddings X, Labels y, Rohtexte texts.
    """
    X = np.load(cache_path)
    mask = np.isin(TEST_LABELS, SELECTED_INTENTS)
    return X[mask], TEST_LABELS[mask], TEST_TEXTS[mask]


def _neighbor_majority(X, y, k=10):
    """Für jeden Punkt: das Mehrheits-Label seiner k nächsten Nachbarn.

    Der Kern der Fehlsitzer-Diagnose. Für jeden Punkt suchen wir die k nächsten
    Nachbarn im **echten 768-dim Raum** (nicht in der t-SNE-Projektion — die verzerrt)
    und schauen, welches Intent-Label dort die Mehrheit stellt.

    - **Metrik cosine**: SBERT-Embeddings leben auf der Kugel — Richtung zählt, nicht
      Länge. Cosine ist die Distanz, für die diese Encoder trainiert sind (nicht Euklid).
    - **k+1 Nachbarn**: der erste Treffer ist der Punkt selbst (Distanz 0) — den werfen
      wir raus, bleiben k echte Nachbarn.
    - „**Fehlsitzer**": Mehrheits-Label ≠ eigenes Label → der Punkt ist von einer
      fremden Klasse umringt. Das ist das Signal, das jeden Klassifikator hier irrt.

    Rückgabe: Array gleicher Länge wie y mit dem jeweiligen Nachbar-Mehrheits-Label.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    maj = []
    for i, row in enumerate(idx):
        neigh = [y[j] for j in row if j != i][:k]  # sich selbst rauswerfen
        vals, counts = np.unique(neigh, return_counts=True)
        maj.append(vals[counts.argmax()])
    return np.array(maj)


def scatter_3d(cache_path, title):
    """768-dim Embeddings → 3D via t-SNE → interaktiver Plotly-Streuplot (drehbar).

    t-SNE (t-distributed Stochastic Neighbor Embedding): nichtlineare Projektion, die
    *Nachbarschaften* erhält — nah-beieinander bleibt nah. Ideal, um Cluster zu SEHEN.
    Vorbehalt: absolute Abstände/Achsen bedeuten nichts, nur die Gruppierung zählt.

    Zusätzlich markiert: **Fehlsitzer** (im 768-dim Raum von fremder Klasse umringt)
    als **Raute** statt Kreis. Beim Drüberfahren (Hover) erscheinen Text, eigenes Label
    und Nachbar-Mehrheit — damit man den Punkt aufmachen und selbst beurteilen kann.
    Achtung: die Rauten-Entscheidung fällt im echten 768-dim Raum; im t-SNE-Bild kann
    eine Raute deshalb gelegentlich gar nicht *aussehen* wie ein Ausreißer (Projektions-
    verzerrung) — dann liegt sie in Wahrheit trotzdem falsch.
    """
    X, y, texts = _subset(cache_path)
    maj = _neighbor_majority(X, y)
    misfit = maj != y
    coords = TSNE(n_components=3, random_state=42, init="pca", perplexity=30).fit_transform(X)

    # Text für den Hover kürzen (lange Anfragen sprengen sonst die Sprechblase).
    short = [t if len(t) <= 80 else t[:77] + "…" for t in texts]
    frame = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        "Intent": y,
        "Text": short,
        "Nachbarn": maj,
        "Sitz": np.where(misfit, "fremd umgeben", "passend"),
    })
    fig = px.scatter_3d(
        frame, x="x", y="y", z="z", color="Intent",
        symbol="Sitz", symbol_map={"passend": "circle", "fremd umgeben": "diamond"},
        hover_data={"Text": True, "Intent": True, "Nachbarn": True,
                    "Sitz": False, "x": False, "y": False, "z": False},
        title=title, opacity=0.85, labels={"color": "Intent"},
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(font=dict(size=9)))
    return fig


def misfit_table():
    """Alle Fehlsitzer beider Encoder als eine Tabelle zum LESEN und Selbst-Beurteilen.

    Pro Encoder jeder Punkt, dessen Nachbar-Mehrheit dem eigenen Label widerspricht —
    mit Rohtext, eigenem Label und Nachbar-Mehrheit. Die **Encoder-Spalte** ist der
    Clou: ein Text, der bei *mpnet (frozen)* als Fehlsitzer auftaucht, beim
    *RoBERTa-ft* aber nicht, wurde durchs Finetunen an die richtige Stelle gezogen.

    Bewusst KEIN Urteil („mislabeled ja/nein") — das ist genau die Lesearbeit, die du
    selbst machen willst: echte Überlappung, schräge Formulierung oder Annotationsfehler?
    """
    rows = []
    for label, path in ENCODERS.items():
        X, y, texts = _subset(path)
        maj = _neighbor_majority(X, y)
        for i in range(len(y)):
            if maj[i] != y[i]:
                rows.append({
                    "Encoder": label,
                    "eigenes Label": y[i],
                    "Nachbar-Mehrheit": maj[i],
                    "Text": texts[i],
                })
    df = pd.DataFrame(rows)
    # nach Text sortieren, damit derselbe Text (beide Encoder) direkt untereinander steht.
    return df.sort_values(["Text", "Encoder"], ignore_index=True)


# --- Fehler-Analyse: der ECHTE Modell-Blick (nicht die Nachbarschafts-Näherung) ---
# Quelle: predictions/<key>.json (y_true/y_pred, in Test-Reihenfolge). Anders als die
# Fehlsitzer-Tabelle (die aus Encoder-Nachbarschaften rät) sind das die *tatsächlichen*
# Fehltipps des fertig getunten Klassifikators — „was wirklich falsch lief".

def available_predictions():
    """Alle Modelle, für die echte Test-Vorhersagen vorliegen (predictions/*.json)."""
    if not PRED_DIR.exists():
        return []
    return sorted(p.stem for p in PRED_DIR.glob("*.json"))


def _load_pred(key):
    """Lädt y_true/y_pred eines Modells als zwei Arrays (Label-Namen, Test-Reihenfolge)."""
    d = json.loads((PRED_DIR / f"{key}.json").read_text(encoding="utf-8"))
    return np.array(d["y_true"]), np.array(d["y_pred"])


def confusion_pairs(key, top=25):
    """Die häufigsten systematischen Verwechslungen: (wahr → fälschlich getippt), gezählt.

    Das ist der Kompass fürs Re-Optimieren: ein oft wiederkehrendes Paar ist ein
    *systematischer* Fehler (zwei kaum trennbare Intents) — kein Zufallsausrutscher.
    """
    yt, yp = _load_pred(key)
    wrong = yt != yp
    c = Counter((a, b) for a, b in zip(yt[wrong], yp[wrong]))
    rows = [{"wahr": a, "getippt (falsch)": b, "Anzahl": n} for (a, b), n in c.most_common(top)]
    return pd.DataFrame(rows)


def weak_classes(key, worst=15):
    """Die schwächsten Intents nach F1 — wo sitzt die Musik?

    F1 je Klasse (Harmonie aus Präzision & Recall). Kleiner F1 = das Modell trifft
    diesen Intent unzuverlässig. Support = wie viele Testbeispiele die Klasse hat
    (kleine Klassen wackeln naturgemäß mehr).
    """
    yt, yp = _load_pred(key)
    labels = sorted(set(yt) | set(yp))
    _, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    df = pd.DataFrame({
        "Intent": labels,
        "F1": np.round(f1, 3),
        "Recall": np.round(rec, 3),
        "Support": sup,
    })
    return df.sort_values("F1", ignore_index=True).head(worst)


def error_examples(key):
    """ALLE falsch getippten Testbeispiele als Tabelle: Text + wahr + getippt.

    Die Lese-Oberfläche zum Selbst-Urteilen — index-gleich zu TEST_TEXTS, weil die
    Vorhersagen in Test-Reihenfolge gespeichert sind.
    """
    yt, yp = _load_pred(key)
    wrong = np.where(yt != yp)[0]
    rows = [{"wahr": yt[i], "getippt (falsch)": yp[i], "Text": TEST_TEXTS[i]} for i in wrong]
    return pd.DataFrame(rows).sort_values(
        ["wahr", "getippt (falsch)"], ignore_index=True
    )


def boundary_2d(cache_path, title):
    """Ehrliche „Ebenen"-Ansicht: t-SNE → 2D, dann ein linearer Kopf, der AUF diesen
    2D-Koordinaten trainiert wird — die farbigen Regionen sind SEINE Grenzen.

    Wichtig: das sind NICHT die echten 768-dim-Trennflächen des Original-Modells (die
    lassen sich auf einer nichtlinearen Projektion nicht originalgetreu zeichnen). Es
    zeigt die *Idee* — ein linearer Klassifikator zerschneidet den Raum in Regionen.
    """
    X, y, _ = _subset(cache_path)
    coords = TSNE(n_components=2, random_state=42, init="pca", perplexity=30).fit_transform(X)
    clf = LogisticRegression(max_iter=2000).fit(coords, y)
    classes = list(clf.classes_)
    cmap = ListedColormap([plt.cm.tab10(i) for i in range(len(classes))])

    pad = 3
    xx, yy = np.meshgrid(
        np.linspace(coords[:, 0].min() - pad, coords[:, 0].max() + pad, 300),
        np.linspace(coords[:, 1].min() - pad, coords[:, 1].max() + pad, 300),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Zi = np.array([classes.index(z) for z in Z]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Zi, levels=np.arange(len(classes) + 1) - 0.5, cmap=cmap, alpha=0.25)
    for i, c in enumerate(classes):
        pts = coords[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=cmap(i), label=c,
                   edgecolors="k", linewidths=0.2)
    ax.legend(fontsize=7, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def build_app():
    """Baut die Gradio-Oberfläche und gibt die Blocks zurück (ohne zu starten).

    Getrennt vom `launch()`, damit man die App auch bloß *konstruieren* kann (z.B. zum
    Testen), ohne gleich einen Server hochzufahren.
    """
    df = load_scoreboard()

    with gr.Blocks(title="Klassifizierung — Dashboard") as app:
        gr.Markdown(
            "# Klassifizierung — Dashboard\n"
            "Textklassifikation auf **banking77** über drei Modell-Generationen, "
            "ehrlich gemessen (test je Modell genau einmal)."
        )

        # Ein Tab. Später kommen „Embedding-Explorer" und „Live-Klassifikator" daneben.
        with gr.Tab("Scoreboard"):
            gr.Markdown(
                "Alle Ansätze nach Test-Accuracy. Farbe = Generation. "
                "Die Latte oben ist der eingefrorene mpnet-Encoder (Gen 2) — "
                "Finetuning (Gen 3) zieht gleich, schlägt ihn aber nicht klar."
            )
            # BarPlot: kategorische Achse x = Modell, Höhe y = Accuracy, Farbe = Generation.
            gr.BarPlot(
                value=df,
                x="Modell",
                y="Accuracy %",
                color="Generation",
                title="Test-Accuracy je Ansatz",
                y_lim=[65, 96],  # unten abschneiden, damit die oberen Unterschiede sichtbar werden
                height=420,
            )
            # Dieselben Daten als sortier-/durchsuchbare Tabelle.
            gr.Dataframe(
                value=df[["Modell", "Generation", "Accuracy %", "Macro-F1 %"]],
                interactive=False,
                wrap=True,
            )

        # Tab 2: Was hat Optimieren gebracht? (plain P2 → getunt P3, je Strategie)
        with gr.Tab("Tuning-Effekt"):
            gr.Markdown(
                "Wie viel brachte das **Optimieren** (P2 *plain* → P3 *getunt*) je "
                "Strategie? Das **Δ** ist der Sprung in Prozentpunkten. Die Lehre steckt "
                "in der Ungleichheit: **Naive Bayes** hatte riesig Luft (+8–10), die "
                "**lineare Sippe** kaum (LinearSVC ~+0), **Finetuning** moderat. Viel "
                "Tuning-Gewinn heißt: die plain-Version ließ Leistung liegen; wenig heißt: "
                "sie war schon nah dran."
            )
            dt = load_tuning_deltas()
            gr.BarPlot(
                value=dt,
                x="Strategie",
                y="Δ Punkte",
                color="Generation",
                title="Tuning-Gewinn (plain → getunt) in Prozentpunkten",
                height=380,
            )
            gr.Dataframe(value=dt, interactive=False, wrap=True)

        # Tab 3: die Embeddings im (projizierten) Raum — Cluster + ehrliche Grenzen.
        with gr.Tab("Raum-Explorer"):
            gr.Markdown(
                "**Die Daten im Raum.** Jeder Punkt ist eine Testanfrage als 768-dim "
                "Embedding, per **t-SNE** auf 3D projiziert (mit der Maus drehbar!). Gezeigt "
                "sind sechs Intents — zwei bekannte Verwechsel-Paare + zwei klar verschiedene.\n\n"
                "Links der **eingefrorene mpnet**, rechts der **finegetunte RoBERTa**: sieh, "
                "wie das Finetunen dieselben Intents zu **engeren, saubereren Häufchen** "
                "zusammenzieht — der Encoder hat die Trennarbeit übernommen.\n\n"
                "**Fehlsitzer** — Punkte, die im echten 768-dim Raum von einer *fremden* "
                "Klasse umringt sind — zeichnen wir als **Rauten** (Kreise = passend "
                "einsortiert). Fahr mit der Maus drüber: Text, eigenes Label und die "
                "Nachbar-Mehrheit erscheinen. Genau die Punkte, an denen jeder "
                "Klassifikator ins Grübeln kommt."
            )
            with gr.Row():
                for _label, _path in ENCODERS.items():
                    gr.Plot(scatter_3d(_path, _label))

            gr.Markdown(
                "### Fehlsitzer zum Nachlesen\n"
                "Dieselben Rauten als Tabelle — **zum Lesen und Selbst-Urteilen.** Ist der "
                "Text wirklich falsch gelabelt? Oder nur echte **Überlappung** zweier "
                "kaum trennbarer Intents? Oder eine **schräge Formulierung**? Das "
                "entscheidet dein Auge, nicht der Algorithmus — er sagt nur, dass die "
                "Nachbarschaft nicht zum Label passt. Die **Encoder-Spalte** verrät obendrein, "
                "welche Fehlsitzer das Finetunen (RoBERTa-ft) gegenüber dem eingefrorenen "
                "mpnet **repariert** hat (dann fehlt die RoBERTa-ft-Zeile zum selben Text)."
            )
            gr.Dataframe(value=misfit_table(), interactive=False, wrap=True)
            gr.Markdown(
                "**Und die Trennflächen — die Ebenen?** Die *echten* leben im 768-dim "
                "Raum — auf einer nichtlinearen Projektion lassen sie sich **nicht "
                "originalgetreu** einzeichnen (das wäre gelogen). Ehrlich geht das so: "
                "t-SNE auf **2D**, dann ein linearer Kopf, der *auf diesen 2D-Koordinaten* "
                "trainiert wird — die farbigen **Regionen** sind *seine* Grenzen in genau "
                "dieser Ansicht. Das zeigt die Idee (linearer Kopf zerschneidet den Raum in "
                "Regionen), ist aber nicht die 768-dim-Ebene des Originalmodells."
            )
            gr.Plot(
                boundary_2d(
                    ENCODERS["RoBERTa-ft (finetuned · Track C)"],
                    "Entscheidungs-Regionen (linearer Kopf auf 2D-t-SNE, RoBERTa-ft)",
                )
            )

        # Tab 4: Fehler-Analyse — der ECHTE Modell-Blick auf „was falsch lief".
        # Erstes INTERAKTIVES Element: das Dropdown wählt ein Modell, drei Tabellen
        # aktualisieren sich per Event (dd.change → Funktion → outputs).
        with gr.Tab("Fehler-Analyse"):
            gr.Markdown(
                "**Was lief falsch?** Hier die *echten* Fehltipps des fertig getunten "
                "Modells (aus `predictions/*.json`) — nicht die Nachbarschafts-Näherung "
                "aus dem Raum-Explorer, sondern die tatsächlichen Verwechslungen. "
                "Das ist die Datengrundlage, um die **nächste Optimierungsrunde** zu "
                "steuern: systematische Paare erkennen, schwache Intents finden, "
                "Einzelfälle lesen.\n\n"
                "Das **Dropdown** listet jedes Modell, für das wir Vorhersagen gesammelt "
                "haben. Noch ist es nur der Champion — die anderen Top-Ansätze füllen sich, "
                "sobald wir ihre Endmessung mit `evaluate_and_save` neu fahren."
            )
            avail = available_predictions()
            default = avail[0] if avail else None
            model_dd = gr.Dropdown(
                choices=avail, value=default,
                label="Modell (echte Test-Vorhersagen)",
            )

            gr.Markdown(
                "**Systematische Verwechslungen** (wahr → fälschlich getippt, gezählt). "
                "Oft wiederkehrende Paare = kaum trennbare Geschwister-Intents."
            )
            conf_out = gr.Dataframe(
                value=confusion_pairs(default) if default else None,
                interactive=False, wrap=True,
            )
            gr.Markdown("**Schwächste Intents** nach F1 — dort sitzt die meiste Fehlerlast.")
            weak_out = gr.Dataframe(
                value=weak_classes(default) if default else None,
                interactive=False, wrap=True,
            )
            gr.Markdown(
                "**Alle Fehl-Beispiele zum Nachlesen** — Text + wahr + getippt. "
                "Die Lese-Oberfläche, um selbst zu urteilen: echte Überlappung, "
                "schräge Formulierung oder fragwürdiges Label?"
            )
            err_out = gr.Dataframe(
                value=error_examples(default) if default else None,
                interactive=False, wrap=True,
            )

            def _refresh(key):
                """Event-Handler: Modell-Key rein → drei aktualisierte Tabellen raus."""
                return confusion_pairs(key), weak_classes(key), error_examples(key)

            model_dd.change(_refresh, inputs=model_dd, outputs=[conf_out, weak_out, err_out])

    return app


if __name__ == "__main__":
    build_app().launch()
