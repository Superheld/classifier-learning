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
from scipy.stats import binomtest
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


def class_report(key):
    """Precision/Recall/F1/Support je Intent — der Standard-Klassifikationsreport.

    - **Precision**: von allen, die das Modell *als X tippte*, wie viele waren X? (Fehlalarm-Maß)
    - **Recall**:    von allen *echten X*, wie viele fand es? (Verpass-Maß)
    - **F1**:        Harmonie aus beidem. Schwächste unten.
    """
    yt, yp = _load_pred(key)
    labels = sorted(set(yt) | set(yp))
    p, r, f1, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return pd.DataFrame({
        "Intent": labels,
        "Precision": np.round(p, 3),
        "Recall": np.round(r, 3),
        "F1": np.round(f1, 3),
        "Support": sup,
    }).sort_values("F1", ignore_index=True)


# --- Signifikanz: ist ein Accuracy-Unterschied echt oder Rauschen? ---

def bootstrap_ci(key, n_boot=2000, seed=0):
    """95%-Konfidenzintervall der Accuracy per Bootstrap.

    Idee: das Testset n_boot-mal MIT Zurücklegen neu ziehen, jedes Mal die Accuracy
    rechnen → eine Verteilung. Deren 2,5-/97,5-Perzentil ist das Intervall. Sagt: „die
    wahre Accuracy liegt mit 95% zwischen lo und hi". Überlappen zweier Intervalle =
    Vorsicht, der Unterschied könnte Zufall sein.
    """
    yt, yp = _load_pred(key)
    correct = (yp == yt).astype(float)
    n = len(correct)
    rng = np.random.default_rng(seed)
    boot = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return correct.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def significance_overview():
    """Je Modell: Accuracy + 95%-Bootstrap-Intervall (als Tabelle)."""
    rows = []
    for k in available_predictions():
        acc, lo, hi = bootstrap_ci(k)
        rows.append({
            "Modell": k,
            "Accuracy %": round(acc * 100, 2),
            "95%-KI unten %": round(lo * 100, 2),
            "95%-KI oben %": round(hi * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("Accuracy %", ascending=False, ignore_index=True)


def mcnemar_pairs():
    """Paarweiser McNemar-Test zwischen allen Modellen mit Vorhersagen.

    McNemar schaut NUR auf die Beispiele, wo genau *einer* der beiden recht hat
    (die „diskordanten Paare"). Unter der Nullhypothese „beide gleich gut" sollten
    A✗B✓ und A✓B✗ etwa gleich häufig sein — binomial mit p=0,5. Der exakte
    Binomialtest gibt den p-Wert. p < 0,05 → der Unterschied ist signifikant.
    """
    keys = available_predictions()
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            yt, ypa = _load_pred(a)
            _, ypb = _load_pred(b)
            ac, bc = (ypa == yt), (ypb == yt)
            n_ab = int((~ac & bc).sum())   # A falsch, B richtig
            n_ba = int((ac & ~bc).sum())   # A richtig, B falsch
            n = n_ab + n_ba
            p = binomtest(min(n_ab, n_ba), n, 0.5).pvalue if n > 0 else 1.0
            rows.append({
                "A": a, "B": b,
                "A✗ B✓": n_ab, "A✓ B✗": n_ba,
                "p-Wert": round(p, 4),
                "signifikant (α=.05)": "ja" if p < 0.05 else "nein",
            })
    return pd.DataFrame(rows)


# --- Scores: Kalibrierung, Top-k, Abstention (braucht predictions/<key>_scores.npy) ---

def available_scores():
    """Modelle, für die eine Score-Matrix vorliegt (predictions/<key>_scores.npy)."""
    if not PRED_DIR.exists():
        return []
    return sorted(p.name[:-len("_scores.npy")] for p in PRED_DIR.glob("*_scores.npy"))


def _scores(key):
    """Lädt (y_true, Score-Matrix, Klassen-Reihenfolge, score_type) eines Modells."""
    d = json.loads((PRED_DIR / f"{key}.json").read_text(encoding="utf-8"))
    yt = np.array(d["y_true"])
    classes = np.array(d["classes"])
    S = np.load(PRED_DIR / f"{key}_scores.npy")
    return yt, S, classes, d.get("score_type")


def reliability_fig(key, bins=10):
    """Reliability-Diagramm + ECE — nur für echte Wahrscheinlichkeiten sinnvoll.

    Konfidenz (höchster Score) in Bins gruppieren; je Bin die *tatsächliche* Accuracy
    gegen die *mittlere Konfidenz* auftragen. Auf der Diagonale = perfekt kalibriert.
    Unter der Diagonale = zu selbstsicher (typisch für finegetunte Netze).
    ECE (Expected Calibration Error) = gewichteter mittlerer Abstand zur Diagonale.
    """
    yt, S, classes, stype = _scores(key)
    if stype != "proba":
        return None
    conf = S.max(1)
    pred = classes[S.argmax(1)]
    correct = (pred == yt).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ece, n = [], [], 0.0, len(yt)
    for i in range(bins):
        hi = conf <= edges[i + 1] if i == bins - 1 else conf < edges[i + 1]
        m = (conf >= edges[i]) & hi
        if m.sum() == 0:
            continue
        c, a = conf[m].mean(), correct[m].mean()
        ece += (m.sum() / n) * abs(a - c)
        xs.append(c)
        ys.append(a)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfekt kalibriert")
    ax.plot(xs, ys, "o-", color="#5B8FF9", label="Modell")
    ax.set_xlabel("mittlere Konfidenz je Bin")
    ax.set_ylabel("tatsächliche Accuracy je Bin")
    ax.set_title(f"Reliability — {key}   (ECE {ece*100:.2f} %)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig


def topk_table(key, kmax=5):
    """Top-k-Accuracy: steht die wahre Klasse unter den k höchstbewerteten? (Rang-basiert)."""
    yt, S, classes, _ = _scores(key)
    order = np.argsort(-S, axis=1)
    n = len(yt)
    rows = []
    for k in range(1, kmax + 1):
        topk = classes[order[:, :k]]
        hit = np.mean([yt[i] in topk[i] for i in range(n)])
        rows.append({"k": k, "Top-k Accuracy %": round(hit * 100, 2)})
    return pd.DataFrame(rows)


def risk_coverage_fig(key, steps=19):
    """Risk-Coverage: nur die sichersten X% beantworten — wie steigt die Accuracy?

    Nach Konfidenz sortieren, die untersichersten weglassen. Zeigt den Deploy-Hebel
    „unsichere an Menschen weiterreichen". Rang-basiert → geht auch für SVM-Margen.
    """
    yt, S, classes, _ = _scores(key)
    conf = S.max(1)
    pred = classes[S.argmax(1)]
    correct = (pred == yt).astype(float)
    order = np.argsort(-conf)
    n = len(yt)
    cov = np.linspace(0.05, 1.0, steps)
    accs = [correct[order[: max(1, int(c * n))]].mean() for c in cov]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cov * 100, np.array(accs) * 100, "o-", color="#3D9970")
    ax.set_xlabel("Abdeckung % (die sichersten X % werden beantwortet)")
    ax.set_ylabel("Accuracy auf den beantworteten %")
    ax.set_title(f"Risk-Coverage — {key}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --- Betriebszahlen: aus 06_synthese/betriebszahlen.json (eigener Benchmark-Lauf) ---

BETRIEB = ROOT / "06_synthese" / "betriebszahlen.json"


def load_betrieb():
    """Betriebszahlen (Parameter, Latenz, Größe) je Modell-Familie + Accuracy dazu.

    Quelle: betriebszahlen.json (fair auf EINER Maschine gemessen, siehe
    betriebszahlen.py). Accuracy wird aus results.json dazugejoint, damit die
    Effizienz-Sicht (Accuracy vs. Kosten) möglich ist.
    """
    if not BETRIEB.exists():
        return pd.DataFrame()
    data = json.loads(BETRIEB.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = []
    for d in data:
        acc = results.get(d.get("result_key", ""), {}).get("accuracy")
        rows.append({
            "Familie": d.get("familie", d.get("result_key", "?")),
            "Parameter (Mio)": d.get("params_mio"),
            "Latenz ms/Anfrage": d.get("latenz_ms"),
            "Größe MB": d.get("groesse_mb"),
            "Accuracy %": round(acc * 100, 2) if acc is not None else None,
        })
    return pd.DataFrame(rows)


def _score_type(key):
    """score_type eines Modells (proba / logits / decision_function / None)."""
    d = json.loads((PRED_DIR / f"{key}.json").read_text(encoding="utf-8"))
    return d.get("score_type")


def _calib_note(key):
    """Kurzer Hinweis über der Reliability-Kurve — ob das Modell überhaupt kalibrierbar ist."""
    if key is None:
        return "_Kein Modell mit Scores vorhanden._"
    st = _score_type(key)
    if st == "proba":
        return f"**{key}** — echte Wahrscheinlichkeiten (`proba`), Reliability-Kurve unten."
    return (f"**{key}** — `score_type={st}`: nur Margen/Ränge, **nicht kalibrierbar**. "
            "Für eine Reliability-Kurve müsste der Klassifikator erst kalibriert werden "
            "(CalibratedClassifierCV, Platt/Isotonic).")


def significance_ci_fig():
    """Accuracy-Punkt je Modell mit 95%-Bootstrap-Intervall als waagerechtem Balken.

    Überlappen sich zwei Balken kräftig, ist der Ranglisten-Unterschied mit Vorsicht
    zu genießen — er könnte Stichproben-Rauschen sein.
    """
    df = significance_overview()
    y = np.arange(len(df))
    acc = df["Accuracy %"].to_numpy()
    lo = df["95%-KI unten %"].to_numpy()
    hi = df["95%-KI oben %"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, max(2, len(df) * 0.6)))
    ax.errorbar(acc, y, xerr=[acc - lo, hi - acc], fmt="o", color="#5B8FF9", capsize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Modell"])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy % (Punkt) mit 95%-Bootstrap-KI (Balken)")
    ax.set_title("Überlappen die Balken, ist der Unterschied evtl. Zufall")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def betrieb_fig():
    """Effizienz-Frontier: Accuracy gegen Parameterzahl (Log-Achse). Oben-links ist gut.

    Die klassische „lohnt sich das große Modell?"-Sicht — kleiner Encoder oben links
    schlägt großes Modell unten rechts an Wirtschaftlichkeit.
    """
    df = load_betrieb().dropna(subset=["Parameter (Mio)", "Accuracy %"])
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(df):
        ax.scatter(df["Parameter (Mio)"], df["Accuracy %"], s=60, color="#5B8FF9")
        for _, r in df.iterrows():
            ax.annotate(r["Familie"], (r["Parameter (Mio)"], r["Accuracy %"]),
                        fontsize=8, xytext=(5, 3), textcoords="offset points")
        ax.set_xscale("log")
    ax.set_xlabel("Parameter (Mio, log)")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Accuracy vs. Modellgröße — Effizienz-Frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
            gr.Markdown(
                "**Per-Klasse-Report** — Precision/Recall/F1/Support je Intent, schwächste "
                "oben. Precision = wie oft stimmt ein *X-Tipp*; Recall = wie viele *echte X* "
                "gefunden. Klaffen sie auseinander, weißt du, ob das Modell zu *vorsichtig* "
                "(hohe Precision, niedriger Recall) oder zu *vorschnell* ist."
            )
            report_out = gr.Dataframe(
                value=class_report(default) if default else None,
                interactive=False, wrap=True,
            )

            def _refresh(key):
                """Event-Handler: Modell-Key rein → vier aktualisierte Tabellen raus."""
                return (confusion_pairs(key), weak_classes(key),
                        error_examples(key), class_report(key))

            model_dd.change(_refresh, inputs=model_dd,
                            outputs=[conf_out, weak_out, err_out, report_out])

        # Tab 5: Signifikanz — ist ein Accuracy-Unterschied echt oder Rauschen?
        with gr.Tab("Signifikanz"):
            gr.Markdown(
                "**Ist der Unterschied echt?** Unsere Spitze liegt bei 94,12 vs. 93,99 vs. … — "
                "bei 3076 Testbeispielen sind **0,13 Punkte ≈ 4 Datensätze**. Das kann Rauschen "
                "sein. Zwei Standardwerkzeuge sagen, ob eine Rangfolge trägt.\n\n"
                "**Bootstrap-KI:** je Modell die Accuracy mit 95%-Konfidenzintervall. "
                "Überlappen zwei Balken, ist der Unterschied mit Vorsicht zu genießen."
            )
            gr.Plot(significance_ci_fig())
            gr.Dataframe(value=significance_overview(), interactive=False, wrap=True)
            gr.Markdown(
                "**McNemar-Test** (paarweise): schaut nur auf die Beispiele, wo genau *einer* "
                "recht hat. `p < 0,05` → der Unterschied ist statistisch signifikant. "
                "Steht hier **nein**, sind die beiden Modelle praktisch gleichauf — egal, wer "
                "die dritte Nachkommastelle gewinnt."
            )
            gr.Dataframe(value=mcnemar_pairs(), interactive=False, wrap=True)

        # Tab 6: Kalibrierung — sagt das Modell ehrlich, wie sicher es ist?
        with gr.Tab("Kalibrierung"):
            gr.Markdown(
                "**Stimmt die Selbstsicherheit?** Wenn ein Modell zu 90 % sicher ist — trifft "
                "es dann in 90 % der Fälle zu? Das **Reliability-Diagramm** trägt die tatsächliche "
                "Accuracy gegen die Konfidenz auf; auf der Diagonale = perfekt kalibriert, "
                "darunter = zu selbstsicher. **ECE** fasst die Abweichung in einer Zahl.\n\n"
                "Geht nur für Modelle mit *echten Wahrscheinlichkeiten* (`score_type=proba`). "
                "Der Champion (LinearSVC) liefert nur **Margen** — nicht kalibrierbar ohne "
                "`CalibratedClassifierCV`. Genau diese Ehrlichkeit ist der Lerneffekt."
            )
            _scored = available_scores()
            _cal_default = next((k for k in _scored
                                 if _score_type(k) == "proba"), None)
            cal_dd = gr.Dropdown(choices=_scored, value=_cal_default, label="Modell (mit Scores)")
            cal_note = gr.Markdown(_calib_note(_cal_default))
            cal_plot = gr.Plot(reliability_fig(_cal_default) if _cal_default else None)

            def _refresh_cal(key):
                return _calib_note(key), reliability_fig(key)

            cal_dd.change(_refresh_cal, inputs=cal_dd, outputs=[cal_note, cal_plot])

        # Tab 7: Konfidenz & Abstention — die Deploy-Sicht.
        with gr.Tab("Konfidenz & Abstention"):
            gr.Markdown(
                "**Die Betriebssicht.** Zwei Hebel, die ein echtes System nutzt:\n"
                "- **Top-k**: bei 77 Intents zeigt man oft *drei* Vorschläge. Top-3 ist meist "
                "viel höher als Top-1 — die ehrlichere Zahl dafür, ob es dem Nutzer hilft.\n"
                "- **Risk-Coverage / Abstention**: die unsichersten Anfragen an einen Menschen "
                "weiterreichen. Die Kurve zeigt, wie die Accuracy auf dem Rest steigt.\n\n"
                "Beides ist rang-basiert und geht auch für den Margen-Champion."
            )
            _sc = available_scores()
            _sc_default = _sc[0] if _sc else None
            conf_dd = gr.Dropdown(choices=_sc, value=_sc_default, label="Modell (mit Scores)")
            topk_out = gr.Dataframe(
                value=topk_table(_sc_default) if _sc_default else None,
                interactive=False, wrap=True,
            )
            rc_plot = gr.Plot(risk_coverage_fig(_sc_default) if _sc_default else None)

            def _refresh_conf(key):
                return topk_table(key), risk_coverage_fig(key)

            conf_dd.change(_refresh_conf, inputs=conf_dd, outputs=[topk_out, rc_plot])

        # Tab 8: Betriebszahlen — Accuracy ist nicht alles (Kosten der Deploy-Entscheidung).
        with gr.Tab("Betriebszahlen"):
            gr.Markdown(
                "**Was kostet das Modell im Betrieb?** Für die Welches-nehmen-Frage zählen "
                "oft Parameter, Latenz und Größe mehr als die dritte Nachkommastelle Accuracy. "
                "Gemessen fair auf **einer** Maschine (siehe `betriebszahlen.py`).\n\n"
                "Die **Effizienz-Frontier** (Accuracy vs. Parameter, Log-Achse): oben-links = "
                "viel Leistung fürs Geld. Ein kleiner eingefrorener Encoder oben links schlägt "
                "ein großes Modell unten rechts wirtschaftlich."
            )
            _bt = load_betrieb()
            if len(_bt):
                gr.Plot(betrieb_fig())
                gr.Dataframe(value=_bt, interactive=False, wrap=True)
            else:
                gr.Markdown(
                    "_Noch keine Betriebszahlen — bitte einmal `python 06_synthese/"
                    "betriebszahlen.py` laufen lassen (misst Parameter, Latenz, Größe)._"
                )

    return app


if __name__ == "__main__":
    build_app().launch()
