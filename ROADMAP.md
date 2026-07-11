# Lernprojekt: Textklassifikation — von TF-IDF bis LLM

Einen Classifier für deutsche Nachrichtenartikel bauen — und dabei das Feld so
lernen, wie es an Universitäten gelehrt und in Teams gearbeitet wird.

**Datensatz:** 10kGNAD (`data/10kGNAD/`) — 10.273 Artikel der Zeitung
„Der Standard", 9 Kategorien: Etat, Inland, International, Kultur, Panorama,
Sport, Web, Wirtschaft, Wissenschaft. `train.csv` (9.245) / `test.csv` (1.028),
stratifiziert (in F1 nachgeprüft).

---

# Die Architektur

Dieses Projekt ist **keine Liste, sondern ein Gebäude** mit drei Ebenen:

```
┌───────────────────────── SYNTHESE ─────────────────────────┐
│   S1 Gesamtvergleich          S2 Mini-App (Deployment)     │
├──────────────────── STRATEGIE-TRACKS ──────────────────────┤
│  Track A        Track B        Track C        Track D      │
│  Gen 1          Gen 2          Gen 3a         Gen 3b       │
│  TF-IDF + ML    Embeddings     Finetuning     LLM-Prompt   │
│                                                            │
│  jeder Track nach demselben 4-Phasen-Schema:               │
│  P1 Vorbereiten → P2 Bauen → P3 Optimieren ⟳ → P4 Einordnen│
├─────────────────────── FUNDAMENT ──────────────────────────┤
│  F1 Daten & EDA      F2 Messwesen     F3 Experimentieren   │
│  F4 Theorie-Werkzeuge                 (Zyklus + Setup)     │
└────────────────────────────────────────────────────────────┘
         ════ dazu 3 QUERSCHNITTS-STRÄNGE (unten) ════
```

- **Fundament** — gilt für alle Tracks gleichermaßen. Kein „Modul unter
  anderen", sondern das Werkzeug, das jeder Track benutzt.
- **Vier Tracks** — die Strategien (Generationen). Parallel und gleichrangig:
  Man kann sie in beliebiger Reihenfolge gehen, sie messen sich alle am selben
  Fundament. Ihr **identisches Innenschema ist selbst Lernstoff**: Wer es
  zweimal durchlaufen hat, erkennt es im dritten Track wieder — *das* ist der
  Beruf, nicht das einzelne Modell.
- **Synthese** — erst sinnvoll, wenn mindestens zwei Tracks durch P3 sind.

**Was davon Standard ist — und was unsere Verpackung:** Die Vierteilung
selbst ist die im Feld übliche (Anthropic/OpenAI/Hugging Face nennen bei
„Text klassifizieren" genau diese vier Optionen). Die Fachbegriffe dafür:
Track A = **classical/traditional ML**, Track B = **feature extraction**
(frozen embeddings + classifier), Track C = **fine-tuning**, Track D =
**zero-/few-shot prompting**. „Feature extraction vs. fine-tuning" ist die
kanonische Unterscheidung der Transfer-Learning-Literatur. *Unsere* Begriffe
sind nur die Etiketten: „Track A–D", das P1–P4-Schema und die
Familien-Nummern — didaktische Verpackung, kein Fachvokabular. Im Gespräch
mit ML-Leuten die Fachbegriffe benutzen.

**Architektur ≠ Chronologie.** Dieses Dokument ist nach Wesen sortiert (was
gehört zu was), nicht nach Reihenfolge. Den empfohlenen Weg durchs Gebäude
zeigt der [Lernpfad](#lernpfad--der-empfohlene-weg-durchs-gebäude) am Ende.

**Vollständigkeit:** Die vier Tracks sind die vier *lebenden Modellfamilien* —
nicht eine Auswahl aus vielen. Das komplette Feld (alle 7 Familien inkl. der
historischen, alle Varianten wie RoBERTa/DistilBERT/BioBERT, Ensembles,
Seitenpfade fastText & CNN/LSTM) zeigt **`MODELL-LANDKARTE.md`** — inklusive
Einordnung, warum 30 Modellnamen aus Rankings auf 5 Varianten-Achsen
zusammenfallen und als „Modellwahl"-Stellschraube in P3 wohnen.

**Die drei Querschnitts-Stränge.** So bauen moderne Kurse (Made With ML,
fast.ai, Full Stack Deep Learning) Tiefe auf: Neben den Modellfamilien laufen
drei Handwerks-Stränge *quer* durch alle Etappen — sie werden im Fundament
eingeführt und in jeder Etappe erneut geübt, auf höherem Niveau:

| Strang | Konzepte (Auswahl) | eingeführt | vertieft in |
|---|---|---|---|
| **Daten-Handwerk** | EDA, Preprocessing, Datenqualität, Labeling & Annotator-Übereinstimmung, Imbalance, Augmentation | F1 | P1 jedes Tracks; E8 (Destillation = Labeling durch LLM) |
| **Experimentier-Handwerk** | Reproduzierbarkeit (Seeds, Versionierung), Experiment-Tracking, Config-Management, Hyperparameter-Suche, Ablation, Daten-/Modell-Tests | F3 | P3 jedes Tracks (jede Runde ist ein getracktes Experiment) |
| **Theorie-Werkzeuge** | Bias/Variance, Learning Curves, Loss & Gradientenabstieg, Regularisierung, Kalibrierung, Signifikanz | F4 | dort, wo sie erklären: Learning Curves in B·P3, Loss in C, Signifikanz in S1 |
| *(Modell-Familien)* | *(die Tracks selbst — siehe unten und `MODELL-LANDKARTE.md`)* | E1–E7 | P3/P4 |

Jedes Konzept der drei Stränge ist im **`KONZEPTE.md`**-Handbuch ausgearbeitet
(Was · Warum · Übung am 10kGNAD · Quelle) — die Antwort auf „wo ist EDA?"
und „wo sind Experiment-Setups?" ist also: als Strang überall, nachschlagbar
dort.

**Die zwei klassischen Karten stecken in der Architektur:**
**CRISP-DM** (das Prozessmodell) lebt in den Phasen — Business/Data
Understanding = F1, Data Preparation = P1, Modeling = P2+P3, Evaluation =
F2/S1, Deployment = S2. **Die drei Generationen** (die Lehrbuch-Taxonomie
nach Jurafsky & Martin) leben in den Tracks. Sie konkurrieren nicht: Die eine
Karte sagt, *wann* man ein Modell wählt, die andere, *welches*.

---

# Fortschritt

| Ebene | Einheit | Ordner | Status | Ergebnis (test-Acc) |
|---|---|---|---|---|
| Fundament | F1 Daten & EDA + Baseline | `stufe0_baseline/` | ✅ Baseline · ⬜ EDA | 16,3 % (Latte) |
| Fundament | F2 Messwesen | `stufe2_evaluation/` | ✅ Basis · ⬜ Val-Split | (Protokoll) |
| Fundament | F3 Experimentieren (Setup + Zyklus) | (Konzept, unten) | ✅ Zyklus · ⬜ Setup | — |
| Fundament | F4 Theorie-Werkzeuge | (Konzept, unten) | ⬜ | — |
| Track A | P1+P2 Bauen | `stufe1_bag_of_words/` | ✅ | 85,2 % |
| Track A | P3 Optimieren | " | ⬜ (0 Runden) | Ziel ~87–88 % |
| Track A | P4 Einordnen | " | ◐ (teils, s. Track A) | — |
| Track B | P1+P2 Bauen | `stufe3_embeddings/` | ✅ | 82,5 → 85,0 %\* |
| Track B | P3 Optimieren | " | ⬜ (0 Runden) | Ziel >85,2 % |
| Track B | P4 Einordnen | " | ◐ (Kontextfenster-Lektion) | — |
| Track C | P1–P4 | `stufe4_finetuning/` | ⬜ nächstes | erwartet ~90 % |
| Track D | P1–P4 | `stufe5_llm/` | ⬜ | erwartet 85–92 % |
| Synthese | S1 Vergleich | `stufe6_vergleich/` | ⬜ | — |
| Synthese | S2 Mini-App | `stufe7_app/` | ⬜ | — |

\* 82,5 % naiv (nur 128 Tokens gelesen); 85,0 % mit Chunking (siehe Track B).
Die Ordnernamen `stufe0`–`stufe3` stammen aus der alten Zählung und bleiben —
die Tabelle ist die Übersetzung.

**Wie man lernt:** Eine Einheit durcharbeiten → den **✓ Checkpoint** ehrlich
beantworten → erst weiter, wenn er sitzt. P3 ist dabei **kein Häkchen,
sondern ein Zähler**: Optimierungsrunden macht man so viele, bis das Plateau
erreicht ist — jede Runde protokolliert. Jede finale Testmessung wandert in
`results.json` für S1.

---

# FUNDAMENT

## F1 — Daten & EDA (+ Baseline)

- **Frage:** Was steckt in den Daten — bevor irgendein Modell sie sieht?
- **Inhalt:** **Explorative Datenanalyse (EDA)** ist in jedem ernsthaften
  Kurs die erste eigene Station (Made With ML widmet ihr eine ganze Lektion):
  Man verbringt Zeit mit den *Daten*, nicht mit Modellen. Danach die Latte
  legen: **Majority-Baseline** (eine Zeile, `DummyClassifier`).
- **EDA-Programm am 10kGNAD:**
  Klassenverteilung (train vs. test — stratifiziert?) · Textlängen-Verteilung
  pro Klasse (Histogramme! Median vs. Ausreißer) · häufigste/seltenste Wörter
  pro Klasse · **Duplikate & Fast-Duplikate** (gibt es Artikel doppelt? auch
  über den Split hinweg? = Leakage!) · Sonderzeichen/Encoding-Artefakte ·
  Vokabular-Überlappung zwischen den Klassen (Vorahnung der späteren
  Verwechslungen: Inland↔International) · ein Dutzend Artikel pro Klasse
  *lesen* — was würdest *du* als Mensch als Signal nutzen?
- **Konzepte:** EDA, Datenqualität, Train/Test-Split, Leakage durch
  Duplikate, Accuracy, unbalancierte Klassen, Majority- vs. Random-Baseline.
- **Quelle:** Made With ML „Exploratory Data Analysis"; Google *Rules of ML*
  Regel 1. Handwerkszeug: pandas + matplotlib.
- **Ergebnis:** Baseline **16,3 %** (Random: 11,3 %). EDA-Befunde: ⬜
  (nachholen — insbesondere Duplikat-Check und Längen-Histogramme; die
  Kontextfenster-Überraschung aus Track B wäre mit EDA *vorher* aufgefallen:
  97 % der Artikel länger als 128 Tokens!)
- **✓ Checkpoint:** Nenne drei Dinge, die dir nur die EDA sagen konnte —
  und welche Modell-Entscheidung jede davon beeinflusst. Und: Warum kann ein
  Spam-Filter mit 95 % Accuracy nutzlos sein?
- **Status:** ✅ Baseline · ⬜ EDA-Programm nachziehen (lohnt auch
  rückwirkend — es erklärt Befunde aus F2 und Track B)

## F2 — Messwesen

- **Frage:** Warum lügt Accuracy allein — und womit misst man ehrlich?
- **Inhalt:** Das Messbesteck, das **jeder** Track benutzt: Confusion Matrix,
  Precision (wie zuverlässig ein Tipp), Recall (wie vollständig gefunden),
  F1, Support, Macro- vs. Weighted-Average — und das **Drei-Sets-Prinzip**:
  **train** (Modell lernt) / **validation** (Varianten vergleichen) /
  **test** (bleibt zu; genau zwei Messungen pro Track: die naive P2-Messung
  und die finale nach dem P3-Plateau — dazwischen nie).
- **Quelle:** Jurafsky & Martin (SLP3), Kap. 4 (Evaluation); scikit-learn
  `classification_report`.
- **Ergebnis (am Track-A-Modell erhoben):** Sport fast perfekt (F1 0,975),
  Etat am schwächsten (Recall 0,63). Kernbefund: **Trennschärfe schlägt
  Größe** (kleine, scharfe Klasse Wissenschaft glänzt; größere, unscharfe
  Etat schwächelt). Panorama ist der „Abfluss". Viele „Fehler" sind
  Grenzfälle → praktische Obergrenze ~90 %.
- **✓ Checkpoint:** Ein Modell tippt „Kultur" nur bei absoluter Sicherheit —
  was ist hoch, Precision oder Recall, und warum?
- **Status:** ✅ Basis (Confusion Matrix als PNG). **⬜ Nachrüsten:**
  Val-Split (15 % aus train, stratifiziert, fester Seed) als
  `load_gnad_val()` in `gnad_utils.py` — Voraussetzung für jedes P3.

## F3 — Experimentieren: Setup + Optimierungszyklus

- **Frage:** Wie verbessert man ein Modell *systematisch* statt zu stochern —
  und wie sorgt man dafür, dass jedes Experiment zählt?
- **Teil 1: Das Experiment-Setup** (was Teams *vor* dem ersten Experiment
  einrichten — bei uns bisher die größte Lücke):
  - **Reproduzierbarkeit:** feste Seeds (`random_state`) überall; gleiche
    Daten + gleicher Code + gleiche Settings = gleiches Ergebnis. Ohne das
    ist jeder Vergleich Rauschen.
  - **Experiment-Tracking:** jede Messung wird automatisch protokolliert —
    Parameter, Metriken, Artefakte. Profi-Werkzeuge: **MLflow** oder
    Weights & Biases; für uns reicht anfangs eine strukturierte
    `experiments.csv` (Datum, Etappe, Config, Val-Score, Test-Score,
    Notiz) via `gnad_utils.log_experiment()` — der Umstieg auf MLflow ist
    dann eine Übung in Track C, wo Läufe teuer werden.
  - **Config statt Magie-Zahlen:** Parameter (min_df, C, max_length …)
    gehören an *eine* benannte Stelle (dict/Datei), nicht verstreut in Code.
  - **Ablation:** rückwärts testen — eine Zutat *weglassen* und messen, ob
    sie wirklich etwas beitrug. (Der Zwilling der „einen Änderung": vorwärts
    eine dazu, rückwärts eine weg.)
- **Teil 2: Der Optimierungszyklus** — der Arbeitszyklus, der in jedem P3
  läuft, in echten Projekten 80 % der Zeit. Strategieunabhängig; nur die
  Stellschrauben wechseln:

  1. **Messlineal fixieren** — Metrik + Val-Set. Am Testset wird nicht gedreht.
  2. **Ist-Stand messen** (Val).
  3. **Fehleranalyse** — falsch klassifizierte Val-Beispiele *lesen*, Muster
     suchen (das F2-Handwerk als Werkzeug).
  4. **Hypothese** — „verwechselt X mit Y, weil …"
  5. **Genau EINE Änderung**, die die Hypothese angreift.
  6. **Neu messen.** Besser → behalten, schlechter → verwerfen. Beides ins
     **Rundenprotokoll** (Hypothese · Änderung · Val-Effekt).
  7. **Wiederholen bis Plateau** (2–3 Runden ohne Verbesserung).
  8. **Erst jetzt: einmal aufs Testset** → `results.json`.

- **Eskalationsregel:** Erst wenn P3 im Plateau ist UND die Metrik das Ziel
  verfehlt, wechselt man den Track. Ein ausgereizter Gen-1-Track schlägt
  regelmäßig einen naiven Gen-3-Track — zum Bruchteil der Kosten.
- **Warum das Val-Set unverhandelbar ist:** Wer Varianten vergleicht und die
  beste behält, hat auf dem Vergleichsdatensatz „mitoptimiert" — die Zahl ist
  geschönt. > **Ehrliche Fußnote:** Genau das ist in Track A/B anfangs
  passiert (min_df-Übungen, Modellvergleich am Testset). Kein Beinbruch — die
  Effekte waren groß, die Rangfolge robust. Aber jedes P3 beginnt damit, den
  Val-Split einzuziehen und die alten Entscheidungen nachzuprüfen. Das ist
  selbst die Lektion: Der Fehler passiert jedem Team; können muss man ihn
  *bemerken und reparieren*.
- **Quelle:** Andrew Ng, *Machine Learning Yearning* (Error-Analysis-Kapitel);
  Google Rules of ML, Phase II.
- **✓ Checkpoint:** Warum genau EINE Änderung pro Runde — und was ist an
  „ich habe drei Sachen verbessert und es wurde besser" wertlos? Zusatz:
  Ein Kollege kann dein 87-%-Ergebnis nicht reproduzieren — nenne die drei
  wahrscheinlichsten Ursachen.
- **Status:** ✅ Zyklus gelesen · ⬜ Setup einrichten (Seeds prüfen,
  `log_experiment()` in `gnad_utils.py`, Config-Dict) · ⬜ angewandt
  (spätestens in Track A P3)

## F4 — Theorie-Werkzeuge (das Erklär-Besteck)

- **Frage:** *Warum* verhält sich ein Modell so — nicht nur *dass*?
- **Inhalt:** Vier Denkwerkzeuge, die jede Beobachtung im Projekt erklärbar
  machen. Kein Mathe-Kurs — jedes Werkzeug wird an unseren echten Zahlen
  eingeführt:
  - **Bias/Variance & Overfitting:** Die Lücke zwischen train- und
    val-Accuracy ist das Diagnose-Signal: große Lücke = Varianz (mehr Daten,
    mehr Regularisierung), kleine Lücke auf niedrigem Niveau = Bias
    (mächtigeres Modell, bessere Features).
  - **Learning Curves:** Accuracy gegen Trainingsmenge plotten (100/500/
    1.000/alle) — zeigt, ob mehr Daten helfen würden oder das Modell satt
    ist. (Die Few-Shot-Kurve aus B·P3 ist genau das!)
  - **Loss & Gradientenabstieg (Intuition):** Was LogReg und BERT beim
    „Lernen" wirklich tun: eine Fehlerfunktion bergab laufen. Erklärt, warum
    die Learning Rate in Track C die empfindlichste Schraube ist.
  - **Kalibrierung & Unsicherheit:** „80 % sicher" sollte in 80 % der Fälle
    stimmen — messbar (Reliability-Diagramm), reparierbar
    (`CalibratedClassifierCV`), und die Grundlage der S2-Kaskade.
- **Quelle:** Google ML Crash Course (Bias/Variance, Learning Curves); SLP3
  Kap. 4 (Loss); scikit-learn „Probability calibration".
- **✓ Checkpoint:** Track A zeigt train 99 % / val 87 % — Bias oder Varianz?
  Welche zwei Stellschrauben aus A·P3 adressieren genau das?
- **Status:** ⬜ (einführen parallel zu A·P3 — dort fallen die Zahlen an,
  die es zu erklären gilt)

---

# DAS TRACK-SCHEMA

Jeder Track durchläuft dieselben vier Phasen. Das Schema einmal verstanden,
liest sich jeder weitere Track von selbst:

| Phase | Frage | Charakter |
|---|---|---|
| **P1 Vorbereiten** | Wie wird Text zu Input für *diese* Strategie? | strategie-spezifische Data Preparation |
| **P2 Bauen** | Läuft ein Basismodell, und was leistet es naiv? | einmalig; endet mit erster Testmessung |
| **P3 Optimieren ⟳** | Wie viel steckt wirklich in der Strategie? | **offener Katalog + Zyklus (F3), n Runden, Protokoll** — kein Einschub, sondern die eigentliche Arbeit |
| **P4 Einordnen** | Wann gewinnt dieser Track, wann verliert er? | Grenzen, Kostenprofil, Einsatzempfehlung |

P3 hat pro Track einen **Stellschrauben-Katalog** — ein Menü, keine
Checkliste. Welche Schraube dran ist, entscheidet die Fehleranalyse, nicht
die Reihenfolge im Dokument.

---

# TRACK A — Gen 1: Klassisches ML (TF-IDF + LogReg/NB)

Ordner: `stufe1_bag_of_words/` · Wissen kommt **aus deinen gelabelten Daten**.
Quelle: Jurafsky & Martin (SLP3) Kap. 4, Anhang B; scikit-learn „Working with
Text Data".

**Die Modelle dieses Tracks:** Naive Bayes (`MultinomialNB`) · **Logistische
Regression** (unser Basismodell) · LinearSVC/SVM (oft 1–2 Punkte stärker) ·
kNN (selten sinnvoll auf TF-IDF) · Random Forest & XGBoost/LightGBM (nur bei
Text + Zusatzfeatures — auf reinem Text gewinnen die linearen).
**Die Features dazu:** rohe Counts · TF-IDF · Wort-/Zeichen-n-Gramme.
Jede Kombination Feature × Modell ist ein Kandidat — das Menü komplett in
`MODELL-LANDKARTE.md` Familie 2.

### A·P1 Vorbereiten — Feature Engineering
Text → gezählte/gewichtete Wörter: Tokenisierung, Vokabular, **TF-IDF**
(häufig im Dokument, selten im Korpus = wichtig), Sparse-Vektoren.
**Stolperfalle:** Vectorizer nur auf train fitten, test/val nur
transformieren — sonst Leakage.

### A·P2 Bauen ✅
Naive Bayes und Logistische Regression auf den Features.
**Ergebnis:** LogReg **85,2 %**, NB 84,3 % (mit rohen Counts; mit TF-IDF nur
67 % — Lektion: Feature-Aufbereitung und Modell müssen zusammenpassen). Der
Sprung 16 → 85 ist der größte des Projekts.
**✓ Checkpoint:** Warum bekommt „der" ein niedriges, „Bundesliga" ein hohes
TF-IDF-Gewicht? Was zeigen train- vs. test-Accuracy über Overfitting?

### A·P3 Optimieren ⟳ (0 Runden · Ziel ~87–88 %)
**Vorbereitung:** Val-Split (F2) — oder gleich 5-fache Kreuzvalidierung,
Gen-1-Training kostet Sekunden.

**Stellschrauben-Katalog** (Menü, per Fehleranalyse wählen):

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Features | Wort-n-Gramme `(1,2)`; Zeichen-n-Gramme; Stoppwörter/Lemmatisierung; `min_df`/`max_df`/`sublinear_tf` | Fehler an Wortformen, Komposita, Füllwörtern hängen |
| Modell | Regularisierung `C`; LinearSVC statt LogReg; `class_weight="balanced"` | Overfitting sichtbar; kleine Klassen untergehen (Etat-Recall 0,63!) |
| Konfidenz | Kalibrierung (`CalibratedClassifierCV`) | für die Kaskade in S2 gebraucht |
| Werkzeug | `Pipeline` + `GridSearchCV` | systematische Suche; fittet automatisch nur auf train-Folds |

**Übungsauftrag:** Mindestens 3 protokollierte Runden; hol +2 Punkte
gegenüber 85,2 % heraus.
**✓ Checkpoint:** Warum hebt `class_weight` den Etat-Recall, und was opfert
es? Warum hast du das am Val-Set entschieden?

### A·P4 Einordnen ◐
Gewinnt bei: viel gelabelten Daten, wortlastigen Themen (Nachrichtenrubriken!),
Budget ~0, Erklärbarkeits-Pflicht (Feature-Gewichte zeigbar). Verliert bei:
Synonymen/Paraphrasen, wenig Daten, kurzen Texten. Kostenprofil: Training
Sekunden, Inferenz ~0. **2026 immer noch ein respektierter Startpunkt.**
**✓ Checkpoint:** Nenne einen Anwendungsfall, in dem du 2026 bewusst bei
Gen 1 bleiben würdest — und die zwei Gründe dafür.

---

# TRACK B — Gen 2: Embeddings

Ordner: `stufe3_embeddings/` · Sprachverständnis kommt **aus einem
vortrainierten Modell**, Kategoriewissen aus deinen Daten.
Quelle: SLP3 Kap. 5; sbert.net.

**Die Modelle dieses Tracks:** Encoder (sentence-transformers:
`paraphrase-multilingual-MiniLM` · `distiluse` · E5/GTE-Familie ·
deutsche/größere Varianten — Wahl entlang Kontextlänge/Dimension/Sprache) ×
Kopf (**Logistische Regression** · kNN · SVM) · **SetFit** (Encoder
nachgeschärft) · alternativ kommerzielle Embedding-APIs (OpenAI, Cohere,
Voyage). Menü komplett in `MODELL-LANDKARTE.md` Familie 6.

### B·P1 Vorbereiten — Encoding
Text → Bedeutungsvektor durchs vortrainierte Netz (ähnliche Bedeutung = nahe
Punkte). **Die stille Falle: das Kontextfenster** (`max_seq_length`) — das
Basismodell liest nur 128 Tokens und schneidet 97 % der Artikel ab. Lange
Texte brauchen **Chunking** (stückeln + Vektoren mitteln). Embeddings cachen.

### B·P2 Bauen ✅
Logistische Regression auf den Embeddings.
**Ergebnis (ehrlich):** 82,5 % naiv — *unter* TF-IDF! Mit Chunking **85,0 %**
— gleichauf. Modellvergleich: Kontextlänge + Dimension schließen die Lücke;
deutsche Spezialisierung half weniger als Kontextlänge. **Lektion: stille
Grenzen vortrainierter Modelle kennen.** Echte Stärke woanders: bei 200
Trainingsbeispielen +21 Punkte gegen TF-IDF.
**✓ Checkpoint:** Warum schlagen Embeddings hier TF-IDF nicht — und in
welchem Szenario täten sie es? (150 kurze Support-Tickets: welcher Track?)

### B·P3 Optimieren ⟳ (0 Runden · Ziel >85,2 %)
**Vorbereitung:** die bisherigen Modellvergleiche am Val-Set nachprüfen (F3,
ehrliche Fußnote).

**Stellschrauben-Katalog:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Encoder | Modellwahl (Kontextlänge, Dimension, Sprache) | Texte abgeschnitten werden; Domäne speziell ist |
| Chunking | Mean- vs. Max-Pooling; Überlappung; nur Anfang+Ende | lange Texte; Signal am Artikelanfang |
| Vektoren | Normalisierung | Kopf kosinus-basiert arbeitet (kNN) |
| Kopf | LogReg-`C`; kNN; SVM — auf denselben Embeddings | billigster Vergleich im ganzen Projekt |
| Datenregime | **Few-Shot-Kurve**: 100/200/500/1.000/alle Beispiele, TF-IDF und Embeddings im selben Diagramm | du wissen willst, *wo* Gen 2 wirklich gewinnt |
| Königsdisziplin | **SetFit**: den Encoder selbst mit ein paar hundert Beispielen kontrastiv nachschärfen | der eingefrorene Encoder das Limit ist — konzeptionelle Brücke zu Track C |

**Übungsauftrag:** (1) Überhole TF-IDF, am Val-Set entschieden. (2) SetFit
mit 500 Beispielen vs. LogReg-Kopf mit 500 — Abstand und Kosten?
**✓ Checkpoint:** Zeichne die Few-Shot-Kurve und lies ab, ab welcher
Datenmenge sich welcher Track lohnt.

### B·P4 Einordnen ◐
Gewinnt bei: wenig gelabelten Daten, kurzen Texten, semantischer Ähnlichkeit
(Synonyme), Produktions-Setups mit Embedding-Cache. Verliert bei: langen
Dokumenten ohne Chunking-Sorgfalt, rein wortlastigen Aufgaben. Kostenprofil:
Encoding einmalig, danach ~0. **Das Standardrezept für günstige
Produktions-Classifier.**
**✓ Checkpoint:** Erkläre einem Kollegen in zwei Sätzen, warum „Embeddings
sind besser als TF-IDF" so pauschal falsch ist.

---

# TRACK C — Gen 3a: Finetuning (BERT & Co.)

Ordner: `stufe4_finetuning/` · Das **ganze** vortrainierte Netz passt sich an
deine Aufgabe an — nicht nur ein Kopf obendrauf (Track B fror ein, hier wird
aufgetaut). Quelle: Hugging Face NLP Course („Fine-tuning"). **Braucht
GPU/`mps`** — hier wird Apples GPU erstmals relevant.

**Die Modelle dieses Tracks:** `deepset/gbert-base` (unser Standard, deutsch)
— und die ganze BERT-Familie entlang der 5 Achsen: RoBERTa/DeBERTa/ELECTRA
(besser vortrainiert) · DistilBERT/ALBERT/MiniLM (kleiner/schneller) ·
XLM-R/mBERT (mehrsprachig) · BioBERT/FinBERT/… (Fachdomänen) ·
Longformer/BigBird (lange Texte). Menü komplett in `MODELL-LANDKARTE.md`
Familie 5 + Teil 2 (Achsen).

### C·P1 Vorbereiten — Tokenizer & Technik
Subword-Tokenisierung mit dem modelleigenen Tokenizer, `max_length` (die
Track-B-Lektion kehrt wieder — diesmal bewusst wählen), Padding, Batches,
Klassifikations-Head.

### C·P2 Bauen ⬜ (nächstes)
Erster Finetuning-Lauf mit Standardwerten.
**Erwartung:** ~90–91 % — Bestwert des Projekts (vgl. Papers-with-Code-
Leaderboard zu 10kGNAD), aber um Größenordnungen mehr Aufwand als Track A.
**✓ Checkpoint:** Unterschied „Classifier auf eingefrorenen Features" (B) vs.
„ganzes Netz finetunen" (C)? Warum braucht C eine GPU, A nicht?

### C·P3 Optimieren ⟳ (0 Runden)
**Stellschrauben-Katalog:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Training | Learning Rate + Warmup/Scheduler (empfindlichste Schraube!); Epochen mit **Early Stopping am Val-Set**; Batch Size | Training divergiert oder overfittet |
| Kapazität | Layer einfrieren vs. alles; **LoRA/PEFT** (nur kleine Zusatzmatrizen — der Industriestandard-Sparansatz) | GPU-Budget knapp; Overfitting bei kleinen Daten |
| Input | `max_length` 128/256/512 | lange Artikel; Speicher vs. Kontext |
| Modell | `gbert-base` vs. `xlm-roberta-base`; Klassengewichte im Loss | Sprachspezifik; kleine Klassen |

**Übungsauftrag:** Drei Läufe mit LR 1e-5/2e-5/5e-5 am Val-Set — einer wird
sichtbar schlechter, genau das ist die Lektion. Dann ein LoRA-Lauf: Wie viel
Accuracy kostet die Ersparnis?
**✓ Checkpoint:** Warum wird Early Stopping am Val-Set entschieden — und was
passierte, wenn man das Testset nähme?

### C·P4 Einordnen ⬜
Gewinnt bei: maximaler Accuracy-Anforderung, viel gelabelten Daten, stabilen
Kategorien, hohem Volumen (nach Training Inferenz billig). Verliert bei:
wenig Daten, häufig wechselnden Kategorien (jedes Mal neu trainieren),
Erklärbarkeits-Pflicht, ohne GPU. 
**✓ Checkpoint:** Dein Team hat 300 gelabelte Beispiele und die Kategorien
ändern sich quartalsweise — warum ist Track C hier die falsche Wahl?

---

# TRACK D — Gen 3b: LLM per Prompt

Ordner: `stufe5_llm/` · **Kein Training**: Sprach- und Weltwissen kommen aus
dem LLM, die Kategorien aus deiner Beschreibung. Quelle: Anthropic
Classification-Guide (docs.claude.com); HF LLM Course. Braucht
`ANTHROPIC_API_KEY`.

**Die Modelle dieses Tracks:** Claude **Haiku** (unser Standard: billig,
schnell) · Claude Sonnet (stärker, teurer — die Kostenkurve ist eine
P3-Übung) · prinzipiell jedes Instruct-LLM (GPT, Gemini, offene Modelle via
Ollama). Menü komplett in `MODELL-LANDKARTE.md` Familie 7.

**Rolle im Projekt: Gegenprobe zu Track C** — war das Finetuning seinen
Aufwand wert? Nachträglich gemessen ist das einwandfrei (gleiches Testset).
Wer streng nach Leitplanke 1 lernen will, zieht D vor C — beide Reihenfolgen
funktionieren; hier gewinnt die konzeptionelle Brücke B→C (eingefroren →
aufgetaut).

### D·P1 Vorbereiten — „Data Prep ist Prompt-Design"
Statt Features: Kategoriedefinitionen schreiben (das eigentliche „Training"!
„Etat" braucht z. B. den Hinweis *Medien-Rubrik des Standard*), strukturierten
JSON-Output erzwingen, Antwort-Validierung + Retries, festes **Val-Sample**
(~150 Artikel aus dem Val-Split) definieren — das Testsample bleibt zu.

### D·P2 Bauen ⬜
Zero-Shot mit Claude (Haiku), Stichprobe klassifizieren, messen.
**Erwartung:** 85–92 % ohne ein einziges Trainingsbeispiel.
**✓ Checkpoint:** Nenne zwei Situationen, in denen der Prompt-Ansatz ein
finegetuntes Modell schlägt, obwohl er „schwächer" misst.

### D·P3 Optimieren ⟳ (0 Runden)
Hier zeigt sich, dass F3 strategieunabhängig ist: **Stellrad ist Sprache
statt Hyperparameter**, der Zyklus bleibt exakt gleich — Fehler lesen →
Definition schärfen → EINE Änderung → am Val-Sample neu messen.

**Stellschrauben-Katalog:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Definitionen | Kategoriebeschreibungen fehleranalysegetrieben schärfen; explizite Grenzfall-Regeln („Politik + Ausland → International, außer …") | systematische Verwechslungen |
| Beispiele | Few-Shot: zufällige vs. gezielte Grenzfälle | Definitionen allein nicht reichen |
| Modell | Haiku vs. Sonnet — Kostenkurve mitzeichnen | Qualität vs. Preis |
| Robustheit | Konfidenz erfragen + „unsicher"-Option; Selbstkonsistenz (3× fragen, Mehrheit) | Grenzfälle; Qualität gegen 3× Kosten |

**Übungsauftrag:** Drei Prompt-Versionen, jede dokumentiert wie ein
Trainingslauf (Hypothese · Änderung · Val-Effekt). Dann hochrechnen: Kosten
der besten Version pro 1.000 Artikel in Cent und Sekunden — ab welchem
Volumen kippt es zu Track C?
**✓ Checkpoint:** Zeig an deinem Protokoll, dass Prompt-Iteration und
Hyperparameter-Tuning derselbe Workflow sind.

### D·P4 Einordnen ⬜
Gewinnt bei: null gelabelten Daten, häufig wechselnden Kategorien (Prompt
ändern statt neu trainieren), Bedarf an Begründungen, kleinen/mittleren
Volumen. Verliert bei: Millionen Texten (laufende Kosten), harten
Latenzanforderungen, Offline-Pflicht.
**✓ Checkpoint:** Ab welchem Monatsvolumen würdest du bei stabilen Kategorien
von D nach B/C „destillieren" (LLM labelt, kleines Modell lernt)?

---

# SYNTHESE

## S1 — Gesamtvergleich

- **Frage:** Welcher Track wann?
- **Inhalt:** Alle `results.json`-Werte in einer Tabelle, ergänzt um die
  P4-Dimensionen: gelabelte Daten nötig · Kosten pro 1.000 Texte ·
  Setup-Aufwand · Erklärbarkeit · „neue Kategorie hinzufügen".
- **Rückblick Business Understanding (schließt den CRISP-DM-Kreis):** War
  Accuracy die richtige Leitmetrik? F2 fand: Der Etat-Recall von 0,63
  verschwindet in der Gesamt-Accuracy spurlos. Alles zusätzlich mit
  **Macro-F1** nachrechnen — ändert sich die Rangfolge?
- **Messrauschen:** Bei n=1.028 ist das 95-%-KI grob **±2 Punkte**. 85,2 vs.
  85,0 ist „gleichauf" — und selbst 90 vs. 88 knapper, als es aussieht.
- **✓ Checkpoint:** Welche zwei Zeilen deiner Tabelle unterscheiden sich
  *nicht* signifikant — und was heißt das für die Modellwahl?
- **Status:** ⬜ (sinnvoll ab: A·P3 und B·P3 abgeschlossen, C/D gemessen)

## S2 — Mini-App: Hybrid-Classifier (Deployment)

- **Frage:** Wie sieht das in einem echten Produkt aus?
- **Inhalt:** **Kaskade** — der billige Track (A oder B, kalibrierte
  Konfidenz aus A·P3!) entscheidet klare Fälle; unsichere gehen an Track D,
  der zusätzlich begründet. Anzeige: Kategorie, Konfidenz, Entscheidungsweg
  (billig/teuer), Begründung.
- **Konzepte:** Konfidenz-Schwellen, Kosten-Qualitäts-Tradeoff in Aktion,
  Modell speichern/laden (`joblib`), UI (Streamlit oder CLI).
- **Warum das der richtige Abschluss ist:** Die App *ist* die Antwort auf S1
  — nicht ein Gewinner, sondern jeder Track für die Fälle, die er am
  besten/billigsten kann. So sehen Produktionssysteme aus.
- **Status:** ⬜

---

# Der vollständige Lernpfad — 12 Etappen durch das ganze Feld

Das komplette Curriculum. **Die Reihenfolge ist die Geschichte des Fachs:**
Jede Etappe behebt die Schwäche der vorherigen — wer den Pfad geht, erlebt
die Entwicklung von 1960 bis heute nach, statt sie erzählt zu bekommen.
Kern-Etappen bauen aufeinander auf; (K)ür-Etappen vertiefen, ohne dass etwas
Späteres von ihnen abhängt. Details je Etappe: Tracks oben bzw.
`MODELL-LANDKARTE.md`.

| # | Etappe | Familie | Art | Aufwand | Status |
|---|---|---|---|---|---|
| 0 | Fundament | — | Kern | 1–2 Tage | ✅ Baseline/F2 · ⬜ EDA, Val-Split, Experiment-Setup, F4 |
| 1 | Regeln & Keywords | 1 | Kern (klein) | 2 Std. | ⬜ |
| 2 | Klassisches ML | 2 | Kern | 1–2 Tage | ✅ Basis · ⬜ P3-Runden |
| 3 | Statische Embeddings & fastText | 3 | Kür | ½–1 Tag | ⬜ |
| 4 | Museum: CNN/LSTM | 4 | Kür | 1–2 Tage | ⬜ |
| 5 | Sentence-Embeddings | 6 | Kern | 1–2 Tage | ✅ Basis · ⬜ P3 + SetFit |
| 6 | Transformer-Finetuning | 5 | Kern | 1–2 Tage | ⬜ |
| 7 | LLM per Prompt | 7 | Kern | 1 Tag | ⬜ |
| 8 | Kombinationen | Meta | Kern | 1 Tag | ⬜ |
| 9 | Gesamtvergleich (S1) | — | Kern | ½ Tag | ⬜ |
| 10 | Mini-App (S2) | — | Kern | 1–2 Tage | ⬜ |
| 11 | Ausblick & angrenzende Felder | — | Kür | offen | ⬜ |

*Zur doppelten Nummerierung: „Familie" (1–7) ist die vollständige Taxonomie
aus `MODELL-LANDKARTE.md`; die „Gen"-Labels der Tracks (Gen 1/2/3a/3b) sind
die gröbere Lehrbuch-Taxonomie. Track B = Gen 2 = Familie 6 — gleiche Sache,
zwei Zoomstufen.*

### Die Etappen

**E0 — Fundament** *(F1 + F2 + F3 + F4)*: EDA (die Daten wirklich kennen —
Duplikate, Längen, Vokabular-Überlappung), Baseline legen (16,3 %),
Messwesen, Experiment-Setup (Seeds, Tracking-Log, Config) und
Optimierungszyklus, Theorie-Besteck. Offen: EDA-Programm, Val-Split +
`log_experiment()` in `gnad_utils.py`, F4. → *Schwäche, die bleibt: Wir
raten noch.*

**E1 — Regeln & Keywords** *(Familie 1)*: Einen Keyword-Classifier von Hand
bauen (30 Zeilen: Wortlisten pro Kategorie, meiste Treffer gewinnt).
**Übung:** Wie weit kommst du in 2 Stunden Regel-Schreiben? (Erwartung:
~50–60 % — deutlich über Baseline!) **✓ Checkpoint:** Beschreibe den Moment,
in dem dir das Regel-Schreiben zu mühsam wurde — das *ist* das Argument für
ML. → *Schwäche: skaliert nicht, jede Ausnahme eine neue Regel.*

**E2 — Klassisches ML** *(Familie 2 = Track A, P1–P4)*: Die Maschine lernt
die „Regeln" selbst aus gelabelten Daten. Basis ✅ (85,2 %), offen: P3-Runden
(+ die Frage aus E1 beantworten: Was hat GridSearch gefunden, das du per Hand
nie gefunden hättest?). → *Schwäche: „Bundesliga" und „Kicker" sind für
TF-IDF fremde Wörter — kein Bedeutungsverständnis.*

**E3 — Statische Embeddings & fastText** *(Familie 3, Kür)*: Wörter bekommen
Bedeutungsvektoren. **Übung 1:** Mit Word2Vec-Vektoren spielen (König −
Mann + Frau ≈ ?), Wortähnlichkeiten im 10kGNAD-Vokabular erkunden.
**Übung 2:** fastText-Classifier trainieren (eine CLI-Zeile!) — Accuracy vs.
Geschwindigkeit gegen Track A. **✓ Checkpoint:** Warum ist „Bank" ein
Problem für statische Vektoren? → *Schwäche: ein Vektor pro Wort, egal im
welchem Satz — kein Kontext.*

**E4 — Museum: CNN/LSTM** *(Familie 4, Kür)*: Die erste Deep-Learning-Welle
nachbauen — ein BiLSTM (oder Text-CNN) mit Word2Vec-Embeddings.
**✓ Checkpoint:** Du hast gespürt, warum der Transformer gewann (Padding-
Mühsal, langsames sequenzielles Training, Ergebnis zwischen A und B).
→ *Schwäche: liest Wort für Wort; Attention war der Ausweg — und wurde zum
Transformer.*

**E5 — Sentence-Embeddings** *(Familie 6 = Track B, P1–P4)*: Kontextuelle
Vektoren für ganze Texte aus vortrainierten Transformern. Basis ✅ (85,0 %),
offen: P3 inkl. Few-Shot-Kurve und SetFit (Brücke zu E6). → *Schwäche: der
Encoder bleibt eingefroren — er passt sich deiner Aufgabe nicht an.*

**E6 — Transformer-Finetuning** *(Familie 5 = Track C, P1–P4)*: Das ganze
Netz auftauen und anpassen. Erwartet ~90 %. Modellwahl über die 5
Varianten-Achsen (Landkarte Teil 2: RoBERTa/DistilBERT/BioBERT & Co. wohnen
alle hier). → *Schwäche: braucht viele Labels, GPU, und bei jeder
Kategorie-Änderung neu trainieren.*

**E7 — LLM per Prompt** *(Familie 7 = Track D, P1–P4)*: Kein Training —
Kategorien werden beschrieben. Gegenprobe zu E6. → *Schwäche: laufende
Kosten und Latenz bei großem Volumen.*

**E8 — Kombinationen** *(Meta-Familie)*: Die Schwächen gegeneinander
ausspielen: **Übung 1 (Ensemble):** A- und B-Modell per Soft-Voting —
schlägt es beide? **Übung 2 (Destillation):** LLM labelt 1.000 ungelabelte
Artikel, kleines Modell lernt daraus — wie nah kommt es dem LLM bei ~0
Kosten? **✓ Checkpoint:** Welche Kombination würdest du für 10 Mio.
Artikel/Monat bauen?

**E9 — Gesamtvergleich (S1):** Jetzt mit *allen* gemessenen Familien in
einer Tabelle (E1-Regeln bis E7-LLM!), Macro-F1-Rückblick, Messrauschen.

**E10 — Mini-App (S2):** Die Kaskade aus E8 wird Produkt.

**E11 — Ausblick** *(Kür, ohne 10kGNAD)*: Multi-Label & hierarchische
Klassifikation, Topic Modeling (BERTopic) für den Fall ohne Kategorien,
Active Learning, MLOps/Monitoring — jede davon ein eigenes kleines Projekt;
Startpunkte in `MODELL-LANDKARTE.md` Teil 6.

**Nächster Schritt:** ⬜ Val-Split (E0 abschließen) → dann wahlweise E1
(kleiner, erhellender Einschub) oder direkt A·P3 (E2 vollenden).

---

# Leitplanken (das, was Teams wirklich tun)

1. **Erst die Baseline, dann eskalieren.** Nie mit dem komplexesten Modell
   starten. (Google Rules of ML, Regel 1.)
2. **Der Test-Split bleibt zu, bis gemessen wird.** Varianten vergleicht man
   am Validation-Set; jede Entscheidung nach einem Blick auf Testdaten ist
   Leakage in Zeitlupe.
3. **Metrik vor Modell.** Erst festlegen, was „gut" heißt (Accuracy oder
   Macro-F1?), dann vergleichen.
4. **Fehleranalyse schlägt Modellwechsel.** Erst verstehen, *wo* es irrt,
   bevor man zum größeren Modell greift.
5. **Kosten und Änderbarkeit zählen, nicht nur Accuracy.** 2 Punkte für das
   100-fache an Rechenkosten ist oft ein schlechter Deal.
6. **Innere vor äußerer Schleife.** Erst den Track ausreizen (P3), dann den
   Track wechseln.
7. **Eine Änderung pro Messung.** Wer drei Schrauben gleichzeitig dreht,
   weiß nicht, welche gewirkt hat.

---

# Quellen (konsolidiert)

- **Jurafsky & Martin, „Speech and Language Processing" (SLP3)** — das
  Standardlehrbuch, kostenlos: `web.stanford.edu/~jurafsky/slp3` (Draft Jan
  2026, verifiziert). Track A → Kap. 4 + Anhang B; Track B → Kap. 5;
  Tracks C/D → Kap. 7; Tokenisierung → Kap. 2.
- **Stanford CS224N** — der kanonische Kurs: `web.stanford.edu/class/cs224n`.
- **Hugging Face NLP/LLM Course** — `huggingface.co/learn` (Track C/D).
- **Andrew Ng, „Machine Learning Yearning"** — frei; Vorlage für F3.
- **Google, „Rules of Machine Learning"** —
  `developers.google.com/machine-learning/guides/rules-of-ml`.
- **CRISP-DM** — Suchbegriff „Cross-industry standard process for data mining".
- **scikit-learn, „Working with Text Data"** — Track A.
- **sbert.net** + **SetFit** (`huggingface.co/docs/setfit`) — Track B.
- **Anthropic Classification-Guide** — `docs.claude.com` (Track D).
- **Papers with Code, „Text Classification on 10kGNAD"** — Leaderboard zum
  Einordnen (BERT-Bestwerte ~90 % = Track C).

---

# Setup

```bash
cd Klassifizierung
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Track D zusätzlich: `export ANTHROPIC_API_KEY=sk-ant-...`
Geteilt: `gnad_utils.py` (Daten laden, Val-Split, Ergebnisse speichern),
`results.json` (Sammelblatt für S1).

---

# Glossar

**Accuracy** – Anteil korrekter Vorhersagen. **Baseline** – trivialer
Vergleichswert. **CRISP-DM** – Standard-Prozessmodell für ML-Projekte.
**Data Leakage** – Testinformation sickert ins Training; häufigster
Anfängerfehler. **Early Stopping** – Training beenden, sobald die
Val-Leistung nicht mehr steigt. **Embedding** – Text als Bedeutungsvektor.
**Epoche** – ein kompletter Durchlauf durch die Trainingsdaten.
**F1** – harmonisches Mittel aus Precision und Recall. **Few-Shot** –
Beispiele im Prompt statt Training. **Finetuning** – ein vortrainiertes Netz
komplett nachtrainieren. **Grid Search** – systematisches Durchprobieren von
Hyperparameter-Kombinationen. **Ground Truth** – die „wahren" Labels.
**Hyperparameter** – Einstellungen, die *du* wählst, statt dass das Modell
sie lernt (`C`, Learning Rate). **Kalibrierung** – Konfidenzen so
korrigieren, dass „80 % sicher" auch in 80 % der Fälle stimmt.
**Kontextfenster** – wie viele Tokens ein Modell maximal liest.
**Kreuzvalidierung** – k-faches Rotieren des Val-Splits. **LoRA/PEFT** –
Finetuning light: nur kleine Zusatzmatrizen trainieren. **n-Gramm** – Folge
von n Wörtern/Zeichen als Feature. **Overfitting** – Trainingsdaten
auswendig gelernt statt Muster. **Precision/Recall** – Zuverlässigkeit eines
Tipps / Vollständigkeit des Findens. **SetFit** – Embedding-Modell mit
wenigen Beispielen kontrastiv nachschärfen. **Stratifizierung** – Split, der
Klassenanteile gleich hält. **TF-IDF** – Wortgewichtung: häufig im Dokument,
selten im Korpus = wichtig. **Transfer Learning** – fremdes Vorwissen
wiederverwenden. **Validation-Set** – drittes Set, an dem die innere
Schleife vergleicht; hält das Testset sauber. **Zero-Shot** – klassifizieren
nur aus Kategoriebeschreibungen.
