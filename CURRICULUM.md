# CURRICULUM — Lernprojekt Textklassifikation

**Ein reines Lernprojekt.** Es gibt kein Ziel-Artefakt — keine App, kein
„fertiger Classifier". Das Ziel ist Können:

**Die vier Lernziele**

1. **Jedes Konzept gesehen haben** — alle Modellfamilien, alle
   Handwerks-Konzepte (EDA bis Signifikanz), alle Theorie-Werkzeuge.
2. **Alles selbst ausprobiert haben** — jede Familie mindestens einmal
   gebaut, optimiert (P3!) und eingeordnet; ehrlich gemessen.
3. **An verschiedenen Datensätzen geübt haben** — was nur an einem
   Datensatz klappt, ist nicht gelernt (→ Datensatz-Rotation, Kap. 10).
4. **Transfer:** Bei jedem künftigen Klassifikations-Problem schnell und
   begründet zum passenden Ansatz kommen. Das ist der eigentliche Ertrag.

Deliverables wie die Mini-App sind **Übungen** (dort: für Deployment),
keine Ziele. Die vollständige Modell-Landkarte des Felds ist Teil dieses
Dokuments (Kap. 2). Handwerks-Handbuch: `KONZEPTE.md` · Projektstand &
aktueller Datensatz: `README.md`.

---

## 1 · Die Architektur

```
┌───────────────────────── SYNTHESE ─────────────────────────┐
│   S1 Gesamtvergleich          S2 Mini-App (Deployment)     │
├──────────────────── STRATEGIE-TRACKS ──────────────────────┤
│  Track A        Track B        Track C        Track D      │
│  Klassisches ML Embeddings     Finetuning     LLM-Prompt   │
│                                                            │
│  jeder Track:  P1 Vorbereiten → P2 Bauen →                 │
│                P3 Optimieren ⟳ → P4 Einordnen              │
├─────────────────────── FUNDAMENT ──────────────────────────┤
│  F1 Daten & EDA   F2 Messwesen   F3 Experimentieren        │
│  F4 Theorie-Werkzeuge                                      │
└────────────────────────────────────────────────────────────┘
          + 3 Querschnitts-Stränge durch alle Ebenen
```

**Die drei Ebenen:**

- **Fundament** — Werkzeug für alle Tracks, kein „Modul unter anderen".
- **Tracks** — die vier lebenden Modellfamilien. Parallel, gleichrangig,
  identisches Innenschema (das Schema selbst ist Lernstoff).
- **Synthese** — sinnvoll, sobald mindestens zwei Tracks durch P3 sind.

**Standard-Begriffe vs. Projekt-Etiketten:**

| Unser Etikett | Fachbegriff (so heißt es draußen) |
|---|---|
| Track A | classical / traditional ML |
| Track B | feature extraction (frozen embeddings + classifier) |
| Track C | fine-tuning (pretrained transformer) |
| Track D | zero-/few-shot prompting |
| P1–P4, „Track", „Familie" | eigene Didaktik-Verpackung — draußen nicht verwenden |

**Die zwei klassischen Karten** stecken in der Architektur:

| Karte | beantwortet | lebt hier |
|---|---|---|
| CRISP-DM (Prozess) | *Wann* wähle ich ein Modell? | F1 = Data Understanding · P1 = Data Prep · P2/P3 = Modeling · F2/S1 = Evaluation · S2 = Deployment |
| Drei Generationen (Lehrbuch) | *Welches* Modell? | die Tracks (A=Gen 1, B=Gen 2, C/D=Gen 3) |

**Die drei Querschnitts-Stränge** (werden im Fundament eingeführt, in jeder
Etappe erneut geübt — Details je Konzept in `KONZEPTE.md`):

| Strang | Konzepte | eingeführt |
|---|---|---|
| 🔍 Daten-Handwerk | EDA · Cleaning · Labeling & Kappa · Imbalance · Augmentation · Data-centric AI | F1 |
| 🧪 Experimentier-Handwerk | Seeds · Tracking · Config · Hyperparameter-Suche · Ablation · Daten-/Modell-Tests | F3 |
| 📐 Theorie-Werkzeuge | Bias/Variance · Learning Curves · Loss & LR · Kalibrierung · Signifikanz · XAI | F4 |

---

## 2 · Die Modell-Landkarte: 7 Familien × 5 Achsen

**Kernthese:** „Best Models"-Artikel listen 20–30 Modellnamen. Das sind
keine 30 Wege — es sind **7 Familien × 5 Varianten-Achsen**. Wer beides
kennt, sortiert jedes neue Modell (auch das von 2027) in fünf Minuten ein.

### Die 7 Familien

Jede Familie beantwortet dieselbe Frage anders: *Wie wird aus Text eine
Vorhersage?*

| # | Familie | seit | Kernidee | Status heute | im Curriculum |
|---|---|---|---|---|---|
| 1 | Regeln & Heuristiken | immer | Keywords, Regex, Wortlisten von Hand | lebt als Vorfilter in Kaskaden | E1 |
| 2 | Klassisches ML | ~1960er | Hand-Features (TF-IDF) + statistisches Modell | quicklebendig: billig, erklärbar | **Track A** (Kap. 5) |
| 3 | Statische Wort-Embeddings | 2013 | ein fester Bedeutungsvektor pro Wort (Word2Vec, GloVe, fastText) | Geschichte — außer fastText als Speed-Classifier | E3 (Kür) |
| 4 | Neuronale Sequenzmodelle | 2014 | Netz liest Wort für Wort (CNN, LSTM, BiLSTM, Attention) | verdrängt vom Transformer | E4 „Museum" (Kür) |
| 5 | Transformer **finetunen** | 2018 | vortrainiertes Netz **auftauen** und umtrainieren | Accuracy-Spitze | **Track C** (Kap. 7) |
| 6 | Sentence-Embeddings + Kopf | 2019 | vortrainiertes Netz **einfrieren**, nur seine Vektoren nutzen | Preis-Leistungs-Standard | **Track B** (Kap. 6) |
| 7 | LLM benutzen | 2020+ | Aufgabe in Sprache beschreiben statt trainieren | flexibelster Ansatz | **Track D** (Kap. 8) |

**Wichtig — die Familien 5, 6 und 7 teilen die Architektur** (alle drei
nutzen vortrainierte Transformer). Was sie unterscheidet, ist die
**Eingriffstiefe** am selben Objekt:

| Familie | Eingriff | es ändern sich | Track |
|---|---|---|---|
| 6 | **einfrieren** — nur Vektoren abgreifen, kleiner Kopf lernt | nichts am Netz | B |
| 5 | **auftauen** — das ganze Netz nachtrainieren | die Gewichte | C |
| 7 | **prompten** — Aufgabe beschreiben | nur der Input | D |

Deshalb sind B und C verschiedene Familien, obwohl beide „Transformer" sind
— und deshalb ist die 7er-Taxonomie die feinere Zoomstufe der groben
„drei Generationen" (die B und C/D nur als Gen 2/3 unterscheidet).
Chronologie-Fußnote: Familie 6 entstand *nach* 5 (SBERT 2019); im
Curriculum kommt B vor C, weil konzeptionell einfacher. Prompting (7) und
LLM-Finetuning sind **Geschwister, nicht Stufen** — Finetuning eines LLM
ist konzeptionell Familie 5 mit größerem Modell.

### Die 5 Varianten-Achsen

Jeder konkrete Modellname ist ein Punkt in diesem Raum — die Achsen sind
orthogonal und kombinieren frei:

| Achse | Frage | Beispiele |
|---|---|---|
| 1 Architektur-Optimierung | besser vortrainiert? | BERT → RoBERTa → DeBERTa → ELECTRA, XLNet |
| 2 Größe/Kompression | kleiner & schneller? | DistilBERT (97 % Leistung, 60 % Größe), TinyBERT, ALBERT, MiniLM, MobileBERT |
| 3 Sprache | welche Sprache(n)? | sprachspezifische BERTs (gbert, camembert) · mBERT · XLM-R |
| 4 Domäne | Fach-Vortraining? | BioBERT, FinBERT, LegalBERT, SciBERT, ClinicalBERT |
| 5 Kontextlänge | wie lange Texte? | 512 Tokens Standard → Longformer, BigBird, ModernBERT |

### Einsortieren üben — externe Modelllisten auf die Landkarte legen

| taucht in Rankings auf als … | Einsortierung |
|---|---|
| SVM, Naive Bayes, LogReg, Random Forest, XGBoost | Familie 2 → Track-A-Menü |
| Word2Vec, GloVe, fastText | Familie 3 → E3 |
| CNN, RNN, LSTM, GRU, BiLSTM | Familie 4 → E4 |
| BERT, RoBERTa, DeBERTa, XLNet, ELECTRA | Familie 5, Achse 1 → Track-C-Menü |
| DistilBERT, TinyBERT, ALBERT, MobileBERT | Familie 5, Achse 2 → Track-C-Menü |
| mBERT, XLM-R · BioBERT, FinBERT … · Longformer | Familie 5, Achsen 3/4/5 → Track-C-Menü |
| SBERT, Universal Sentence Encoder, Embedding-APIs, SetFit | Familie 6 → Track-B-Menü |
| GPT-x, Claude & Co., Prompt Engineering, In-Context Learning | Familie 7 → Track D |
| Ensembles, Voting, Stacking | Meta-Familie → E8 |
| Metriken, Cross-Validation, Tuning, Deployment | keine Modelle — Handwerk: F2/F3/S2 |

### Angrenzende Felder (bewusst außerhalb — je ein eigenes Lernprojekt)

Multi-Label- & hierarchische Klassifikation · Extreme Classification
(Millionen Labels) · Sequenz-Labeling (NER) · Clustering/Topic Modeling
(wenn Kategorien erst gefunden werden müssen: LDA, BERTopic) · Multimodal
(Text+Bild/Audio) · MLOps im Großen (Drift, CI/CD, Active Learning). → E11.

---

## 3 · Fundament

### F1 — Daten & EDA (+ Baseline)

**Frage:** Was steckt in den Daten — bevor irgendein Modell sie sieht?

**EDA-Checkliste** (für jeden neuen Datensatz):

- [ ] Klassenverteilung (train vs. test — stratifiziert?)
- [ ] Textlängen-Verteilung pro Klasse (Histogramm; Median vs. Ausreißer)
- [ ] Häufigste/seltenste Wörter pro Klasse
- [ ] Duplikate & Fast-Duplikate — auch über den Split hinweg (= Leakage!)
- [ ] Encoding-Artefakte, Boilerplate, Muster, die das Label verraten
- [ ] Vokabular-Überlappung zwischen Klassen (Vorahnung der Verwechslungen)
- [ ] Ein Dutzend Beispiele pro Klasse *lesen* — was nutzt ein Mensch als Signal?

**Baseline danach:** Majority-Klasse tippen (`DummyClassifier`, eine Zeile) —
die Latte, die jedes Modell schlagen muss.

| | |
|---|---|
| Konzepte | EDA, Datenqualität, Split, Leakage, unbalancierte Klassen, Majority-/Random-Baseline |
| Werkzeug | pandas + matplotlib |
| Quelle | Made With ML „EDA"; Google Rules of ML, Regel 1 |
| ✓ Checkpoint | Drei Dinge nennen, die nur die EDA zeigen konnte — und welche Modell-Entscheidung jedes beeinflusst. Warum kann 95 % Accuracy nutzlos sein? |

### F2 — Messwesen

**Frage:** Warum lügt Accuracy allein — und womit misst man ehrlich?

| Werkzeug | zeigt |
|---|---|
| Confusion Matrix | *welche* Klasse mit *welcher* verwechselt wird |
| Precision (pro Klasse) | wie zuverlässig ein Tipp ist |
| Recall (pro Klasse) | wie vollständig eine Klasse gefunden wird |
| F1 / Macro vs. Weighted | Kompromiss; Macro schützt kleine Klassen |
| Fehleranalyse (Beispiele lesen) | *warum* — die Grundlage jedes P3 |

**Das Drei-Sets-Prinzip:**

| Set | Zweck | Regel |
|---|---|---|
| train | Modell lernt | — |
| validation | Varianten vergleichen (innere Schleife) | jede Tuning-Entscheidung hier |
| test | ehrliche Endmessung | zu; genau 2 Messungen pro Track (naive P2, finale P3) |

| | |
|---|---|
| Quelle | SLP3 Kap. 4; scikit-learn `classification_report` |
| ✓ Checkpoint | Modell tippt Klasse X nur bei absoluter Sicherheit — was ist hoch, Precision oder Recall, und warum? |

### F3 — Experimentieren: Setup + Optimierungszyklus

**Frage:** Wie verbessert man systematisch statt zu stochern — und wie zählt
jedes Experiment?

**Teil 1 — das Setup** (vor dem ersten Experiment einrichten):

| Baustein | bedeutet | Minimal-Variante |
|---|---|---|
| Reproduzierbarkeit | Seeds überall; Code/Daten/Umgebung fixiert | `random_state`, requirements mit Versionen |
| Experiment-Tracking | jeder Lauf protokolliert (Config, Metriken) | `experiments.csv` via Helper; Upgrade: MLflow |
| Config-Management | alle Parameter an *einer* benannten Stelle | Config-Dict oben im Script |
| Ablation | Zutat *weglassen* → trug sie wirklich bei? | Tabelle „ohne X → Val-Score" |

**Teil 2 — der Zyklus** (läuft in jedem P3; nur die Stellschrauben wechseln):

1. Messlineal fixieren (Metrik + Val-Set) — am Testset wird nicht gedreht
2. Ist-Stand messen (Val)
3. Fehleranalyse — falsch klassifizierte Val-Beispiele *lesen*
4. Hypothese — „verwechselt X mit Y, weil …"
5. Genau **eine** Änderung, die die Hypothese angreift
6. Neu messen; behalten oder verwerfen — beides ins Rundenprotokoll
7. Wiederholen bis Plateau (2–3 Runden ohne Verbesserung)
8. Erst jetzt: einmal aufs Testset → Ergebnis-Datei

**Eskalationsregel:** Track wechseln erst bei Plateau UND verfehltem Ziel —
ein ausgereizter einfacher Track schlägt oft einen naiven großen.

| | |
|---|---|
| Quelle | Ng, *ML Yearning*; Google Rules of ML Phase II; Made With ML „Tracking/Reproducibility" |
| ✓ Checkpoint | Warum EINE Änderung pro Runde? Kollege kann dein Ergebnis nicht reproduzieren — die drei wahrscheinlichsten Ursachen? |

### F4 — Theorie-Werkzeuge

**Frage:** *Warum* verhält sich ein Modell so — nicht nur *dass*?

| Werkzeug | Signal | Konsequenz |
|---|---|---|
| Bias/Variance | Lücke train- vs. val-Score | große Lücke → Regularisierung/mehr Daten; kleine Lücke, niedrig → mächtigeres Modell/Features |
| Learning Curves | Score vs. Trainingsmenge | zeigt, ob mehr Labeln sich lohnt |
| Loss & Gradientenabstieg | Lernen = Fehlerfunktion bergab | erklärt Learning-Rate-Empfindlichkeit (Track C) |
| Kalibrierung | „80 % sicher" stimmt in 80 %? | Reliability-Diagramm; Basis der S2-Kaskade |

| | |
|---|---|
| Quelle | Google ML Crash Course; SLP3 Kap. 4; scikit-learn „Calibration" |
| ✓ Checkpoint | train 99 % / val 87 % — Bias oder Varianz? Welche zwei P3-Schrauben adressieren genau das? |

---

## 4 · Das Track-Schema

| Phase | Frage | Charakter |
|---|---|---|
| **P1 Vorbereiten** | Wie wird Text zu Input für *diese* Strategie? | strategie-spezifische Data Prep |
| **P2 Bauen** | Was leistet ein Basismodell naiv? | einmalig; endet mit erster Testmessung |
| **P3 Optimieren ⟳** | Wie viel steckt wirklich drin? | Stellschrauben-Katalog + F3-Zyklus, n Runden, Protokoll |
| **P4 Einordnen** | Wann gewinnt dieser Track, wann verliert er? | Grenzen, Kostenprofil, Einsatzempfehlung |

Der P3-Katalog ist ein **Menü, keine Checkliste** — welche Schraube dran
ist, entscheidet die Fehleranalyse.

---

## 5 · Track A — Klassisches ML

| Steckbrief | |
|---|---|
| Fachbegriff | classical / traditional ML |
| Wissensquelle | deine gelabelten Daten |
| Modelle | siehe Modell-Menü unten |
| Features | Counts · TF-IDF · Wort-/Zeichen-n-Gramme |
| Werkzeug | scikit-learn |
| Quellen | SLP3 Kap. 4 + Anhang B; scikit-learn „Working with Text Data" |

**Das Modell-Menü.** Auf hochdimensionalen, spärlichen TF-IDF-Features gilt
die Faustregel: **linear gewinnt** — der Rest kostet meist Zeit und Punkte:

| Klassifikator | Modellfamilie | auf TF-IDF-Text | probieren? |
|---|---|---|---|
| **Logistische Regression** | linear, probabilistisch | stark; liefert Wahrscheinlichkeiten (→ Kalibrierung, Kaskade) | ✅ Standard |
| **LinearSVC** | linear, Margin-basiert | oft 1–2 Punkte stärker; keine Wahrscheinlichkeiten | ✅ Standard |
| SGDClassifier | linear, gradient-trainiert | ~gleichwertig; skaliert auf Riesen-Daten; viele Loss-/Penalty-Optionen | bei sehr großen Daten |
| MultinomialNB | Naive Bayes | der Lehrbuch-Textklassifikator; blitzschnelle Baseline; mag rohe Counts | ✅ als Baseline |
| **ComplementNB** | Naive Bayes | die NB-Variante für **unbalancierte** Klassen — oft besser als MultinomialNB | ✅ bei Klassen-Unwucht |
| kNN | abstandsbasiert | funktioniert (Cosinus), aber langsam und mittelmäßig bei vielen Klassen | Lehr-Experiment |
| RandomForest / XGBoost / LightGBM | Baum-Ensembles | auf spärlichem Text meist schwächer als linear und langsamer; stark erst bei Text **+ Zusatzfeatures** | nur bei Mischfeatures |

**P1 Vorbereiten:** Tokenisierung → Vokabular → TF-IDF-Gewichtung →
Sparse-Vektoren. ⚠️ Vectorizer nur auf train fitten (sonst Leakage).

**P2 Bauen:** NB und LogReg trainieren, erste Testmessung. Lektion:
Feature-Aufbereitung und Modell müssen zusammenpassen (NB mag rohe Counts).
*✓ Warum bekommt ein Füllwort niedriges, ein Fachwort hohes TF-IDF-Gewicht?*

**P3 Optimieren — Stellschrauben:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Features | n-Gramme (1,2) · Zeichen-n-Gramme · Stoppwörter/Lemma · min_df/max_df/sublinear_tf | Fehler an Wortformen/Komposita hängen |
| Modell | `C` · LinearSVC · `class_weight="balanced"` | Overfitting; kleine Klassen gehen unter |
| Konfidenz | Kalibrierung (`CalibratedClassifierCV`) | Kaskade (S2) braucht ehrliche Konfidenzen |
| Werkzeug | `Pipeline` + GridSearchCV / CV | systematisch statt stochern |

*✓ Warum hebt class_weight den Recall kleiner Klassen, und was opfert es?*

**P4 Einordnen:**

| Gewinnt bei | Verliert bei |
|---|---|
| viel gelabelten Daten · wortlastigen Themen · Budget ~0 · Erklärbarkeits-Pflicht | Synonymen/Paraphrasen · wenig Daten · kurzen Texten |

→ **Schwäche, die zu Track B führt:** kennt keine Bedeutung — Synonyme sind
fremde Wörter.

---

## 6 · Track B — Embeddings

| Steckbrief | |
|---|---|
| Fachbegriff | feature extraction (frozen embeddings + classifier) |
| Wissensquelle | Sprachverständnis: vortrainiertes Modell · Kategorien: deine Daten |
| Modelle | siehe Modell-Menü unten (Encoder × Kopf) |
| Werkzeug | sentence-transformers, scikit-learn |
| Quellen | SLP3 Kap. 5; sbert.net; SetFit-Doku |

**Das Modell-Menü.** Zwei unabhängige Entscheidungen — Encoder und Kopf:

| Baustein | Optionen | Einschätzung | probieren? |
|---|---|---|---|
| Encoder klein | MiniLM-Familie (z. B. `all-MiniLM`, `paraphrase-multilingual-MiniLM`) | schnell, guter Start; ⚠️ kurzes Kontextfenster | ✅ Start |
| Encoder stark | **E5 / GTE / BGE-Familien**, größere SBERT-Modelle | mehr Kontext + Dimension; Retrieval-optimiert — führen die Embedding-Leaderboards (MTEB) | ✅ P3 |
| Encoder API | OpenAI / Cohere / Voyage Embeddings | stark, kein Setup; laufende Kosten, Daten gehen raus | optional |
| Kopf | **LogReg** · LinearSVC · **kNN (Cosinus)** · kleines MLP | LogReg Standard; kNN glänzt bei vielen Klassen × wenig Beispielen pro Klasse | ✅ LogReg, dann vergleichen |
| Nachschärfen | **SetFit** (kontrastives Finetuning des Encoders mit wenigen Labels) | oft +3–8 Punkte im Low-Data-Regime; Brücke zu Track C | ✅ Königsdisziplin |

**P1 Vorbereiten:** Text → Bedeutungsvektor. ⚠️ **Kontextfenster**
(`max_seq_length`) prüfen — lange Texte brauchen Chunking (stückeln +
mitteln). Embeddings cachen.

**P2 Bauen:** Kopf auf eingefrorenen Embeddings, Testmessung. Typische
Lektion: naive Nutzung unterschätzt den Ansatz (Kontextfenster!); echte
Stärke im Low-Data-Regime.
*✓ In welchem Szenario schlagen Embeddings Bag-of-Words deutlich — und warum?*

**P3 Optimieren — Stellschrauben:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Encoder | Modellwahl: Kontextlänge · Dimension · Sprache | Texte abgeschnitten; Domäne speziell |
| Chunking | Mean-/Max-Pooling · Überlappung · Anfang+Ende | lange Texte |
| Kopf | LogReg-`C` · kNN · SVM | billigster Vergleich überhaupt |
| Datenregime | **Few-Shot-Kurve** (Score vs. Trainingsmenge, beide Familien) | wissen, *wo* Gen 2 gewinnt |
| Königsdisziplin | **SetFit** — Encoder kontrastiv nachschärfen | der eingefrorene Encoder limitiert (Brücke zu C) |

*✓ Few-Shot-Kurve zeichnen: ab welcher Datenmenge lohnt welcher Track?*

**P4 Einordnen:**

| Gewinnt bei | Verliert bei |
|---|---|
| wenig Labels · kurzen Texten · Semantik/Synonymen · Embedding-Cache-Setups | langen Dokumenten ohne Chunking · rein wortlastigen Aufgaben |

→ **Schwäche, die zu Track C führt:** der Encoder bleibt eingefroren.

---

## 7 · Track C — Finetuning

| Steckbrief | |
|---|---|
| Fachbegriff | fine-tuning (pretrained transformer) |
| Wissensquelle | vortrainiertes Netz, komplett an die Aufgabe angepasst |
| Modelle | siehe Modell-Menü unten |
| Werkzeug | Hugging Face transformers · **GPU nötig** |
| Quellen | HF NLP Course „Fine-tuning"; die 5 Achsen: Kap. 2 |

**Das Modell-Menü** (alles Encoder-Transformer; Auswahl über die 5 Achsen):

| Modell | Achse | Einschätzung | probieren? |
|---|---|---|---|
| BERT-base | Referenz | der Klassiker; heute selten erste Wahl | Lehr-Referenz |
| **RoBERTa / DeBERTa-v3** | besser vortrainiert | Standard-Empfehlung für Maximal-Accuracy | ✅ |
| **DistilBERT / MiniLM** | Kompression | ~97 % der Leistung, ~40 % schneller — Produktions-Favorit | ✅ Vergleichslauf |
| **ModernBERT** | Neuauflage (2024) | schneller + langes Kontextfenster; der moderne Default | ✅ wenn verfügbar |
| XLM-R · sprachspezifische BERTs (gbert, camembert …) | Sprache | nach Datensprache wählen; multilingual kostet meist etwas Accuracy | je nach Datensatz |
| BioBERT / FinBERT / LegalBERT … | Domäne | nur bei echten Fachtexten spürbar | nur bei Fachdomäne |
| Longformer / BigBird | Kontextlänge | nur wenn Dokumente wirklich lang sind | bei Langtext |
| + **LoRA/PEFT** | Trainings-Sparmodus | Standard bei knapper GPU; minimaler Qualitätsverlust | ✅ als P3-Übung |

**P1 Vorbereiten:** Subword-Tokenisierung (modelleigener Tokenizer),
`max_length` bewusst wählen, Padding/Batches, Klassifikations-Head.

**P2 Bauen:** erster Lauf mit Standardwerten — meist Bestwert-Kandidat, aber
um Größenordnungen teurer als Track A.
*✓ Unterschied „Kopf auf eingefrorenen Features" (B) vs. „Netz auftauen" (C)?
Warum braucht C eine GPU?*

**P3 Optimieren — Stellschrauben:**

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Training | Learning Rate + Warmup (empfindlichste Schraube!) · Epochen + **Early Stopping am Val-Set** · Batch Size | divergiert oder overfittet |
| Kapazität | Layer-Freezing · **LoRA/PEFT** | GPU-Budget knapp; kleine Datenmengen |
| Input | `max_length` (128/256/512) | lange Texte; Speicher vs. Kontext |
| Modell | Achsen-Wahl · Klassengewichte im Loss | Sprache/Domäne; kleine Klassen |

*✓ Warum Early Stopping am Val-Set — was passierte am Testset?*

**P4 Einordnen:**

| Gewinnt bei | Verliert bei |
|---|---|
| Maximal-Accuracy · viel Labels · stabilen Kategorien · hohem Volumen | wenig Daten · wechselnden Kategorien · Erklärbarkeits-Pflicht · ohne GPU |

→ **Schwäche, die zu Track D führt:** Labels + GPU nötig; jede
Kategorie-Änderung = neu trainieren.

---

## 8 · Track D — LLM per Prompt

| Steckbrief | |
|---|---|
| Fachbegriff | zero-/few-shot prompting |
| Wissensquelle | Sprach- & Weltwissen: LLM · Kategorien: deine *Beschreibung* |
| Modelle | siehe Modell-Menü unten |
| Werkzeug | LLM-API + strukturierter JSON-Output |
| Quellen | Anthropic Classification-Guide; HF LLM Course |
| Rolle | Gegenprobe zum Finetuning: War der Aufwand es wert? (Wer streng „Baseline zuerst" lernen will: D vor C — beide Reihenfolgen gehen.) |

**Das Modell-Menü:**

| Modell | Einschätzung | probieren? |
|---|---|---|
| **kleines LLM** (Claude Haiku o. ä.) | billig, schnell; für Klassifikation meist ausreichend | ✅ Standard |
| mittleres LLM (Claude Sonnet o. ä.) | besser bei Grenzfällen + Begründungen; Kostenkurve messen! | ✅ Vergleichslauf |
| offenes LLM lokal (via Ollama) | offline/Datenschutz; schwächer, aber ohne API-Kosten | optional |
| feingetuntes kleines LLM | Spezialfall — wenn man hier landet, ist meist Track C die bessere Antwort | Einordnung kennen |

**P1 Vorbereiten — „Data Prep ist Prompt-Design":** Kategoriedefinitionen
schreiben (das eigentliche „Training"!), JSON-Output erzwingen,
Antwort-Validierung + Retries, festes Val-Sample definieren.

**P2 Bauen:** Zero-Shot auf einer Stichprobe, messen — ohne ein einziges
Trainingsbeispiel.
*✓ Zwei Situationen, in denen Prompting ein finegetuntes Modell schlägt,
obwohl es „schwächer" misst?*

**P3 Optimieren — Stellschrauben** (Stellrad ist Sprache, Zyklus identisch):

| Bereich | Schrauben | greift wenn … |
|---|---|---|
| Definitionen | fehleranalysegetrieben schärfen · Grenzfall-Regeln | systematische Verwechslungen |
| Beispiele | Few-Shot: zufällig vs. gezielte Grenzfälle | Definitionen reichen nicht |
| Modell | klein vs. groß — Kostenkurve mitzeichnen | Qualität vs. Preis |
| Robustheit | Konfidenz + „unsicher"-Option · Selbstkonsistenz (3×, Mehrheit) | Grenzfälle |

*✓ Am eigenen Protokoll zeigen: Prompt-Iteration = Hyperparameter-Tuning,
nur das Stellrad ist anders.*

**P4 Einordnen:**

| Gewinnt bei | Verliert bei |
|---|---|
| null Labels · wechselnden Kategorien · Begründungs-Bedarf · kleinen/mittleren Volumen | Massenvolumen (laufende Kosten) · harter Latenz · Offline-Pflicht |

→ **Schwäche, die zur Synthese führt:** Kosten/Latenz — Kombinationen lösen es.

---

## 9 · Synthese

### S1 — Gesamtvergleich

- Alle Testmessungen in **einer Tabelle**, ergänzt um die P4-Dimensionen:
  Labels nötig · Kosten/1.000 Texte · Setup-Aufwand · Erklärbarkeit ·
  „neue Kategorie hinzufügen"
- **Metrik-Rückblick** (schließt den CRISP-DM-Kreis): War die Leitmetrik
  richtig? Alles zusätzlich mit Macro-F1 — ändert sich die Rangfolge?
- **Messrauschen:** Konfidenzintervall der Testmenge kennen; Unterschiede
  darunter sind Rauschen. Schärfer: McNemar-Test (`KONZEPTE.md`).
- *✓ Welche zwei Zeilen unterscheiden sich NICHT signifikant — und was heißt
  das für die Wahl?*

### S2 — Mini-App: Hybrid-Classifier (Deployment)

*Lernübung für Deployment — kein Projektziel. Wer S2 weglässt, hat trotzdem
alle vier Lernziele erreichbar; wer sie baut, lernt den Betriebs-Aspekt.*

- **Kaskade:** billiger Track (A/B, **kalibrierte** Konfidenz) entscheidet
  klare Fälle; unsichere gehen ans LLM (D) mit Begründung.
- Anzeige: Kategorie · Konfidenz · Entscheidungsweg · Begründung.
- Konzepte: Konfidenz-Schwellen · Kosten-Qualitäts-Tradeoff · Modell
  speichern/laden · UI (Streamlit/CLI) · Daten-/Modell-Tests vor Deployment.
- Die App *ist* die Antwort auf S1: kein Gewinner — jeder Track für die
  Fälle, die er am besten/billigsten kann.

---

## 10 · Der Lernpfad — 12 Etappen

**Reihenfolge = Geschichte des Fachs**: Jede Etappe behebt die Schwäche der
vorherigen. Kern baut aufeinander auf; Kür vertieft, ohne dass Späteres
davon abhängt.

**Datensatz-Rotation (Lernziel 3):** Konzepte an mindestens zwei
*kontrastierenden* Datensätzen üben — z. B. Intent-Klassifikation (kurze
Sätze, sehr viele Klassen: banking77) und Themen-Klassifikation (lange
Artikel, wenige Klassen: 10kGNAD o. ä.). Der Kontrast ist selbst Lernstoff:
Bei Langtext tut das Kontextfenster weh, bei 77 Intents die Klassenzahl und
das Few-Shot-Regime; kNN-Köpfe und ComplementNB verhalten sich je nach
Datensatz völlig anders. **Was an beiden funktioniert, ist verstanden.**
Praktisch: erster Durchlauf komplett an einem Datensatz; danach gezielte
Wiederholung einzelner Etappen (mind. E0, E2, E5, E7) am Kontrast-Datensatz.

| # | Etappe | Familie | Art | Aufwand | Kern-Übung |
|---|---|---|---|---|---|
| E0 | Fundament (F1–F4) | — | Kern | 1–2 Tage | EDA-Checkliste · Val-Split · Experiment-Log |
| E1 | Regeln & Keywords | 1 | Kern (klein) | 2 Std. | Keyword-Classifier von Hand — bis es wehtut |
| E2 | Klassisches ML = Track A | 2 | Kern | 1–2 Tage | P1–P4 + mind. 3 P3-Runden |
| E3 | Word2Vec & fastText | 3 | Kür | ½–1 Tag | Analogien spielen · fastText vs. Track A |
| E4 | Museum: CNN/LSTM | 4 | Kür | 1–2 Tage | BiLSTM nachbauen — spüren, warum Transformer gewann |
| E5 | Embeddings = Track B | 6 | Kern | 1–2 Tage | P1–P4 + Few-Shot-Kurve + SetFit |
| E6 | Finetuning = Track C | 5 | Kern | 1–2 Tage | P1–P4 + LR-Dreier + LoRA |
| E7 | LLM = Track D | 7 | Kern | 1 Tag | P1–P4 + 3 Prompt-Versionen + Kostenrechnung |
| E8 | Kombinationen | Meta | Kern | 1 Tag | Ensemble (Soft-Voting) · Destillation (LLM labelt → klein lernt) |
| E9 | Gesamtvergleich = S1 | — | Kern | ½ Tag | Vergleichstabelle · Macro-F1 · Signifikanz |
| E10 | Mini-App = S2 | — | Kern | 1–2 Tage | Kaskade als Streamlit/CLI |
| E11 | Ausblick | — | Kür | offen | Multi-Label · Topic Modeling · Active Learning · MLOps |

*Nummerierungs-Hinweis: „Familie" (1–7) = Taxonomie aus Kap. 2;
Gen-Labels = Lehrbuch-Zoomstufe. Track B = Gen 2 = Familie 6 — gleiche Sache.*

### Die Etappen ohne eigenen Track — im Detail

Die Kern-Tracks (E2/E5/E6/E7) sind oben in Kap. 5–8 ausgearbeitet. Die
übrigen Etappen hier:

**E1 — Regeln & Keywords** *(Familie 1)*
- **Bauen:** Keyword-Classifier von Hand (~30 Zeilen): Wortlisten pro
  Klasse, meiste Treffer gewinnt. Zeitbudget: 2 Stunden Regel-Schreiben.
- **Konzepte:** Regeln vs. Lernen · warum Ausnahmen explodieren ·
  Regex/Wortlisten als Vorfilter-Handwerk (lebt in Kaskaden weiter).
- **✓ Checkpoint:** Beschreibe den Moment, in dem dir das Regel-Schreiben
  zu mühsam wurde — das *ist* das Argument für ML.

**E3 — Statische Wort-Embeddings & fastText** *(Familie 3, Kür)*
- **Modelle:** Word2Vec · GloVe (je: ein fester Vektor pro Wort) ·
  **fastText** (Subword-Einheiten; zugleich eigener, extrem schneller
  Classifier — Produktionsklassiker für Massendurchsatz).
- **Übung 1:** Mit Wortvektoren spielen: Analogien (König − Mann + Frau ≈ ?),
  nächste Nachbarn im Vokabular des eigenen Datensatzes.
- **Übung 2:** fastText-Classifier trainieren (eine CLI-Zeile) — Accuracy
  und Texte/Sekunde gegen Track A messen.
- **✓ Checkpoint:** Warum ist ein mehrdeutiges Wort („Bank") für statische
  Vektoren unlösbar — und wie lösen es die kontextuellen aus Track B?

**E4 — Museum: CNN & LSTM** *(Familie 4, Kür)*
- **Modelle:** Text-CNN (Kim 2014: Faltungsfilter über Wortfolgen) ·
  RNN/LSTM/GRU/**BiLSTM** (sequenzielles Lesen mit Gedächtnis) · Attention
  als Vorstufe des Transformers · ULMFiT/ELMo als BERT-Vorläufer.
- **Übung:** Ein BiLSTM (oder Text-CNN) mit statischen Embeddings auf dem
  eigenen Datensatz trainieren — inklusive der ganzen Mühsal: Padding,
  Sequenzlängen, langsames Training.
- **Warum Kür:** kostet Trainingsaufwand wie Track C, liefert Qualität
  zwischen A und B — 2026 fast nie die richtige Wahl. Der Lernwert ist
  historisch: *spüren*, welche Probleme der Transformer löste.
- **✓ Checkpoint:** Nenne die zwei Gründe, warum der Transformer die
  Sequenzmodelle verdrängte (Parallelisierung, Kontextzugriff) — belegt
  mit deiner eigenen Trainingszeit-Messung.

**E8 — Kombinationen** *(Meta-Familie)*
- **Übung 1 — Ensemble:** zwei Tracks per Soft-Voting kombinieren —
  schlägt es beide Einzelmodelle? Was kostet es?
- **Übung 2 — Destillation:** LLM (Track D) labelt ~1.000 ungelabelte
  Texte, ein kleines Modell (A/B) lernt daraus — wie nah kommt es dem LLM
  bei ~0 laufenden Kosten? (Der Standardweg vom Prototyp zur Produktion.)
- **Konzepte:** Voting/Stacking · Kaskaden (→ S2) · Destillation ·
  Regel-Vorfilter (E1 kehrt zurück).
- **✓ Checkpoint:** Welche Kombination baust du für 10 Mio. Texte/Monat —
  und warum genau diese?

**E11 — Ausblick** *(Kür)*: Multi-Label & hierarchische Klassifikation ·
Topic Modeling (BERTopic) für den Fall ohne Kategorien · Active Learning ·
MLOps/Monitoring — jedes ein eigenes kleines Projekt; Überblick in Kap. 2
(„Angrenzende Felder").

**Der rote Faden (Schwäche → nächste Etappe):**
Raten (E0) → Regeln skalieren nicht (E1) → ML kennt keine Bedeutung (E2) →
statische Vektoren kennen keinen Kontext (E3) → Sequenzmodelle sind mühsam
(E4) → eingefrorene Encoder passen sich nicht an (E5) → Finetuning ist teuer
und starr (E6) → Prompting kostet laufend (E7) → Kombinationen (E8) →
Vergleich (E9) → Produkt (E10).

---

## 11 · Leitplanken

1. **Erst die Baseline, dann eskalieren.**
2. **Test-Split bleibt zu** — Varianten vergleicht man am Val-Set.
3. **Metrik vor Modell** — erst definieren, was „gut" heißt.
4. **Fehleranalyse schlägt Modellwechsel.**
5. **Kosten & Änderbarkeit zählen, nicht nur Accuracy.**
6. **Innere vor äußerer Schleife** — erst den Track ausreizen (P3).
7. **Eine Änderung pro Messung.**

---

## 12 · Quellen

| Quelle | wofür | wo |
|---|---|---|
| Jurafsky & Martin, SLP3 | Lehrbuch-Kanon (Kap. 2, 4, 5, 7) | web.stanford.edu/~jurafsky/slp3 |
| Stanford CS224N | der kanonische NLP-Kurs | web.stanford.edu/class/cs224n |
| Hugging Face NLP/LLM Course | Praxis Track C/D | huggingface.co/learn |
| Made With ML (MLOps Course) | Handwerks-Stränge, EDA, Tracking | madewithml.com |
| Ng, ML Yearning | Fehleranalyse, Val/Test-Disziplin (F3) | frei verfügbar |
| Google, Rules of ML | Baseline-first, Iteration | developers.google.com/machine-learning/guides/rules-of-ml |
| Google ML Crash Course | Bias/Variance, Learning Curves (F4) | developers.google.com/machine-learning/crash-course |
| scikit-learn Guides | Track A, Tuning, Kalibrierung | scikit-learn.org |
| sbert.net + SetFit | Track B | sbert.net · huggingface.co/docs/setfit |
| Anthropic Classification-Guide | Track D | docs.claude.com |
| CRISP-DM | Prozessmodell | Suchbegriff „CRISP-DM" |

---

## 13 · Glossar

**Accuracy** – Anteil korrekter Vorhersagen. **Baseline** – trivialer
Vergleichswert. **CRISP-DM** – Standard-Prozessmodell für ML-Projekte.
**Data Leakage** – Testinformation sickert ins Training. **Early Stopping** –
Training beenden, sobald die Val-Leistung nicht mehr steigt. **EDA** –
explorative Datenanalyse. **Embedding** – Text als Bedeutungsvektor.
**Epoche** – ein Durchlauf durch die Trainingsdaten. **F1** – harmonisches
Mittel aus Precision und Recall. **Few-Shot** – Beispiele im Prompt statt
Training. **Finetuning** – vortrainiertes Netz nachtrainieren.
**Grid Search** – systematisches Durchprobieren von Hyperparametern.
**Ground Truth** – die „wahren" Labels. **Hyperparameter** – Einstellungen,
die *du* wählst. **Kalibrierung** – Konfidenzen ehrlich machen.
**Kontextfenster** – wie viele Tokens ein Modell maximal liest.
**Kreuzvalidierung** – k-faches Rotieren des Val-Splits. **LoRA/PEFT** –
Finetuning light. **n-Gramm** – Folge von n Wörtern/Zeichen als Feature.
**Overfitting** – auswendig gelernt statt Muster. **Precision/Recall** –
Zuverlässigkeit eines Tipps / Vollständigkeit des Findens. **SetFit** –
Encoder mit wenigen Beispielen kontrastiv nachschärfen. **Stratifizierung** –
Split mit gleichen Klassenanteilen. **TF-IDF** – häufig im Dokument, selten
im Korpus = wichtig. **Transfer Learning** – fremdes Vorwissen nutzen.
**Validation-Set** – drittes Set für Tuning-Entscheidungen. **Zero-Shot** –
klassifizieren nur aus Kategoriebeschreibungen.
